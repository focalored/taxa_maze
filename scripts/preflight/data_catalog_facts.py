#!/usr/bin/env python
"""Preflight data checks for specs/pilot1.md: the ToL-10M catalog, the shard id
lists, and the BioCLIP 1 ToL-EOL cache id lists (checks C1-C6 and C8-C12).

Read-only inputs:
  /u/liv/bdbk/data/tol10m/metadata/catalog.csv
  /projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/
      ids.txt, vocab.json, shards_index.json, shards/image_set_NN.ids.txt

Outputs (all under /projects/bdbk/liv/repos/taxa_maze/audit/preflight/):
  data_catalog_facts.json      every number, keyed by check id
  data_shard_composition.csv   per-shard table (C5)
  data_cache_missing_ids.csv   catalog EOL train/val ids absent from the cache (C6)
  data_species_top100.csv      largest species by N_s (C9)
  (with --smoke every name gets a _smoke suffix)

Definitions used throughout:
  * A rank value is MISSING if it is null, the empty string, or whitespace only.
    Present values are compared after str.strip().
  * A row is COMPLETE if none of the seven rank columns is missing.
  * EOL rows are rows with a non-missing eol_content_id. The local shards hold
    EOL images only, so EOL rows are the training/validation population.
  * Taxon keys are full lineage prefixes joined by '|', e.g.
    'Animalia|Chordata|Mammalia'. The species key uses the raw epithet:
    'kingdom|phylum|class|order|family|genus|epithet'.
  * The cache's vocab.json writes the species element as the binomial
    'Genus epithet'; we build that form only when comparing against vocab.json.

Run through Slurm on the cpu partition; see the job log for timing.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
CACHE = "/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1"
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
SEP = "|"
BATCH = 8192
KS = (8, 16)
SPEC = {"train": 5_908_775, "val": 310_899}

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def blank(c):
    return pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")


def to_py(o):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def key_exprs(prefix="k_"):
    """Full-lineage prefix key for every rank (null if any element is null)."""
    return [
        pl.concat_str([pl.col(c) for c in RANKS[: i + 1]], separator=SEP).alias(f"{prefix}{r}")
        for i, r in enumerate(RANKS)
    ]


def raw_csv_scan(path):
    """Independent record count with the csv module (catches embedded newlines
    and ragged rows that polars would silently pad)."""
    n_lines = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 26)
            if not b:
                break
            n_lines += b.count(b"\n")
    field_hist = {}
    n_rec = 0
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for row in rd:
            n_rec += 1
            k = len(row)
            field_hist[k] = field_hist.get(k, 0) + 1
    return {"n_newlines": n_lines, "n_header_fields": len(header),
            "n_records_csv_module": n_rec, "fields_per_record_hist": field_hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2%% hash sample of catalog ids and 2 shards only; outputs get a _smoke suffix")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    F = {"script": os.path.abspath(__file__), "argv": sys.argv, "catalog": CATALOG, "cache": CACHE,
         "definitions": {
             "missing": "null, empty string, or whitespace-only after str.strip()",
             "complete": "all of kingdom..species present",
             "eol_row": "eol_content_id present",
             "taxon_key": "'|'-joined full lineage prefix; species element = raw epithet",
         }}

    # ------------------------------------------------------------------ load
    cols = ["split", "treeoflife_id", "eol_content_id", "bioscan_filename", "inat21_filename"] + RANKS
    df = pl.read_csv(CATALOG, columns=cols, infer_schema=False)
    log("catalog loaded", df.shape)
    if args.smoke:  # ~2% of ids; all rows of a kept id stay together
        df = df.filter(pl.col("treeoflife_id").hash(seed=0) % 50 == 0)
        log("smoke sample", df.shape)
    F["catalog_shape_polars"] = list(df.shape)
    if not args.smoke:
        F["catalog_raw_scan"] = raw_csv_scan(CATALOG)
        log("raw csv scan", F["catalog_raw_scan"])

    # Missing-value kinds and odd tokens, before cleaning.
    kinds = {}
    for c in RANKS + ["split", "treeoflife_id", "eol_content_id"]:
        s = df[c]
        st = s.str.strip_chars()
        kinds[c] = {
            "null": int(s.is_null().sum()),
            "empty_string": int((s == "").sum()),
            "whitespace_only": int(((s != "") & (st == "")).sum()),
            "padded_nonblank": int(((st != s) & (st != "")).sum()),
        }
    F["value_kinds_before_cleaning"] = kinds
    toks = ["nan", "none", "null", "na", "n/a", "unknown", "incertae sedis", "unclassified",
            "undefined", "-", "?", "sp", "sp.", "spp", "spp.", "xx", "x"]
    odd = {}
    for c in RANKS:
        low = df[c].str.strip_chars().str.to_lowercase()
        vc = {t: int((low == t).sum()) for t in toks}
        odd[c] = {t: v for t, v in vc.items() if v}
    F["placeholder_tokens_rows"] = odd

    # Clean.
    df = df.with_columns(
        [pl.when(blank(c)).then(None).otherwise(pl.col(c).str.strip_chars()).alias(c) for c in RANKS]
        + [(~blank("eol_content_id")).alias("is_eol"),
           (~blank("inat21_filename")).alias("is_inat"),
           (~blank("bioscan_filename")).alias("is_bioscan"),
           pl.col("treeoflife_id").str.strip_chars().alias("treeoflife_id")]
    )
    df = df.with_columns(
        pl.concat_str([pl.when(pl.col(c).is_null()).then(pl.lit("-")).otherwise(pl.lit("x"))
                       for c in RANKS]).alias("pattern")
    ).with_columns((pl.col("pattern") == "xxxxxxx").alias("complete"))
    log("cleaned")
    eol = df.filter(pl.col("is_eol"))

    # ------------------------------------------------------------------ C1
    def split_table(frame):
        t = frame.group_by("split").agg(pl.len().alias("rows"),
                                        pl.col("treeoflife_id").n_unique().alias("distinct_ids"))
        return {r["split"]: {"rows": r["rows"], "distinct_ids": r["distinct_ids"]}
                for r in t.sort("split").to_dicts()}

    src = (df.group_by(["split", "is_eol", "is_inat", "is_bioscan"]).len()
           .sort(["split", "len"], descending=[False, True]).to_dicts())
    F["C1"] = {"all_rows": split_table(df), "eol_rows": split_table(eol),
               "n_rows_total": df.height, "n_eol_rows_total": eol.height,
               "source_flags_by_split": src, "spec_values": SPEC}
    log("C1", F["C1"]["all_rows"], F["C1"]["eol_rows"])

    # ------------------------------------------------------------------ C2, C3
    def idset(frame, s):
        return frame.filter(pl.col("split") == s).select("treeoflife_id").unique()

    F["C2"], F["C3"] = {}, {}
    for scope, frame in (("all", df), ("eol", eol)):
        tr, ts, va = idset(frame, "train"), idset(frame, "train_small"), idset(frame, "val")
        ts_tr = ts.join(tr, on="treeoflife_id", how="semi").height
        ts_va = ts.join(va, on="treeoflife_id", how="semi").height
        ts_only = ts.join(tr, on="treeoflife_id", how="anti").join(va, on="treeoflife_id", how="anti")
        F["C2"][scope] = {
            "train_small_distinct_ids": ts.height,
            "train_distinct_ids": tr.height,
            "train_small_ids_also_train": ts_tr,
            "train_small_ids_also_val": ts_va,
            "train_small_ids_in_neither": ts_only.height,
            "train_small_only_examples": ts_only.head(5)["treeoflife_id"].to_list(),
            "train_small_over_train": ts.height / max(tr.height, 1),
        }
        c3 = {}
        for s in ("train", "val", "train_small"):
            sub = frame.filter(pl.col("split") == s)
            vc = sub.group_by("treeoflife_id").len().filter(pl.col("len") > 1)
            c3[s] = {"rows": sub.height, "distinct_ids": sub["treeoflife_id"].n_unique(),
                     "ids_with_more_than_one_row": vc.height,
                     "examples": vc.head(5).to_dicts()}
        c3["train_and_val_shared_ids"] = tr.join(va, on="treeoflife_id", how="semi").height
        c3["train_and_val_shared_examples"] = (tr.join(va, on="treeoflife_id", how="semi")
                                               .head(5)["treeoflife_id"].to_list())
        F["C3"][scope] = c3
    # Do the duplicated train_small rows carry the same lineage and source as their train row?
    a = df.filter(pl.col("split") == "train_small").select(["treeoflife_id", "is_eol"] + RANKS)
    b = df.filter(pl.col("split") == "train").select(["treeoflife_id", "is_eol"] + RANKS)
    j = a.join(b, on="treeoflife_id", how="inner", suffix="_tr")
    mism = j.filter(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(c + "_tr"))
                                       for c in RANKS + ["is_eol"]]))
    F["C2"]["train_small_vs_train_row_pairs"] = j.height
    F["C2"]["train_small_vs_train_lineage_or_source_mismatches"] = mism.height
    log("C2", F["C2"]["eol"], "C3", F["C3"]["eol"]["train_and_val_shared_ids"])

    # ------------------------------------------------------------------ C4
    def pattern_table(frame):
        res = {}
        for s in sorted(frame["split"].unique().to_list()):
            sub = frame.filter(pl.col("split") == s)
            pt = sub.group_by("pattern").len().sort("len", descending=True).to_dicts()
            n_inc = sum(r["len"] for r in pt if r["pattern"] != "xxxxxxx")
            res[s] = {
                "rows": sub.height,
                "incomplete_rows": n_inc,
                "complete_rows": sub.height - n_inc,
                "complete_distinct_ids": sub.filter(pl.col("complete"))["treeoflife_id"].n_unique(),
                "patterns": [{"pattern": r["pattern"], "rows": r["len"],
                              "missing": [RANKS[i] for i, ch in enumerate(r["pattern"]) if ch == "-"]}
                             for r in pt],
            }
        return res

    F["C4"] = {"pattern_legend": "x = present, - = missing, in order " + ",".join(RANKS),
               "eol": pattern_table(eol), "all": pattern_table(df)}
    # Rows whose present ranks are not a prefix (a gap followed by a present rank).
    F["C4"]["eol_rows_with_gap"] = int(eol.filter(
        ~pl.col("pattern").str.contains(r"^x*-*$")).height)
    log("C4 eol train/val complete:",
        F["C4"]["eol"].get("train", {}).get("complete_rows"),
        F["C4"]["eol"].get("val", {}).get("complete_rows"))

    # ------------------------------------------------------------------ per-id flags
    flags = df.group_by("treeoflife_id").agg(
        (pl.col("split") == "train").any().alias("has_train"),
        (pl.col("split") == "val").any().alias("has_val"),
        (pl.col("split") == "train_small").any().alias("has_ts"),
        pl.col("is_eol").any().alias("has_eol"),
        ((pl.col("split") == "train") & pl.col("is_eol") & pl.col("complete")).any().alias("tr_complete"),
        ((pl.col("split") == "val") & pl.col("is_eol") & pl.col("complete")).any().alias("va_complete"),
        pl.len().alias("n_cat_rows"),
    )
    log("flags", flags.height)

    # ------------------------------------------------------------------ shard ids, cache ids
    idx = json.load(open(f"{CACHE}/shards_index.json"))
    if args.smoke:
        idx = [idx[59], idx[62]]
    parts, shard_lines = [], {}
    for e in idx:
        with open(f"{CACHE}/shards/{e['shard']}.ids.txt") as fh:
            u = [ln.strip() for ln in fh if ln.strip()]
        shard_lines[e["shard"]] = len(u)
        parts.append(pl.DataFrame({"shard": [e["shard"]] * len(u), "uuid": u,
                                   "pos": np.arange(len(u), dtype=np.int64)}))
    sh = pl.concat(parts)
    with open(f"{CACHE}/ids.txt") as fh:
        cache_ids = [ln.strip() for ln in fh if ln.strip()]
    ids_chk = {
        "n_cache_ids_txt_lines": len(cache_ids),
        "n_cache_ids_distinct": len(set(cache_ids)),
        "n_shard_uuid_total": sh.height,
        "n_shard_uuid_distinct": sh["uuid"].n_unique(),
        "shard_lines_equal_index_n_rows": all(shard_lines[e["shard"]] == e["n_rows"] for e in idx),
    }
    if not args.smoke:
        ids_chk["concat_shard_ids_equals_cache_ids_txt"] = (sh["uuid"].to_list() == cache_ids)
    log("ids", ids_chk)

    # ------------------------------------------------------------------ C5
    shj = sh.join(flags, left_on="uuid", right_on="treeoflife_id", how="left")
    f = lambda c: pl.col(c).fill_null(False)
    per = shj.group_by("shard").agg(
        pl.len().alias("n_ids"),
        f("has_train").sum().alias("n_train"),
        f("has_val").sum().alias("n_val"),
        (f("has_ts") & ~f("has_train") & ~f("has_val")).sum().alias("n_train_small_only"),
        pl.col("n_cat_rows").is_null().sum().alias("n_not_in_catalog"),
        (f("has_train") & f("has_val")).sum().alias("n_train_and_val"),
        (pl.col("n_cat_rows").is_not_null() & ~f("has_eol")).sum().alias("n_in_catalog_non_eol"),
        f("tr_complete").sum().alias("n_train_complete"),
        f("va_complete").sum().alias("n_val_complete"),
    ).sort("shard")
    per.write_csv(f"{OUT}/data_shard_composition{sfx}.csv")
    rows = per.to_dicts()
    kind = {}
    for r in rows:
        if r["n_train"] == r["n_ids"]:
            kind[r["shard"]] = "all_train"
        elif r["n_val"] == r["n_ids"]:
            kind[r["shard"]] = "all_val"
        else:
            kind[r["shard"]] = "mixed"
    s60 = shj.filter(pl.col("shard") == "image_set_60")
    pos60 = {}
    if s60.height:
        ptr = s60.filter(f("has_train"))["pos"].to_numpy()
        pva = s60.filter(f("has_val"))["pos"].to_numpy()
        pos60 = {"train_pos_min": int(ptr.min()) if len(ptr) else None,
                 "train_pos_max": int(ptr.max()) if len(ptr) else None,
                 "val_pos_min": int(pva.min()) if len(pva) else None,
                 "val_pos_max": int(pva.max()) if len(pva) else None,
                 "train_positions_contiguous": bool(len(ptr) and ptr.max() - ptr.min() + 1 == len(ptr))}
    totals = {k: int(per[k].sum()) for k in per.columns if k != "shard"}
    F["C5"] = {"per_shard": rows, "shard_kind": kind, "totals": totals,
               "shard_60_positions": pos60, "ids_checks": ids_chk}
    log("C5 totals", totals)

    # ------------------------------------------------------------------ C6
    E = eol.filter(pl.col("split").is_in(["train", "val"])).select("treeoflife_id").unique()
    C = pl.DataFrame({"treeoflife_id": cache_ids}).unique()
    missing = E.join(C, on="treeoflife_id", how="anti")
    extra = C.join(E, on="treeoflife_id", how="anti")
    mrows = (eol.filter(pl.col("split").is_in(["train", "val"]))
             .join(missing, on="treeoflife_id", how="semi")
             .select(["treeoflife_id", "split", "eol_content_id", "pattern"] + RANKS)
             .sort(["split", "treeoflife_id"]))
    mrows.write_csv(f"{OUT}/data_cache_missing_ids{sfx}.csv")
    ex = extra.join(flags, on="treeoflife_id", how="left")
    F["C6"] = {
        "catalog_eol_train_val_distinct_ids": E.height,
        "cache_distinct_ids": C.height,
        "catalog_ids_missing_from_cache": missing.height,
        "missing_by_split": mrows.group_by("split").len().sort("split").to_dicts(),
        "missing_by_split_and_complete": mrows.group_by(["split", pl.col("pattern") == "xxxxxxx"])
        .len().sort("split").to_dicts(),
        "missing_examples": mrows.head(8).to_dicts(),
        "cache_ids_not_in_catalog_eol_train_val": extra.height,
        "of_which_not_in_catalog": int(ex["n_cat_rows"].is_null().sum()),
        "of_which_train_small_only": int(ex.select(
            (f("has_ts") & ~f("has_train") & ~f("has_val")).sum()).item()),
        "of_which_non_eol": int(ex.select((pl.col("n_cat_rows").is_not_null() & ~f("has_eol")).sum()).item()),
        "extra_examples": ex.head(5).to_dicts(),
    }
    log("C6", {k: v for k, v in F["C6"].items() if "examples" not in k})

    # ------------------------------------------------------------------ populations
    T = eol.filter((pl.col("split") == "train") & pl.col("complete")).with_columns(key_exprs())
    V = eol.filter((pl.col("split") == "val") & pl.col("complete")).with_columns(key_exprs())
    Tc = T.join(C, on="treeoflife_id", how="semi")
    Vc = V.join(C, on="treeoflife_id", how="semi")
    TV = pl.concat([T, V])
    F["populations"] = {"T_eol_train_complete_rows": T.height,
                        "T_distinct_ids": T["treeoflife_id"].n_unique(),
                        "V_eol_val_complete_rows": V.height,
                        "V_distinct_ids": V["treeoflife_id"].n_unique(),
                        "T_in_cache": Tc.height, "V_in_cache": Vc.height}
    log("populations", F["populations"])

    # ------------------------------------------------------------------ C8
    def taxa_counts(frame):
        return {r: frame[f"k_{r}"].n_unique() for r in RANKS}

    F["C8"] = {"train_complete": taxa_counts(T),
               "train_complete_by_name_only": {r: T[r].n_unique() for r in RANKS},
               "train_complete_in_cache": taxa_counts(Tc),
               "val_complete": taxa_counts(V),
               "train_val_complete": taxa_counts(TV),
               "kingdoms_train_complete": T.group_by("kingdom").len().sort("len", descending=True).to_dicts()}

    # vocab.json comparison
    voc = json.load(open(f"{CACHE}/vocab.json"))
    vocab = voc["vocab"]
    F["C8"]["vocab_ranks"] = voc.get("ranks")
    F["C8"]["vocab_sep"] = voc.get("sep")
    F["C8"]["vocab_list_lengths"] = {r: len(vocab[r]) for r in RANKS}
    F["C8"]["vocab_list_distinct"] = {r: len(set(vocab[r])) for r in RANKS}

    def vocab_form_sets(frame, truncate):
        """Prefix sets in vocab.json's convention (species element = 'Genus epithet').
        truncate=True: a row contributes rank r if ranks 0..r are all present."""
        out = {}
        for i, r in enumerate(RANKS):
            sub = frame.filter(pl.col("pattern").str.starts_with("x" * (i + 1))) if truncate else \
                frame.filter(pl.col("complete"))
            if r == "species":
                e = pl.concat_str([pl.col(c) for c in RANKS[:6]]
                                  + [pl.concat_str([pl.col("genus"), pl.col("species")], separator=" ")],
                                  separator=SEP)
            else:
                e = pl.concat_str([pl.col(c) for c in RANKS[: i + 1]], separator=SEP)
            out[r] = set(sub.select(e.alias("k"))["k"].unique().to_list())
        return out

    eol_tr = eol.filter(pl.col("split") == "train")
    eol_tv = eol.filter(pl.col("split").is_in(["train", "val"]))
    cache_rows = eol_tv.join(C, on="treeoflife_id", how="semi")
    cands = {
        "train_complete": vocab_form_sets(eol_tr, False),
        "train_val_complete": vocab_form_sets(eol_tv, False),
        "train_truncated_at_gap": vocab_form_sets(eol_tr, True),
        "train_val_truncated_at_gap": vocab_form_sets(eol_tv, True),
        "cache_ids_truncated_at_gap": vocab_form_sets(cache_rows, True),
    }
    cmp_ = {}
    for name, sets in cands.items():
        cmp_[name] = {}
        for r in RANKS:
            vs = set(vocab[r])
            cs = sets[r]
            cmp_[name][r] = {"n_candidate": len(cs), "n_vocab": len(vs),
                             "vocab_minus_candidate": len(vs - cs),
                             "candidate_minus_vocab": len(cs - vs),
                             "vocab_minus_candidate_examples": sorted(vs - cs)[:3],
                             "candidate_minus_vocab_examples": sorted(cs - vs)[:3]}
        cmp_[name]["exact_match_all_ranks"] = all(
            cmp_[name][r]["vocab_minus_candidate"] == 0 and cmp_[name][r]["candidate_minus_vocab"] == 0
            for r in RANKS)
    F["C8"]["vocab_vs_candidates"] = cmp_
    del cands
    log("C8", F["C8"]["train_complete"], F["C8"]["vocab_list_lengths"],
        {k: v["exact_match_all_ranks"] for k, v in cmp_.items()})

    # ------------------------------------------------------------------ C9
    def species_sizes(frame, tag):
        ns = frame.group_by("k_species").len().rename({"len": "N_s"}).sort("N_s", descending=True)
        arr = ns["N_s"].to_numpy().astype(np.int64)
        N = int(arr.sum())
        steps = N // BATCH
        res = {"population": tag, "n_images": N, "n_species": int(len(arr)),
               "N_s_max": int(arr.max()), "N_s_mean": float(arr.mean()),
               "N_s_median": float(np.median(arr)),
               "N_s_p90": float(np.percentile(arr, 90)), "N_s_p99": float(np.percentile(arr, 99)),
               "N_s_p99_9": float(np.percentile(arr, 99.9)),
               "percentile_method": "numpy default (linear interpolation)",
               "n_species_N_s_eq_1": int((arr == 1).sum()),
               "n_species_N_s_le_8": int((arr <= 8).sum()),
               "n_species_N_s_le_16": int((arr <= 16).sum()),
               "top10": ns.head(10).to_dicts(),
               "global_batch": BATCH, "steps_per_epoch": int(steps), "K": {}}
        for K in KS:
            g = (arr + K - 1) // K
            G = int(g.sum())
            q, rem = arr // g, arr % g
            single = int(np.where(q == 1, g - rem, 0).sum())
            sizes = np.bincount(q, weights=g - rem, minlength=K + 2) + \
                np.bincount(q + 1, weights=rem, minlength=K + 2)
            over = g > steps
            res["K"][str(K)] = {
                "G": G, "mean_group_size": N / G,
                "groups_with_one_image": single, "share_groups_with_one_image": single / G,
                "group_size_hist": {str(i): int(sizes[i]) for i in range(1, K + 1) if sizes[i]},
                "n_species_single_group": int((g == 1).sum()),
                "images_in_single_group_species": int(arr[g == 1].sum()),
                "max_groups_per_species": int(g.max()),
                "n_species_groups_exceed_steps_per_epoch": int(over.sum()),
                "images_in_species_exceeding": int(arr[over].sum()),
                "groups_per_batch_mean": G / max(steps, 1),
            }
        return res, ns

    c9, ns = species_sizes(T, "EOL train, complete lineage (catalog)")
    c9c, _ = species_sizes(Tc, "EOL train, complete lineage, present in BioCLIP 1 cache")
    ns.head(100).write_csv(f"{OUT}/data_species_top100{sfx}.csv")
    F["C9"] = {"catalog": c9, "in_cache": c9c}
    log("C9", {k: v for k, v in c9.items() if k != "top10"})

    # ------------------------------------------------------------------ C10
    sp = T.select([f"k_{r}" for r in RANKS]).unique()
    c10 = {"n_species": sp.height, "per_rank": {}, "penalty_terms": {}, "monitor_pairs": {}}
    tot_S = tot_C = 0
    ch = {}
    for i in range(6):
        par, chi = f"k_{RANKS[i]}", f"k_{RANKS[i + 1]}"
        t = sp.group_by(par).agg(pl.col(chi).n_unique().alias("n_child"), pl.len().alias("n_species"))
        ch[RANKS[i]] = t
        c10["per_rank"][RANKS[i]] = {
            "n_taxa": t.height,
            "one_child_taxon": int((t["n_child"] == 1).sum()),
            "one_species_n_a_eq_1": int((t["n_species"] == 1).sum()),
            "one_child_but_ge2_species": int(((t["n_child"] == 1) & (t["n_species"] >= 2)).sum()),
        }
        terms = sp.select(par).join(t, on=par, how="left")
        keep_S = int((terms["n_species"] >= 2).sum())
        keep_C = int((terms["n_child"] >= 2).sum())
        c10["penalty_terms"][RANKS[i]] = {"possible": sp.height,
                                          "kept_rule_S_n_a_ge_2": keep_S,
                                          "kept_rule_C_n_child_ge_2": keep_C,
                                          "kept_by_S_dropped_by_C": keep_S - keep_C}
        tot_S += keep_S
        tot_C += keep_C
    c10["penalty_terms"]["total"] = {"possible": 6 * sp.height, "kept_rule_S": tot_S,
                                     "kept_rule_C": tot_C, "difference": tot_S - tot_C}
    # |P_s| distribution: number of kept ancestor terms per species under each rule.
    kS = np.zeros(sp.height, np.int64)
    kC = np.zeros(sp.height, np.int64)
    for i in range(6):
        par = f"k_{RANKS[i]}"
        t = sp.select(par).join(ch[RANKS[i]], on=par, how="left", maintain_order="left")
        kS += (t["n_species"].to_numpy() >= 2)
        kC += (t["n_child"].to_numpy() >= 2)
    c10["kept_terms_per_species_hist"] = {
        "rule_S": {str(k): int((kS == k).sum()) for k in range(7)},
        "rule_C": {str(k): int((kC == k).sum()) for k in range(7)},
    }
    # Genus-level single-child taxa = genera with one species, by definition.
    # Bank monitor pairs (spec line 181), p=1..5: (rank p node, its rank p+1 child).
    for p in range(5):
        par, chi = RANKS[p], RANKS[p + 1]
        pairs = sp.select([f"k_{par}", f"k_{chi}"]).unique()
        pairs = pairs.join(ch[par].select([f"k_{par}", pl.col("n_child").alias("par_n_child")]),
                           on=f"k_{par}", how="left")
        pairs = pairs.join(ch[chi].select([f"k_{chi}", pl.col("n_child").alias("chi_n_child")]),
                           on=f"k_{chi}", how="left")
        c10["monitor_pairs"][f"{par}->{chi}"] = {
            "n_pairs": pairs.height,
            "kept_parent_has_ge2_children": int((pairs["par_n_child"] >= 2).sum()),
            "kept_parent_and_child_have_ge2_children":
                int(((pairs["par_n_child"] >= 2) & (pairs["chi_n_child"] >= 2)).sum()),
        }
    F["C10"] = c10
    log("C10", c10["per_rank"], c10["penalty_terms"]["total"])

    # ------------------------------------------------------------------ C11
    def homonyms(frame):
        res = {}
        for i in range(1, 7):
            r = RANKS[i]
            par = f"k_{RANKS[i - 1]}"
            u = frame.select([par, r]).unique()
            u = u.with_columns(pl.col(par).str.split(SEP).list.last().alias("par_name"))
            h = (u.group_by(r).agg(pl.col(par).n_unique().alias("n_parents"),
                                   pl.col("par_name").n_unique().alias("n_parent_names"),
                                   pl.col(par).sort().head(3).alias("parents"))
                 .filter(pl.col("n_parents") > 1).sort("n_parents", descending=True))
            res[r] = {"n_names_total": frame[r].n_unique(), "n_names_under_gt1_parent": h.height,
                      # same immediate-parent name, different path above it: one taxon whose
                      # upper lineage is labelled inconsistently, not a true homonym
                      "n_same_parent_name_diff_upper_path": int((h["n_parent_names"] == 1).sum()),
                      "n_diff_parent_names": int((h["n_parent_names"] > 1).sum()),
                      "examples": h.head(3).to_dicts()}
        # Binomial 'Genus epithet' under more than one full genus prefix.
        u = frame.select([pl.concat_str([pl.col("genus"), pl.col("species")], separator=" ").alias("binomial"),
                          "k_genus"]).unique()
        h = (u.group_by("binomial").agg(pl.col("k_genus").n_unique().alias("n_parents"),
                                         pl.col("k_genus").sort().head(3).alias("parents"))
             .filter(pl.col("n_parents") > 1).sort("n_parents", descending=True))
        res["species_binomial"] = {"n_binomials_total": u["binomial"].n_unique(),
                                   "n_binomials_under_gt1_genus_prefix": h.height,
                                   "examples": h.head(3).to_dicts()}
        return res

    F["C11"] = {"train_complete": homonyms(T), "train_val_complete": homonyms(TV)}
    log("C11", {r: v.get("n_names_under_gt1_parent", v.get("n_binomials_under_gt1_genus_prefix"))
                for r, v in F["C11"]["train_complete"].items()})

    # ------------------------------------------------------------------ C12
    c12 = {}
    for r in RANKS:
        vk = V.group_by(f"k_{r}").len()
        absent = vk.join(T.select(f"k_{r}").unique(), on=f"k_{r}", how="anti")
        c12[r] = {"val_taxa": vk.height, "val_taxa_absent_from_train": absent.height,
                  "val_images_in_absent_taxa": int(absent["len"].sum()),
                  "examples": absent.sort("len", descending=True).head(3).to_dicts()}
    c12["val_complete_images"] = V.height
    F["C12"] = c12
    log("C12 species", c12["species"]["val_taxa_absent_from_train"],
        c12["species"]["val_images_in_absent_taxa"])

    # ------------------------------------------------------------------ epithet notes
    eps = T.select("species").unique()["species"]
    F["epithet_notes_train_complete"] = {
        "n_distinct_epithets": len(eps),
        "multiword": int(eps.str.contains(" ").sum()),
        "has_uppercase": int(eps.str.contains("[A-Z]").sum()),
        "has_digit": int(eps.str.contains("[0-9]").sum()),
        "len_le_2": int((eps.str.len_chars() <= 2).sum()),
        "multiword_examples": eps.filter(eps.str.contains(" ")).head(5).to_list(),
        "rows_with_multiword_epithet": int(T["species"].str.contains(" ").sum()),
    }

    F["elapsed_s"] = time.time() - T0
    with open(f"{OUT}/data_catalog_facts{sfx}.json", "w") as fh:
        json.dump(to_py(F), fh, indent=1, default=str)
    log("wrote", f"{OUT}/data_catalog_facts{sfx}.json")


if __name__ == "__main__":
    main()
