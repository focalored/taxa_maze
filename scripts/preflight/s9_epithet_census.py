#!/usr/bin/env python
"""S9 census: put every species key of the complete-lineage ToL-10M EOL train/val rows into one category
by its raw epithet (P placeholder, A author/year, S infraspecific, O other multi-word, C clean), check the
author calls against EOL taxon.tab and GBIF, and compute what dropping or merging them changes.
Run: srun <cpu flags> --output=<repo>/audit/preflight/s9_census_%j.log bash s9_run.sh s9_epithet_census.py [--no-gbif]
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import os
import random
import re
import socket
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

import numpy as np
import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
TAXON = "/u/liv/bdbk/data/tol10m/metadata/taxon.tab"
HF_META = "/u/liv/bdbk/data/tol10m/.cache/huggingface/download/metadata/catalog.csv.metadata"
INAT_VAL = "/u/liv/bdbk/data/inat21/meta/val.json"
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
GBIF_CACHE = f"{OUT}/s9_gbif_cache.json"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
B = 8192
KS = (8, 16)
CATS = ["P", "A", "S", "O", "C"]
T0 = time.time()

# Whole-epithet placeholders. Matching is on the raw (already lowercase) string.
PH_WHOLE = {"sp", "sp.", "spp", "spp.", "ssp", "ssp.", "subsp", "subsp.", "sp.nov", "sp.nov.", "sp. nov.",
            "nov", "nov.", "n.sp.", "n. sp.", "x", "×", "xx", "xxx", "indet", "indet.", "unknown",
            "unidentified", "undetermined", "undet.", "cf", "cf.", "aff", "aff.", "nr", "nr.", "gen.", "spec.",
            "species", "ss.", "s.l.", "s.s.", "complex", "group", "hybrid", "cv", "cv.", "var", "var.", "f.",
            "form", "morph", "sect.", "subg.", "ser.", "ined.", "na", "none", "null", "-"}
# First tokens that mean "species not identified" (qualifier, rank marker, cultivar-only, morph label).
P_LEAD = {"sp", "sp.", "spp", "spp.", "ssp.", "cf", "cf.", "aff", "aff.", "nr", "nr.", "near", "nov.", "sp.nov",
          "n.sp.", "indet.", "subg.", "subgen.", "sect.", "subsect.", "ser.", "gen.", "morph", "cv", "cv.", "ss.",
          "-"}
P_INNER = {"sp", "sp.", "spp", "spp.", "prob."}
S_MARK = {"ssp", "ssp.", "subsp", "subsp.", "var", "var.", "f.", "fo.", "forma", "form", "ab.", "ab", "race",
          "nothosubsp.", "nothovar.", "subvar.", "lusus", "morph"}
O_MARK = {"group": "cultivar_or_species_group", "complex": "complex_or_aggregate", "agg.": "complex_or_aggregate",
          "agg": "complex_or_aggregate", "cv.": "cultivar", "cv": "cultivar", "nom.": "provisional_name",
          "prov.": "provisional_name", "comb.": "provisional_name", "ined.": "provisional_name",
          "strain": "strain_or_virus", "virus": "strain_or_virus", "pv.": "strain_or_virus"}
UNKNOWN_WORDS = {"unknown", "unidentified", "unident.", "indet", "indet.", "indeterminate", "undetermined", "undet."}
NOT_AUTHOR_DOT = set(S_MARK) | set(O_MARK) | PH_WHOLE | P_LEAD | P_INNER
CONNECT = {"ex", "et", "al.", "al", "in", "and", "&", "von", "van", "der", "den", "de", "du", "la", "le", "da",
           "di", "del", "della", "dos", "das", "y", "auct.", "non", "sensu", "emend.", "hort."}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fold(s):
    s = s.replace("’", "'").replace("‘", "'")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


# ---------------------------------------------------------------- EOL taxon.tab evidence
def load_taxon_evidence():
    tx = pl.read_csv(TAXON, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True,
                     columns=["taxonRank", "canonicalName", "authority"])
    tx = tx.with_row_index("rid")
    log("taxon.tab", tx.shape)
    # Author tokens: lowercased authority strings without years and separators; dotted tokens also
    # contribute their dotted parts ("barb.rodr." -> "barb.", "rodr.").
    au = (tx.filter(pl.col("authority").is_not_null())
          .select("rid", pl.col("authority").str.to_lowercase().str.replace_all(r"[’‘]", "'")
                  .str.replace_all(r"\d+", " ").str.replace_all(r"[()\[\],;&:\"]", " ")
                  .str.split(" ").alias("t"))
          .explode("t").filter(pl.col("t").str.len_chars() > 0))
    parts = (au.filter(pl.col("t").str.contains(r"\.[^.]"))
             .with_columns(pl.col("t").str.split(".")).explode("t")
             .filter(pl.col("t").str.len_chars() > 0).with_columns(pl.col("t") + "."))
    apos = (au.filter(pl.col("t").str.contains("'")).with_columns(pl.col("t").str.split("'")).explode("t")
            .filter(pl.col("t").str.len_chars() > 1))
    au = pl.concat([au, parts, apos]).unique()
    ut = au.select("t").unique()
    ut = ut.with_columns(pl.col("t").map_elements(fold, return_dtype=pl.String).alias("ft"))
    au = au.join(ut, on="t").select("rid", "ft").unique()
    auth_count = dict(au.group_by("ft").len().iter_rows())
    # Genus authorities: author tokens of every genus-rank row, keyed by the folded genus name.
    gr = tx.filter(pl.col("taxonRank") == "genus").select("rid", pl.col("canonicalName").alias("g"))
    ga = au.join(gr, on="rid").with_columns(pl.col("g").map_elements(fold, return_dtype=pl.String))
    genus_auth = {}
    for g, t in ga.select("g", "ft").iter_rows():
        genus_auth.setdefault(g, set()).add(t)
    # Epithet usage, binomials and trinomials from canonical names with two or more words.
    cn = (tx.filter(pl.col("canonicalName").is_not_null())
          .select(pl.col("canonicalName").str.to_lowercase().str.split(" ").alias("w"))
          .filter(pl.col("w").list.len() >= 2))
    ep = cn.select(pl.col("w").list.slice(1).alias("e")).explode("e").filter(pl.col("e").str.len_chars() > 0)
    ue = ep.select("e").unique().with_columns(pl.col("e").map_elements(fold, return_dtype=pl.String).alias("fe"))
    epi_count = dict(ep.join(ue, on="e").group_by("fe").len().iter_rows())
    bi = cn.select(pl.col("w").list.slice(0, 2).list.join(" ").alias("b")).unique()
    bi = bi.with_columns(pl.col("b").map_elements(fold, return_dtype=pl.String))
    tri = (cn.filter(pl.col("w").list.len() >= 3)
           .select(pl.col("w").list.slice(0, 3).list.join(" ").alias("b")).unique())
    tri = tri.with_columns(pl.col("b").map_elements(fold, return_dtype=pl.String))
    ev = {"auth_count": auth_count, "epi_count": epi_count, "genus_auth": genus_auth,
          "binomials": set(bi["b"].to_list()), "trinomials": set(tri["b"].to_list()),
          "n_rows": tx.height, "n_authority": int(tx["authority"].is_not_null().sum()),
          "n_genus_rows": gr.height}
    log("taxon evidence: author tokens", len(auth_count), "epithet tokens", len(epi_count),
        "genera with authority", len(genus_auth), "binomials", len(ev["binomials"]), "trinomials", len(ev["trinomials"]))
    return ev


# ---------------------------------------------------------------- classifier
def author_reason(t, pos, g, ev, last=False):
    """Why token t (at position pos of the epithet) is author or year material, else None."""
    ft = fold(t)
    if re.search(r"\d", t):
        return "digit"
    if re.search(r"[()\[\]&,;]", t):
        return "punctuation"
    if pos >= 1 and t in CONNECT:
        return "author_connector"
    if t.endswith("-") and len(t) > 1:
        return "truncated_author"
    if "'" in ft and not (t.startswith("'") or t.endswith("'")):
        return "apostrophe_name"
    if "." in t and t not in NOT_AUTHOR_DOT:
        return "abbreviation"
    ac, ec = ev["auth_count"].get(ft, 0), ev["epi_count"].get(ft, 0)
    if ft in ev["genus_auth"].get(g, ()):
        if pos >= 1 or f"{g} {ft}" not in ev["binomials"]:
            return "genus_authority"
    if ac >= 2 and ec == 0 and (pos >= 1 or f"{g} {ft}" not in ev["binomials"]):
        return "author_name"
    if pos >= 1 and ac >= 5 and ac >= 20 * ec:
        return "author_name_mostly"
    sx = ev.get("suffix", {}).get(t)
    if pos >= 1 and last and sx and ((sx[1] >= 3 and ac >= 1 and ac >= ec) or (sx[1] >= 5 and ec <= 2)):
        return "author_name_catalog_suffix"
    return None


def classify(genus, s, ev):
    """Return (category, subtype, info) for a raw epithet s under genus."""
    g = fold(genus)
    toks = s.split(" ")
    info = {}
    if s in PH_WHOLE:
        return "P", "bare_token", info
    if re.fullmatch(r"\d+", s):
        return "P", "bare_number", info
    if "?" in s:
        return "P", "question_mark", info
    if re.match(r"(undescribed|undefined)", s) or s in UNKNOWN_WORDS:
        return "P", "undescribed_or_undefined", info
    if toks[0] in P_LEAD or s.startswith("'") or s.startswith("- "):
        return "P", "qualified_or_rank_marker", {"lead": toks[0]}
    if "_or_" in s or " or " in s:
        return "P", "either_or", info
    if any(t in P_INNER for t in toks[1:]):
        return "P", "inner_sp_or_prob", info
    if re.search(r"(taxonomy|geo):", s):
        return "O", "machine_tag", info
    if re.fullmatch(r"\d+-[a-z]+", s):
        return "O", "numeral_epithet", info
    if re.search(r"(virus|viroid|phytoplasma)", s) or any(t in ("strain", "pv.", "bv.") for t in toks):
        return "O", "strain_or_virus", info
    if "=" in toks or "_x_" in s or any(t in ("x", "×") for t in toks[1:-1]) or \
            (len(toks) >= 3 and toks[-1] in ("x", "×")):
        return "O", "hybrid_formula", info
    hybrid = False
    if toks[0] in ("x", "×") and len(toks) >= 2:
        toks = ["×" + toks[1]] + toks[2:]
        hybrid = True
    elif toks[0].startswith("×"):
        hybrid = True
    info["hybrid"] = hybrid
    core0 = toks[0].lstrip("×")
    # A: author or year material.
    r0 = author_reason(core0, 0, g, ev) if core0 else None
    if r0:
        return "A", ("author_only_single" if len(toks) == 1 else "author_only_multi"), {**info, "reason": r0}
    for j in range(1, len(toks)):
        r = author_reason(toks[j], j, g, ev, last=(j == len(toks) - 1))
        if r:
            return "A", "epithet_plus_author", {**info, "reason": r, "epithet_tokens": toks[:j]}
    if len(toks) == 1:
        if "_" in s:
            return "O", "underscore_label", info
        if "'" in s:
            return "O", "stray_quote", info
        return "C", ("hybrid_nothospecies" if hybrid else "single_word"), info
    for t in toks[1:]:
        if t in O_MARK:
            return "O", O_MARK[t], info
        if t.startswith("'") or t.endswith("'"):
            return "O", "cultivar", info
    if any(t in S_MARK for t in toks[1:]):
        return "S", "with_rank_marker", info
    if all(re.fullmatch(r"[^\W\d_]+(-[^\W\d_]+)*", t.lstrip("×")) for t in toks):
        if len(toks) == 2:
            return "S", "two_words", info
        return "O", "three_plus_plain_words", info
    return "O", "other", info


def gbif_get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(1.5 * (k + 1))
    return {"error": err}


def gbif_query(q):
    name, rank, kingdom = q
    p = {"name": name, "strict": "true"}
    if rank:
        p["rank"] = rank
    if kingdom:
        p["kingdom"] = kingdom
    d = gbif_get("https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode(p))
    keep = ["matchType", "rank", "status", "confidence", "scientificName", "canonicalName", "usageKey",
            "kingdom", "genus", "error"]
    return {k: d.get(k) for k in keep if k in d}


def run_gbif(queries, use_net):
    cache = json.load(open(GBIF_CACHE)) if os.path.exists(GBIF_CACHE) else {}
    todo = [q for q in queries if "|".join(map(str, q)) not in cache]
    log("gbif queries", len(queries), "cached", len(queries) - len(todo), "to fetch", len(todo) if use_net else 0)
    if use_net and todo:
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for q, res in zip(todo, ex.map(gbif_query, todo)):
                if res.get("error"):
                    log("gbif error", q, res["error"])
                    continue
                res["queried_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                cache["|".join(map(str, q))] = res
        with open(GBIF_CACHE, "w") as fh:
            json.dump(cache, fh, indent=0, ensure_ascii=False)
    return {q: cache.get("|".join(map(str, q))) for q in queries}


def gbif_is_species(res):
    return bool(res) and res.get("matchType") == "EXACT" and res.get("rank") in (
        "SPECIES", "SUBSPECIES", "VARIETY", "FORM", "INFRASPECIFIC_NAME") and res.get("status") in ("ACCEPTED", "SYNONYM")


def authorship_tokens(res):
    if not res or not res.get("scientificName") or not res.get("canonicalName"):
        return set()
    a = res["scientificName"][len(res["canonicalName"]):]
    a = re.sub(r"[()\[\],;&:\d]", " ", a.lower())
    return {fold(t) for t in a.split() if t}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gbif", action="store_true", help="use only cached GBIF answers")
    ap.add_argument("--n-gbif-single", type=int, default=100000, help="single-word author calls sent to GBIF")
    args = ap.parse_args()
    R = {"meta": {"script": os.path.abspath(__file__), "argv": sys.argv, "host": socket.gethostname(),
                  "slurm_job": os.environ.get("SLURM_JOB_ID"), "start_utc": datetime.datetime.utcnow().isoformat(),
                  "inputs": {"catalog": CATALOG, "taxon_tab": TAXON, "inat21_val": INAT_VAL}}}
    R["meta"]["sha256"] = {"catalog": sha256(CATALOG), "taxon_tab": sha256(TAXON)}
    R["meta"]["hf_download_metadata_catalog"] = open(HF_META).read().split()
    log("sha256", R["meta"]["sha256"])

    # ------------------------------------------------------------ catalog
    cols = ["split", "treeoflife_id", "eol_content_id", "inat21_filename", "bioscan_filename"] + RANKS
    df = pl.read_csv(CATALOG, columns=cols, infer_schema=False)
    log("catalog", df.shape)
    src = (pl.when(pl.col("eol_content_id").is_not_null()).then(pl.lit("eol"))
           .when(pl.col("inat21_filename").is_not_null()).then(pl.lit("inat21"))
           .when(pl.col("bioscan_filename").is_not_null()).then(pl.lit("bioscan")).otherwise(pl.lit("none")))
    df = df.with_columns(src.alias("source"),
                         pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]).alias("complete"))
    sp = pl.col("species")
    casing = (df.filter(sp.is_not_null()).group_by(["source", "split"])
              .agg(pl.len().alias("rows_with_species"),
                   sp.str.contains(r"^\p{Lu}").sum().alias("starts_uppercase"),
                   sp.str.contains(r"\p{Lu}").sum().alias("any_uppercase"),
                   (sp != sp.str.strip_chars()).sum().alias("leading_or_trailing_space"),
                   sp.str.contains(r"\s\s").sum().alias("double_space"),
                   sp.str.contains(" ").sum().alias("contains_space"),
                   sp.str.contains(r"[^\x00-\x7F]").sum().alias("non_ascii"))
              .sort(["source", "split"]))
    R["casing_by_source_split"] = casing.to_dicts()
    R["genus_casing_eol_complete"] = {
        "starts_uppercase": int(df.filter(pl.col("complete") & (pl.col("source") == "eol"))["genus"]
                                .str.contains(r"^\p{Lu}").sum()),
        "rows": int(df.filter(pl.col("complete") & (pl.col("source") == "eol")).height)}
    full = df.filter(pl.col("complete"))
    R["full_label_counts"] = {
        "all_rows_all_splits": int(full.height),
        "excluding_train_small": int(full.filter(pl.col("split") != "train_small").height),
        "by_split": dict(full.group_by("split").len().iter_rows()),
        "card_quote": "There are 8,455,243 images with full taxonomic labels. (README.md, Data Instances)"}
    E = df.filter((pl.col("source") == "eol") & pl.col("split").is_in(["train", "val"]))
    pat = pl.concat_str([pl.when(pl.col(c).is_null()).then(pl.lit("-")).otherwise(pl.lit("x")) for c in RANKS])
    E = E.with_columns(pat.alias("pattern"))
    R["eol_train_patterns"] = {
        "rows": int(E.filter(pl.col("split") == "train").height),
        "complete": int(E.filter((pl.col("split") == "train") & pl.col("complete")).height),
        "genus_level_rows_species_missing_only": int(E.filter((pl.col("split") == "train") & (pl.col("pattern") == "xxxxxx-")).height),
        "other_incomplete": int(E.filter((pl.col("split") == "train") & ~pl.col("complete") & (pl.col("pattern") != "xxxxxx-")).height)}
    TV = (E.filter(pl.col("complete"))
          .with_columns(pl.concat_str([pl.col(c) for c in RANKS], separator="|").alias("key"),
                        pl.concat_str([pl.col(c) for c in RANKS[:6]], separator="|").alias("gkey")))
    T = TV.filter(pl.col("split") == "train")
    V = TV.filter(pl.col("split") == "val")
    log("T", T.height, "V", V.height)
    R["populations"] = {"T_eol_train_complete": T.height, "V_eol_val_complete": V.height,
                        "key_format": "kingdom|phylum|class|order|family|genus|species, raw strings, same as the preflight (no value has padding, so raw == stripped)"}

    # ------------------------------------------------------------ classify distinct (genus, epithet)
    ev = load_taxon_evidence()
    R["taxon_tab"] = {"rows": ev["n_rows"], "authority_non_null": ev["n_authority"], "genus_rows": ev["n_genus_rows"],
                      "distinct_author_tokens": len(ev["auth_count"]), "distinct_epithet_tokens": len(ev["epi_count"]),
                      "binomials": len(ev["binomials"]), "trinomials": len(ev["trinomials"])}
    mw = (TV.select("species").unique().filter(pl.col("species").str.contains(" "))
          .with_columns(pl.col("species").str.split(" ").alias("w"))
          .with_columns(pl.col("w").list.last().alias("last"), pl.col("w").list.first().alias("first")))
    sfx = mw.group_by("last").agg(pl.len().alias("n_epithets"), pl.col("first").n_unique().alias("n_first"))
    ev["suffix"] = {t: (n, f) for t, n, f in sfx.iter_rows()}
    dbg = ["kaila", "hita", "branstetter", "beau", "graef", "weld", "cazanavelle", "magnus", "mérida-rivas", "curtis",
           "dognin", "minor", "alba", "occidentalis", "intermedia", "domestica", "signatus", "italicus", "roosevelti",
           "nannodes", "peleides", "qinlingensis", "barley", "bernardi", "orbigny"]
    R["debug_token_stats"] = {t: {"auth_count": ev["auth_count"].get(fold(t), 0), "epi_count": ev["epi_count"].get(fold(t), 0),
                                  "catalog_suffix": ev["suffix"].get(t)} for t in dbg}
    log("debug token stats", R["debug_token_stats"])
    pairs = TV.group_by(["genus", "species"]).agg(
        (pl.col("split") == "train").sum().alias("n_tr"), (pl.col("split") == "val").sum().alias("n_va"),
        pl.col("kingdom").first().alias("kingdom"))
    rows = []
    for genus, s, n_tr, n_va, kingdom in pairs.select("genus", "species", "n_tr", "n_va", "kingdom").iter_rows():
        cat, sub, info = classify(genus, s, ev)
        g = fold(genus)
        first = s.split(" ")[0] if not s.startswith(("x ", "× ")) else "×" + s.split(" ")[1]
        ft = fold(first.lstrip("×"))
        rows.append({"genus": genus, "species": s, "cat": cat, "sub": sub, "reason": info.get("reason"),
                     "hybrid": bool(info.get("hybrid")), "kingdom": kingdom,
                     "epithet_part": " ".join(info["epithet_tokens"]) if "epithet_tokens" in info else None,
                     "first_word": first, "n_tr_pair": n_tr, "n_va_pair": n_va,
                     "auth_count": ev["auth_count"].get(ft, 0), "epi_count": ev["epi_count"].get(ft, 0),
                     "first_in_genus_authority": ft in ev["genus_auth"].get(g, ()),
                     "binomial_in_eol_dh": f"{g} {ft}" in ev["binomials"],
                     "trinomial_in_eol_dh": len(s.split(" ")) >= 2 and fold(" ".join([genus] + s.split(" ")[:2])) in ev["trinomials"]})
    cl = pl.DataFrame(rows, infer_schema_length=None)
    log("classified pairs", cl.height, cl.group_by("cat").len().sort("cat").to_dicts())

    # ------------------------------------------------------------ GBIF checks
    single_a = (cl.filter((pl.col("cat") == "A") & (pl.col("sub") == "author_only_single")
                          & pl.col("reason").is_in(["genus_authority", "author_name"]))
                .sort(["n_tr_pair", "genus", "species"], descending=[True, False, False]))
    weak = (cl.filter((pl.col("cat") == "C") & (pl.col("sub") == "single_word") & ~pl.col("binomial_in_eol_dh")
                      & (pl.col("auth_count") >= 5) & (pl.col("auth_count") >= 20 * pl.col("epi_count")))
            .sort(["n_tr_pair", "genus", "species"], descending=[True, False, False]))
    rnd = random.Random(0)
    c_single = cl.filter((pl.col("cat") == "C") & (pl.col("sub") == "single_word")).sort(["genus", "species"])
    def pick(frame, n):
        idx = sorted(rnd.sample(range(frame.height), min(n, frame.height)))
        return frame.with_row_index("_i").filter(pl.col("_i").is_in(idx)).drop("_i")

    c_sample = pick(c_single, 200)
    s_all = cl.filter(pl.col("cat") == "S").sort(["n_tr_pair", "genus", "species"], descending=[True, False, False])
    s_sample = pl.concat([s_all.head(100), pick(s_all.slice(100), 100)])
    a1 = (cl.filter((pl.col("cat") == "A") & (pl.col("sub") == "epithet_plus_author"))
          .sort(["n_tr_pair", "genus", "species"], descending=[True, False, False]))
    a1_sample = pl.concat([a1.head(100), pick(a1.slice(100), 100)])
    groups = {"A_single_all": single_a.head(args.n_gbif_single), "C_weak_author_candidates": weak,
              "C_random_single_word": c_sample, "S_top100_plus_random100": s_sample,
              "A_epithet_plus_author_top100_plus_random100": a1_sample}
    queries = set()
    for name, g in groups.items():
        for r in g.iter_rows(named=True):
            if name.startswith("S_"):
                queries.add((f"{r['genus']} {r['species']}", None, r["kingdom"]))
                queries.add((f"{r['genus']} {r['first_word']}", None, r["kingdom"]))
            elif name.startswith("A_epithet"):
                queries.add((f"{r['genus']} {r['epithet_part'].split(' ')[0].lstrip('×')}", None, r["kingdom"]))
            else:
                queries.add((f"{r['genus']} {r['species']}", None, r["kingdom"]))
                queries.add((r["genus"], "GENUS", r["kingdom"]))
    gb = run_gbif(sorted(queries, key=lambda q: tuple(map(str, q))), not args.no_gbif)
    checks, overrides = {}, []
    for name, g in groups.items():
        out = []
        for r in g.iter_rows(named=True):
            if name.startswith("S_"):
                full_ = gb.get((f"{r['genus']} {r['species']}", None, r["kingdom"]))
                base = gb.get((f"{r['genus']} {r['first_word']}", None, r["kingdom"]))
                out.append({**{k: r[k] for k in ("genus", "species", "n_tr_pair", "n_va_pair", "trinomial_in_eol_dh")},
                            "gbif_full": full_, "gbif_first_word": base,
                            "full_is_infraspecific_name": gbif_is_species(full_) and full_.get("rank") != "SPECIES",
                            "first_word_is_species": gbif_is_species(base)})
                continue
            if name.startswith("A_epithet"):
                e0 = r["epithet_part"].split(" ")[0].lstrip("×")
                res = gb.get((f"{r['genus']} {e0}", None, r["kingdom"]))
                out.append({**{k: r[k] for k in ("genus", "species", "epithet_part", "reason", "n_tr_pair", "n_va_pair")},
                            "gbif_epithet_part": res, "epithet_part_is_species": gbif_is_species(res)})
                continue
            res = gb.get((f"{r['genus']} {r['species']}", None, r["kingdom"]))
            gen = gb.get((r["genus"], "GENUS", r["kingdom"]))
            in_auth = fold(r["species"]) in authorship_tokens(gen)
            rec = {**{k: r[k] for k in ("genus", "species", "kingdom", "n_tr_pair", "n_va_pair", "reason", "auth_count",
                                        "epi_count", "first_in_genus_authority", "binomial_in_eol_dh")},
                   "gbif_binomial": res, "gbif_binomial_is_species": gbif_is_species(res),
                   "gbif_genus": gen, "epithet_in_gbif_genus_authorship": in_auth}
            out.append(rec)
            if name == "A_single_all" and rec["gbif_binomial_is_species"]:
                overrides.append((r["genus"], r["species"], "C", "single_word", "gbif_exact_species"))
            if name == "C_weak_author_candidates" and not rec["gbif_binomial_is_species"] and in_auth:
                overrides.append((r["genus"], r["species"], "A", "author_only_single", "gbif_genus_authority"))
        checks[name] = out
    R["gbif"] = {"endpoint": "https://api.gbif.org/v1/species/match?name=...&strict=true[&rank=GENUS]&kingdom=...",
                 "cache_file": GBIF_CACHE, "n_queries": len(queries),
                 "rule": "real species = EXACT match at SPECIES/SUBSPECIES/VARIETY/FORM with status ACCEPTED or SYNONYM",
                 "checks": checks,
                 "overrides": [dict(zip(["genus", "species", "new_cat", "new_sub", "why"], o)) for o in overrides]}
    for gname, gs, ncat, nsub, why in overrides:
        m = (pl.col("genus") == gname) & (pl.col("species") == gs)
        cl = cl.with_columns(pl.when(m).then(pl.lit(ncat)).otherwise(pl.col("cat")).alias("cat"),
                             pl.when(m).then(pl.lit(nsub)).otherwise(pl.col("sub")).alias("sub"),
                             pl.when(m).then(pl.lit(why)).otherwise(pl.col("reason")).alias("reason"))
    log("overrides", len(overrides), "after:", cl.group_by("cat").len().sort("cat").to_dicts())

    # ------------------------------------------------------------ per-key tables
    attach = cl.select("genus", "species", "cat", "sub", "reason", "hybrid", "epithet_part", "first_word")
    T = T.join(attach, on=["genus", "species"], how="left")
    V = V.join(attach, on=["genus", "species"], how="left")
    assert T["cat"].null_count() == 0 and V["cat"].null_count() == 0
    kt = T.group_by("key").agg(pl.len().alias("n_tr"), pl.col("cat", "sub", "reason", "species", "gkey",
                                                                   "epithet_part", "first_word").first())
    kv = V.group_by("key").agg(pl.len().alias("n_va"), pl.col("cat", "sub", "reason", "species", "gkey",
                                                                   "epithet_part", "first_word").first())
    keys = (kt.join(kv.select("key", "n_va"), on="key", how="full", coalesce=True)
            .with_columns(pl.col("n_tr").fill_null(0), pl.col("n_va").fill_null(0)))
    meta = pl.concat([kt.select("key", "cat", "sub", "reason", "species", "gkey", "epithet_part", "first_word"),
                      kv.select("key", "cat", "sub", "reason", "species", "gkey", "epithet_part", "first_word")]).unique("key")
    keys = keys.select("key", "n_tr", "n_va").join(meta, on="key")

    def top(frame, n=20, by="n_tr"):
        return frame.sort([by, "key"], descending=[True, False]).head(n).select(
            "key", "species", "sub", "reason", "n_tr", "n_va").to_dicts()

    cats = {}
    for c in CATS:
        f = keys.filter(pl.col("cat") == c)
        subs = (f.group_by("sub").agg(pl.len().alias("n_keys"), (pl.col("n_tr") > 0).sum().alias("n_train_keys"),
                                      (pl.col("n_va") > 0).sum().alias("n_val_keys"),
                                      pl.col("n_tr").sum().alias("train_images"), pl.col("n_va").sum().alias("val_images"))
                .sort("train_images", descending=True).to_dicts())
        cats[c] = {"n_keys_union": f.height, "n_train_keys": int((f["n_tr"] > 0).sum()),
                   "n_val_keys": int((f["n_va"] > 0).sum()), "train_images": int(f["n_tr"].sum()),
                   "val_images": int(f["n_va"].sum()), "by_subtype": subs, "top20_by_train_images": top(f),
                   "top10_by_subtype": {sb: top(f.filter(pl.col("sub") == sb), 10) for sb in f["sub"].unique().sort().to_list()},
                   "top20_by_val_images": top(f, by="n_va") if c != "C" else None}
        if c in ("A", "P"):
            cats[c]["by_reason_or_token"] = (f.with_columns(pl.when(pl.col("cat") == "P").then(pl.col("species"))
                                                            .otherwise(pl.col("reason")).alias("rt"))
                                             .group_by("rt").agg(pl.len().alias("n_keys"), pl.col("n_tr").sum().alias("train_images"),
                                                                  pl.col("n_va").sum().alias("val_images"))
                                             .sort("train_images", descending=True).head(60).to_dicts())
    R["categories"] = cats
    R["category_rules"] = {
        "order": "P, then O (machine tag, numeral epithet, hybrid formula), then A, then C for one word, then O markers, then S, then O",
        "P": "whole string in PH_WHOLE, a bare number, contains '?', starts with undescribed/undefined/unknown/unident/indet, first token in P_LEAD or a leading quote, '_or_'/' or ', or an inner sp./spp./prob.",
        "A_suffix_rule": "a last word (after the first) that ends >= 3 catalog epithets with >= 3 different first words and is used at least as often as an EOL author as an EOL epithet, or ends >= 5 such epithets and is an EOL epithet at most twice",
        "A": "a token with a digit, ()[]&,; an author connector (ex, et, al., in, and, de, le, ...) after the first word, a trailing hyphen, an apostrophe name, a dotted abbreviation that is not a rank marker, a surname from the EOL taxon.tab authority column that is never an epithet there (>= 2 authority rows, 0 epithet uses), a token of the genus's own EOL authority, or after the first word a surname used >= 20 times more as an author than as an epithet. For the first word, the author tests apply only when 'Genus word' is not an EOL canonical name.",
        "S": "two or more plain words with no author token, or any words with ssp/subsp/var/f./forma/form/ab./race/morph",
        "O": "machine tags, numeral epithets (7-punctata), hybrid formulas (a x b =), cultivar/group/agg./complex/nom. prov./strain/virus labels, three or more plain words, underscores, stray quotes",
        "C": "one plain word (including named hybrids such as ×sinensis or 'x hispanica')",
        "sets": {"PH_WHOLE": sorted(PH_WHOLE), "P_LEAD": sorted(P_LEAD), "P_INNER": sorted(P_INNER),
                 "S_MARK": sorted(S_MARK), "O_MARK": O_MARK, "CONNECT": sorted(CONNECT)}}
    R["P_exact_tokens"] = (keys.filter(pl.col("cat") == "P").group_by("species")
                           .agg(pl.len().alias("n_keys"), pl.col("n_tr").sum().alias("train_images"),
                                pl.col("n_va").sum().alias("val_images"))
                           .sort("train_images", descending=True).to_dicts())
    single = keys.filter(pl.col("sub") == "author_only_single").sort("n_tr", descending=True)
    R["A_single_word_top200"] = (single.head(200).join(cl.select("genus", "species", "auth_count", "epi_count",
                                                                 "first_in_genus_authority", "binomial_in_eol_dh"),
                                                       left_on=[pl.col("gkey").str.split("|").list.last(), "species"],
                                                       right_on=["genus", "species"], how="left")
                                 .select("key", "species", "reason", "n_tr", "n_va", "auth_count", "epi_count",
                                         "first_in_genus_authority", "binomial_in_eol_dh").to_dicts())

    # ------------------------------------------------------------ effects on train
    def sizes(frame):
        ns = frame.group_by("key").len().rename({"len": "N_s"}).sort(["N_s", "key"], descending=[True, False])
        arr = ns["N_s"].to_numpy().astype(np.int64)
        N = int(arr.sum())
        st_f, st_c = N // B, math.ceil(N / B)
        res = {"rows": N, "species_keys": int(len(arr)), "largest": ns.head(5).to_dicts(),
               "steps_per_epoch_floor": st_f, "steps_per_epoch_ceil": st_c, "K": {}}
        for K in KS:
            g = (arr + K - 1) // K
            res["K"][str(K)] = {"max_groups": int(g.max()), "total_groups": int(g.sum()),
                                "species_groups_gt_steps_floor": int((g > st_f).sum()),
                                "species_groups_gt_steps_ceil": int((g > st_c).sum()),
                                "those_species": ns.filter(pl.Series(g > st_f)).head(10).to_dicts()}
        return res

    eff = {"baseline_keep_all": sizes(T)}
    scen = {"drop_P": ["P"], "drop_P_and_A": ["P", "A"]}
    for name, cs in scen.items():
        eff[name] = sizes(T.filter(~pl.col("cat").is_in(cs)))
        eff[name]["train_rows_dropped"] = int(T.filter(pl.col("cat").is_in(cs)).height)
        eff[name]["val_rows_dropped"] = int(V.filter(pl.col("cat").is_in(cs)).height)
    keepA1 = ~pl.col("cat").is_in(["P", "A"]) | (pl.col("sub") == "epithet_plus_author")
    eff["drop_P_and_author_only_A_keep_epithet_plus_author"] = sizes(T.filter(keepA1))
    eff["drop_P_and_author_only_A_keep_epithet_plus_author"]["train_rows_dropped"] = int(T.filter(~keepA1).height)
    eff["drop_P_and_author_only_A_keep_epithet_plus_author"]["val_rows_dropped"] = int(V.filter(~keepA1).height)

    # Higher-rank taxa that lose every image, and penalty terms that exist only through a P/A sibling.
    Tk = T.filter(~pl.col("cat").is_in(["P", "A"]))
    lost = {}
    for i, r in enumerate(RANKS[:6]):
        pre = pl.concat_str([pl.col(c) for c in RANKS[:i + 1]], separator="|")
        a = set(T.select(pre.alias("k"))["k"].unique().to_list())
        b = set(Tk.select(pre.alias("k"))["k"].unique().to_list())
        lost[r] = {"before": len(a), "after": len(b), "lost": len(a - b), "examples": sorted(a - b)[:8]}
    eff["drop_P_and_A_taxa_lost_by_rank"] = lost
    iv_ = json.load(open(INAT_VAL))["categories"]
    inat_prefix = {RANKS[i]: {"|".join([c_[r_] for r_ in ["kingdom", "phylum", "class", "order", "family", "genus"][:i + 1]])
                              for c_ in iv_} for i in range(6)}
    inat_name = {RANKS[i]: {c_[["kingdom", "phylum", "class", "order", "family", "genus"][i]] for c_ in iv_} for i in range(6)}
    Tk0 = T.filter(keepA1)
    lost0, ov = {}, {}
    for i, r in enumerate(RANKS[:6]):
        pre = pl.concat_str([pl.col(c) for c in RANKS[:i + 1]], separator="|")
        a_ = set(T.select(pre.alias("k"))["k"].unique().to_list())
        b_ = set(Tk0.select(pre.alias("k"))["k"].unique().to_list())
        lost0[r] = {"before": len(a_), "after": len(b_), "lost": len(a_ - b_), "examples": sorted(a_ - b_)[:8]}
        la = {k for k in a_ - set(Tk.select(pre.alias("k"))["k"].unique().to_list())}
        ov[r] = {"lost_under_drop_P_and_A": len(la),
                 "of_which_full_prefix_in_inat21_val": len(la & inat_prefix[r]),
                 "of_which_name_in_inat21_val": len({k.split("|")[-1] for k in la} & inat_name[r]),
                 "examples_full_prefix": sorted(la & inat_prefix[r])[:10],
                 "lost_under_drop_P_and_author_only_A": len(a_ - b_),
                 "of_which_full_prefix_in_inat21_val_P_A0": len((a_ - b_) & inat_prefix[r]),
                 "examples_full_prefix_P_A0": sorted((a_ - b_) & inat_prefix[r])[:10],
                 "inat21_val_species_under_lost_taxa_P_and_A": sum(1 for c_ in iv_ if "|".join(
                     [c_[r_] for r_ in ["kingdom", "phylum", "class", "order", "family", "genus"][:i + 1]]) in la),
                 "inat21_val_species_under_lost_taxa_P_and_A0": sum(1 for c_ in iv_ if "|".join(
                     [c_[r_] for r_ in ["kingdom", "phylum", "class", "order", "family", "genus"][:i + 1]]) in (a_ - b_))}
    eff["drop_P_and_author_only_A_taxa_lost_by_rank"] = lost0
    eff["lost_taxa_vs_inat21_val"] = ov
    pseudo = T.filter(pl.col("cat").is_in(["P"]) | pl.col("sub").is_in(["author_only_single", "author_only_multi"]))
    realg = (T.filter(~(pl.col("cat").is_in(["P"]) | pl.col("sub").is_in(["author_only_single", "author_only_multi"])))
             .group_by("gkey").agg(pl.len().alias("real_images"), pl.col("key").n_unique().alias("real_species")))
    pg = (pseudo.group_by("gkey").agg(pl.len().alias("pseudo_images"), pl.col("key").n_unique().alias("pseudo_keys"))
          .join(realg, on="gkey", how="left").with_columns(pl.col("real_images").fill_null(0), pl.col("real_species").fill_null(0)))
    vb = V.filter(pl.col("cat").is_in(["P"]) | pl.col("sub").is_in(["author_only_single", "author_only_multi"]))
    vreal = V.filter(~(pl.col("cat").is_in(["P"]) | pl.col("sub").is_in(["author_only_single", "author_only_multi"]))).select("gkey").unique()
    eff["pseudo_species_P_and_author_only_A"] = {
        "train_images": pseudo.height, "train_keys": pseudo["key"].n_unique(), "genera": pg.height,
        "genera_with_no_real_species": int((pg["real_species"] == 0).sum()),
        "pseudo_images_in_genera_with_real_species": int(pg.filter(pl.col("real_species") > 0)["pseudo_images"].sum()),
        "real_images_in_those_genera": int(pg["real_images"].sum()),
        "top_genera": pg.sort("pseudo_images", descending=True).head(10).to_dicts(),
        "val_images": vb.height,
        "val_images_in_genera_with_real_val_species": int(vb.join(vreal, on="gkey", how="semi").height)}
    sp_all = T.select("key", "cat", *RANKS).unique("key")
    terms = {}
    for i, r in enumerate(RANKS[:6]):
        pre = pl.concat_str([pl.col(c) for c in RANKS[:i + 1]], separator="|").alias("a")
        x = sp_all.with_columns(pre)
        n_all = x.group_by("a").agg(pl.len().alias("n_all"), (~pl.col("cat").is_in(["P", "A"])).sum().alias("n_kept"))
        y = x.filter(~pl.col("cat").is_in(["P", "A"])).join(n_all, on="a")
        terms[r] = {"kept_species_terms_before": int((y["n_all"] >= 2).sum()),
                    "kept_species_terms_after": int((y["n_kept"] >= 2).sum()),
                    "terms_that_exist_only_through_a_P_or_A_sibling": int(((y["n_all"] >= 2) & (y["n_kept"] < 2)).sum())}
    eff["drop_P_and_A_penalty_terms_of_remaining_species"] = terms

    # Merges: S keys truncated to their first word; A epithet_plus_author keys cut to their first epithet word.
    def merges(frame, which):
        base = set(frame.filter(pl.col("cat").is_in(["C", "O"]))["key"].unique().to_list())
        if which == "S":
            src_ = frame.filter(pl.col("cat") == "S")
            src_ = src_.with_columns((pl.col("gkey") + "|" + pl.col("first_word")).alias("target"))
        else:
            src_ = frame.filter(pl.col("sub") == "epithet_plus_author")
            src_ = src_.with_columns((pl.col("gkey") + "|" + pl.col("epithet_part").str.split(" ").list.first()).alias("target"))
        kk = src_.group_by("key").agg(pl.len().alias("n"), pl.col("target").first())
        kk = kk.with_columns(pl.col("target").is_in(list(base)).alias("exists"))
        new_t = kk.filter(~pl.col("exists"))
        merged = pl.concat([frame.filter(pl.col("cat").is_in(["C", "O"])).select("key"),
                            src_.select(pl.col("target").alias("key"))])
        ns = merged.group_by("key").len()
        return {"source_keys": kk.height, "source_images": int(kk["n"].sum()),
                "keys_merging_into_existing": int(kk["exists"].sum()),
                "images_merging_into_existing": int(kk.filter(pl.col("exists"))["n"].sum()),
                "keys_becoming_new": new_t.height, "distinct_new_keys": new_t["target"].n_unique(),
                "images_in_new_keys": int(new_t["n"].sum()),
                "species_keys_after_merge_of_C_O_plus_merged": ns.height,
                "largest_after_merge": ns.sort("len", descending=True).head(3).to_dicts(),
                "examples_existing": kk.filter(pl.col("exists")).sort("n", descending=True).head(8).to_dicts(),
                "examples_new": new_t.sort("n", descending=True).head(8).to_dicts()}

    eff["S_truncated_to_first_word_train"] = merges(T, "S")
    eff["A_epithet_plus_author_cut_to_first_epithet_train"] = merges(T, "A1")
    eff["S_truncated_to_first_word_val"] = merges(V, "S")
    trk = set(T["key"].unique().to_list())
    vk = keys.filter(pl.col("n_va") > 0)
    val = {}
    for c in CATS:
        f = vk.filter(pl.col("cat") == c)
        in_tr = f.filter(pl.col("key").is_in(list(trk)))
        val[c] = {"val_keys": f.height, "val_images": int(f["n_va"].sum()),
                  "val_keys_with_train_images": in_tr.height,
                  "val_images_whose_key_has_train_images": int(in_tr["n_va"].sum()),
                  "val_images_whose_key_has_no_train_images": int(f["n_va"].sum() - in_tr["n_va"].sum())}
    R["val_by_category"] = val
    R["val_note"] = ("The 0.98 dedup (spec line 42) has not been run. It compares a val image only with train images of "
                     "the same species key, so it can remove at most val_images_whose_key_has_train_images of each category.")
    R["effects"] = eff

    # ------------------------------------------------------------ iNat21 val categories
    iv = json.load(open(INAT_VAL))
    icats = iv["categories"]
    irows = []
    for c in icats:
        gname = c.get("genus") or c["name"].split(" ")[0]
        ep_ = c.get("specific_epithet") or " ".join(c["name"].split(" ")[1:])
        cat, sub, info = classify(gname, ep_.lower(), ev)
        irows.append({"id": c["id"], "name": c["name"], "genus": gname, "specific_epithet": ep_, "cat": cat, "sub": sub,
                      "reason": info.get("reason"), "n_words": len(ep_.split()),
                      "starts_upper": ep_[:1].isupper()})
    idf = pl.DataFrame(irows)
    flagged = idf.filter(pl.col("cat") != "C")
    iq = [(f"{r['genus']} {r['specific_epithet']}", None, None) for r in flagged.iter_rows(named=True)]
    ig = run_gbif(iq, not args.no_gbif) if iq else {}
    R["inat21_val"] = {"n_categories": len(icats), "category_keys": sorted(icats[0].keys()),
                       "by_category": dict(idf.group_by("cat").len().iter_rows()),
                       "multi_word_epithets": int((idf["n_words"] > 1).sum()),
                       "epithet_starts_uppercase": int(idf["starts_upper"].sum()),
                       "flagged": [{**r, "gbif": ig.get((f"{r['genus']} {r['specific_epithet']}", None, None))}
                                   for r in flagged.iter_rows(named=True)]}
    log("inat21", R["inat21_val"]["by_category"], "flagged", flagged.height)

    # ------------------------------------------------------------ full non-C key list
    R["non_clean_keys"] = (keys.filter(pl.col("cat") != "C").sort(["cat", "n_tr", "key"], descending=[False, True, False])
                           .select("key", "species", "cat", "sub", "reason", "n_tr", "n_va").to_dicts())
    R["meta"]["end_utc"] = datetime.datetime.utcnow().isoformat()
    R["meta"]["elapsed_s"] = time.time() - T0
    with open(f"{OUT}/s9_epithet_census.json", "w") as fh:
        json.dump(to_py(R), fh, indent=1, ensure_ascii=False, default=str)
    brief = {k: R[k] for k in ("populations", "full_label_counts", "eol_train_patterns", "val_by_category")}
    brief["categories"] = {c: {k: v for k, v in R["categories"][c].items() if k in ("n_keys_union", "n_train_keys", "n_val_keys", "train_images", "val_images", "by_subtype")} for c in CATS}
    brief["effects"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("examples",)} if isinstance(v, dict) else v
                        for k, v in R["effects"].items()}
    print(json.dumps(to_py(brief), indent=1, ensure_ascii=False, default=str))
    log("done")


if __name__ == "__main__":
    main()
