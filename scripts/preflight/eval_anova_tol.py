"""Preflight check E7 (ToL-EOL part): ANOVA split of BioCLIP 1 image embeddings on
ToL-EOL train, species = full 7-rank lineage from catalog.csv joined by uuid.

Inputs (read-only):
  /projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/
      emb.f16.npy (6,219,459 x 512 fp16, not normalized), ids.txt (uuid per row),
      codes.i32.npy + vocab.json (used only for a cross-check of the species grouping)
  /u/liv/bdbk/data/tol10m/metadata/catalog.csv (2 GB)

Row selection: catalog rows with split == "train" (not train_small) whose seven
rank columns (kingdom..species) are all non-null and non-empty after stripping
whitespace; inner-joined to ids.txt on treeoflife_id == uuid.
Species key: "|".join(kingdom, phylum, class, order, family, genus, species).

Computation (float64 throughout, chunked over the memory-mapped cache):
  pass 1: L2-normalize each selected row; accumulate the global sum and the
          per-species sums and counts.
  pass 2: T = mean_i ||x_i - xbar||^2, W = mean_i ||x_i - m_s(i)||^2 (direct).
  B_iw = T - W (image-weighted between); B_su = mean_s ||m_s - mbar||^2 with
  mbar = mean_s m_s (species-uniform between). Identity cross-checks and an
  exploratory raw (unnormalized) variant are recorded too.
Extras recorded because they are free in the same pass (not part of E7):
  full-cache row-norm min/max (spec line 120 quotes 5.95-17.91), and distinct
  lineage-prefix counts per rank on the selected rows (spec line 181).

Writes /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_anova_tol.json
Launched by scripts/preflight/eval_anova_tol.sh via sbatch on the cpu partition.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

CACHE = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1")
CATALOG = Path("/u/liv/bdbk/data/tol10m/metadata/catalog.csv")
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_anova_tol.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
CHUNK = 1 << 17
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def build_selection(res):
    ids = (CACHE / "ids.txt").read_text().splitlines()
    n_rows = len(ids)
    res["ids_count"] = n_rows
    res["ids_distinct"] = len(set(ids))
    log("ids.txt rows", n_rows, "distinct", res["ids_distinct"])
    ids_df = pl.DataFrame({"uuid": ids, "row": np.arange(n_rows, dtype=np.int64)})
    del ids

    cols = ["split", "treeoflife_id"] + RANKS
    cat = pl.read_csv(CATALOG, columns=cols, infer_schema_length=0)
    res["catalog_rows"] = cat.height
    res["catalog_split_counts"] = {r["split"]: r["len"]
                                   for r in cat.group_by("split").len().to_dicts()}
    log("catalog rows", cat.height, res["catalog_split_counts"])

    tr = cat.filter(pl.col("split") == "train").drop("split")
    res["catalog_train_rows"] = tr.height
    res["catalog_train_distinct_ids"] = tr["treeoflife_id"].n_unique()
    # any uuid listed as both train and val?
    val_ids = cat.filter(pl.col("split") == "val").select("treeoflife_id")
    res["catalog_ids_in_both_train_and_val"] = tr.join(val_ids, on="treeoflife_id",
                                                       how="semi").height
    # context: cache rows that are train / val in the catalog, any lineage
    res["extra_cache_rows_train_any_lineage"] = ids_df.join(
        tr.select("treeoflife_id"), left_on="uuid", right_on="treeoflife_id",
        how="semi").height
    res["extra_cache_rows_val_any_lineage"] = ids_df.join(
        val_ids, left_on="uuid", right_on="treeoflife_id", how="semi").height
    log("cache rows train/val (any lineage):", res["extra_cache_rows_train_any_lineage"],
        res["extra_cache_rows_val_any_lineage"])
    del cat, val_ids
    complete = pl.all_horizontal([pl.col(c).is_not_null()
                                  & (pl.col(c).str.strip_chars() != "") for c in RANKS])
    res["catalog_train_rank_whitespace_padded_values"] = int(
        tr.select(pl.sum_horizontal([(pl.col(c).is_not_null()
                                      & (pl.col(c) != pl.col(c).str.strip_chars()))
                                     .cast(pl.Int64) for c in RANKS])).to_series().sum())
    trc = tr.filter(complete)
    res["catalog_train_complete_rows"] = trc.height
    res["catalog_train_complete_distinct_ids"] = trc["treeoflife_id"].n_unique()
    del tr
    trc = trc.with_columns(pl.concat_str([pl.col(c) for c in RANKS], separator="|")
                           .alias("sp_key"))
    j = ids_df.join(trc, left_on="uuid", right_on="treeoflife_id", how="inner").sort("row")
    res["selected_rows"] = j.height
    res["selected_rows_distinct"] = j["row"].n_unique()
    res["catalog_train_complete_ids_not_in_cache"] = trc.height - j.height
    log("selected (train, complete lineage, in cache):", j.height)

    # extras: distinct lineage-prefix counts per rank (paths) and bare names
    res["extra_distinct_per_rank_on_selected"] = {
        "prefix_paths": {c: j.select(pl.concat_str([pl.col(x) for x in RANKS[: i + 1]],
                                                   separator="|")).to_series().n_unique()
                         for i, c in enumerate(RANKS)},
        "bare_names": {c: j[c].n_unique() for c in RANKS},
    }
    log("distinct per rank:", res["extra_distinct_per_rank_on_selected"])

    sp = j["sp_key"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    # re-index to 0..S-1 densely (physical codes of a fresh Categorical may be global)
    uniq, sp = np.unique(sp, return_inverse=True)
    rows = j["row"].to_numpy().astype(np.int64)
    res["n_species"] = int(uniq.size)

    # cross-check grouping against the cache's own codes / vocab
    codes = np.load(CACHE / "codes.i32.npy", mmap_mode="r")
    csp = np.asarray(codes[:, 6])[rows].astype(np.int64)
    res["xcheck_codes_species_negative_on_selected"] = int((csp < 0).sum())
    pairs = np.unique(sp * (int(csp.max()) + 2) + (csp + 1)).size
    res["xcheck_distinct_codes_species_on_selected"] = int(np.unique(csp).size)
    res["xcheck_distinct_catalog_code_pairs"] = int(pairs)
    res["xcheck_grouping_is_bijection"] = bool(pairs == uniq.size == np.unique(csp).size)
    vocab = json.loads((CACHE / "vocab.json").read_text())
    vsp = vocab["vocab"]["species"]
    want = j.select(pl.concat_str([pl.col(c) for c in RANKS[:6]]
                                  + [pl.concat_str([pl.col("genus"), pl.col("species")],
                                                   separator=" ")],
                                  separator="|")).to_series()
    got = pl.Series([vsp[c] if c >= 0 else None for c in csp.tolist()])
    res["xcheck_vocab_species_path_mismatches"] = int((want != got).fill_null(True).sum())
    all_codes_ok = np.asarray((np.asarray(codes) >= 0).all(axis=1))
    res["xcheck_cache_rows_with_all_7_codes"] = int(all_codes_ok.sum())
    log("xcheck:", {k: v for k, v in res.items() if k.startswith("xcheck")})
    return rows, sp, int(uniq.size), n_rows


def passes(rows, sp, S, n_rows, res):
    emb = np.load(CACHE / "emb.f16.npy", mmap_mode="r")
    assert emb.shape[0] == n_rows, (emb.shape, n_rows)
    D = emb.shape[1]
    res["emb_shape"], res["emb_dtype"] = list(emb.shape), str(emb.dtype)
    sel = np.zeros(n_rows, dtype=bool)
    sel[rows] = True
    lab_full = np.full(n_rows, -1, dtype=np.int64)
    lab_full[rows] = sp
    N = int(sel.sum())

    sums = torch.zeros((S, D), dtype=torch.float64)
    sums_raw = torch.zeros((S, D), dtype=torch.float64)
    gsum = torch.zeros(D, dtype=torch.float64)
    gsum_raw = torch.zeros(D, dtype=torch.float64)
    sq_raw = 0.0
    sq_n = 0.0
    nmin_all, nmax_all, nmin_sel, nmax_sel = np.inf, -np.inf, np.inf, -np.inf
    nonfinite = 0
    for a in range(0, n_rows, CHUNK):
        b = min(a + CHUNK, n_rows)
        x = torch.from_numpy(np.asarray(emb[a:b], dtype=np.float64))
        nonfinite += int((~torch.isfinite(x)).sum())
        nrm = torch.linalg.vector_norm(x, dim=1)
        nmin_all, nmax_all = min(nmin_all, float(nrm.min())), max(nmax_all, float(nrm.max()))
        m = torch.from_numpy(sel[a:b])
        if m.any():
            xs, ns = x[m], nrm[m]
            nmin_sel, nmax_sel = min(nmin_sel, float(ns.min())), max(nmax_sel, float(ns.max()))
            lab = torch.from_numpy(lab_full[a:b][sel[a:b]])
            xn = xs / ns[:, None]
            sums.index_add_(0, lab, xn)
            sums_raw.index_add_(0, lab, xs)
            gsum += xn.sum(0)
            gsum_raw += xs.sum(0)
            sq_n += float((xn * xn).sum())
            sq_raw += float((xs * xs).sum())
        if (a // CHUNK) % 10 == 0:
            log(f"pass1 rows {b:,}/{n_rows:,}")
    cnt = torch.from_numpy(np.bincount(sp, minlength=S).astype(np.float64))
    res["row_norm_all_rows"] = dict(min=nmin_all, max=nmax_all)
    res["row_norm_selected_rows"] = dict(min=nmin_sel, max=nmax_sel)
    res["n_nonfinite_all_rows"] = nonfinite
    res["images_per_species"] = dict(min=int(cnt.min()), max=int(cnt.max()),
                                     median=float(cnt.median()),
                                     n_species_with_1_image=int((cnt == 1).sum()))

    out = {}
    for tag, S_, G_, SQ in (("l2_normalized_float64", sums, gsum, sq_n),
                            ("exploratory_raw_unnormalized_float64", sums_raw, gsum_raw,
                             sq_raw)):
        out[tag] = dict(means=S_ / cnt[:, None], xbar=G_ / N, sq=SQ)
    del sums, sums_raw

    # pass 2: direct residuals
    acc = {k: dict(T=0.0, W=0.0) for k in out}
    for a in range(0, n_rows, CHUNK):
        b = min(a + CHUNK, n_rows)
        m = sel[a:b]
        if not m.any():
            continue
        xs = torch.from_numpy(np.asarray(emb[a:b], dtype=np.float64)[m])
        lab = torch.from_numpy(lab_full[a:b][m])
        for k, o in out.items():
            x = xs / torch.linalg.vector_norm(xs, dim=1)[:, None] if k.startswith("l2") else xs
            acc[k]["T"] += float(((x - o["xbar"]) ** 2).sum())
            acc[k]["W"] += float(((x - o["means"].index_select(0, lab)) ** 2).sum())
        if (a // CHUNK) % 10 == 0:
            log(f"pass2 rows {b:,}/{n_rows:,}")

    for k, o in out.items():
        T, W = acc[k]["T"] / N, acc[k]["W"] / N
        mu, xbar = o["means"], o["xbar"]
        B_iw_direct = float((cnt * ((mu - xbar) ** 2).sum(1)).sum()) / N
        mbar = mu.mean(0)
        B_su = float(((mu - mbar) ** 2).sum(1).mean())
        T_id = o["sq"] / N - float(xbar @ xbar)
        W_id = o["sq"] / N - float((cnt * (mu ** 2).sum(1)).sum()) / N
        SSW, SSB = W * N, B_iw_direct * N
        res[f"anova_{k}"] = dict(
            N=N, S=S, T=T, W=W, B_iw=T - W, B_iw_direct=B_iw_direct, B_su=B_su,
            within_frac_W_over_T=W / T, between_frac_1_minus_W_over_T=1 - W / T,
            B_su_over_T=B_su / T,
            within_frac_species_uniform_W_over_W_plus_Bsu=W / (W + B_su),
            between_frac_species_uniform_Bsu_over_W_plus_Bsu=B_su / (W + B_su),
            identity_check=dict(T_identity=T_id, W_identity=W_id, abs_err_T=abs(T_id - T),
                                abs_err_W=abs(W_id - W),
                                abs_err_B=abs((T - W) - B_iw_direct)),
            exploratory_unbiased=dict(W_over_N_minus_S=SSW / (N - S),
                                      B_over_S_minus_1=SSB / (S - 1),
                                      within_frac=(SSW / (N - S))
                                      / (SSW / (N - S) + SSB / (S - 1))),
            mean_row_sq_norm=o["sq"] / N)
        log(k, json.dumps(res[f"anova_{k}"]))


def main():
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    res = dict(host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               torch=torch.__version__, polars=pl.__version__, numpy=np.__version__,
               chunk_rows=CHUNK)
    log("start", res)
    rows, sp, S, n_rows = build_selection(res)
    passes(rows, sp, S, n_rows, res)
    res["seconds_total"] = time.time() - T0
    OUT.write_text(json.dumps(res, indent=1))
    log("wrote", OUT)


if __name__ == "__main__":
    main()
