#!/usr/bin/env python
"""Preflight check C7 (specs/pilot1.md lines 50 and 120): integrity and row norms
of the BioCLIP 1 ToL-EOL embedding cache.

Read-only inputs:
  /projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/
      emb.f16.npy, ids.txt, manifest.json, shards_index.json,
      shards/image_set_NN.ids.txt, shards/image_set_NN.done.json,
      shards/image_set_NN.bioclip1.f16.raw (first/last rows only)
Output:
  /projects/bdbk/liv/repos/taxa_maze/audit/preflight/data_cache_norms.json

Norms are computed in float32, chunk by chunk, from np.load(mmap_mode='r').
The extreme rows are re-checked in float64.
"""
import hashlib
import json
import os
import time

import numpy as np

CACHE = "/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1"
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
CHUNK = 25_000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    R = {"script": os.path.abspath(__file__)}
    path = f"{CACHE}/emb.f16.npy"
    emb = np.load(path, mmap_mode="r")
    man = json.load(open(f"{CACHE}/manifest.json"))
    idx = json.load(open(f"{CACHE}/shards_index.json"))
    N, D = emb.shape
    R.update({"emb_path": path, "file_bytes": os.path.getsize(path), "shape": [N, D],
              "dtype": str(emb.dtype), "fortran_order": bool(np.isfortran(emb)),
              "header_bytes": os.path.getsize(path) - emb.nbytes,
              "manifest_N": man["N"], "manifest_D": man["D"], "manifest_dtype": man["dtype"],
              "manifest_normalized": man["normalized"]})
    log("emb", R["shape"], R["dtype"])

    # ids.txt
    with open(f"{CACHE}/ids.txt", "rb") as fh:
        raw_ids = fh.read()
    lines = raw_ids.decode().split("\n")
    trailing_newline = lines[-1] == ""
    ids = [x for x in lines if x != ""]
    R["ids_txt"] = {
        "n_lines": len(ids), "n_distinct": len(set(ids)), "ends_with_newline": trailing_newline,
        "n_blank_lines_inside": sum(1 for x in lines[:-1] if x == ""),
        "all_len_36": all(len(x) == 36 for x in ids),
        "sha256_file_bytes": hashlib.sha256(raw_ids).hexdigest(),
        "sha256_joined_no_trailing_newline": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "manifest_ids_sha256": man["ids_sha256"],
        "rows_equal_ids": len(ids) == N,
    }
    R["ids_txt"]["manifest_sha_matches"] = man["ids_sha256"] in (
        R["ids_txt"]["sha256_file_bytes"], R["ids_txt"]["sha256_joined_no_trailing_newline"])
    log("ids", R["ids_txt"])

    norms = np.empty(N, np.float32)
    nan_rows = inf_rows = 0
    per_shard = []
    for e in idx:
        s, n = e["start_row"], e["n_rows"]
        blk = np.ascontiguousarray(emb[s:s + n])
        sha = hashlib.sha256(blk.tobytes()).hexdigest()
        for j in range(0, n, CHUNK):
            x = blk[j:j + CHUNK].astype(np.float32)
            nan_rows += int(np.isnan(x).any(axis=1).sum())
            inf_rows += int(np.isinf(x).any(axis=1).sum())
            norms[s + j:s + j + len(x)] = np.sqrt(np.einsum("ij,ij->i", x, x))
        # spot-check against the per-shard raw file (first and last 4 rows)
        rawp = f"{CACHE}/shards/{e['shard']}.bioclip1.f16.raw"
        raw = np.memmap(rawp, dtype=np.float16, mode="r").reshape(-1, D)
        spot = bool(np.array_equal(raw[:4], blk[:4]) and np.array_equal(raw[-4:], blk[-4:]))
        done = json.load(open(f"{CACHE}/shards/{e['shard']}.done.json"))
        sid = open(f"{CACHE}/shards/{e['shard']}.ids.txt", "rb").read()
        nb = norms[s:s + n]
        per_shard.append({
            "shard": e["shard"], "start_row": s, "n_rows": n, "raw_rows": raw.shape[0],
            "emb_slice_sha256_equals_index_sha256_raw": sha == e["sha256_raw"],
            "spot_rows_equal_raw_file": spot,
            "shard_ids_sha256_equals_done_json": hashlib.sha256(sid).hexdigest() == done["ids_sha256"],
            "norm_min": float(nb.min()), "norm_max": float(nb.max()),
            "done_json_norm_min": done["row_norm_min"], "done_json_norm_max": done["row_norm_max"],
        })
        log(e["shard"], per_shard[-1]["emb_slice_sha256_equals_index_sha256_raw"], spot,
            round(float(nb.min()), 4), round(float(nb.max()), 4))

    i_min, i_max = int(np.argmin(norms)), int(np.argmax(norms))
    f64 = lambda i: float(np.linalg.norm(np.asarray(emb[i], dtype=np.float64)))
    qs = [0, 1e-6, 1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999, 1 - 1e-6, 1]
    shard_of = lambda i: next(p["shard"] for p in per_shard if p["start_row"] <= i < p["start_row"] + p["n_rows"])
    R["norms"] = {
        "computed_in": "float32 per 25k-row chunk via einsum; extremes re-checked in float64",
        "min": float(norms[i_min]), "max": float(norms[i_max]),
        "min_float64": f64(i_min), "max_float64": f64(i_max),
        "argmin_row": i_min, "argmin_uuid": ids[i_min], "argmin_shard": shard_of(i_min),
        "argmax_row": i_max, "argmax_uuid": ids[i_max], "argmax_shard": shard_of(i_max),
        "mean": float(norms.astype(np.float64).mean()),
        "quantiles": {str(q): float(np.quantile(norms, q)) for q in qs},
        "n_rows_with_nan": nan_rows, "n_rows_with_inf": inf_rows,
        "n_zero_norm": int((norms == 0).sum()), "n_norm_lt_1": int((norms < 1).sum()),
        "n_below_5_95": int((norms < 5.95).sum()), "n_above_17_91": int((norms > 17.91).sum()),
        "n_within_0_99_to_1_01": int(((norms > 0.99) & (norms < 1.01)).sum()),
        "spec_range": [5.95, 17.91],
    }
    R["per_shard"] = per_shard
    R["all_slices_match_index_sha"] = all(p["emb_slice_sha256_equals_index_sha256_raw"] for p in per_shard)
    R["all_spot_checks_equal"] = all(p["spot_rows_equal_raw_file"] for p in per_shard)
    R["all_shard_ids_sha_match_done_json"] = all(p["shard_ids_sha256_equals_done_json"] for p in per_shard)
    R["done_json_norm_min_over_shards"] = min(p["done_json_norm_min"] for p in per_shard)
    R["done_json_norm_max_over_shards"] = max(p["done_json_norm_max"] for p in per_shard)
    R["elapsed_s"] = time.time() - T0
    with open(f"{OUT}/data_cache_norms.json", "w") as fh:
        json.dump(R, fh, indent=1)
    log("norms", {k: v for k, v in R["norms"].items() if k != "quantiles"})
    log("wrote", f"{OUT}/data_cache_norms.json")


if __name__ == "__main__":
    main()
