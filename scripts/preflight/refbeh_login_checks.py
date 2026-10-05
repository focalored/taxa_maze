#!/usr/bin/env python3
"""
refbeh_login_checks.py -- light checks (small files only) for
audit/2026-10-02_reference_behaviour.md. Safe on the login node.

Reads, read-only:
  - the 63 shards/*.done.json and shards_index.json of probe_cache_b16/tol-eol/bioclip1 (KB each)
  - vocab.json (1.3 MB), codes.i32.npy (2.8 MB) and parents.i32.npz (67 KB) of
    probe_cache_b16/inat21-val/bioclip1
  - manifest.json of the eight reference caches
Writes audit/preflight/refbeh_login_checks.json. Imports nothing from tol_embed.
"""

import json
import statistics as st
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
TOLC = RES / "probe_cache_b16/tol-eol/bioclip1"
INATC = RES / "probe_cache_b16/inat21-val/bioclip1"
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_login_checks.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

out = {}

# --- done.json: rates and task grouping -------------------------------------------------
rows = [json.loads(p.read_text()) for p in sorted((TOLC / "shards").glob("*.done.json"))]
full = [r for r in rows if r["shard"] != "image_set_63"]
ips = [r["img_per_s"] for r in full]
out["done_json"] = dict(
    n=len(rows),
    fields=sorted(rows[0]),
    full_shards_img_per_s_min_median_max=[min(ips), st.median(ips), max(ips)],
    full_shards_mb_per_s_min_median_max=[min(r["mb_per_s"] for r in full), st.median(r["mb_per_s"] for r in full),
                                         max(r["mb_per_s"] for r in full)],
    image_set_63=dict((k, r[k]) for r in rows if r["shard"] == "image_set_63"
                      for k in ("n_rows", "elapsed_s", "img_per_s")),
    sum_n_rows=sum(r["n_rows"] for r in rows),
    sum_n_skipped=sum(r["n_skipped"] for r in rows),
    sum_decode_errors=sum(r["decode_errors"] for r in rows),
    any_suspect=any(r["suspect"] for r in rows),
    sampled_row_norm_min=min(r["row_norm_min"] for r in rows),
    sampled_row_norm_max=max(r["row_norm_max"] for r in rows),
    max_abs=max(r["max_abs"] for r in rows),
    has_precision_field=any("precision" in r for r in rows),
)
groups = {}
for i, r in enumerate(rows):
    w = datetime.strptime(r["written_utc"], "%Y-%m-%dT%H:%M:%SZ")
    s = w - timedelta(seconds=r["elapsed_s"])
    groups.setdefault(i % 8, []).append(dict(shard=r["shard"], written=r["written_utc"],
                                             start=s.strftime("%H:%M:%S")))
out["done_json"]["groups_by_index_mod_8"] = {
    str(k): dict(shards=[g["shard"] for g in v],
                 written_minutes=sorted({g["written"][11:16] for g in v}),
                 start_minutes=sorted({g["start"][:5] for g in v}))
    for k, v in sorted(groups.items())}
idx = json.loads((TOLC / "shards_index.json").read_text())
out["shards_index"] = dict(n=len(idx), first=idx[0], last=idx[-1],
                           sorted_stems=[e["shard"] for e in idx] == sorted(e["shard"] for e in idx),
                           contiguous=all(idx[i + 1]["start_row"] == idx[i]["start_row"] + idx[i]["n_rows"]
                                          for i in range(len(idx) - 1)))

# --- vocab ordering and parents ------------------------------------------------------------
v = json.loads((INATC / "vocab.json").read_text())
V = v["vocab"]
order = {}
for r in RANKS:
    L = V[r]
    order[r] = dict(n=len(L), string_sorted=L == sorted(L),
                    tuple_sorted=L == sorted(L, key=lambda p: tuple(p.split("|"))))
out["inat_vocab_order"] = order
sp = V["species"]
out["inat_vocab_species_string_sort_inversions"] = sum(sp[i] > sp[i + 1] for i in range(len(sp) - 1))
codes = np.load(INATC / "codes.i32.npy")
par = np.load(INATC / "parents.i32.npz")
pc = {}
for ri in range(1, 7):
    L, P = V[RANKS[ri]], V[RANKS[ri - 1]]
    pos = {p: i for i, p in enumerate(P)}
    a = par[RANKS[ri]]
    pc[RANKS[ri]] = dict(parent_equals_prefix_index=all(pos[L[i].rsplit("|", 1)[0]] == a[i] for i in range(len(L))),
                         rows_consistent=bool((a[codes[:, ri]] == codes[:, ri - 1]).all()))
pc["kingdom_all_zero"] = bool((par["kingdom"] == 0).all())
out["inat_parents"] = dict(keys=par.files, checks=pc)
out["inat_genus_paths_vs_names"] = dict(paths=len(V["genus"]), names=len({p.split("|")[-1] for p in V["genus"]}))

# --- manifests -------------------------------------------------------------------------------
mans = {}
for p in [RES / "probe_cache/inat21-val/bioclip2/manifest.json"] + \
         [RES / f"probe_cache_b16/inat21-val/{m}/manifest.json" for m in
          ("bioclip1", "rcme", "openclip-b16", "clip-l14-laion2b", "bfl-euclidean", "bfl-hyperbolic")] + \
         [TOLC / "manifest.json"]:
    m = json.loads(p.read_text())
    mans[str(p.relative_to(RES))] = {k: m.get(k) for k in (
        "checkpoint", "model", "N", "D", "dtype", "normalized", "precision", "batch_size", "workers",
        "backend", "argv", "slurm_job_id", "ids_sha256", "weight_sha256", "preprocess_cfg", "source_run",
        "pct_from_expected", "n_skipped_total")}
out["manifests"] = mans

OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps({k: out[k] for k in ("done_json", "shards_index", "inat_vocab_order",
                                       "inat_vocab_species_string_sort_inversions", "inat_parents",
                                       "inat_genus_paths_vs_names")}, indent=1, default=str))
print("wrote", OUT)
