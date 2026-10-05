#!/usr/bin/env python
"""One-time pilot 1 inputs under <p1_data_dir>/p1/: held-out species, the 206 fresh encodes, the bank seed, the ToL-val dedup list, and the step-0 drift floor (Amendment 1 S25, S26, S52).

Usage: `python scripts/p1/prepare.py tree|oversize|seed|dedup`, then `floor --part i --parts 4` and `floor-merge` (scripts/p1/prepare.sh runs them in order).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import assert_outside_projects, load_paths  # noqa: E402

DEDUP_COS = 0.98
CACHE_SUBDIR = "results/probe_cache_b16/tol-eol/bioclip1"


def out_dir(paths) -> Path:
    d = assert_outside_projects(Path(paths.p1_data_dir) / "p1")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_tree(paths):
    from src.data.taxonomy import build_train_tree
    return build_train_tree(Path(paths.p1_data_dir) / "store", s9_drop=True, K=16,
                            heldout_file=out_dir(paths) / "heldout_species_s9.txt")


def keys_sha(keys) -> str:
    return hashlib.sha256("".join(k + "\n" for k in keys).encode()).hexdigest()


def tree_uuids(paths, tree):
    """uuid of every train image of the tree, in the tree's (species, shard, row) order."""
    import polars as pl
    idx = pl.read_parquet(Path(paths.p1_data_dir) / "store" / "index.parquet", columns=["uuid", "shard", "row"])
    order = pl.DataFrame({"shard": tree.img_shard.astype(np.int16), "row": tree.img_row.astype(np.int32),
                          "pos": np.arange(len(tree.img_row))})
    j = order.join(idx, on=["shard", "row"], how="left").sort("pos")
    if j["uuid"].null_count():
        raise RuntimeError("a tree image has no store uuid")
    return j["uuid"].to_list()


def load_embeddings(paths, uuids):
    """BioCLIP 1 embeddings (fp16 rows) for `uuids`: the cache, else the fresh encodes (S52)."""
    cache = Path(paths.tol_embed_dir) / CACHE_SUBDIR
    ids = (cache / "ids.txt").read_text().split()
    row = {u: i for i, u in enumerate(ids)}
    emb = np.load(cache / "emb.f16.npy")
    od = out_dir(paths)
    fresh_ids = (od / "oversize_uuids.txt").read_text().split()
    fresh = np.load(od / "oversize_emb.f16.npy")
    frow = {u: i for i, u in enumerate(fresh_ids)}
    out = np.empty((len(uuids), emb.shape[1]), dtype=np.float16)
    n_cache = n_fresh = 0
    for i, u in enumerate(uuids):
        r = row.get(u)
        if r is not None:
            out[i] = emb[r]
            n_cache += 1
        else:
            out[i] = fresh[frow[u]]
            n_fresh += 1
    return out, {"from_cache": n_cache, "fresh": n_fresh}


def normalize64(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def step_tree(paths):
    from src.data.taxonomy import save_stats
    tree = get_tree(paths)
    save_stats(out_dir(paths) / "data_stats_train.json", tree.stats)
    print(json.dumps(tree.stats, indent=1, default=str))


def step_oversize(paths):
    """Fresh fp16-autocast encodes of the store rows the cache lacks (fp32 weights, as the cache was made)."""
    import polars as pl
    import torch
    import torch.nn.functional  # noqa: F401
    from src.data.tol_store import ImageStore, to_model_input
    from src.models.bioclip_lora import load_bioclip1

    store_dir = Path(paths.p1_data_dir) / "store"
    miss = pl.read_csv(store_dir / "not_in_cache.csv")
    store = ImageStore(store_dir)
    u8 = store.read_many(miss["shard"].to_numpy(), miss["row"].to_numpy())
    model, _, _ = load_bioclip1(paths.bioclip1_ckpt, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        f, _ = model.encode_image(to_model_input(torch.from_numpy(u8).cuda()))
    od = out_dir(paths)
    np.save(od / "oversize_emb.f16.npy", f.half().cpu().numpy())
    (od / "oversize_uuids.txt").write_text("".join(u + "\n" for u in miss["uuid"].to_list()))
    norms = f.float().norm(dim=1)
    print(json.dumps({"n": len(miss), "split": miss["split"].value_counts().to_dicts(),
                      "norm_min": float(norms.min()), "norm_max": float(norms.max())}))


def step_seed(paths):
    tree = get_tree(paths)
    uuids = tree_uuids(paths, tree)
    emb, src = load_embeddings(paths, uuids)
    S, D = tree.n_species_total, emb.shape[1]
    sums = np.zeros((S, D), dtype=np.float64)
    starts = tree.sp_start
    chunk = 1 << 18
    for a in range(0, len(uuids), chunk):
        b = min(a + chunk, len(uuids))
        v = normalize64(emb[a:b])
        sp = tree.img_species[a:b]
        seg = np.flatnonzero(np.r_[True, np.diff(sp) != 0])
        sums[sp[seg]] += np.add.reduceat(v, seg, axis=0)
    m_s = sums / tree.N_s[:, None]
    od = out_dir(paths)
    np.save(od / "bank_seed_s9.npy", m_s)
    meta = {"n_species": S, "species_keys_head": tree.node_keys[6][:3], "species_keys_sha256": keys_sha(tree.node_keys[6]),
            "n_images": len(uuids), "sources": src, "cache": str(Path(paths.tol_embed_dir) / CACHE_SUBDIR),
            "normalization": "fp16 cache rows -> fp64 -> L2-normalize -> species mean"}
    (od / "bank_seed_s9.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def step_dedup(paths):
    """S26: val images with cosine >= 0.98 to any same-species train image (full seven-rank key)."""
    import polars as pl
    from src.data.taxonomy import _train_frame
    tree = get_tree(paths)
    df, _ = _train_frame(Path(paths.p1_data_dir) / "store", True)
    va = df.filter(pl.col("split") == "val").select("uuid", "k6").sort("uuid")
    tr_uuids = tree_uuids(paths, tree)
    tr_emb, src_tr = load_embeddings(paths, tr_uuids)
    va_emb, src_va = load_embeddings(paths, va["uuid"].to_list())
    sp_id = {k: i for i, k in enumerate(tree.node_keys[6])}
    va_sp = np.array([sp_id.get(k, -1) for k in va["k6"].to_list()])
    drop, n_checked, max_cos = [], 0, np.full(len(va_sp), np.nan)
    order = np.argsort(va_sp, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(va_sp[order]) != 0, True])
    uu = va["uuid"].to_list()
    for i in range(len(bounds) - 1):
        rows = order[bounds[i]: bounds[i + 1]]
        s = va_sp[rows[0]]
        if s < 0:
            continue
        a, b = tree.sp_start[s], tree.sp_start[s] + tree.N_s[s]
        c = normalize64(va_emb[rows]) @ normalize64(tr_emb[a:b]).T
        mx = c.max(axis=1)
        max_cos[rows] = mx
        n_checked += len(rows)
        drop += [uu[r] for r in rows[mx >= DEDUP_COS]]
    od = out_dir(paths)
    (od / "tolval_dedup_drop.txt").write_text("".join(u + "\n" for u in sorted(drop)))
    stats = {"threshold": DEDUP_COS, "n_val_after_s9": len(va_sp), "n_val_species_in_train_images": n_checked,
             "n_dropped": len(drop), "n_kept": len(va_sp) - len(drop), "sources_val": src_va, "sources_train": src_tr,
             "precision": "fp16 cache rows -> fp64, L2-normalized, fp64 dot products"}
    (od / "tolval_dedup.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def step_floor(paths, part: int, parts: int):
    """S25 drift floor: step-0 species means under fp16 and bf16 autocast, one share of the species."""
    import torch
    from src.data.tol_datamodule import ChunkDataset
    from src.data.tol_store import ImageStore
    from src.models.bank import species_means_pass
    from src.models.bioclip_lora import add_qv_lora, load_bioclip1
    from torch.utils.data import DataLoader

    tree = get_tree(paths)
    cum = np.cumsum(tree.N_s)
    T = int(cum[-1])
    cuts = [0] + [int(np.searchsorted(cum, T * r // parts, side="left")) + 1 for r in range(1, parts)] + [len(cum)]
    s_lo, s_hi = cuts[part], cuts[part + 1]
    a, b = int(tree.sp_start[s_lo]), (int(tree.sp_start[s_hi]) if s_hi < len(cum) else T)
    model, _, _ = load_bioclip1(paths.bioclip1_ckpt, device="cuda")
    add_qv_lora(model)  # B = 0: the step-0 training model, same function as BioCLIP 1
    model = model.cuda()
    store = ImageStore(Path(paths.p1_data_dir) / "store")
    import os
    nw = max(1, len(os.sched_getaffinity(0)) - 2)
    sp = torch.as_tensor(tree.img_species[a:b], dtype=torch.long, device="cuda")
    res, t0 = {}, time.time()
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        ds = ChunkDataset(store, tree.img_shard[a:b], tree.img_row[a:b], 1024)
        dl = DataLoader(ds, batch_size=None, num_workers=nw, pin_memory=True)
        sums = species_means_pass(model, dl, sp, tree.n_species_total, "cuda", dt)
        res[name] = (sums[s_lo:s_hi] / torch.as_tensor(tree.N_s[s_lo:s_hi], dtype=torch.float64, device="cuda")[:, None]).cpu().numpy()
    od = out_dir(paths) / "floor_parts"
    od.mkdir(exist_ok=True)
    np.savez(od / f"part{part}of{parts}.npz", s_lo=s_lo, s_hi=s_hi, fp16=res["fp16"], bf16=res["bf16"],
             seconds=time.time() - t0, n_images=b - a)
    print(json.dumps({"part": part, "species": [s_lo, s_hi], "images": b - a, "seconds": round(time.time() - t0, 1)}))


def step_floor_merge(paths, parts: int):
    import torch
    from src.models.bank import Bank
    tree = get_tree(paths)
    S = tree.n_species_total
    m = {k: np.zeros((S, 512), dtype=np.float64) for k in ("fp16", "bf16")}
    secs, imgs = [], 0
    for p in range(parts):
        z = np.load(out_dir(paths) / "floor_parts" / f"part{p}of{parts}.npz")
        for k in m:
            m[k][int(z["s_lo"]): int(z["s_hi"])] = z[k]
        secs.append(float(z["seconds"]))
        imgs += int(z["n_images"])
    if imgs != int(tree.N_s.sum()):
        raise RuntimeError(f"floor parts cover {imgs} images, expected {int(tree.N_s.sum())}")
    banks = {}
    for k in m:
        banks[k] = Bank(tree, 512, "cpu")
        banks[k].set_species_means(torch.from_numpy(m[k]))
    seed = Bank(tree, 512, "cpu")
    seed.set_species_means(torch.from_numpy(np.load(out_dir(paths) / "bank_seed_s9.npy")))
    floor = {k.replace("drift/", ""): v for k, v in banks["bf16"].drift(banks["fp16"]).items()}
    info = {f"fp16_vs_seed/{k.replace('drift/', '')}": v for k, v in banks["fp16"].drift(seed).items()}
    out = {**floor, **info, "pass_seconds_max_part": max(secs), "parts": parts, "n_images": imgs,
           "note": "bf16 vs fp16 step-0 species means (S25 drift floor); fp16_vs_seed compares the fp16 pass with the cache seed"}
    (out_dir(paths) / "drift_floor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["tree", "oversize", "seed", "dedup", "floor", "floor-merge"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=4)
    a = ap.parse_args()
    paths = load_paths()
    t0 = time.time()
    {"tree": lambda: step_tree(paths), "oversize": lambda: step_oversize(paths), "seed": lambda: step_seed(paths),
     "dedup": lambda: step_dedup(paths), "floor": lambda: step_floor(paths, a.part, a.parts),
     "floor-merge": lambda: step_floor_merge(paths, a.parts)}[a.step]()
    print(f"[{a.step}] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
