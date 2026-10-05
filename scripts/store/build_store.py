#!/usr/bin/env python
"""Build pilot 1's uint8 image store from the 63 ToL-10M EOL shards (Amendment 1 S1, S52), decoding exactly as the BioCLIP 1 cache did.

Usage: `bash scripts/store/build_store.sh` (catalog, then 63 shard tasks, then merge); one trial step:
`python scripts/store/build_store.py shard --shard 63 --limit 3000 --out <dir>`.
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import assert_outside_projects, load_paths  # noqa: E402

CROP = 224
ROW_BYTES = CROP * CROP * 3
N_SHARDS = 63
MEMBERS_PER_SHARD_MAX = 100_000  # shards 01-62 hold 100,000 members, shard 63 holds 19,674
N_MEMBERS_TOTAL = 6_219_674      # EOL train + val ids (preflight C6a)
PIL_DEFAULT_MAX_PIXELS = 89_478_485  # PIL warns above this and refuses above twice this
MEMBER_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$")


def shard_name(i: int) -> str:
    return f"image_set_{i:02d}"


def build_transform():
    """Resize and CenterCrop of open_clip's val transform, checked against the cache manifest's repr."""
    from src.open_clip.transform import image_transform

    pv = image_transform(
        224, is_train=False, mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711], resize_mode="shortest", interpolation="bicubic",
    )
    got = re.sub(r" at 0x[0-9a-f]+", "", repr(pv))
    want = (
        "Compose(\n    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)\n"
        "    CenterCrop(size=(224, 224))\n    <function _convert_to_rgb>\n    ToTensor()\n"
        "    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])\n)"
    )
    if got != want:
        raise RuntimeError(f"val transform differs from the cache manifest's:\n{got}\n!=\n{want}")
    resize, crop, to_rgb = pv.transforms[0], pv.transforms[1], pv.transforms[2]
    return resize, crop, to_rgb


def step_catalog(out: Path, paths) -> None:
    from src.data.tol_catalog import read_eol_trainval

    t0 = time.time()
    df = read_eol_trainval(Path(paths.tol10m_dir) / "metadata" / "catalog.csv")
    df.write_parquet(out / "catalog.parquet")
    counts = {
        s: {"eol": int((df["split"] == s).sum()), "complete": int(((df["split"] == s) & df["complete"]).sum())}
        for s in ("train", "val")
    }
    print(json.dumps({"rows": df.height, "counts": counts, "seconds": round(time.time() - t0, 1)}))


def _decode_worker(wid, q_in, q_out, u8_path, cap):
    from PIL import Image, ImageFile

    Image.MAX_IMAGE_PIXELS = None          # Amendment 1 S52: keep the oversize images
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # truncated JPEGs must raise, as in the cache
    resize, crop, to_rgb = build_transform()
    mm = np.memmap(u8_path, dtype=np.uint8, mode="r+", shape=(cap, CROP, CROP, 3))
    errors, big, n, n_black = [], [], 0, 0
    while True:
        item = q_in.get()
        if item is None:
            break
        row, uuid, data = item
        try:
            with io.BytesIO(data) as stream:
                img = Image.open(stream)
                img.load()
                src_mode, (w, h) = img.mode, img.size
                img = img.convert("RGB")
            img = to_rgb(crop(resize(img)))
            arr = np.asarray(img, dtype=np.uint8)
            if arr.shape != (CROP, CROP, 3):
                raise ValueError(f"crop shape {arr.shape}")
            mm[row] = arr
            n += 1
            n_black += int(arr.max() == 0)
            if w * h > PIL_DEFAULT_MAX_PIXELS:
                big.append({"row": row, "uuid": uuid, "w": w, "h": h, "mode": src_mode,
                            "refused_by_default": w * h > 2 * PIL_DEFAULT_MAX_PIXELS})
        except Exception as e:  # recorded and fails the shard; the row stays zero
            errors.append({"row": row, "uuid": uuid, "error": repr(e)[:300]})
    mm.flush()
    del mm
    q_out.put({"wid": wid, "n": n, "errors": errors, "big": big, "n_black": n_black})


def _put(q, item, procs):
    """Queue put that fails instead of hanging when a decode worker has died (e.g. OOM-killed)."""
    while True:
        try:
            q.put(item, timeout=60)
            return
        except queue.Full:
            dead = [p.pid for p in procs if not p.is_alive()]
            if dead:
                raise RuntimeError(f"decode worker(s) {dead} died; aborting the shard")


def _get(q, procs):
    while True:
        try:
            return q.get(timeout=60)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                raise RuntimeError("all decode workers exited without reporting")


def step_shard(out: Path, paths, idx: int, workers: int, limit: int = 0) -> None:
    build_transform()  # fail fast on a transform mismatch, before any worker starts
    name = shard_name(idx)
    sdir = out / "shards"
    sdir.mkdir(parents=True, exist_ok=True)
    u8_path, ids_path, done_path = sdir / f"{name}.u8", sdir / f"{name}.uuids.txt", sdir / f"{name}.done.json"
    for p in (done_path, ids_path):
        if p.exists():
            p.unlink()  # a rerun rebuilds the shard from scratch
    with open(u8_path, "wb") as f:
        f.truncate(MEMBERS_PER_SHARD_MAX * ROW_BYTES)  # sparse; cut to the real row count at the end

    # Workers start before the population table is loaded, so they do not inherit its memory.
    ctx = mp.get_context("fork")
    q_in, q_out = ctx.Queue(maxsize=4 * workers), ctx.Queue()
    procs = [ctx.Process(target=_decode_worker, args=(w, q_in, q_out, str(u8_path), MEMBERS_PER_SHARD_MAX))
             for w in range(workers)]
    for p in procs:
        p.start()

    import polars as pl  # imported after the fork so workers never inherit its thread pool

    cat = pl.read_parquet(out / "catalog.parquet", columns=["uuid", "split", "complete"])
    status = dict(zip(cat["uuid"].to_list(), zip(cat["split"].to_list(), cat["complete"].to_list())))
    del cat

    shard = Path(paths.tol10m_dir) / "dataset" / "EOL" / f"{name}.tar.gz"
    t0 = time.time()
    pigz = subprocess.Popen(["pigz", "-dc", str(shard)], stdout=subprocess.PIPE, bufsize=1 << 20)
    kept, n_members, n_incomplete, n_unknown, bad_names = [], 0, 0, 0, []
    split_counts = {"train": 0, "val": 0}
    split_counts_incomplete = {"train": 0, "val": 0}
    with tarfile.open(fileobj=pigz.stdout, mode="r|", bufsize=1 << 20) as tf:
        for m in tf:
            if not m.isfile() or not MEMBER_RE.match(m.name):
                bad_names.append(m.name)
                continue
            n_members += 1
            uuid = m.name[:-4]
            st = status.get(uuid)
            if st is None:
                n_unknown += 1
                continue
            split, complete = st
            if not complete:
                n_incomplete += 1
                split_counts_incomplete[split] += 1
                continue
            data = tf.extractfile(m).read()
            _put(q_in, (len(kept), uuid, data), procs)
            kept.append(uuid)
            split_counts[split] += 1
            if limit and len(kept) >= limit:
                break
    t_read = time.time() - t0
    if limit:
        pigz.kill()
    rc = pigz.wait()
    for _ in procs:
        _put(q_in, None, procs)
    results = [_get(q_out, procs) for _ in procs]
    for p in procs:
        p.join()
    t_all = time.time() - t0

    n_kept = len(kept)
    os.truncate(u8_path, n_kept * ROW_BYTES)
    ids_path.write_text("".join(u + "\n" for u in kept))
    errors = [e for r in results for e in r["errors"]]
    big = sorted((b for r in results for b in r["big"]), key=lambda b: b["row"])
    n_decoded = sum(r["n"] for r in results)
    summary = {
        "shard": name, "pigz_returncode": rc, "n_members": n_members, "n_kept": n_kept,
        "n_decoded": n_decoded, "n_incomplete": n_incomplete, "n_unknown": n_unknown,
        "bad_member_names": bad_names[:20], "n_bad_member_names": len(bad_names),
        "split_counts_kept": split_counts, "split_counts_incomplete": split_counts_incomplete,
        "n_errors": len(errors), "errors": errors[:50], "n_all_black_crops": sum(r["n_black"] for r in results),
        "n_over_pil_default_limit": len(big), "n_refused_by_pil_default": sum(b["refused_by_default"] for b in big),
        "big_images": big, "u8_bytes": os.path.getsize(u8_path), "workers": workers,
        "seconds_read": round(t_read, 1), "seconds_total": round(t_all, 1),
        "img_per_s": round(n_kept / max(t_all, 1e-9), 1), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    summary["limit"] = limit
    ok = ((rc == 0 or limit) and not errors and not bad_names and n_unknown == 0 and n_decoded == n_kept)
    summary["ok"] = ok
    (done_path if ok else sdir / f"{name}.failed.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("big_images", "errors")}))
    if not ok:
        sys.exit(f"{name}: FAILED, see {sdir / (name + '.failed.json')}")


def step_merge(out: Path, paths) -> None:
    import polars as pl

    from src.data.tol_catalog import N_COMPLETE

    sdir = out / "shards"
    frames, dones = [], []
    for i in range(1, N_SHARDS + 1):
        name = shard_name(i)
        done = json.loads((sdir / f"{name}.done.json").read_text())
        ids = (sdir / f"{name}.uuids.txt").read_text().splitlines()
        if len(ids) != done["n_kept"] or done["u8_bytes"] != len(ids) * ROW_BYTES:
            raise RuntimeError(f"{name}: uuids/u8 size disagree with done.json")
        if os.path.getsize(sdir / f"{name}.u8") != len(ids) * ROW_BYTES:
            raise RuntimeError(f"{name}: .u8 file size is not {len(ids)} rows")
        frames.append(pl.DataFrame({"uuid": ids, "shard": [i] * len(ids), "row": list(range(len(ids)))},
                                   schema={"uuid": pl.Utf8, "shard": pl.Int16, "row": pl.Int32}))
        dones.append(done)
    idx = pl.concat(frames)
    cat = pl.read_parquet(out / "catalog.parquet")
    comp = cat.filter(pl.col("complete")).select("uuid", "split")
    if idx["uuid"].n_unique() != idx.height:
        raise RuntimeError("a uuid appears in more than one store row")
    joined = comp.join(idx, on="uuid", how="full", coalesce=True)
    missing = joined.filter(pl.col("shard").is_null()).height
    extra = joined.filter(pl.col("split").is_null()).height
    if missing or extra:
        raise RuntimeError(f"store vs catalog: {missing} complete uuids missing, {extra} store rows not complete")
    counts = {s: joined.filter(pl.col("split") == s).height for s in ("train", "val")}
    if counts != N_COMPLETE:
        raise RuntimeError(f"store split counts {counts} != {N_COMPLETE}")
    n_members = sum(d["n_members"] for d in dones)
    if n_members != N_MEMBERS_TOTAL:
        raise RuntimeError(f"{n_members} tar members, expected {N_MEMBERS_TOTAL}")
    index = joined.select("uuid", "split", "shard", "row").sort(["shard", "row"])
    index.write_parquet(out / "index.parquet")

    big = [dict(b, shard=d["shard"]) for d in dones for b in d["big_images"]]
    refused = [b for b in big if b["refused_by_default"]]
    # The cache lacks exactly the images PIL refused by default (preflight C6b); check against ids.txt.
    cache_ids = set((Path(paths.tol_embed_dir) / "results/probe_cache_b16/tol-eol/bioclip1/ids.txt").read_text().split())
    not_in_cache = index.filter(~pl.col("uuid").is_in(list(cache_ids)))
    refused_ids = {b["uuid"] for b in refused}
    summary = {
        "n_rows": index.height, "split_counts": counts, "n_members": n_members,
        "n_incomplete_members": sum(d["n_incomplete"] for d in dones),
        "n_over_pil_default_limit_kept": len(big), "n_refused_by_pil_default_kept": len(refused),
        "n_store_rows_not_in_cache": not_in_cache.height,
        "store_rows_not_in_cache_split": {s: not_in_cache.filter(pl.col("split") == s).height for s in ("train", "val")},
        "not_in_cache_equals_refused": set(not_in_cache["uuid"].to_list()) == refused_ids,
        "n_all_black_crops": sum(d["n_all_black_crops"] for d in dones),
        "shard_seconds_max": max(d["seconds_total"] for d in dones),
        "u8_bytes_total": sum(d["u8_bytes"] for d in dones),
    }
    not_in_cache.select("uuid", "split", "shard", "row").write_csv(out / "not_in_cache.csv")
    (out / "store_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["catalog", "shard", "merge"])
    ap.add_argument("--shard", type=int, help="1..63 (defaults to $SLURM_ARRAY_TASK_ID)")
    # sched_getaffinity sees the Slurm allocation; os.cpu_count() would see all 192 node CPUs.
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    ap.add_argument("--limit", type=int, default=0, help="trial run: stop after this many kept images")
    ap.add_argument("--out", default=None, help="output dir (default: <p1_data_dir>/store)")
    args = ap.parse_args()
    paths = load_paths()
    out = assert_outside_projects(args.out or Path(paths.p1_data_dir) / "store")
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "catalog":
        step_catalog(out, paths)
    elif args.step == "shard":
        idx = args.shard or int(os.environ["SLURM_ARRAY_TASK_ID"])
        step_shard(out, paths, idx, args.workers, args.limit)
    else:
        step_merge(out, paths)


if __name__ == "__main__":
    main()
