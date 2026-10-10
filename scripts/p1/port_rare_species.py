"""Port the Rare Species dataset (HF imageomics/rare-species, a CC0 benchmark of 400 IUCN-listed species, EOL images excluded from
TreeOfLife-10M) into the eval harness's format: images as files, plus ids.txt, codes.i32.npy (N x 7) and vocab.json as the iNat21
cache has them, so `src.eval.zeroshot.evaluate_ranks` runs unchanged. Usage: `python scripts/p1/port_rare_species.py --root /u/liv/data/taxa_maze/rare_species`.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/u/liv/data/taxa_maze/rare_species"))
    args = ap.parse_args()
    hf, img_root, lab = args.root / "hf", args.root / "images", args.root / "labels"
    img_root.mkdir(exist_ok=True); lab.mkdir(exist_ok=True)
    t0 = time.time()
    meta = list(csv.DictReader(open(hf / "metadata.csv", encoding="utf-8")))
    by_id = {r["rarespecies_id"]: r for r in meta}
    if len(by_id) != len(meta):
        raise RuntimeError("rarespecies_id is not unique in metadata.csv")

    # 1. images: parquet struct bytes -> files at metadata.csv's file_name (dataset/<taxon>/<file>.jpg), joined on rarespecies_id
    n_written, n_existing, bad = 0, 0, []
    for shard in sorted((hf / "data").glob("train-*.parquet")):
        df = pl.read_parquet(shard, columns=["file_name", "rarespecies_id"])
        for row in df.iter_rows(named=True):
            r = by_id[row["rarespecies_id"]]
            out = img_root / r["file_name"]
            if row["file_name"]["path"] != Path(r["file_name"]).name:
                bad.append((row["rarespecies_id"], row["file_name"]["path"], r["file_name"]))
            if out.exists() and out.stat().st_size == len(row["file_name"]["bytes"]):
                n_existing += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(row["file_name"]["bytes"])
            n_written += 1
    if bad:
        raise RuntimeError(f"{len(bad)} rows whose parquet file name differs from metadata.csv, e.g. {bad[:2]}")
    print(f"images: {n_written} written, {n_existing} already present; {time.time() - t0:.0f} s", flush=True)

    # 2. labels in the iNat21 cache's format; rows follow metadata.csv's order; the species leaf is the binomial, as iNat21's vocab has it
    paths = {k: [] for k in RANKS}
    index = {k: {} for k in RANKS}
    codes = np.full((len(meta), 7), -1, dtype=np.int32)
    ids = []
    for i, r in enumerate(meta):
        parts = [r[k] for k in RANKS]
        if any(not x for x in parts):
            raise RuntimeError(f"row {i} has an empty rank: {parts}")
        parts[6] = f"{r['genus']} {r['species']}"
        if parts[6] != r["sciName"]:
            raise RuntimeError(f"row {i}: genus+species {parts[6]!r} differs from sciName {r['sciName']!r}")
        for d, k in enumerate(RANKS):
            key = "|".join(parts[: d + 1])
            if key not in index[k]:
                index[k][key] = len(paths[k]); paths[k].append(key)
            codes[i, d] = index[k][key]
        ids.append(r["file_name"])
    (lab / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    np.save(lab / "codes.i32.npy", codes)
    (lab / "vocab.json").write_text(json.dumps({"sep": "|", "vocab": paths}, indent=0, ensure_ascii=False), encoding="utf-8")

    # 3. checks: every file opens, counts per rank, and the manifest
    t1 = time.time(); sizes = []
    for f in ids:
        with Image.open(img_root / f) as im:
            im.verify(); sizes.append(im.size)
    per_species = Counter(codes[:, 6].tolist())
    manifest = {"source": "https://huggingface.co/datasets/imageomics/rare-species (CC0-1.0)", "downloaded": "2026-10-10 via huggingface_hub snapshot_download, anonymous",
                "metadata_csv_sha256": hashlib.sha256((hf / "metadata.csv").read_bytes()).hexdigest(), "n_images": len(ids),
                "classes_per_rank": {k: len(paths[k]) for k in RANKS}, "images_per_species": {"min": min(per_species.values()), "median": float(np.median(list(per_species.values()))), "max": max(per_species.values())},
                "image_size_px": {"min_side_min": min(min(s) for s in sizes), "max_side_max": max(max(s) for s in sizes)},
                "format": "ids.txt (relative to images/), codes.i32.npy (N x 7, index into vocab[rank]), vocab.json ({sep, vocab: {rank: [path]}}); species leaf is the binomial",
                "ported": datetime.datetime.now().isoformat(timespec="seconds")}
    (lab / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"labels: {len(ids)} rows; classes per rank {manifest['classes_per_rank']}; images per species {manifest['images_per_species']}; all files verified in {time.time() - t1:.0f} s; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
