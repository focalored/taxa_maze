#!/usr/bin/env python
"""Preflight check C5: compare a shard's tar member list with the cache's ids.txt.

Inputs (read-only):
  audit/preflight/data_tar_members_image_set_NN.txt   (written by data_tar_list.sh)
  /projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/shards/image_set_NN.ids.txt
  audit/preflight/data_cache_missing_ids.csv           (optional; written by data_catalog_facts.py)
Output:
  audit/preflight/data_tar_image_set_NN.json

If --t0/--t1 are omitted, timing is reused from an existing output JSON, so the
script can be re-run after data_catalog_facts.py has written the missing-id list.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter

CACHE = "/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1"
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
TARDIR = "/u/liv/bdbk/data/tol10m/dataset/EOL"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="two-digit shard number, e.g. 60")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    a = ap.parse_args()
    name = f"image_set_{a.shard}"
    out_json = f"{OUT}/data_tar_{name}.json"

    if a.t0 is None or a.t1 is None:
        prev = json.load(open(out_json))
        t0, t1 = prev["wall_t0"], prev["wall_t1"]
    else:
        t0, t1 = a.t0, a.t1

    with open(f"{OUT}/data_tar_members_{name}.txt") as fh:
        members = [ln.rstrip("\n") for ln in fh if ln.rstrip("\n")]
    dirs = [m for m in members if m.endswith("/")]
    files = [m for m in members if not m.endswith("/")]
    jpgs = [m for m in files if m.endswith(".jpg")]
    non_jpg = [m for m in files if not m.endswith(".jpg")]
    prefixes = Counter(os.path.dirname(m) for m in files)
    tar_uuids = [os.path.basename(m)[: -len(".jpg")] for m in jpgs]
    bad_uuid = [u for u in tar_uuids if not UUID_RE.match(u)]

    with open(f"{CACHE}/shards/{name}.ids.txt") as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]

    st, si = set(tar_uuids), set(ids)
    only_tar = sorted(st - si)
    only_ids = sorted(si - st)
    dup_tar = [u for u, c in Counter(tar_uuids).items() if c > 1]

    missing_in_tar = None
    miss_path = f"{OUT}/data_cache_missing_ids.csv"
    if os.path.exists(miss_path):
        with open(miss_path) as fh:
            miss = [r["treeoflife_id"] for r in csv.DictReader(fh)]
        hits = sorted(set(miss) & st)
        missing_in_tar = {"n_cache_missing_ids_checked": len(miss),
                          "n_found_in_this_tar": len(hits), "examples": hits[:10]}

    wall = t1 - t0
    nbytes = os.path.getsize(f"{TARDIR}/{name}.tar.gz")
    res = {
        "shard": name,
        "tar_path": f"{TARDIR}/{name}.tar.gz",
        "tar_bytes": nbytes,
        "wall_t0": t0,
        "wall_t1": t1,
        "tar_wall_s": wall,
        "read_MB_per_s": nbytes / wall / 1e6,
        "members_per_s": len(jpgs) / wall,
        "n_members": len(members),
        "n_dir_members": len(dirs),
        "n_jpg_members": len(jpgs),
        "n_non_jpg_file_members": len(non_jpg),
        "non_jpg_examples": non_jpg[:10],
        "member_dir_prefixes": dict(prefixes.most_common(5)),
        "n_member_names_not_uuid": len(bad_uuid),
        "bad_uuid_examples": bad_uuid[:10],
        "n_duplicate_tar_uuids": len(dup_tar),
        "n_ids_txt": len(ids),
        "sets_equal": st == si,
        "order_equal": tar_uuids == ids,
        "order_equal_after_dropping_tar_only": [u for u in tar_uuids if u in si] == ids,
        "tar_positions_of_tar_only": [i for i, u in enumerate(tar_uuids) if u not in si][:20],
        "n_only_in_tar": len(only_tar),
        "only_in_tar_examples": only_tar[:10],
        "n_only_in_ids_txt": len(only_ids),
        "only_in_ids_txt_examples": only_ids[:10],
        "first_members": members[:3],
        "last_members": members[-3:],
        "cache_missing_ids_vs_this_tar": missing_in_tar,
    }
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
