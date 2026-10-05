#!/usr/bin/env python3
"""
refbeh_census.py -- full member census of the 63 ToL-EOL shards versus the
BioCLIP 1 cache (audit/2026-10-02_reference_behaviour.md, R1).

Why: image_set_63.tar.gz holds 19,674 <uuid>.jpg members but the cache's
shards/image_set_63.ids.txt has 19,673, while that shard's done.json reports
n_skipped = 0. In encode_cache.py the per-shard counter only sees exceptions
whose args carry the shard url; webdataset 0.2.86's decode stage
(filters._decode) passes exceptions to the handler without the url, so decode
failures land in the handler's "unattributed" slot (encode_cache.py:1037-1041,
1398-1399) and never reach done.json. This census finds every member that the
cache lacks and records why it fails to decode.

For each shard (run in parallel, one process per shard, read-only):
  - stream with tarfile 'r|gz' (sequential, like webdataset's 'r|*'),
  - key = webdataset base_plus_ext rule (path up to the first dot of the basename),
  - members not in the cached ids: try the reference decode
    (PIL.Image.open(BytesIO); load(); convert("RGB") with LOAD_TRUNCATED_IMAGES=False),
    then again with LOAD_TRUNCATED_IMAGES=True, and note whether the bytes end in FFD9,
  - check that tar order minus the dropped members equals the cached ids order.

Writes audit/preflight/refbeh_census.json. Imports nothing from tol_embed.
"""

import io
import json
import os
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

EOL = Path("/u/liv/bdbk/data/tol10m/dataset/EOL")
TOLC = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1")
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_census.json")


def wds_key(name):
    m = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", name)
    return (None, None) if not m else (m.group(1), m.group(2).lower())


def try_decode(data, allow_truncated):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    try:
        with io.BytesIO(data) as s:
            img = Image.open(s)
            img.load()
            img = img.convert("RGB")
            return dict(ok=True, mode_size=[img.mode, list(img.size)])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def census(stem):
    t0 = time.time()
    ids = (TOLC / "shards" / f"{stem}.ids.txt").read_text().splitlines()
    idset = set(ids)
    keys, dropped, types, nbytes = [], [], {}, 0
    err = None
    try:
        with open(EOL / f"{stem}.tar.gz", "rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for ti in tf:
                t = "reg" if ti.isreg() else ("dir" if ti.isdir() else f"type_{ti.type!r}")
                types[t] = types.get(t, 0) + 1
                if not ti.isreg():
                    continue
                data = tf.extractfile(ti).read()
                nbytes += len(data)
                key, suffix = wds_key(ti.name)
                keys.append(key)
                if key not in idset:
                    dropped.append(dict(
                        name=ti.name, suffix=suffix, bytes=len(data), position_in_tar=len(keys) - 1,
                        ends_with_ffd9=data[-2:] == b"\xff\xd9", starts_with_ffd8=data[:2] == b"\xff\xd8",
                        reference_decode=try_decode(data, False),
                        decode_with_truncated_allowed=try_decode(data, True)))
            comp = raw.tell()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        comp = None
    kept = [k for k in keys if k in idset]
    el = time.time() - t0
    return dict(shard=stem, n_members=sum(types.values()), types=types, n_cached=len(ids),
                n_dropped=len(dropped), dropped=dropped,
                n_cached_not_in_tar=len(idset - set(keys)),
                order_equal_after_removing_dropped=kept == ids,
                n_duplicate_keys=len(keys) - len(set(keys)),
                seconds=el, member_mb=nbytes / 1e6,
                compressed_mb_per_s=(comp / 1e6 / el) if comp else None,
                members_per_s=len(keys) / max(el, 1e-9), stream_error=err)


def main():
    stems = sorted(p.name[: -len(".tar.gz")] for p in EOL.glob("image_set_*.tar.gz"))
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(time.strftime("[%H:%M:%S]"), f"{len(stems)} shards, {nproc} processes", flush=True)
    res = []
    # one process per shard; results arrive in completion order and are sorted at the end
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(census, stems):
            res.append(r)
            print(time.strftime("[%H:%M:%S]"), r["shard"], "members", r["n_members"], "cached", r["n_cached"],
                  "dropped", r["n_dropped"], "order_ok", r["order_equal_after_removing_dropped"],
                  "%.0fs %.0f MB/s" % (r["seconds"], r["compressed_mb_per_s"] or -1),
                  ("ERR " + r["stream_error"]) if r["stream_error"] else "", flush=True)
            OUT.write_text(json.dumps(dict(partial=True, shards=sorted(res, key=lambda x: x["shard"])), indent=1))
    res.sort(key=lambda x: x["shard"])
    summary = dict(
        n_shards=len(res),
        total_members=sum(r["n_members"] for r in res),
        total_cached=sum(r["n_cached"] for r in res),
        total_dropped=sum(r["n_dropped"] for r in res),
        total_cached_not_in_tar=sum(r["n_cached_not_in_tar"] for r in res),
        all_order_equal=all(r["order_equal_after_removing_dropped"] for r in res),
        any_stream_error=[r["shard"] for r in res if r["stream_error"]],
        any_non_regular=[r["shard"] for r in res if set(r["types"]) != {"reg"}],
        any_duplicate_keys=[r["shard"] for r in res if r["n_duplicate_keys"]],
        dropped_list=[dict(shard=r["shard"], **d) for r in res for d in r["dropped"]],
        compressed_mb_per_s_min_median_max=None,
    )
    rates = sorted(r["compressed_mb_per_s"] for r in res if r["compressed_mb_per_s"])
    if rates:
        summary["compressed_mb_per_s_min_median_max"] = [rates[0], rates[len(rates) // 2], rates[-1]]
    mps = sorted(r["members_per_s"] for r in res)
    summary["members_per_s_min_median_max"] = [mps[0], mps[len(mps) // 2], mps[-1]]
    OUT.write_text(json.dumps(dict(partial=False, summary=summary, shards=res), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "dropped_list"}, indent=1))
    for d in summary["dropped_list"]:
        print("DROPPED", d["shard"], d["name"], d["bytes"], "ffd9" if d["ends_with_ffd9"] else "no-ffd9",
              d["reference_decode"], d["decode_with_truncated_allowed"])
    print(time.strftime("[%H:%M:%S]"), "wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
