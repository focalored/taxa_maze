#!/usr/bin/env python
"""Follow-up to C5/C6: stream one shard and try to decode the members that the
BioCLIP 1 cache dropped (tar members absent from shards/image_set_NN.ids.txt).

Input:  audit/preflight/data_tar_image_set_NN.json (from data_tar_compare.py)
Output: audit/preflight/data_tar_probe_image_set_NN.json
Images are inspected in memory only; nothing is extracted to disk.
"""
import argparse
import io
import json
import os
import tarfile
import time

from PIL import Image

OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
TARDIR = "/u/liv/bdbk/data/tol10m/dataset/EOL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--no-limit", action="store_true",
                    help="set PIL.Image.MAX_IMAGE_PIXELS=None, time the full decode and a 224-px resize")
    a = ap.parse_args()
    name = f"image_set_{a.shard}"
    if a.no_limit:
        Image.MAX_IMAGE_PIXELS = None
    cmp_ = json.load(open(f"{OUT}/data_tar_{name}.json"))
    targets = set(cmp_["only_in_tar_examples"])
    assert len(targets) == cmp_["n_only_in_tar"], "more targets than examples stored"
    t0 = time.time()
    found, n_seen = {}, 0
    with tarfile.open(f"{TARDIR}/{name}.tar.gz", mode="r|gz") as tf:
        for m in tf:
            n_seen += 1
            u = os.path.basename(m.name)[:-4]
            if u not in targets:
                continue
            b = tf.extractfile(m).read()
            rec = {"member": m.name, "tar_size": m.size, "bytes_read": len(b),
                   "magic_hex": b[:12].hex(), "position": n_seen - 1}
            try:
                im = Image.open(io.BytesIO(b))
                rec.update({"pil_format": im.format, "pil_mode": im.mode, "pil_size": list(im.size)})
                td = time.time()
                im.load()
                rgb = im.convert("RGB")
                rec["decode_ok"] = True
                rec["rgb_size"] = list(rgb.size)
                rec["decode_s"] = time.time() - td
                if a.no_limit:  # shortest side -> 224, bicubic, then centre crop (as in open_clip val)
                    tr = time.time()
                    w, h = rgb.size
                    s_ = 224 / min(w, h)
                    small = rgb.resize((max(224, round(w * s_)), max(224, round(h * s_))), Image.BICUBIC)
                    rec["resize_224_s"] = time.time() - tr
                    rec["resized_size"] = list(small.size)
            except Exception as ex:  # record, do not raise
                rec["decode_ok"] = False
                rec["error"] = f"{type(ex).__name__}: {ex}"
            found[u] = rec
            print(json.dumps({u: rec}), flush=True)
            if len(found) == len(targets):
                break
    res = {"shard": name, "targets": sorted(targets), "found": found,
           "members_scanned": n_seen, "elapsed_s": time.time() - t0}
    sfx = "_nolimit" if a.no_limit else ""
    res["pil_max_image_pixels"] = Image.MAX_IMAGE_PIXELS
    with open(f"{OUT}/data_tar_probe_{name}{sfx}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
