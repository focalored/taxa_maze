#!/usr/bin/env python3
"""
refbeh_gpu_jpeg.py -- follow-up to refbeh_gpu.py part E (audit/2026-10-02_reference_behaviour.md).

Part E found that re-saving the transformed 224 px crop as JPEG drops the
cosine to the cached BioCLIP 1 row to a median of about 0.80 (q95). That is
low enough to want a control. This script re-checks it on the first 200
members of image_set_01.tar.gz with:
  - PSNR of each re-saved crop against the original uint8 crop,
  - cosine of each variant both to the cache row and to a fresh fp16-autocast
    encoding of the untouched crop (which equals the cache bit for bit),
  - JPEG with the default 4:2:0 chroma subsampling and with 4:4:4,
  - JPEG re-save of the full-resolution original (before the transform),
  - lossless PNG and WebP round trips of the crop.
Writes audit/preflight/refbeh_gpu_jpeg.json. Imports nothing from tol_embed.
"""

import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

TAXA = Path("/projects/bdbk/liv/repos/taxa_maze")
sys.path.insert(0, str(TAXA))
from src.open_clip import create_model_and_transforms  # noqa: E402

OUT = TAXA / "audit/preflight/refbeh_gpu_jpeg.json"
EOL = Path("/u/liv/bdbk/data/tol10m/dataset/EOL")
TOLC = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1")


def cosine(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def psnr(u8a, u8b):
    mse = np.mean((u8a.astype(np.float64) - u8b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    from PIL import Image
    from torchvision.transforms import Compose
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    model, _, pv = create_model_and_transforms(f"local-dir:/u/liv/bdbk/ckpt/bioclip1", None,
                                               precision="amp", device="cuda", weights_only=True)
    model.eval()
    geo, to_t = Compose(pv.transforms[:3]), Compose(pv.transforms[3:])

    items = []
    with open(EOL / "image_set_01.tar.gz", "rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tf:
        for ti in tf:
            if ti.isreg():
                items.append((ti.name[:-4], tf.extractfile(ti).read()))
                if len(items) >= 200:
                    break
    ids = (TOLC / "shards/image_set_01.ids.txt").read_text().splitlines()[:200]
    assert [u for u, _ in items] == ids
    cached = np.asarray(np.load(TOLC / "emb.f16.npy", mmap_mode="r")[:200])

    def enc(X):
        outs = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(0, len(X), 256):
                f, extra = model.encode_image(X[i:i + 256].cuda())
                outs.append(f.float().cpu().numpy())
        return np.concatenate(outs)

    pil = []
    for _, data in items:
        im = Image.open(io.BytesIO(data)); im.load(); pil.append(im.convert("RGB"))
    crops = [geo(im) for im in pil]
    X0 = torch.stack([to_t(c) for c in crops])
    f0 = enc(X0)
    res = {"base_bit_identical_to_cache_rows": int((f0.astype(np.float16) == cached).all(1).sum()), "n": len(items)}

    def roundtrip(img, fmt, **kw):
        b = io.BytesIO(); img.save(b, format=fmt, **kw); b.seek(0)
        out = Image.open(b); out.load()
        return out.convert("RGB"), b.getbuffer().nbytes

    variants = {
        "crop_jpeg_q95_420": lambda c: roundtrip(c, "JPEG", quality=95),
        "crop_jpeg_q95_444": lambda c: roundtrip(c, "JPEG", quality=95, subsampling=0),
        "crop_jpeg_q100_444": lambda c: roundtrip(c, "JPEG", quality=100, subsampling=0),
        "crop_png": lambda c: roundtrip(c, "PNG"),
        "crop_webp_lossless": lambda c: roundtrip(c, "WEBP", lossless=True),
    }
    for name, fn in variants.items():
        t0 = time.time()
        outs = [fn(c) for c in crops]
        ps = [psnr(np.asarray(c), np.asarray(o)) for c, (o, _) in zip(crops, outs)]
        X = torch.stack([to_t(o) for o, _ in outs])
        f = enc(X)
        cc, cf = cosine(f, cached), cosine(f, f0)
        res[name] = dict(psnr_db_median=float(np.median(ps)), psnr_db_min=float(np.min(ps)),
                         bytes_median=float(np.median([n for _, n in outs])),
                         tensor_bit_identical=bool(torch.equal(X, X0)),
                         cos_to_cache_min=float(cc.min()), cos_to_cache_median=float(np.median(cc)),
                         n_cos_to_cache_below_0_9999=int((cc < 0.9999).sum()),
                         cos_to_fresh_median=float(np.median(cf)), seconds=time.time() - t0)
        print(name, res[name], flush=True)
    # full-resolution JPEG re-save before the transform
    outs = [roundtrip(im, "JPEG", quality=95) for im in pil]
    X = torch.stack([pv(o) for o, _ in outs])
    f = enc(X)
    cc = cosine(f, cached)
    res["fullres_jpeg_q95_420_then_transform"] = dict(
        cos_to_cache_min=float(cc.min()), cos_to_cache_median=float(np.median(cc)),
        n_cos_to_cache_below_0_9999=int((cc < 0.9999).sum()))
    print("fullres", res["fullres_jpeg_q95_420_then_transform"], flush=True)
    # sanity: cosine between two DIFFERENT images, for scale
    cdiff = cosine(f0[:-1], f0[1:])
    res["scale_cos_between_consecutive_different_images_median"] = float(np.median(cdiff))
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
