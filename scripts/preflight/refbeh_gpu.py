#!/usr/bin/env python3
"""
refbeh_gpu.py -- GPU checks for the 2026-10-02 reference-behaviour audit
(audit/2026-10-02_reference_behaviour.md). Run through Slurm (refbeh_gpu.sh).

It uses the COPIED open_clip at taxa_maze/src/open_clip. It imports nothing
from tol_embed; the reference rules it tests are re-implemented here from the
reading of tol_embed/scripts/encode_cache.py and zeroshot_ranks.py.

Parts:
  E  ToL-EOL BioCLIP 1 re-encode (spec line 50). The first 1,000 members of
     image_set_01.tar.gz (and 500 of image_set_63) are read with tarfile + PIL,
     transformed with the factory's preprocess_val, encoded, and compared by
     uuid with probe_cache_b16/tol-eol/bioclip1/emb.f16.npy. Variants: fp16
     autocast (batch 256 and 1000), fp32, bf16 autocast, pure-fp16 weights,
     JPEG re-save of the 224 px crop (q 75/95/100), lossless uint8/PNG store,
     EXIF transpose. Also checks that webdataset 0.2.86's pipeline yields the
     same keys and bit-identical tensors as tarfile + PIL.
  F  iNat21-val BioCLIP 1 re-encode of 512 seeded rows vs the inat21-val cache.
  G  Zero-shot reproduction: our own scoring code, run on the cached image
     embeddings for all 21 (checkpoint, form) cells in the three JSONs, compared
     on correct counts, n_classes, n_paths and n_eval. Plus sensitivity runs
     (text batch, TF32 / "medium" matmul precision, fp16 text autocast, chunk).

Writes only audit/preflight/refbeh_gpu.json (and the Slurm log).
"""

import io
import json
import os
import platform
import random
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TAXA = Path("/projects/bdbk/liv/repos/taxa_maze")
sys.path.insert(0, str(TAXA))
from src.open_clip import create_model_and_transforms, get_tokenizer  # noqa: E402  (copied package)

OUT = TAXA / "audit/preflight/refbeh_gpu.json"
RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
EOL = Path("/u/liv/bdbk/data/tol10m/dataset/EOL")
INAT = Path("/u/liv/bdbk/data/inat21")
TOLC = RES / "probe_cache_b16/tol-eol/bioclip1"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
JSONS = ["ZEROSHOT_RANKS_inat21-val.json", "ZEROSHOT_RANKS_inat21-val__forms.json",
         "ZEROSHOT_RANKS_inat21-val__bfl.json"]
CKPT = "/u/liv/bdbk/ckpt"
# Re-stated from tol_embed/scripts/data_paths.py:326-379 (not imported).
MODEL = {
    "bioclip2": "hf-hub:imageomics/bioclip-2",
    "bioclip1": f"local-dir:{CKPT}/bioclip1",
    "rcme": f"local-dir:{CKPT}/rcme",
    "openclip-b16": f"local-dir:{CKPT}/openclip-b16",
    "clip-l14-laion2b": f"local-dir:{CKPT}/clip-l14-laion2b",
    "bfl-euclidean": f"local-dir:{CKPT}/bfl-euclidean",
    "bfl-hyperbolic": f"local-dir:{CKPT}/bfl-hyperbolic-vanilla",
}
DEV = "cuda"
out = {"env": {}, "E_tol_reencode": {}, "F_inat_reencode": {}, "G_zeroshot": {}}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def cache_dir(ck, route="inat21-val"):
    return RES / ("probe_cache" if ck == "bioclip2" else "probe_cache_b16") / route / ck


def set_reference_backend():
    """Mirror encode_cache.set_determinism (encode_cache.py:452-478) with seed 0."""
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("highest")
    return {"cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def compare(fresh32, cached16):
    f = fresh32.astype(np.float64)
    c = cached16.astype(np.float64)
    cos = (f * c).sum(1) / np.maximum(np.linalg.norm(f, axis=1) * np.linalg.norm(c, axis=1), 1e-30)
    return dict(n=int(len(cos)), min_cos=float(cos.min()), p01_cos=float(np.percentile(cos, 1)),
                median_cos=float(np.median(cos)), n_cos_below_0_9999=int((cos < 0.9999).sum()),
                max_abs=float(np.abs(f - c).max()),
                n_rows_bit_identical_after_fp16_cast=int((fresh32.astype(np.float16) == cached16).all(axis=1).sum()),
                fp16_spacing_at_max_abs=float(np.spacing(np.float16(np.abs(cached16).max()))))


def encode_images(model, X, batch, mode):
    feats = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch].to(DEV, non_blocking=True)
            if mode == "fp16_autocast":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f, extra = model.encode_image(xb)
            elif mode == "bf16_autocast":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    f, extra = model.encode_image(xb)
            elif mode == "fp32":
                f, extra = model.encode_image(xb)
            elif mode == "fp16_weights":
                f, extra = model.encode_image(xb.half())
            else:
                raise ValueError(mode)
            assert extra is None, "continual head is live"
            feats.append(f.float().cpu().numpy())
    out_dtype = str(f.dtype)
    return np.concatenate(feats), out_dtype


def read_tar_prefix(path, n):
    items = []
    with open(path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tf:
        for ti in tf:
            if not ti.isreg():
                continue
            items.append((ti.name, tf.extractfile(ti).read()))
            if len(items) >= n:
                break
    return items


# ---------------------------------------------------------------------------
# E. ToL re-encode
# ---------------------------------------------------------------------------
def part_e():
    from PIL import Image, ImageOps
    import webdataset as wds
    from torchvision.transforms import Compose
    E = out["E_tol_reencode"]
    backend = set_reference_backend()
    E["backend"] = backend
    model, _, pv = create_model_and_transforms(MODEL["bioclip1"], None, precision="amp", device=DEV,
                                               weights_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    E["param_count"] = int(sum(p.numel() for p in model.parameters()))
    E["visual_output_dim"] = int(model.visual.output_dim)
    E["param_dtype"] = str(next(model.parameters()).dtype)
    E["preprocess_cfg"] = dict(model.visual.preprocess_cfg)
    E["preprocess_val_repr"] = re.sub(r" at 0x[0-9a-f]+", "", repr(pv))
    log("E: model", E["param_count"], E["visual_output_dim"], E["param_dtype"])

    ids = (TOLC / "ids.txt").read_text().splitlines()
    row_of = {u: i for i, u in enumerate(ids)}
    emb = np.load(TOLC / "emb.f16.npy", mmap_mode="r")
    E["cache_shape"] = list(emb.shape)

    def load_shard(stem, n):
        t0 = time.time()
        items = read_tar_prefix(EOL / f"{stem}.tar.gz", n)
        uu = [nm[:-4] for nm, _ in items]
        pil = []
        for nm, data in items:
            im = Image.open(io.BytesIO(data)); im.load(); im = im.convert("RGB")
            pil.append(im)
        X = torch.stack([pv(im) for im in pil])
        rows = np.array([row_of[u] for u in uu])
        log(f"E: read+decode+transform {len(items)} from {stem} in {time.time() - t0:.1f}s")
        return items, uu, pil, X, rows

    items, uu, pil, X, rows = load_shard("image_set_01", 1000)
    cached = np.asarray(emb[rows])
    E["image_set_01_rows_first_last"] = [int(rows[0]), int(rows[-1])]
    E["image_set_01_rows_are_0_to_999"] = bool((rows == np.arange(1000)).all())

    # webdataset 0.2.86 reference path on the same shard: same keys, same tensors?
    t0 = time.time()
    pipe = wds.DataPipeline(
        wds.SimpleShardList([str(EOL / "image_set_01.tar.gz")]),
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),
        wds.to_tuple("jpg", "__key__"),
        wds.map_tuple(pv, lambda k: k),
    )
    wk, wx = [], []
    for x, k in pipe:
        wk.append(k); wx.append(x)
        if len(wk) >= 1000:
            break
    WX = torch.stack(wx)
    E["webdataset_keys_equal_tarfile_uuids"] = wk == uu
    E["webdataset_tensors_bit_identical_to_tarfile_pil"] = bool(torch.equal(WX, X))
    E["webdataset_read_seconds"] = time.time() - t0
    log("E: webdataset parity", E["webdataset_keys_equal_tarfile_uuids"],
        E["webdataset_tensors_bit_identical_to_tarfile_pil"])

    variants = {}
    for name, batch, mode in [("fp16_autocast_b256", 256, "fp16_autocast"),
                              ("fp16_autocast_b1000", 1000, "fp16_autocast"),
                              ("fp16_autocast_b64", 64, "fp16_autocast"),
                              ("fp32_b256", 256, "fp32"),
                              ("bf16_autocast_b256", 256, "bf16_autocast")]:
        t0 = time.time()
        fresh, odt = encode_images(model, X, batch, mode)
        variants[name] = dict(compare(fresh, cached), output_dtype=odt, seconds=time.time() - t0)
        log("E:", name, variants[name])
    # pure fp16 weights (precision="fp16" in the factory) -- a different mode from autocast
    model_h, _, _ = create_model_and_transforms(MODEL["bioclip1"], None, precision="fp16", device=DEV,
                                                weights_only=True)
    model_h.eval()
    fresh, odt = encode_images(model_h, X, 256, "fp16_weights")
    variants["fp16_weights_b256"] = dict(compare(fresh, cached), output_dtype=odt)
    del model_h
    torch.cuda.empty_cache()
    log("E: fp16_weights", variants["fp16_weights_b256"])

    # storage variants of the transformed 224 px crop
    tfs = pv.transforms
    geo = Compose(tfs[:3])          # Resize, CenterCrop, _convert_to_rgb
    to_t = Compose(tfs[3:])         # ToTensor, Normalize
    crops = [geo(im) for im in pil]
    Xu8 = torch.stack([to_t(Image.fromarray(np.asarray(c).copy())) for c in crops])
    E["uint8_crop_store_tensor_bit_identical"] = bool(torch.equal(Xu8, X))
    pngs = []
    for c in crops:
        b = io.BytesIO(); c.save(b, format="PNG"); b.seek(0)
        pngs.append(to_t(Image.open(b).convert("RGB")))
    E["png_crop_store_tensor_bit_identical"] = bool(torch.equal(torch.stack(pngs), X))
    E["crop_size"] = list(crops[0].size)
    for q in (75, 95, 100):
        xs = []
        for c in crops:
            b = io.BytesIO(); c.save(b, format="JPEG", quality=q); b.seek(0)
            xs.append(to_t(Image.open(b).convert("RGB")))
        fresh, odt = encode_images(model, torch.stack(xs), 256, "fp16_autocast")
        variants[f"jpeg_q{q}_resave_fp16_autocast_b256"] = compare(fresh, cached)
        log(f"E: jpeg q{q}", variants[f"jpeg_q{q}_resave_fp16_autocast_b256"])
    # EXIF transpose (the reference does NOT apply it)
    orient = []
    for nm, data in items:
        with Image.open(io.BytesIO(data)) as im0:
            try:
                orient.append(int(im0.getexif().get(274, 1)))
            except Exception:
                orient.append(-1)
    orient = np.array(orient)
    sel = np.flatnonzero(orient > 1)
    E["n_exif_orientation_gt1_in_first1000"] = int(len(sel))
    E["exif_orientation_values"] = {str(k): int(v) for k, v in zip(*np.unique(orient, return_counts=True))}
    if len(sel):
        xs = []
        for i in sel:
            im = Image.open(io.BytesIO(items[i][1])); im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
            xs.append(pv(im))
        fresh, _ = encode_images(model, torch.stack(xs), 256, "fp16_autocast")
        variants["exif_transpose_applied_on_rotated_only"] = compare(fresh, cached[sel])
        fresh0, _ = encode_images(model, X[sel], 256, "fp16_autocast")
        variants["no_exif_transpose_same_rows"] = compare(fresh0, cached[sel])
    E["variants_image_set_01_first1000"] = variants
    save()

    # a second shard: first 500 of image_set_63 (rows near the end of the cache)
    items63, uu63, pil63, X63, rows63 = load_shard("image_set_63", 500)
    fresh, _ = encode_images(model, X63, 256, "fp16_autocast")
    E["image_set_63_first500_fp16_autocast_b256"] = dict(compare(fresh, np.asarray(emb[rows63])),
                                                         rows_first_last=[int(rows63[0]), int(rows63[-1])])
    log("E: image_set_63", E["image_set_63_first500_fp16_autocast_b256"])
    del model
    torch.cuda.empty_cache()
    save()


# ---------------------------------------------------------------------------
# F. iNat21-val re-encode
# ---------------------------------------------------------------------------
def part_f():
    from PIL import Image
    Fo = out["F_inat_reencode"]
    set_reference_backend()
    model, _, pv = create_model_and_transforms(MODEL["bioclip1"], None, precision="amp", device=DEV,
                                               weights_only=True)
    model.eval()
    cdir = cache_dir("bioclip1")
    names = (cdir / "ids.txt").read_text().splitlines()
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(names), size=512, replace=False))
    X = torch.stack([pv(Image.open(INAT / names[i]).convert("RGB")) for i in rows])
    emb = np.load(cdir / "emb.f16.npy", mmap_mode="r")
    fresh, odt = encode_images(model, X, 256, "fp16_autocast")
    Fo["bioclip1_512_seeded_rows_fp16_autocast_b256"] = dict(compare(fresh, np.asarray(emb[rows])), output_dtype=odt)
    fresh32, _ = encode_images(model, X, 256, "fp32")
    Fo["bioclip1_512_seeded_rows_fp32_b256"] = compare(fresh32, np.asarray(emb[rows]))
    log("F:", Fo)
    del model
    torch.cuda.empty_cache()
    save()


# ---------------------------------------------------------------------------
# G. zero-shot reproduction
# ---------------------------------------------------------------------------
def render(path, rank_idx, form, sep="|"):
    parts = path.split(sep)[: rank_idx + 1]
    if rank_idx == len(RANKS) - 1 and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    if form == "bare":
        return parts[-1]
    text = " ".join(parts)
    return f"a photo of {text}." if form == "photo" else text


def encode_texts(model, tok, texts, batch=256, autocast_dtype=None):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = tok(texts[i:i + batch]).to(DEV)
            if autocast_dtype is None:
                e = model.encode_text(t)
            else:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    e = model.encode_text(t)
            outs.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(outs)


def zs_eval(model, tok, ck, form, text_batch=256, autocast_dtype=None, chunk=8192, extra_stats=False):
    cdir = cache_dir(ck)
    emb = np.load(cdir / "emb.f16.npy")              # 100k x D fp16, 100-150 MB
    codes = np.load(cdir / "codes.i32.npy")
    vocab = json.loads((cdir / "vocab.json").read_text())
    sep, V = vocab.get("sep", "|"), vocab["vocab"]
    n = emb.shape[0]
    res = {}
    for r, rank in enumerate(RANKS):
        paths = V[rank]
        strings = [render(p, r, form, sep) for p in paths]
        uniq, first = [], {}
        path2class = np.empty(len(paths), dtype=np.int64)
        for i, s in enumerate(strings):
            if s not in first:
                first[s] = len(uniq)
                uniq.append(s)
            path2class[i] = first[s]
        T = encode_texts(model, tok, uniq, batch=text_batch, autocast_dtype=autocast_dtype).to(DEV)
        gold_path = codes[:n, r].astype(np.int64)
        valid = gold_path >= 0
        gold = np.where(valid, path2class[np.clip(gold_path, 0, None)], -1)
        correct, ties = 0, 0
        preds = []
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            x = F.normalize(torch.from_numpy(emb[a:b].astype(np.float32)).to(DEV), dim=-1)
            s = x @ T.T
            pred = s.argmax(dim=1)
            if extra_stats and s.shape[1] >= 2:
                top2 = s.topk(2, dim=1).values
                ties += int((top2[:, 0] == top2[:, 1]).sum())
            pred = pred.cpu().numpy()
            preds.append(pred)
            m = valid[a:b]
            correct += int((pred[m] == gold[a:b][m]).sum())
        cell = dict(correct=correct, n_classes=len(uniq), n_paths=len(paths), n_eval=int(valid.sum()),
                    top1=100.0 * correct / max(int(valid.sum()), 1))
        if extra_stats:
            tok_len = (tok(uniq) != 0).sum(dim=1)
            cell.update(n_images_with_exact_top2_tie=ties,
                        n_distinct_text_rows=int(torch.unique(T.cpu(), dim=0).shape[0]),
                        max_tokens_incl_sot_eot=int(tok_len.max()),
                        n_strings_truncated_at_77=int((tok_len >= 77).sum()))
        res[rank] = cell
        res[rank]["_pred_sha"] = __import__("hashlib").sha256(np.concatenate(preds).tobytes()).hexdigest()[:16]
        del T
    res["average"] = dict(top1=float(np.mean([res[r]["top1"] for r in RANKS])))
    return res


def part_g():
    G = out["G_zeroshot"]
    G["backend"] = set_reference_backend()
    want = {}  # (ck, form) -> {rank: cell} from the JSONs
    for name in JSONS:
        j = json.loads((RES / name).read_text())
        for form, d in j["results"].items():
            for ck, cells in d.items():
                want.setdefault((ck, form), {})[name] = cells
    order = ["openclip-b16", "bioclip1", "rcme", "bioclip2", "clip-l14-laion2b", "bfl-euclidean", "bfl-hyperbolic"]
    results, summary = {}, []
    for ck in order:
        forms = sorted({f for (c, f) in want if c == ck})
        if not forms:
            continue
        t0 = time.time()
        model, _, _ = create_model_and_transforms(MODEL[ck], pretrained=None)   # precision fp32, as the reference
        model = model.to(DEV).eval()
        tok = get_tokenizer(MODEL[ck])
        G.setdefault("tokenizers", {})[ck] = type(tok).__name__
        G.setdefault("text_param_dtype", {})[ck] = str(model.token_embedding.weight.dtype)
        log(f"G: loaded {ck} in {time.time() - t0:.1f}s tokenizer={type(tok).__name__}")
        for form in forms:
            t1 = time.time()
            res = zs_eval(model, tok, ck, form, extra_stats=True)
            results[f"{ck}|{form}"] = res
            for name, cells in want[(ck, form)].items():
                for rank in RANKS:
                    ref = cells[rank]
                    ref_correct = int(round(ref["top1"] * ref["n_eval"] / 100.0))
                    got = res[rank]
                    summary.append(dict(json=name, ckpt=ck, form=form, rank=rank,
                                        ref_correct=ref_correct, got_correct=got["correct"],
                                        ok_correct=ref_correct == got["correct"],
                                        ok_n_classes=ref["n_classes"] == got["n_classes"],
                                        ok_n_paths=ref["n_paths"] == got["n_paths"],
                                        ok_n_eval=ref["n_eval"] == got["n_eval"]))
                avg_ref = cells["average"]["top1"]
                summary.append(dict(json=name, ckpt=ck, form=form, rank="average",
                                    ref_top1=avg_ref, got_top1=res["average"]["top1"],
                                    ok_average=abs(avg_ref - res["average"]["top1"]) < 1e-9))
            n_ok = sum(1 for s in summary if s["ckpt"] == ck and s["form"] == form and s.get("ok_correct"))
            log(f"G: {ck} {form} in {time.time() - t1:.1f}s; correct-count matches so far for this cell: {n_ok}")
            save()
        # sensitivity on three cells
        if ck in ("bioclip1", "rcme", "bfl-hyperbolic"):
            sform = "lineage" if ck == "rcme" else "photo"
            base = results[f"{ck}|{sform}"]
            sens = {}
            for vname, kw, prec in [("text_batch_1024", dict(text_batch=1024), "highest"),
                                    ("text_batch_1", dict(text_batch=1), "highest"),
                                    ("chunk_1000", dict(chunk=1000), "highest"),
                                    ("tf32_high", {}, "high"),
                                    ("medium", {}, "medium"),
                                    ("text_fp16_autocast", dict(autocast_dtype=torch.float16), "highest")]:
                if vname == "text_batch_1" and ck != "bioclip1":
                    continue
                torch.set_float32_matmul_precision(prec)
                t1 = time.time()
                r2 = zs_eval(model, tok, ck, sform, **kw)
                torch.set_float32_matmul_precision("highest")
                sens[vname] = {rank: dict(correct=r2[rank]["correct"],
                                          delta_vs_reference_setting=r2[rank]["correct"] - base[rank]["correct"],
                                          same_predictions=r2[rank]["_pred_sha"] == base[rank]["_pred_sha"])
                               for rank in RANKS}
                log(f"G: sensitivity {ck} {sform} {vname} ({time.time() - t1:.1f}s):",
                    {r: sens[vname][r]["delta_vs_reference_setting"] for r in RANKS})
            G.setdefault("sensitivity", {})[f"{ck}|{sform}"] = sens
            save()
        del model
        torch.cuda.empty_cache()
    G["results"] = results
    G["comparison"] = summary
    G["n_rank_cells_compared"] = sum(1 for s in summary if s["rank"] != "average")
    G["n_rank_cells_all_ok"] = sum(1 for s in summary if s["rank"] != "average" and s["ok_correct"]
                                   and s["ok_n_classes"] and s["ok_n_paths"] and s["ok_n_eval"])
    G["n_average_ok"] = sum(1 for s in summary if s["rank"] == "average" and s["ok_average"])
    G["n_average_compared"] = sum(1 for s in summary if s["rank"] == "average")
    G["mismatching_cells"] = [s for s in summary if s["rank"] != "average" and not (
        s["ok_correct"] and s["ok_n_classes"] and s["ok_n_paths"] and s["ok_n_eval"])]
    log("G: rank cells all-ok", G["n_rank_cells_all_ok"], "of", G["n_rank_cells_compared"],
        "; averages ok", G["n_average_ok"], "of", G["n_average_compared"])
    for s in G["mismatching_cells"][:40]:
        log("   MISMATCH", s)
    save()


if __name__ == "__main__":
    import PIL, torchvision, webdataset
    out["env"] = dict(host=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      torchvision=torchvision.__version__, pillow=PIL.__version__,
                      webdataset=webdataset.__version__, numpy=np.__version__,
                      hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"))
    log("env", out["env"])
    if not torch.cuda.is_available():
        sys.exit("no GPU")
    which = sys.argv[1:] or ["E", "F", "G"]
    OUT = TAXA / f"audit/preflight/refbeh_gpu_{''.join(which)}.json"   # one file per job
    for p in which:
        t0 = time.time()
        try:
            {"E": part_e, "F": part_f, "G": part_g}[p]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            out.setdefault("errors", {})[p] = repr(e)
            save()
        log(f"part {p} finished in {time.time() - t0:.1f}s")
    save()
    log("wrote", OUT)
