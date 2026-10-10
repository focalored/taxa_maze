"""Diagnostic zero-shot evaluation of pilot-1 checkpoints and the baseline models under the M3 encode setting (Amendment 2 A2.4:
fp32 weights under fp16 autocast, batch 512, cudnn.benchmark off, deterministic on), one JSON per model; never used for checkpoint
selection (S27) or the LR choice (line 161). Datasets: iNat21-val (default) and Rare Species (`--dataset rare_species`, ported by
`scripts/p1/port_rare_species.py`). Usage: `python scripts/p1/eval_inat21.py --run-dir logs/p1/<run_name>` (every `checkpoints/epoch_*.ckpt`
not yet evaluated), `--baseline bfl-euclidean` (a registered baseline, `src/eval/checkpoints.py`), or `--untouched` (BioCLIP 1 via
the training loader; expect 70,193 species-correct on iNat21, photo form).
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from PIL import Image  # noqa: E402

from src.data.inat21 import INat21Val, inat21_cache_dir, load_inat21_labels  # noqa: E402
from src.eval.checkpoints import CHECKPOINTS, load_checkpoint  # noqa: E402
from src.eval.zeroshot import RANKS, evaluate_ranks  # noqa: E402
from src.models.bioclip_lora import add_qv_lora, load_bioclip1  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

FORMS = ("photo", "lineage")
BATCH = 512
RARE_ROOT = Path("/u/liv/data/taxa_maze/rare_species")


class ImageList(torch.utils.data.Dataset):
    """Images listed in ids.txt under root, each item (tensor, row index), like INat21Val."""

    def __init__(self, transform, root, ids_file):
        self.transform, self.root = transform, Path(root)
        self.files = Path(ids_file).read_text(encoding="utf-8").splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.root / self.files[idx]) as im:
            return self.transform(im.convert("RGB")), idx


def dataset_spec(name):
    """(dataset factory taking a transform, labels cache dir, description)."""
    if name == "inat21":
        return (lambda tf: INat21Val(tf)), inat21_cache_dir("bioclip1"), "iNat21-val, 100,000 images, 10,000 species"
    if name == "rare_species":
        return (lambda tf: ImageList(tf, RARE_ROOT / "images", RARE_ROOT / "labels" / "ids.txt")), RARE_ROOT / "labels", "Rare Species (HF imageomics/rare-species), 11,983 images, 400 species"
    raise ValueError(f"unknown dataset {name!r}")


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt, device, baseline=None):
    """BioCLIP 1 with the checkpoint's LoRA and logit_scale (strict load); untouched BioCLIP 1 when ckpt is None; or a registered baseline."""
    if baseline is not None:
        model, preprocess, tokenizer = load_checkpoint(baseline, device)
        spec = CHECKPOINTS[baseline]
        return model, preprocess, tokenizer, {"checkpoint": None, "baseline": baseline, "model": spec["model"], "upstream": spec.get("upstream")}
    if ckpt is None:
        ckpt_dir = load_paths().bioclip1_ckpt
        model, preprocess, tokenizer = load_bioclip1(ckpt_dir, device)
        return model, preprocess, tokenizer, {"checkpoint": None, "ckpt_dir": str(ckpt_dir)}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model, preprocess, tokenizer = load_bioclip1(hp["ckpt_dir"], device)
    add_qv_lora(model, r=hp["lora_r"], alpha=hp["lora_alpha"], seed=hp["seed"], towers=hp.get("lora_towers", "both"))
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    info = {"checkpoint": str(ckpt), "sha256": sha256(ckpt), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "arm": hp.get("arm"), "lr": hp.get("lr"), "lam": hp.get("lam"), "lora_r": hp["lora_r"], "lora_alpha": hp["lora_alpha"],
            "lora_towers": hp.get("lora_towers", "both"),
            "logit_scale": float(model.logit_scale.detach().float())}
    del ck
    return model, preprocess, tokenizer, info


@torch.no_grad()
def encode_images(model, preprocess, device, workers, make_dataset):
    ds = make_dataset(preprocess)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    out = torch.zeros((len(ds), int(model.visual.output_dim)), dtype=torch.float32, device=device)
    seen = np.zeros(len(ds), dtype=bool)
    for images, idx in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            f, _ = model.encode_image(images.to(device, non_blocking=True))
        out[idx.to(device)] = F.normalize(f.float(), dim=-1)
        seen[idx.numpy()] = True
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} iNat21 rows were not encoded")
    return out


def evaluate(model, preprocess, tokenizer, device, workers, make_dataset, labels_dir):
    t0 = time.time()
    img = encode_images(model, preprocess, device, workers, make_dataset)
    t_enc = time.time() - t0
    codes, vocab = load_inat21_labels(labels_dir)
    res = {form: evaluate_ranks(img, codes, vocab, model, tokenizer, form, device) for form in FORMS}
    return res, {"n_images": int(img.shape[0]), "encode_s": round(t_enc, 1), "eval_s": round(time.time() - t0, 1)}


def setting(workers, labels_dir):
    return {"batch": BATCH, "autocast": "float16", "weights": "float32", "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark, "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "fp32_matmul_precision": torch.get_float32_matmul_precision(), "loader_workers": workers,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "ids_order": str(Path(labels_dir) / "ids.txt")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, help="run directory holding checkpoints/epoch_*.ckpt; results go to <run-dir>/inat21_diag/")
    ap.add_argument("--untouched", action="store_true", help="evaluate BioCLIP 1 itself (validation of this script)")
    ap.add_argument("--baseline", choices=sorted(CHECKPOINTS), default=None, help="a registered baseline model instead of a run")
    ap.add_argument("--dataset", choices=("inat21", "rare_species"), default="inat21")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <run-dir>/inat21_diag or logs/p1/inat21_diag_untouched)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device: run inside a GPU allocation (scripts/p1/eval_inat21.sh)")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    workers = args.workers if args.workers is not None else max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2)

    make_dataset, labels_dir, desc = dataset_spec(args.dataset)
    tag = "inat21" if args.dataset == "inat21" else args.dataset
    if args.baseline:
        targets, out = [args.baseline], args.out or Path(f"logs/p1/{tag}_diag_baselines")
        if (out / f"{args.baseline}.json").exists():
            print("already evaluated:", out / f"{args.baseline}.json"); return
    elif args.untouched:
        targets, out = [None], args.out or Path(f"logs/p1/{tag}_diag_untouched")
    else:
        if args.run_dir is None:
            sys.exit("give --run-dir, --baseline or --untouched")
        out = args.out or args.run_dir / f"{tag}_diag"
        targets = sorted((args.run_dir / "checkpoints").glob("epoch_*.ckpt"))
        targets = [t for t in targets if not (out / f"{t.stem}.json").exists()]
        if not targets:
            print("nothing to evaluate: every epoch_*.ckpt already has a result in", out)
            return
    out.mkdir(parents=True, exist_ok=True)
    for ckpt in targets:
        is_base = isinstance(ckpt, str)
        name = ckpt if is_base else ("bioclip1_untouched" if ckpt is None else ckpt.stem)
        t0 = time.time()
        model, preprocess, tokenizer, info = load_model(None if is_base else ckpt, device, baseline=ckpt if is_base else None)
        res, timing = evaluate(model, preprocess, tokenizer, device, workers, make_dataset, labels_dir)
        rec = {"purpose": f"diagnostic zero-shot evaluation on {desc} (Amendment 3 A3.1 for iNat21; Rare Species is outside the spec); not for checkpoint selection (S27) or the LR choice (line 161)",
               "dataset": args.dataset, "model": info, "setting": setting(workers, labels_dir), "timing": {**timing, "total_s": round(time.time() - t0, 1)},
               "finished": datetime.datetime.now().isoformat(timespec="seconds"), "results": res}
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        sp = res["photo"]["species"]
        print(f"{name}: photo species {sp['correct']:,} / {sp['n_eval']:,} = {sp['top1']:.2f}%, average {res['photo']['average']['top1']:.2f}%; "
              f"lineage species {res['lineage']['species']['top1']:.2f}%; {timing['eval_s']} s (encode {timing['encode_s']} s) -> {out / (name + '.json')}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
