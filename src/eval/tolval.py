"""ToL-val zero-shot top-1 per rank (Amendment 1 S4): candidates are the taxa of the deduplicated val set, images in fp16 autocast, text and scoring in fp32.

Usage: `res = tolval_eval(model, tokenizer, dm, device, rank, world)`; `res["photo"]["species"]["correct"]`.
"""
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.tol_catalog import RANKS
from src.data.tol_store import to_model_input
from src.eval.zeroshot import class_text, encode_texts, top1_correct
from src.models.losses import all_reduce_sum, gather_varlen_nograd


@torch.no_grad()
def encode_store_images(model, loader, device) -> torch.Tensor:
    """L2-normalized fp32 embeddings of a store loader's images: fp32 weights under fp16 autocast."""
    out = []
    for batch in loader:
        x = to_model_input(batch["u8"].to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            f, _ = model.encode_image(x)
        out.append(F.normalize(f.float(), dim=-1))
    return torch.cat(out) if out else torch.zeros(0, model.visual.output_dim, device=device)


@torch.no_grad()
def encode_candidates(model, tokenizer, strings: Sequence[str], device, rank: int, world: int) -> torch.Tensor:
    """Each rank encodes a contiguous share of the strings (fp32, no autocast); all ranks get all rows."""
    n = len(strings)
    a, b = (n * rank) // world, (n * (rank + 1)) // world
    with torch.autocast(device_type="cuda", enabled=False):
        if b > a:
            mine = encode_texts(model, tokenizer, list(strings[a:b]), device).to(device)
        else:  # fewer candidates than ranks (possible at kingdom)
            mine = torch.zeros(0, model.visual.output_dim, dtype=torch.float32, device=device)
    return gather_varlen_nograd(mine)


@torch.no_grad()
def tolval_eval(model, tokenizer, dm, device, rank: int, world: int,
                forms: Sequence[str] = ("photo", "lineage")) -> Dict:
    val = dm.val
    loader, a, b = dm.val_loader(rank, world)
    was_training = model.training
    model.eval()
    emb = encode_store_images(model, loader, device)
    labels, seen = val.labels[a:b], val.seen[a:b]
    res = {}
    for form in forms:
        res[form] = {}
        for d, rank_name in enumerate(RANKS):
            strings = [class_text(k.split("|"), form) for k in val.node_keys[d]]
            txt = encode_candidates(model, tokenizer, strings, device, rank, world)
            correct, n_valid, pred = top1_correct(emb, txt, labels[:, d], device)
            hit = pred == labels[:, d]
            c = torch.tensor([correct, n_valid, int((hit & seen).sum()), int(seen.sum())],
                             dtype=torch.float64, device=device)
            c = all_reduce_sum(c).tolist()
            res[form][rank_name] = {"correct": int(c[0]), "n_eval": int(c[1]), "n_classes": len(strings),
                                    "top1": 100.0 * c[0] / c[1], "seen_correct": int(c[2]), "seen_n": int(c[3]),
                                    "seen_top1": 100.0 * c[2] / max(c[3], 1)}
        res[form]["mean7"] = float(np.mean([res[form][r]["top1"] for r in RANKS]))
    if was_training:
        model.train()
    return res
