#!/usr/bin/env python
"""
Preflight checks M1-M5 for specs/pilot1.md, section 3 (Model).

What this script checks, all on CPU:
  M1  sha256 of the BioCLIP 1 weight file against SHA256SUMS and the tol-eol cache manifest.
  M2  visual.output_dim and the exact parameter count of the loaded model; also that the
      loaded state equals the checkpoint file tensor-for-tensor.
  M3  attention-module facts for both towers (class, in_proj shapes, batch_first, heads),
      the text tower's causal mask and EOT pooling, logit_scale, logit_bias, and a numerical
      proof of the q,k,v row order inside in_proj_weight.
  M4  the LoRA budget for r=16 adapters on the q and v row blocks of every in_proj_weight.
  M5  the eval transform repr, compared to the cache manifest's preprocess_val_repr.
  Extra: token counts of the "a photo of <lineage>." strings for every ToL-EOL vocab entry,
      so the 32-token truncation probe in the GPU job can be judged against real strings.

Reads only: the checkpoint dir, taxa_maze/src/open_clip, and two allowed tol_embed artifacts
(probe_cache_b16/tol-eol/bioclip1/manifest.json and vocab.json). Writes only
audit/preflight/model_cpu_facts.json.

Run from a Slurm CPU job:
  PYTHONPATH=/projects/bdbk/liv/repos/taxa_maze python scripts/preflight/model_cpu_checks.py
"""
import copy
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/projects/bdbk/liv/repos/taxa_maze")
CKPT = Path("/u/liv/bdbk/ckpt/bioclip1")
MODEL_NAME = "local-dir:/u/liv/bdbk/ckpt/bioclip1"
TOL_CACHE = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1")
MANIFEST = TOL_CACHE / "manifest.json"
VOCAB = TOL_CACHE / "vocab.json"
OUT = REPO / "audit/preflight/model_cpu_facts.json"
EXPECTED_SHA = "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"

sys.path.insert(0, str(REPO))
from src.open_clip import create_model_and_transforms, get_tokenizer  # noqa: E402
import src.open_clip as oc  # noqa: E402
from src.open_clip import constants as oc_constants  # noqa: E402

T0 = time.time()
facts = {
    "script": str(Path(__file__).resolve()),
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "open_clip_file": oc.__file__,
    "open_clip_version": getattr(oc, "__version__", None),
}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def qualname(obj):
    t = obj if isinstance(obj, type) else type(obj)
    return f"{t.__module__}.{t.__qualname__}"


# ---------------------------------------------------------------- M1
log("M1: hashing checkpoint files")
sums = {}
for line in (CKPT / "SHA256SUMS").read_text().splitlines():
    if line.strip():
        hx, name = line.split(maxsplit=1)
        sums[name.strip().lstrip("*")] = hx
manifest = json.loads(MANIFEST.read_text())
m1 = {"files": {}, "sha256sums_file": str(CKPT / "SHA256SUMS"), "manifest_file": str(MANIFEST)}
for name in ("open_clip_pytorch_model.bin", "open_clip_config.json"):
    p = CKPT / name
    t = time.time()
    hx = sha256_file(p)
    m1["files"][name] = {
        "path": str(p),
        "realpath": os.path.realpath(p),
        "bytes": p.stat().st_size,
        "sha256_computed": hx,
        "sha256_in_SHA256SUMS": sums.get(name),
        "equal_to_SHA256SUMS": hx == sums.get(name),
        "hash_seconds": round(time.time() - t, 2),
    }
w = m1["files"]["open_clip_pytorch_model.bin"]
m1["manifest_weight_sha256"] = manifest.get("weight_sha256")
m1["manifest_weight_sha256_source"] = manifest.get("weight_sha256_source")
m1["manifest_model"] = manifest.get("model")
m1["task_expected_sha256"] = EXPECTED_SHA
m1["weight_equal_manifest"] = w["sha256_computed"] == manifest.get("weight_sha256")
m1["weight_equal_task_value"] = w["sha256_computed"] == EXPECTED_SHA
m1["provenance_txt"] = (CKPT / "PROVENANCE.txt").read_text()
facts["M1"] = m1
log(f"M1: weight sha256 {w['sha256_computed']} (SHA256SUMS equal: {w['equal_to_SHA256SUMS']}, "
    f"manifest equal: {m1['weight_equal_manifest']})")

# ---------------------------------------------------------------- M2
log("M2: loading model with create_model_and_transforms")
torch.manual_seed(0)
model, preprocess_train, preprocess_val = create_model_and_transforms(MODEL_NAME)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
named = dict(model.named_parameters())
visual_params = sum(p.numel() for n, p in named.items() if n.startswith("visual."))
other = {n: p for n, p in named.items() if not n.startswith("visual.")}
logit_params = sum(p.numel() for n, p in other.items() if n.startswith("logit_"))
text_params = sum(p.numel() for n, p in other.items() if not n.startswith("logit_"))
buffers = {n: list(b.shape) for n, b in model.named_buffers()}
dtypes = sorted({str(p.dtype) for p in model.parameters()})

# Compare loaded state with the raw checkpoint, tensor for tensor.
sd = torch.load(CKPT / "open_clip_pytorch_model.bin", map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
msd = model.state_dict()
ck_keys, m_keys = set(sd.keys()), set(msd.keys())
mismatch = [k for k in sorted(ck_keys & m_keys) if not torch.equal(sd[k].to(msd[k].dtype), msd[k])]
facts["M2"] = {
    "model_class": qualname(model),
    "visual_output_dim": int(model.visual.output_dim),
    "n_params_total": int(n_params),
    "n_params_total_millions": round(n_params / 1e6, 4),
    "n_params_visual": int(visual_params),
    "n_params_text": int(text_params),
    "n_params_logit": int(logit_params),
    "param_dtypes": dtypes,
    "buffers": buffers,
    "checkpoint_n_tensors": len(sd),
    "checkpoint_n_elements": int(sum(v.numel() for v in sd.values())),
    "checkpoint_dtypes": sorted({str(v.dtype) for v in sd.values()}),
    "keys_only_in_checkpoint": sorted(ck_keys - m_keys),
    "keys_only_in_model_state": sorted(m_keys - ck_keys),
    "n_tensors_not_equal_after_load": len(mismatch),
    "tensors_not_equal_examples": mismatch[:10],
}
log(f"M2: output_dim={model.visual.output_dim}, n_params={n_params:,} "
    f"(visual {visual_params:,}, text {text_params:,}, logit {logit_params})")
del sd

# ---------------------------------------------------------------- M3
log("M3: architecture facts")


def tower_summary(name, transformer):
    rows = []
    for i, blk in enumerate(transformer.resblocks):
        a = blk.attn
        mlp_act = None
        if hasattr(blk, "mlp"):
            for m in blk.mlp:
                if "gelu" in type(m).__name__.lower():
                    mlp_act = qualname(m)
        rows.append({
            "block": i,
            "block_class": qualname(blk),
            "attn_class": qualname(a),
            "is_torch_MultiheadAttention": isinstance(a, torch.nn.MultiheadAttention),
            "is_open_clip_Attention": isinstance(a, oc.transformer.Attention),
            "in_proj_weight_shape": list(a.in_proj_weight.shape),
            "in_proj_bias_shape": None if a.in_proj_bias is None else list(a.in_proj_bias.shape),
            "batch_first": getattr(a, "batch_first", None),
            "embed_dim": getattr(a, "embed_dim", None),
            "num_heads": getattr(a, "num_heads", None),
            "head_dim": getattr(a, "head_dim", None),
            "qkv_same_embed_dim": getattr(a, "_qkv_same_embed_dim", None),
            "bias_k_is_None": getattr(a, "bias_k", None) is None,
            "add_zero_attn": getattr(a, "add_zero_attn", None),
            "dropout": getattr(a, "dropout", None),
            "out_proj_weight_shape": list(a.out_proj.weight.shape),
            "out_proj_has_bias": a.out_proj.bias is not None,
            "mlp_activation": mlp_act,
            "ls_1": qualname(blk.ls_1),
        })
    keys = [k for k in rows[0] if k != "block"]
    uniform = {k: all(r[k] == rows[0][k] for r in rows) for k in keys}
    return {
        "tower": name,
        "transformer_class": qualname(transformer),
        "transformer_batch_first": getattr(transformer, "batch_first", None),
        "transformer_width": getattr(transformer, "width", None),
        "transformer_layers": getattr(transformer, "layers", None),
        "n_blocks": len(rows),
        "block0": rows[0],
        "all_blocks_identical_on": uniform,
    }


vis = tower_summary("visual", model.visual.transformer)
txt = tower_summary("text", model.transformer)
mask = model.attn_mask
expected_mask = torch.full((model.context_length, model.context_length), float("-inf")).triu_(1)
tok = get_tokenizer(MODEL_NAME)
probe = tok(["a photo of Animalia."])
eot_pos = int(probe[0].argmax())
n_nonzero = int((probe[0] != 0).sum())
logit_scale_raw = float(model.logit_scale.detach())
facts["M3"] = {
    "visual": vis,
    "text": txt,
    "visual_extra": {
        "image_size": list(model.visual.image_size),
        "patch_size": list(model.visual.patch_size),
        "grid_size": list(model.visual.grid_size),
        "n_tokens": int(model.visual.positional_embedding.shape[0]),
        "pool_type": model.visual.pool_type,
        "attn_pool_is_None": model.visual.attn_pool is None,
        "final_ln_after_pool": model.visual.final_ln_after_pool,
        "proj_shape": list(model.visual.proj.shape),
        "is_continual": model.visual.is_continual,
        "continual_proj_is_None": model.visual.continual_proj is None,
        "patch_dropout": qualname(model.visual.patch_dropout),
        "conv1": repr(model.visual.conv1),
        "ln_pre": qualname(model.visual.ln_pre),
    },
    "text_extra": {
        "context_length": int(model.context_length),
        "vocab_size": int(model.vocab_size),
        "positional_embedding_shape": list(model.positional_embedding.shape),
        "text_projection_type": qualname(model.text_projection),
        "text_projection_shape": list(model.text_projection.shape),
        "text_pool_type": model.text_pool_type,
        "text_eos_id_attr": model.text_eos_id,
        "attn_mask_is_None": mask is None,
        "attn_mask_shape": None if mask is None else list(mask.shape),
        "attn_mask_dtype": None if mask is None else str(mask.dtype),
        "attn_mask_equals_full_neg_inf_triu1": bool(torch.equal(mask, expected_mask)),
        "attn_mask_n_neg_inf": int(torch.isinf(mask).sum()),
        "attn_mask_n_zero": int((mask == 0).sum()),
        "encode_text_source_lines": None,
        "tokenizer_class": qualname(tok),
        "tokenizer_context_length": getattr(tok, "context_length", None),
        "sot_token_id": getattr(tok, "sot_token_id", None),
        "eot_token_id": getattr(tok, "eot_token_id", None),
        "probe_string": "a photo of Animalia.",
        "probe_tokens_nonzero": probe[0][:n_nonzero].tolist(),
        "probe_argmax_position": eot_pos,
        "probe_argmax_is_eot": int(probe[0][eot_pos]) == getattr(tok, "eot_token_id", -1),
    },
    "logit_scale_raw": logit_scale_raw,
    "logit_scale_exp": math.exp(logit_scale_raw),
    "logit_scale_shape": list(model.logit_scale.shape),
    "logit_bias_is_None": model.logit_bias is None,
}
src_lines, start = inspect.getsourcelines(type(model).encode_text)
facts["M3"]["text_extra"]["encode_text_source_lines"] = {
    "file": inspect.getsourcefile(type(model).encode_text),
    "first_line": start,
    "source": "".join(src_lines),
}
blk_src, blk_start = inspect.getsourcelines(type(model.transformer.resblocks[0]).attention)
facts["M3"]["text_extra"]["ResidualAttentionBlock_attention_source"] = {
    "file": inspect.getsourcefile(type(model.transformer.resblocks[0])),
    "first_line": blk_start,
    "source": "".join(blk_src),
}


def manual_mha(x, mha, attn_mask=None, order=(0, 1, 2)):
    """Self-attention computed by hand, taking q,k,v from the in_proj_weight row blocks
    listed in `order` (0 = rows 0:d, 1 = rows d:2d, 2 = rows 2d:3d)."""
    d, h = mha.embed_dim, mha.num_heads
    hd = d // h
    W, b = mha.in_proj_weight, mha.in_proj_bias
    Wb = [W[0:d], W[d:2 * d], W[2 * d:3 * d]]
    bb = [b[0:d], b[d:2 * d], b[2 * d:3 * d]]
    q = x @ Wb[order[0]].T + bb[order[0]]
    k = x @ Wb[order[1]].T + bb[order[1]]
    v = x @ Wb[order[2]].T + bb[order[2]]
    N, L, _ = x.shape
    q, k, v = (t.reshape(N, L, h, hd).transpose(1, 2) for t in (q, k, v))
    att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    if attn_mask is not None:
        att = att + attn_mask
    att = att.softmax(dim=-1)
    o = (att @ v).transpose(1, 2).reshape(N, L, d)
    return o @ mha.out_proj.weight.T + mha.out_proj.bias


row_order = {}
g = torch.Generator().manual_seed(0)
for tower, blk, L, use_mask in (
    ("visual", model.visual.transformer.resblocks[0], 197, False),
    ("text", model.transformer.resblocks[0], 77, True),
):
    mha = copy.deepcopy(blk.attn).double()
    d = mha.embed_dim
    x = torch.randn(2, L, d, generator=g, dtype=torch.float64)
    m = mask.double() if use_mask else None
    res = {}
    for mode in ("eval", "train"):
        mha.train(mode == "train")
        with torch.no_grad():
            ref = mha(x, x, x, need_weights=False, attn_mask=m)[0]
        for label, order in (("q,k,v (assumed)", (0, 1, 2)), ("k,q,v", (1, 0, 2)),
                             ("q,v,k", (0, 2, 1)), ("v,k,q", (2, 1, 0))):
            with torch.no_grad():
                man = manual_mha(x, mha, m, order)
            res[f"{mode}: {label}"] = float((man - ref).abs().max())
    row_order[tower] = {"input_shape": [2, L, d], "dtype": "float64", "mask": use_mask,
                        "max_abs_diff_vs_module": res,
                        "module_output_abs_max": float(ref.abs().max())}
facts["M3"]["row_order_check"] = row_order
log("M3: row-order check " + json.dumps({k: v["max_abs_diff_vs_module"] for k, v in row_order.items()}))

# ---------------------------------------------------------------- M4
log("M4: LoRA budget")
r, alpha = 16, 32
adapters = []
for tower, tr in (("visual", model.visual.transformer), ("text", model.transformer)):
    for i, blk in enumerate(tr.resblocks):
        W = blk.attn.in_proj_weight
        d_in = W.shape[1]
        d_out = W.shape[0] // 3
        for which, rows in (("q", (0, d_out)), ("v", (2 * d_out, 3 * d_out))):
            A = torch.zeros(r, d_in)    # A: r x d_in
            B = torch.zeros(d_out, r)   # B: d_out x r
            adapters.append({"tower": tower, "block": i, "which": which, "rows": list(rows),
                             "A_shape": list(A.shape), "B_shape": list(B.shape),
                             "n": int(A.numel() + B.numel())})
n_adapt = sum(a["n"] for a in adapters)
n_adapt_vis = sum(a["n"] for a in adapters if a["tower"] == "visual")
n_adapt_txt = sum(a["n"] for a in adapters if a["tower"] == "text")
n_logit = int(model.logit_scale.numel())
trainable = n_adapt + n_logit
total_with_adapters = n_params + n_adapt
n_vproj = int(model.visual.proj.numel())
n_tproj = int(model.text_projection.numel())
trainable_ii = trainable + n_vproj + n_tproj
facts["M4"] = {
    "r": r, "alpha": alpha, "scaling_alpha_over_r": alpha / r,
    "n_adapter_pairs_AB": len(adapters),
    "n_adapter_matrices": 2 * len(adapters),
    "per_adapter_visual": adapters[0],
    "per_adapter_text": next(a for a in adapters if a["tower"] == "text"),
    "n_adapter_params_visual": n_adapt_vis,
    "n_adapter_params_text": n_adapt_txt,
    "n_adapter_params": n_adapt,
    "n_logit_scale": n_logit,
    "n_trainable": trainable,
    "n_frozen_base": n_params - n_logit,
    "n_total_frozen_plus_adapters": total_with_adapters,
    "fraction_trainable_of_total_with_adapters": trainable / total_with_adapters,
    "fraction_trainable_of_base_params": trainable / n_params,
    "arm_ii_extra_visual_proj": n_vproj,
    "arm_ii_extra_text_projection": n_tproj,
    "arm_ii_trainable": trainable_ii,
    "arm_ii_fraction_of_total_with_adapters": trainable_ii / total_with_adapters,
    "full_finetune_trainable": int(n_params),
}
log(f"M4: adapters={n_adapt:,} trainable={trainable:,} fraction={trainable / total_with_adapters:.6f}")

# ---------------------------------------------------------------- M5
log("M5: preprocess")


def norm_repr(s):
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)


val_repr = repr(preprocess_val)
man_repr = manifest.get("preprocess_val_repr")
steps = []
for t in getattr(preprocess_val, "transforms", []):
    if inspect.isfunction(t):
        steps.append({"function": t.__name__, "module": t.__module__})
        continue
    d = {"class": qualname(t)}
    for attr in ("size", "interpolation", "max_size", "antialias", "mean", "std"):
        if hasattr(t, attr):
            v = getattr(t, attr)
            d[attr] = v if isinstance(v, (int, float, list, tuple, type(None), bool)) else str(v)
    steps.append(d)
facts["M5"] = {
    "preprocess_val_repr": val_repr,
    "manifest_preprocess_val_repr": man_repr,
    "repr_equal_exact": val_repr == man_repr,
    "repr_equal_after_masking_function_address": norm_repr(val_repr) == norm_repr(man_repr),
    "preprocess_val_steps": steps,
    "openai_mean": list(oc_constants.OPENAI_DATASET_MEAN),
    "openai_std": list(oc_constants.OPENAI_DATASET_STD),
    "model_visual_preprocess_cfg": dict(getattr(model.visual, "preprocess_cfg", {}) or {}),
    "manifest_preprocess_cfg": manifest.get("preprocess_cfg"),
    "preprocess_cfg_equal": dict(getattr(model.visual, "preprocess_cfg", {}) or {}) == manifest.get("preprocess_cfg"),
    "preprocess_train_repr": repr(preprocess_train),
}
facts["M5"]["model_visual_preprocess_cfg"] = json.loads(json.dumps(facts["M5"]["model_visual_preprocess_cfg"], default=str))
log("M5: repr equal after masking address: "
    f"{facts['M5']['repr_equal_after_masking_function_address']}")

# ---------------------------------------------------------------- token lengths of real class strings
log("Extra: token counts of ToL-EOL vocab strings")
vocab = json.loads(VOCAB.read_text())
RANKS = vocab["ranks"]
SEP = vocab.get("sep", "|")


def render(path, rank_idx, form):
    # Same rule as tol_embed/scripts/zeroshot_ranks.py render(): strip the doubled genus
    # from the binomial species leaf, space-join, wrap in the photo template.
    parts = path.split(SEP)[: rank_idx + 1]
    if rank_idx == len(RANKS) - 1 and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    text = " ".join(parts)
    return f"a photo of {text}." if form == "photo" else text


_TOK = None


def _init():
    global _TOK
    _TOK = get_tokenizer(MODEL_NAME)


def _count(texts):
    return [len(_TOK.encode(t)) + 2 for t in texts]   # + SOT + EOT


tok_stats = {}
ctx = mp.get_context("fork")
n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
with ctx.Pool(n_workers, initializer=_init) as pool:
    for ri, rank in enumerate(RANKS):
        paths = vocab["vocab"][rank]
        for form in ("photo", "lineage"):
            texts = [render(p, ri, form) for p in paths]
            chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
            counts = np.array([c for part in pool.map(_count, chunks) for c in part])
            longest = int(np.argmax(counts))
            tok_stats[f"{rank}/{form}"] = {
                "n_strings": int(len(counts)),
                "max_tokens_incl_sot_eot": int(counts.max()),
                "p50": float(np.percentile(counts, 50)),
                "p99": float(np.percentile(counts, 99)),
                "p999": float(np.percentile(counts, 99.9)),
                "n_gt_32": int((counts > 32).sum()),
                "n_gt_77": int((counts > 77).sum()),
                "frac_gt_32": float((counts > 32).mean()),
                "longest_example": texts[longest],
            }
            log(f"  {rank}/{form}: n={len(counts)} max={counts.max()} >32: {(counts > 32).sum()} >77: {(counts > 77).sum()}")
facts["token_counts_tol_eol_vocab"] = {"vocab_file": str(VOCAB), "ranks": RANKS, "stats": tok_stats}

facts["elapsed_s"] = round(time.time() - T0, 1)
OUT.write_text(json.dumps(facts, indent=2, default=str))
log(f"wrote {OUT}")
