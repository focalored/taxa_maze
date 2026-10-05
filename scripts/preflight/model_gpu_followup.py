#!/usr/bin/env python
"""
Follow-up to model_gpu_checks.py (job 245676), one A100, default SDPA flags, bf16 autocast.

Why: in job 245676 the merged-weight LoRA probes with gradient checkpointing failed with
torch.utils.checkpoint.CheckpointError ("A different number of tensors was saved during the
original forward and recomputation: 27 vs 23"). That failure came from wrapping the step in
torch.nn.utils.parametrize.cached(): the first forward computes the parametrized weight and
caches it, the recomputation reuses the cache, so fewer tensors are saved the second time.
This script repeats those probes WITHOUT parametrize.cached(), and adds:
  - a combined image (2048, checkpointed) + text (2048) LoRA step, the per-GPU share of a
    global batch of 8192 on 4 GPUs;
  - a small full fine-tuning probe (spec line 64, capacity control), all weights trainable;
  - a repeat of one frozen-weight proxy probe, to show run-to-run variation;
  - a direct reproduction of the cached()+checkpoint error, recorded with its exact text.

Writes only audit/preflight/model_gpu_followup_facts.json (plus stdout captured by Slurm).
"""
import copy
import gc
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

REPO = Path("/projects/bdbk/liv/repos/taxa_maze")
MODEL_NAME = "local-dir:/u/liv/bdbk/ckpt/bioclip1"
VOCAB = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/vocab.json")
OUT = REPO / "audit/preflight/model_gpu_followup_facts.json"
SOFT_BUDGET_S = 18 * 60

sys.path.insert(0, str(REPO))
from src.open_clip import create_model_and_transforms, get_tokenizer  # noqa: E402

T0 = time.time()
GiB = float(1 << 30)
DEV = torch.device("cuda")
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
props = torch.cuda.get_device_properties(0)
facts = {"script": str(Path(__file__).resolve()), "host": socket.gethostname(),
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
         "gpu": props.name, "gpu_total_memory_GiB": props.total_memory / GiB,
         "flags": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                   "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                   "math": torch.backends.cuda.math_sdp_enabled(),
                   "float32_matmul_precision": torch.get_float32_matmul_precision()},
         "autocast": "bf16", "probes": {}}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save():
    OUT.write_text(json.dumps(facts, indent=2, default=str))


def free():
    gc.collect()
    torch.cuda.empty_cache()


class QVLoRA(torch.nn.Module):
    """Same parametrization as model_gpu_checks.py: W -> W + [s*B_q A_q ; 0 ; s*B_v A_v]."""

    def __init__(self, d, r=16, alpha=32):
        super().__init__()
        self.A_q = torch.nn.Parameter(torch.empty(r, d))
        self.B_q = torch.nn.Parameter(torch.zeros(d, r))
        self.A_v = torch.nn.Parameter(torch.empty(r, d))
        self.B_v = torch.nn.Parameter(torch.zeros(d, r))
        torch.nn.init.kaiming_uniform_(self.A_q, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))
        self.s = alpha / r

    def forward(self, W):
        d = W.shape[1]
        dq = (self.B_q @ self.A_q) * self.s
        dv = (self.B_v @ self.A_v) * self.s
        z = torch.zeros(d, d, dtype=W.dtype, device=W.device)
        return W + torch.cat([dq.to(W.dtype), z, dv.to(W.dtype)], dim=0)


base, _, _ = create_model_and_transforms(MODEL_NAME, device=DEV)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)

lora = copy.deepcopy(base)
torch.manual_seed(0)
for tr in (lora.visual.transformer, lora.transformer):
    for blk in tr.resblocks:
        parametrize.register_parametrization(blk.attn, "in_proj_weight",
                                              QVLoRA(blk.attn.embed_dim).to(DEV))
lora.logit_scale.requires_grad_(True)
facts["lora_trainable_params"] = int(sum(p.numel() for p in lora.parameters() if p.requires_grad))

full = copy.deepcopy(base)
for p in full.parameters():
    p.requires_grad_(True)
facts["full_ft_trainable_params"] = int(sum(p.numel() for p in full.parameters() if p.requires_grad))
log(f"models ready on {facts['host']}: lora trainable {facts['lora_trainable_params']:,}, "
    f"full trainable {facts['full_ft_trainable_params']:,}")

tok = get_tokenizer(MODEL_NAME)
vocab = json.loads(VOCAB.read_text())
RANKS, SEP = vocab["ranks"], vocab.get("sep", "|")


def render(path):
    parts = path.split(SEP)[: len(RANKS)]
    if len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    return f"a photo of {' '.join(parts)}."


species = vocab["vocab"]["species"]
pick = np.random.default_rng(0).choice(len(species), size=2048, replace=False)
tokens77 = tok([render(species[i]) for i in pick]).to(DEV)
del vocab, species


def encode_text_len(m, text, L):
    text = text[:, :L]
    cast_dtype = m.transformer.get_cast_dtype()
    x = m.token_embedding(text).to(cast_dtype)
    x = x + m.positional_embedding[:L].to(cast_dtype)
    x = m.transformer(x, attn_mask=m.attn_mask[:L, :L])
    x = m.ln_final(x)
    x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    return x @ m.text_projection


def timed(step, warmup, iters):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def probe(key, make_step, n_items, warmup=2, iters=5):
    if time.time() - T0 > SOFT_BUDGET_S:
        facts["probes"][key] = {"skipped": "soft time budget reached"}
        return
    free()
    torch.cuda.reset_peak_memory_stats()
    rec = {"n_items": n_items, "allocated_before_GiB": torch.cuda.memory_allocated() / GiB}
    step = None
    try:
        step = make_step()
        dt = timed(step, warmup, iters)
        rec.update(ok=True, step_s=dt, items_per_s=n_items / dt)
    except torch.cuda.OutOfMemoryError as e:
        rec.update(ok=False, oom=True, error=str(e).split("\n")[0][:400])
    except Exception as e:
        rec.update(ok=False, oom=False, error=f"{type(e).__name__}: {str(e)[:600]}")
    rec["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / GiB
    rec["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / GiB
    step = None
    free()
    facts["probes"][key] = rec
    save()
    log(f"  {key}: " + (f"{rec['items_per_s']:.0f} items/s, step {rec['step_s']:.3f} s, "
                        if rec.get("ok") else f"FAILED ({rec['error'][:140]}), ")
        + f"peak {rec['peak_allocated_GiB']:.2f} GiB")


def image_step(m, bs, ckpt, input_grad=False, use_cached=False):
    def mk():
        m.train()
        m.set_grad_checkpointing(ckpt)
        x = torch.randn(bs, 3, 224, 224, device=DEV)
        if input_grad:
            x.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]

        def step():
            ctx = parametrize.cached() if use_cached else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.normalize(m.encode_image(x)[0].float(), dim=-1).sum()
                loss.backward()
            x.grad = None
            for p in params:
                p.grad = None
        return step
    return mk


def combined_step(m, bs_img, img_ckpt, L, txt_ckpt):
    def mk():
        m.train()
        m.visual.set_grad_checkpointing(img_ckpt)
        m.transformer.grad_checkpointing = txt_ckpt
        x = torch.randn(bs_img, 3, 224, 224, device=DEV)
        params = [p for p in m.parameters() if p.requires_grad]

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fi = F.normalize(m.encode_image(x)[0].float(), dim=-1)
                ft = F.normalize(encode_text_len(m, tokens77, L).float(), dim=-1)
                logits = m.logit_scale.exp() * fi @ ft.T
                loss = logits.logsumexp(dim=-1).mean()
            loss.backward()
            for p in params:
                p.grad = None
        return step
    return mk


log("LoRA (merged weight, no parametrize.cached) image tower")
probe("lora_image/bs1024/no_ckpt/uncached", image_step(lora, 1024, False), 1024)
probe("lora_image/bs2048/ckpt/uncached", image_step(lora, 2048, True), 2048)
probe("lora_image/bs4096/ckpt/uncached", image_step(lora, 4096, True), 4096)
probe("lora_image/bs1024/ckpt/cached_REPRODUCE_ERROR", image_step(lora, 1024, True, use_cached=True), 1024,
      warmup=1, iters=1)

log("LoRA combined image + text step")
probe("lora_combined/img2048_ckpt+txt2048_L77_no_ckpt", combined_step(lora, 2048, True, 77, False), 2048)
probe("lora_combined/img2048_ckpt+txt2048_L77_ckpt", combined_step(lora, 2048, True, 77, True), 2048)
probe("lora_combined/img2048_ckpt+txt2048_L32_no_ckpt", combined_step(lora, 2048, True, 32, False), 2048)
probe("lora_combined/img1024_no_ckpt+txt2048_L32_no_ckpt", combined_step(lora, 1024, False, 32, False), 1024)

log("frozen-weight proxy repeat (run-to-run variation)")
probe("proxy_image/bs2048/ckpt/repeat", image_step(base, 2048, True, input_grad=True), 2048)

log("full fine-tuning (all 149.6M weights trainable; AdamW state not allocated)")
probe("full_ft_image/bs512/no_ckpt", image_step(full, 512, False), 512)
probe("full_ft_image/bs1024/no_ckpt", image_step(full, 1024, False), 1024)
probe("full_ft_image/bs2048/ckpt", image_step(full, 2048, True), 2048)
probe("full_ft_combined/img2048_ckpt+txt2048_L77_ckpt", combined_step(full, 2048, True, 77, True), 2048)

facts["elapsed_s"] = round(time.time() - T0, 1)
save()
log(f"done; wrote {OUT}")
