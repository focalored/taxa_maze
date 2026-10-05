#!/usr/bin/env python
"""
Preflight checks M6-M7 for specs/pilot1.md on one A100 (sections 3 and 6).

M6  Which encode_text / encode_image calls run under the SDPA flags of src/train.py lines 12-15
    (flash on, mem-efficient off, math off), in fp32 and under autocast; exact error text of
    each failure; then the same with all SDPA backends enabled (PyTorch defaults).
M7  Throughput and peak memory with default SDPA flags and bf16 autocast:
    (a) inference encode_image at batch 1024;
    (b) a LoRA-like training step on the image tower: all weights frozen, input pixels
        require grad, forward + backward of sum(L2-normalized features), at several batch
        sizes with and without gradient checkpointing;
    (c) the same for the text tower at batch 2048 with 77-token and 32-token inputs, and a
        check that truncating trailing padding does not change the EOT embedding.
Extras (clearly labelled "extra" in the output):
    - the same training step with real merged-weight LoRA adapters (r=16, alpha=32) added to
      the q and v row blocks of every in_proj_weight via torch parametrize, as spec line 61
      describes, to bound the error of the frozen-weight proxy;
    - bit-identity of LoRA-at-init vs the frozen model, and of the MHA fast path vs slow path
      in fp32 (relevant to the smoke test on spec line 63);
    - what set_float32_matmul_precision("medium"/"high"/"highest") does to an fp32 matmul.

Writes only audit/preflight/model_gpu_facts.json (plus stdout, captured by Slurm).
"""
import copy
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

REPO = Path("/projects/bdbk/liv/repos/taxa_maze")
MODEL_NAME = "local-dir:/u/liv/bdbk/ckpt/bioclip1"
VOCAB = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/vocab.json")
OUT = REPO / "audit/preflight/model_gpu_facts.json"
SOFT_BUDGET_S = 20 * 60   # do not start new heavy probes after this many seconds

sys.path.insert(0, str(REPO))
from src.open_clip import create_model_and_transforms, get_tokenizer  # noqa: E402

T0 = time.time()
GiB = float(1 << 30)
DEV = torch.device("cuda")
facts = {"script": str(Path(__file__).resolve()), "host": socket.gethostname(),
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save():
    OUT.write_text(json.dumps(facts, indent=2, default=str))


def ac(prec):
    if prec == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[prec])


def get_flags():
    d = {"flash": torch.backends.cuda.flash_sdp_enabled(),
         "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
         "math": torch.backends.cuda.math_sdp_enabled(),
         "float32_matmul_precision": torch.get_float32_matmul_precision(),
         "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
         "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
         "mha_fastpath": torch.backends.mha.get_fastpath_enabled()}
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        d["cudnn_sdp"] = torch.backends.cuda.cudnn_sdp_enabled()
    return d


def set_flags(flash, mem, math_, matmul):
    torch.set_float32_matmul_precision(matmul)
    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(mem)
    torch.backends.cuda.enable_math_sdp(math_)


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ environment
facts["default_flags_at_start"] = get_flags()
props = torch.cuda.get_device_properties(0)
facts["gpu"] = {"name": props.name, "total_memory_GiB": props.total_memory / GiB,
                "capability": f"{props.major}.{props.minor}", "sm_count": props.multi_processor_count,
                "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,power.limit,clocks.max.sm,"
                        "driver_version,pci.bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    facts["gpu"]["nvidia_smi"] = q.stdout.strip()
except Exception as e:  # pragma: no cover
    facts["gpu"]["nvidia_smi"] = f"failed: {e}"
log(f"host {facts['host']} gpu {props.name} {props.total_memory / GiB:.1f} GiB; defaults {facts['default_flags_at_start']}")

# ------------------------------------------------------------------ model and inputs
model, _, _ = create_model_and_transforms(MODEL_NAME, device=DEV)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = get_tokenizer(MODEL_NAME)
strings4 = ["a photo of Animalia.", "a photo of Animalia Chordata.",
            "a photo of Animalia Chordata Mammalia.", "a photo of Animalia Chordata Mammalia Carnivora."]
tokens4 = tok(strings4).to(DEV)
g = torch.Generator(device="cuda").manual_seed(0)
images4 = torch.randn(4, 3, 224, 224, device=DEV, generator=g)
facts["inputs_M6"] = {"strings": strings4, "tokens_shape": list(tokens4.shape),
                      "images": "torch.randn(4,3,224,224), cuda generator seed 0"}
log("model loaded")


def run(fn):
    """Run fn, return a record with success flag, output summary, exact error text, warnings."""
    rec = {}
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t = time.time()
        try:
            out = fn()
            torch.cuda.synchronize()
            if isinstance(out, tuple):
                out = out[0]
            rec.update(ok=True, out_shape=list(out.shape), out_dtype=str(out.dtype),
                       finite=bool(torch.isfinite(out).all()))
            rec["_out"] = out.detach().float().cpu()
        except Exception as e:
            rec.update(ok=False, error_type=type(e).__name__, error=str(e))
        rec["seconds"] = round(time.time() - t, 3)
    msgs = []
    for w in wl:
        s = str(w.message)
        if s not in msgs:
            msgs.append(s)
    rec["warnings"] = msgs
    return rec


def attention_ops(fn):
    """Names and counts of attention-related aten ops that fn dispatches (CPU-side profiler)."""
    from torch.profiler import profile, ProfilerActivity
    try:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            fn()
            torch.cuda.synchronize()
        keys = ("scaled_dot_product", "multi_head_attention", "flash", "efficient_attention",
                "attention_math", "cudnn_attention", "_fused_sdp_choice")
        return {e.key: e.count for e in prof.key_averages() if any(k in e.key for k in keys)}
    except Exception as e:
        return {"profiler_error": f"{type(e).__name__}: {e}"}


def make_text_grad_hook(m):
    # Frozen weights + integer tokens: make the token-embedding output a grad-requiring leaf so
    # backward runs activation gradients through every text block (no weight gradients).
    return m.token_embedding.register_forward_hook(lambda mod, inp, out: out.requires_grad_(True))


def m6_cases(m):
    def img(prec):
        def f():
            with torch.no_grad(), ac(prec):
                return m.encode_image(images4)
        return f

    def txt(prec):
        def f():
            with torch.no_grad(), ac(prec):
                return m.encode_text(tokens4)
        return f

    def img_fp32_slowpath():
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad():
                return m.encode_image(images4)
        finally:
            torch.backends.mha.set_fastpath_enabled(True)

    def train_img_bf16():
        m.train()
        try:
            x = images4.clone().requires_grad_(True)
            with ac("bf16"):
                f = F.normalize(m.encode_image(x)[0].float(), dim=-1)
            f.sum().backward()
            return f
        finally:
            m.eval()

    def train_txt_bf16():
        m.train()
        h = make_text_grad_hook(m)
        try:
            with ac("bf16"):
                f = F.normalize(m.encode_text(tokens4).float(), dim=-1)
            f.sum().backward()
            return f
        finally:
            h.remove()
            m.eval()

    q = torch.randn(4, 8, 77, 64, device=DEV, dtype=torch.bfloat16)

    def sdpa_causal_bf16():
        return F.scaled_dot_product_attention(q, q, q, is_causal=True)

    def sdpa_floatmask_bf16():
        return F.scaled_dot_product_attention(q, q, q, attn_mask=m.attn_mask.to(torch.bfloat16))

    return {
        "encode_text/fp32": txt("fp32"),
        "encode_text/bf16_autocast": txt("bf16"),
        "encode_image/fp32": img("fp32"),
        "encode_image/fp16_autocast": img("fp16"),
        "encode_image/bf16_autocast": img("bf16"),
        "extra/encode_text/fp16_autocast": txt("fp16"),
        "extra/encode_image/fp32_mha_fastpath_disabled": img_fp32_slowpath,
        "extra/train_fwd_bwd/encode_image/bf16_autocast": train_img_bf16,
        "extra/train_fwd_bwd/encode_text/bf16_autocast": train_txt_bf16,
        "extra/sdpa_direct/bf16_is_causal_True_no_mask(4,8,77,64)": sdpa_causal_bf16,
        "extra/sdpa_direct/bf16_float_causal_mask(4,8,77,64)": sdpa_floatmask_bf16,
    }


# ------------------------------------------------------------------ M6
log("M6: SDPA flag matrix")
configs = [
    ("train_py_HEAD_flags(flash=1,mem_eff=0,math=0,matmul=medium)", dict(flash=True, mem=False, math_=False, matmul="medium")),
    ("train_py_worktree_flags(flash=1,mem_eff=0,math=0,matmul=highest)", dict(flash=True, mem=False, math_=False, matmul="highest")),
    ("default_flags(all_backends_enabled,matmul=highest)", dict(flash=True, mem=True, math_=True, matmul="highest")),
]
m6 = {}
outs = {}
for cname, kw in configs:
    set_flags(**kw)
    block = {"flags_in_effect": get_flags(), "cases": {}}
    for case, fn in m6_cases(model).items():
        rec = run(fn)
        outs[(cname, case)] = rec.pop("_out", None)
        block["cases"][case] = rec
        status = "OK" if rec["ok"] else f"FAIL {rec['error_type']}: {rec['error'][:160]!r}"
        log(f"  [{cname}] {case}: {status}")
    m6[cname] = block
    free()

# Which attention ops run under the default flags (eval, no_grad unless noted).
set_flags(True, True, True, "highest")
ops = {}
for case, fn in m6_cases(model).items():
    if case.startswith("extra/sdpa_direct"):
        continue
    ops[case] = attention_ops(fn)
m6["default_flags_attention_ops"] = ops
set_flags(True, False, False, "medium")
ops_fo = {}
for case in ("encode_image/fp16_autocast", "encode_image/bf16_autocast"):
    ops_fo[case] = attention_ops(m6_cases(model)[case])
m6["train_py_flags_attention_ops"] = ops_fo
set_flags(True, True, True, "highest")

# Cross-config agreement of successful outputs.
agree = {}
cn = [c for c, _ in configs]
for case in ("encode_image/fp16_autocast", "encode_image/bf16_autocast"):
    a, b = outs.get((cn[0], case)), outs.get((cn[2], case))
    if a is not None and b is not None:
        agree[case + " : HEAD flags vs default flags"] = {
            "bit_identical": bool(torch.equal(a, b)), "max_abs_diff": float((a - b).abs().max())}
m6["output_agreement"] = agree
facts["M6"] = m6
save()

# ------------------------------------------------------------------ extra: fp32 fast path vs slow path
log("extra: MHA fast path vs slow path, fp32 eval")
set_flags(True, True, True, "highest")
with torch.no_grad():
    fast = model.encode_image(images4)[0].clone()
    torch.backends.mha.set_fastpath_enabled(False)
    slow = model.encode_image(images4)[0].clone()
    torch.backends.mha.set_fastpath_enabled(True)
    fast2 = model.encode_image(images4)[0].clone()
facts["extra_fastpath_vs_slowpath_fp32"] = {
    "fast_vs_fast_rerun_bit_identical": bool(torch.equal(fast, fast2)),
    "fast_vs_slow_bit_identical": bool(torch.equal(fast, slow)),
    "fast_vs_slow_max_abs_diff": float((fast - slow).abs().max()),
    "fast_vs_slow_min_cosine": float(F.cosine_similarity(fast, slow, dim=-1).min()),
    "fast_ops": attention_ops(lambda: model.encode_image(images4)),
}
save()


# ------------------------------------------------------------------ LoRA (merged weight) helper
class QVLoRA(torch.nn.Module):
    """Parametrization W -> W + [s*B_q A_q ; 0 ; s*B_v A_v] on a packed (3d, d) in_proj_weight."""

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


def make_lora_model(base, seed=0):
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    loras = []
    for tr in (m.visual.transformer, m.transformer):
        for blk in tr.resblocks:
            mha = blk.attn
            lp = QVLoRA(mha.embed_dim).to(mha.in_proj_weight.device)
            parametrize.register_parametrization(mha, "in_proj_weight", lp)
            loras.append(lp)
    m.logit_scale.requires_grad_(True)
    return m, loras


log("extra: LoRA-at-init bit identity")
lora_model, loras = make_lora_model(model)
lora_model.eval()
n_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
ident = {"n_trainable_params_in_lora_model": int(n_trainable)}
for prec in ("fp32", "fp16", "bf16"):
    with torch.no_grad(), ac(prec):
        a = model.encode_image(images4)[0].float()
        b = lora_model.encode_image(images4)[0].float()
        ta = model.encode_text(tokens4).float()
        tb = lora_model.encode_text(tokens4).float()
    ident[prec] = {"image_bit_identical": bool(torch.equal(a, b)), "image_max_abs_diff": float((a - b).abs().max()),
                   "text_bit_identical": bool(torch.equal(ta, tb)), "text_max_abs_diff": float((ta - tb).abs().max())}
with torch.no_grad():
    loras[0].B_q[0, 0] = 1e-3      # visual block 0, q adapter
    loras[12].B_v[0, 0] = 1e-3     # text block 0, v adapter
    a = model.encode_image(images4)[0].float()
    b = lora_model.encode_image(images4)[0].float()
    ta = model.encode_text(tokens4).float()
    tb = lora_model.encode_text(tokens4).float()
    ident["after_one_entry_set_1e-3_fp32"] = {"image_max_abs_diff": float((a - b).abs().max()),
                                              "text_max_abs_diff": float((ta - tb).abs().max())}
    loras[0].B_q.zero_()
    loras[12].B_v.zero_()
facts["extra_lora_init_identity"] = ident
log(f"  {json.dumps(ident)}")
save()

# ------------------------------------------------------------------ extra: matmul precision
log("extra: fp32 matmul precision modes")
mm = {}
A = torch.randn(4096, 4096, device=DEV)
B = torch.randn(4096, 4096, device=DEV)
ref = (A.double() @ B.double())
for mode in ("highest", "high", "medium"):
    torch.set_float32_matmul_precision(mode)
    C = A @ B
    torch.cuda.synchronize()
    mm[mode] = {"rel_fro_err_vs_fp64": float((C.double() - ref).norm() / ref.norm()),
                "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32}
torch.set_float32_matmul_precision("highest")
facts["extra_matmul_precision"] = mm
del A, B, C, ref
free()
save()

# ------------------------------------------------------------------ M7 (default flags, bf16 autocast)
set_flags(True, True, True, "highest")
facts["M7_flags"] = get_flags()
M7 = {}


def timed(step, warmup, iters):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def probe(label, make_step, n_items, warmup=2, iters=5):
    """Run a step function and report items/s and peak memory, catching OOM."""
    if time.time() - T0 > SOFT_BUDGET_S:
        return {"label": label, "skipped": "soft time budget reached"}
    free()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rec = {"label": label, "n_items": n_items}
    try:
        step = make_step()
        dt = timed(step, warmup, iters)
        rec.update(ok=True, step_s=dt, items_per_s=n_items / dt)
    except torch.cuda.OutOfMemoryError as e:
        rec.update(ok=False, oom=True, error=str(e).split("\n")[0][:400])
    except Exception as e:
        rec.update(ok=False, oom=False, error=f"{type(e).__name__}: {str(e)[:400]}")
    rec["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / GiB
    rec["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / GiB
    rec["allocated_before_GiB"] = base / GiB
    step = None
    free()
    log(f"  {label}: " + (f"{rec.get('items_per_s', 0):.0f} items/s, " if rec.get("ok") else
                          f"FAILED ({rec.get('error', rec.get('skipped'))[:120]}), ")
        + f"peak {rec['peak_allocated_GiB']:.2f} GiB")
    return rec


# (a) inference throughput
log("M7(a): inference encode_image")
inf = {}
for prec in ("bf16", "fp16"):
    for bs in (1024,):
        def mk(bs=bs, prec=prec):
            x = torch.randn(bs, 3, 224, 224, device=DEV)

            def step():
                with torch.no_grad(), ac(prec):
                    model.encode_image(x)
            return step
        inf[f"encode_image/{prec}_autocast/bs{bs}"] = probe(f"infer image {prec} bs{bs}", mk, bs, warmup=3, iters=10)
M7["a_inference_image"] = inf
facts["M7"] = M7
save()


# (b) image tower training-like step
def make_image_step(m, bs, ckpt, lora):
    def mk():
        m.train()
        m.set_grad_checkpointing(ckpt)
        x = torch.randn(bs, 3, 224, 224, device=DEV)
        if not lora:
            x.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]

        def step():
            with (parametrize.cached() if lora else nullcontext()):
                with ac("bf16"):
                    f = m.encode_image(x)[0]
                    loss = F.normalize(f.float(), dim=-1).sum()
                loss.backward()
            x.grad = None
            for p in params:
                p.grad = None
        return step
    return mk


log("M7(b): image tower training-like step (frozen weights, input requires grad)")
img_proxy = {}
for bs, ckpt in ((256, False), (512, False), (1024, False), (2048, False),
                 (1024, True), (2048, True), (4096, True)):
    key = f"bs{bs}/{'ckpt' if ckpt else 'no_ckpt'}"
    img_proxy[key] = probe(f"proxy image {key}", make_image_step(model, bs, ckpt, False), bs)
model.set_grad_checkpointing(False)
model.eval()
M7["b_image_train_proxy"] = img_proxy
save()

log("M7(b) extra: image tower with real merged-weight LoRA adapters")
img_lora = {}
for bs, ckpt in ((512, False), (1024, False), (2048, False), (2048, True), (4096, True)):
    key = f"bs{bs}/{'ckpt' if ckpt else 'no_ckpt'}"
    img_lora[key] = probe(f"lora image {key}", make_image_step(lora_model, bs, ckpt, True), bs)
lora_model.set_grad_checkpointing(False)
lora_model.eval()
M7["b_extra_image_train_merged_lora"] = img_lora
save()

# (c) text tower
log("M7(c): text tower")
vocab = json.loads(VOCAB.read_text())
RANKS, SEP = vocab["ranks"], vocab.get("sep", "|")


def render(path, rank_idx):
    parts = path.split(SEP)[: rank_idx + 1]
    if rank_idx == len(RANKS) - 1 and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    return f"a photo of {' '.join(parts)}."


species = vocab["vocab"]["species"]
rng = np.random.default_rng(0)
pick = rng.choice(len(species), size=2048, replace=False)
texts = [render(species[i], len(RANKS) - 1) for i in pick]
tokens77 = tok(texts).to(DEV)
eot = tokens77.argmax(dim=-1)
fits32 = eot < 32
del vocab


def encode_text_len(m, text, L):
    """CLIP.encode_text with the sequence cut to its first L tokens (positional embedding and
    causal mask sliced to L). For L = context_length it is the same op sequence as encode_text."""
    text = text[:, :L]
    cast_dtype = m.transformer.get_cast_dtype()
    x = m.token_embedding(text).to(cast_dtype)
    x = x + m.positional_embedding[:L].to(cast_dtype)
    x = m.transformer(x, attn_mask=m.attn_mask[:L, :L])
    x = m.ln_final(x)
    x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    return x @ m.text_projection


txt = {"n_strings": len(texts), "source": f"2048 species-level 'a photo of ...' strings sampled (seed 0) from {VOCAB}",
       "eot_index_max": int(eot.max()), "eot_index_median": float(eot.float().median()),
       "n_fit_in_32_tokens": int(fits32.sum()), "examples": texts[:3]}

# Stock encode_text cannot take a 32-token input; record its error.
txt["stock_encode_text_on_32_tokens"] = {k: v for k, v in run(lambda: model.encode_text(tokens77[:, :32])).items()
                                         if k != "_out"}
with torch.no_grad():
    ref77 = model.encode_text(tokens77)
    mine77 = encode_text_len(model, tokens77, 77)
txt["encode_text_len77_equals_stock_fp32"] = bool(torch.equal(ref77, mine77))

# Truncation equivalence on strings whose EOT index is < 32.
sub = tokens77[fits32]
eq = {}
for prec in ("fp32", "bf16"):
    with torch.no_grad(), ac(prec):
        e77 = encode_text_len(model, sub, 77).float()
        e32 = encode_text_len(model, sub, 32).float()
    cos = F.cosine_similarity(e77, e32, dim=-1)
    eq[prec] = {"n": int(sub.shape[0]), "max_abs_diff": float((e77 - e32).abs().max()),
                "max_abs_value": float(e77.abs().max()),
                "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                "n_rows_bit_identical": int((e77 == e32).all(dim=-1).sum())}
eq["fp32_attention_ops_77"] = attention_ops(lambda: encode_text_len(model, sub, 77))
txt["truncation_77_vs_32"] = eq
log(f"  truncation check: {json.dumps({k: v for k, v in eq.items() if k in ('fp32', 'bf16')})}")


def make_text_step(m, L, ckpt, lora, train=True):
    def mk():
        if not train:
            m.eval()

            def step():
                with torch.no_grad(), ac("bf16"):
                    encode_text_len(m, tokens77, L)
            return step
        m.train()
        m.set_grad_checkpointing(ckpt)
        h = None if lora else make_text_grad_hook(m)
        params = [p for p in m.parameters() if p.requires_grad]
        hooks.append(h)

        def step():
            with (parametrize.cached() if lora else nullcontext()):
                with ac("bf16"):
                    f = encode_text_len(m, tokens77, L)
                    loss = F.normalize(f.float(), dim=-1).sum()
                loss.backward()
            for p in params:
                p.grad = None
        return step
    return mk


hooks = []
tp = {}
for L in (77, 32):
    tp[f"infer/L{L}"] = probe(f"infer text L{L} bs2048", make_text_step(model, L, False, False, train=False), 2048,
                              warmup=3, iters=10)
    for ckpt in (False, True):
        key = f"train_proxy/L{L}/{'ckpt' if ckpt else 'no_ckpt'}"
        tp[key] = probe(f"proxy text {key} bs2048", make_text_step(model, L, ckpt, False), 2048)
        for h in hooks:
            if h is not None:
                h.remove()
        hooks.clear()
    key = f"extra_merged_lora/L{L}/no_ckpt"
    tp[key] = probe(f"lora text {key} bs2048", make_text_step(lora_model, L, False, True), 2048)
model.set_grad_checkpointing(False)
lora_model.set_grad_checkpointing(False)
model.eval()
lora_model.eval()
txt["throughput_bs2048"] = tp
M7["c_text"] = txt
save()


# ------------------------------------------------------------------ extra: one combined LoRA step
def make_combined_step(m, bs_img, img_ckpt, L):
    def mk():
        m.train()
        m.visual.set_grad_checkpointing(img_ckpt)
        m.transformer.grad_checkpointing = False
        x = torch.randn(bs_img, 3, 224, 224, device=DEV)
        params = [p for p in m.parameters() if p.requires_grad]

        def step():
            with parametrize.cached():
                with ac("bf16"):
                    fi = F.normalize(m.encode_image(x)[0].float(), dim=-1)
                    ft = F.normalize(encode_text_len(m, tokens77, L).float(), dim=-1)
                    logits = m.logit_scale.exp() * fi @ ft.T
                    loss = logits.logsumexp(dim=-1).mean()
                loss.backward()
            for p in params:
                p.grad = None
        return step
    return mk


log("extra: combined image+text merged-LoRA step (per-GPU share of global batch 8192 on 4 GPUs)")
comb = {}
for L in (77, 32):
    key = f"img2048_ckpt+text2048_L{L}_no_ckpt"
    comb[key] = probe(f"combined {key}", make_combined_step(lora_model, 2048, True, L), 2048)
M7["extra_combined_lora_step"] = comb
facts["M7"] = M7
facts["elapsed_s"] = round(time.time() - T0, 1)
save()
log(f"done; wrote {OUT}")
