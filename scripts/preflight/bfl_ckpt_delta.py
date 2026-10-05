"""Compare the released BFL checkpoints against BioCLIP 1, tensor by tensor.

Purpose (lit-review evidence for agent/litreview/bfl-level-restricted-loss.md):
  The BFL paper (arXiv 2606.21838) says it "fine-tunes" BioCLIP ViT-B/16 but does not say
  whether the fine-tune is full or parameter-efficient (LoRA). The weights answer that:
  - full fine-tune  -> every tensor moves (LayerNorms, MLPs, k-slice of in_proj, embeddings).
  - merged LoRA     -> only the adapted slices move, and each delta has rank <= r.
  It also reads logit_scale, the single shared temperature, from all checkpoints.

Read-only inputs (never written):
  /u/liv/bdbk/ckpt/bioclip1/open_clip_pytorch_model.bin
  /u/liv/bdbk/ckpt/bfl-euclidean/open_clip_model.safetensors
  /u/liv/bdbk/ckpt/bfl-hyperbolic/open_clip_model.safetensors
Output:
  /projects/bdbk/liv/repos/taxa_maze/audit/preflight/bfl_ckpt_delta.json
"""

import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

BIO = "/u/liv/bdbk/ckpt/bioclip1/open_clip_pytorch_model.bin"
BIO_SHA_EXPECTED = "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"
BFL = {
    "bfl-euclidean": "/u/liv/bdbk/ckpt/bfl-euclidean/open_clip_model.safetensors",
    "bfl-hyperbolic": "/u/liv/bdbk/ckpt/bfl-hyperbolic/open_clip_model.safetensors",
}
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight/bfl_ckpt_delta.json"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_bioclip(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}


def group_of(key):
    """Coarse parameter group used for the summary table."""
    tower = "visual" if key.startswith("visual.") else ("text" if not key in ("logit_scale",) else "scalar")
    if "resblocks" in key:
        if "attn.in_proj" in key:
            part = "attn.in_proj"
        elif "attn.out_proj" in key:
            part = "attn.out_proj"
        elif ".mlp." in key:
            part = "mlp"
        elif ".ln_" in key:
            part = "ln(block)"
        else:
            part = "other(block)"
    else:
        part = key[len("visual."):] if tower == "visual" else key
    return f"{tower}:{part}"


def delta_stats(a, b):
    a64 = a.double()
    b64 = b.double()
    d = b64 - a64
    na = a64.norm().item()
    nd = d.norm().item()
    return {
        "shape": list(a.shape),
        "norm_base": na,
        "norm_delta": nd,
        "rel_delta": (nd / na) if na > 0 else float("nan"),
        "frac_elems_changed": (d != 0).double().mean().item(),
        "max_abs_delta": d.abs().max().item() if d.numel() else 0.0,
    }


def svd_rank_stats(d2):
    """Singular-value profile of a 2-D delta. A merged rank-r LoRA has sv[r:] ~ 0."""
    s = torch.linalg.svdvals(d2.double()).numpy()
    if s[0] == 0:
        return {"sv_max": 0.0}
    sn = s / s[0]
    energy = np.cumsum(s**2) / np.sum(s**2)
    return {
        "sv_max": float(s[0]),
        "sv_norm_at": {str(i): float(sn[i]) for i in (1, 4, 8, 15, 16, 31, 32, 63, 64, 127) if i < len(sn)},
        "rank_99pct_energy": int(np.searchsorted(energy, 0.99) + 1),
        "rank_sv_gt_1e-3_of_max": int((sn > 1e-3).sum()),
        "full_dim": int(min(d2.shape)),
    }


def main():
    t0 = time.time()
    res = {"inputs": {"bioclip1": BIO, **BFL}}
    got = sha256(BIO)
    res["bioclip1_sha256"] = got
    res["bioclip1_sha256_matches_hf_7b4abf1_lfs"] = (got == BIO_SHA_EXPECTED)
    print(f"[sha256] bioclip1 .bin = {got}  matches HF rev 7b4abf1 LFS hash: {got == BIO_SHA_EXPECTED}", flush=True)

    base = load_bioclip(BIO)
    print(f"[load] bioclip1 tensors: {len(base)}  params: {sum(v.numel() for v in base.values()):,}", flush=True)
    ls_b = base["logit_scale"].float().item()
    res["bioclip1_logit_scale"] = {"raw": ls_b, "exp": math.exp(ls_b)}
    res["float32_log100"] = float(np.float32(math.log(100)))
    print(f"[logit_scale] bioclip1 raw={ls_b!r} exp={math.exp(ls_b)!r}; float32(ln 100)={np.float32(math.log(100))!r}", flush=True)

    for name, path in BFL.items():
        sd = load_file(path)
        only_bfl = sorted(set(sd) - set(base))
        only_base = sorted(set(base) - set(sd))
        rec = {"n_tensors": len(sd), "only_in_bfl": only_bfl, "only_in_bioclip1": only_base}
        ls = sd["logit_scale"].float().item()
        rec["logit_scale"] = {"raw": ls, "exp": math.exp(ls), "equals_float32_ln100": ls == float(np.float32(math.log(100))),
                              "equals_bioclip1": ls == ls_b}
        per = {}
        groups = {}
        for k in sorted(set(sd) & set(base)):
            st = delta_stats(base[k], sd[k])
            per[k] = st
            g = group_of(k)
            G = groups.setdefault(g, {"n": 0, "n_changed": 0, "sum_sq_delta": 0.0, "sum_sq_base": 0.0, "min_frac_changed": 1.0})
            G["n"] += 1
            G["n_changed"] += int(st["norm_delta"] > 0)
            G["sum_sq_delta"] += st["norm_delta"] ** 2
            G["sum_sq_base"] += st["norm_base"] ** 2
            G["min_frac_changed"] = min(G["min_frac_changed"], st["frac_elems_changed"])
        for g, G in groups.items():
            G["rel_delta_group"] = math.sqrt(G["sum_sq_delta"] / G["sum_sq_base"]) if G["sum_sq_base"] > 0 else float("nan")
        rec["groups"] = groups
        rec["n_common"] = len(per)
        rec["n_unchanged_tensors"] = [k for k, st in per.items() if st["norm_delta"] == 0]
        # q / k / v split of the packed in_proj weights, plus singular-value profiles of the deltas
        qkv = {}
        for k in sorted(per):
            if k.endswith("attn.in_proj_weight"):
                W0, W1 = base[k], sd[k]
                dim = W0.shape[1]
                for j, nm in enumerate("qkv"):
                    a = W0[j * dim:(j + 1) * dim]
                    b = W1[j * dim:(j + 1) * dim]
                    st = delta_stats(a, b)
                    st.update(svd_rank_stats(b.double() - a.double()))
                    qkv[f"{k}[{nm}]"] = st
            if k.endswith("attn.out_proj.weight") or k.endswith("mlp.c_fc.weight"):
                st = dict(per[k])
                st.update(svd_rank_stats(sd[k].double() - base[k].double()))
                per[k] = st
        rec["qkv_split"] = qkv
        rec["per_tensor"] = per
        res[name] = rec

        # console summary
        print(f"\n===== {name}: {len(sd)} tensors; only_in_bfl={only_bfl}; only_in_bioclip1={only_base}", flush=True)
        print(f"  logit_scale raw={ls!r} exp={math.exp(ls)!r} equals float32(ln100)={rec['logit_scale']['equals_float32_ln100']} equals bioclip1={ls == ls_b}")
        print(f"  unchanged tensors ({len(rec['n_unchanged_tensors'])}): {rec['n_unchanged_tensors'][:20]}")
        print(f"  {'group':34s} {'n':>4s} {'changed':>8s} {'rel_delta':>10s} {'min_frac_elems_changed':>22s}")
        for g in sorted(groups):
            G = groups[g]
            print(f"  {g:34s} {G['n']:4d} {G['n_changed']:8d} {G['rel_delta_group']:10.4f} {G['min_frac_changed']:22.4f}")
        for tw in ("visual.transformer.resblocks", "transformer.resblocks"):
            for layer in (0, 5, 11):
                for nm in "qkv":
                    key = f"{tw}.{layer}.attn.in_proj_weight[{nm}]"
                    if key in qkv:
                        s = qkv[key]
                        print(f"  {key:52s} rel={s['rel_delta']:.4f} frac_changed={s['frac_elems_changed']:.4f} "
                              f"rank99={s.get('rank_99pct_energy')} rank>1e-3={s.get('rank_sv_gt_1e-3_of_max')}/{s.get('full_dim')} "
                              f"sv[16]/sv[0]={s.get('sv_norm_at', {}).get('16', float('nan')):.4f}")
                key = f"{tw}.{layer}.mlp.c_fc.weight"
                if key in per:
                    s = per[key]
                    print(f"  {key:52s} rel={s['rel_delta']:.4f} frac_changed={s['frac_elems_changed']:.4f} "
                          f"rank99={s.get('rank_99pct_energy')} rank>1e-3={s.get('rank_sv_gt_1e-3_of_max')}/{s.get('full_dim')}")
        sys.stdout.flush()

    res["elapsed_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[done] wrote {OUT} in {res['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(16)
    main()
