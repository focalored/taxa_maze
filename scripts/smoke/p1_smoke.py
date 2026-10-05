#!/usr/bin/env python
"""Pilot 1 smoke tests and gates (spec lines 50, 63, 186-188; Amendment 1 S12; preflight readings S14, S17, S35); each writes result.json and log.txt under logs/smoke/<test>/<stamp>_j<jobid>/.

Usage: `python scripts/smoke/p1_smoke.py cache|adapter|zero_sum|penalty_twice|penalty_forms|one_step|grad_1v4 [--world-run 4]` (Slurm: scripts/smoke/p1_smoke.sh).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import REPO_ROOT, assert_outside_projects, load_paths  # noqa: E402

PATHS = load_paths()
STORE = Path(PATHS.p1_data_dir) / "store"
PREP = Path(PATHS.p1_data_dir) / "p1"


class Run:
    """Output directory plus a tee of stdout into log.txt."""

    def __init__(self, test: str):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = assert_outside_projects(REPO_ROOT / "logs" / "smoke" / test /
                                           f"{stamp}_j{os.environ.get('SLURM_JOB_ID', 'local')}")
        self.rank0 = int(os.environ.get("RANK", "0")) == 0
        if self.rank0:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.log = open(self.dir / "log.txt", "w")
        self.t0 = time.time()

    def p(self, *a):
        msg = " ".join(str(x) for x in a)
        if self.rank0:
            print(msg, flush=True)
            self.log.write(msg + "\n")
            self.log.flush()

    def finish(self, result: dict) -> None:
        import torch
        result.update({"seconds": round(time.time() - self.t0, 1), "host": os.uname().nodename,
                       "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        if self.rank0:
            (self.dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
            self.p(f"PASS={result.get('pass')}  -> {self.dir / 'result.json'}")
        if not result.get("pass"):
            sys.exit(1)


def tree_and_bank(device):
    import torch
    from src.data.taxonomy import build_train_tree
    from src.models.bank import Bank
    tree = build_train_tree(STORE, s9_drop=True, K=16, heldout_file=PREP / "heldout_species_s9.txt")
    bank = Bank(tree, 512, device)
    bank.set_species_means(torch.from_numpy(np.load(PREP / "bank_seed_s9.npy")).to(device))
    return tree, bank


def plan_batch(tree, store, B: int, world: int, rank_bins=None):
    """Step 0 of the real epoch-0 plan at batch B; returns the requested rank bins concatenated."""
    import torch
    from src.data.tol_datamodule import _read_sorted
    from src.data.tol_sampler import make_epoch_plan
    plan = make_epoch_plan(tree.sp_start, tree.N_s, tree.K, B, world, 42, 0)
    bins = plan.steps[0] if rank_bins is None else [plan.steps[0][r] for r in rank_bins]
    idx = np.concatenate([b[0] for b in bins])
    gsp = np.concatenate([b[1] for b in bins])
    gsz = np.concatenate([b[2] for b in bins])
    u8 = _read_sorted(store, tree.img_shard[idx], tree.img_row[idx])
    return {"u8": torch.from_numpy(u8), "img": torch.from_numpy(idx), "grp_species": torch.from_numpy(gsp),
            "grp_sizes": torch.from_numpy(gsz)}


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def make_module(arm: str, lam: float, dev):
    from src.models.p1_module import P1Module
    m = P1Module(arm=arm, lr=1e-4, lam=lam, ckpt_dir=PATHS.bioclip1_ckpt, bank_seed_file=str(PREP / "bank_seed_s9.npy"))
    return m.to(dev)


# ------------------------------------------------------------------------------------------------
def test_cache(run: Run):
    """Spec line 50 and Amendment 1 S52: 1,000 store images through the new pipeline vs the cache rows."""
    import polars as pl
    import torch
    import torch.nn.functional as F
    from src.data.tol_store import ImageStore, to_model_input
    from src.models.bioclip_lora import add_qv_lora, load_bioclip1
    idx = pl.read_parquet(STORE / "index.parquet")
    miss = set(pl.read_csv(STORE / "not_in_cache.csv")["uuid"].to_list())
    cand = idx.filter(~pl.col("uuid").is_in(list(miss)))
    pick = np.sort(np.random.default_rng(42).choice(cand.height, 1000, replace=False))
    rows = cand[pick]
    store = ImageStore(STORE)
    u8 = store.read_many(rows["shard"].to_numpy(), rows["row"].to_numpy())
    cache = Path(PATHS.tol_embed_dir) / "results/probe_cache_b16/tol-eol/bioclip1"
    pos = {u: i for i, u in enumerate((cache / "ids.txt").read_text().split())}
    emb = np.load(cache / "emb.f16.npy", mmap_mode="r")
    ref = torch.from_numpy(np.stack([emb[pos[u]] for u in rows["uuid"].to_list()]).astype(np.float64))
    res = {"n": 1000, "splits": rows["split"].value_counts().to_dicts(), "bar": 0.9999}
    for name, lora in (("bioclip1", False), ("lora_at_init", True)):
        model, _, _ = load_bioclip1(PATHS.bioclip1_ckpt, device="cuda")
        if lora:
            add_qv_lora(model)
            model = model.cuda()
        model.eval()
        outs = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for a in range(0, 1000, 250):
                f, _ = model.encode_image(to_model_input(torch.from_numpy(u8[a:a + 250]).cuda()))
                outs.append(f.float().cpu())
        new = torch.cat(outs).double()
        cos = (F.normalize(new, dim=-1) * F.normalize(ref, dim=-1)).sum(1)
        res[name] = {"min_cos": float(cos.min()), "median_cos": float(cos.median()), "n_below_bar": int((cos < 0.9999).sum())}
        run.p(name, res[name])
    res["pass"] = all(res[k]["n_below_bar"] == 0 for k in ("bioclip1", "lora_at_init"))
    run.finish(res)


def test_adapter(run: Run):
    """Spec lines 60-63, preflight reading S50: bit-identical at init in each precision mode; one nonzero adapter changes the output."""
    import torch
    from src.data.tol_store import ImageStore, to_model_input
    from src.eval.zeroshot import class_text
    from src.models.bioclip_lora import add_qv_lora, load_bioclip1, lora_modules, param_counts
    store = ImageStore(STORE)
    u8 = store.read_many(np.full(64, 1), np.arange(64))
    x = to_model_input(torch.from_numpy(u8).cuda())
    base, _, tok = load_bioclip1(PATHS.bioclip1_ckpt, device="cuda")
    lora, _, _ = load_bioclip1(PATHS.bioclip1_ckpt, device="cuda")
    add_qv_lora(lora)
    lora = lora.cuda()
    texts = tok([class_text(["Animalia", "Chordata", "Aves", "Passeriformes"][: i % 4 + 1], "photo") for i in range(64)]).cuda()
    modes = {"fp32_eval": (None, False), "fp16_autocast_eval": (torch.float16, False),
             "bf16_autocast_train_grad": (torch.bfloat16, True)}

    def run_mode(model, dt, train):
        model.train(train)
        ctx = torch.autocast(device_type="cuda", dtype=dt) if dt else torch.autocast(device_type="cuda", enabled=False)
        with torch.set_grad_enabled(train), ctx:
            img, _ = model.encode_image(x)
            txt = model.encode_text(texts)
        return img.detach().float(), txt.detach().float()

    res = {"params": param_counts(lora), "visual_output_dim": int(lora.visual.output_dim), "modes": {}}
    for name, (dt, train) in modes.items():
        bi, bt = run_mode(base, dt, train)
        li, lt = run_mode(lora, dt, train)
        res["modes"][name] = {"image_bit_identical": bool(torch.equal(bi, li)), "text_bit_identical": bool(torch.equal(bt, lt))}
    vis = [m for n, m in lora_modules(lora) if n.startswith("visual.")][0]
    txt_m = [m for n, m in lora_modules(lora) if not n.startswith("visual.")][0]
    mha = lora.visual.transformer.resblocks[0].attn
    w0 = mha.in_proj_weight.detach().clone()
    with torch.no_grad():
        vis.B_q[0, 0] = 1e-3
        vis.B_v[0, 0] = 1e-3
        txt_m.B_v[0, 0] = 1e-3
    dw = (mha.in_proj_weight.detach() - w0).abs()
    d = mha.embed_dim
    res["rows_moved"] = {"q": float(dw[:d].max()), "k": float(dw[d: 2 * d].max()), "v": float(dw[2 * d:].max())}
    for name, (dt, train) in modes.items():
        bi, bt = run_mode(base, dt, train)
        li, lt = run_mode(lora, dt, train)
        res["modes"][name]["image_changed_after_nonzero_adapter"] = float((bi - li).abs().max())
        res["modes"][name]["text_changed_after_nonzero_adapter"] = float((bt - lt).abs().max())
    run.p(json.dumps(res, indent=1))
    m = res["modes"]
    rm = res["rows_moved"]
    res["pass"] = (rm["q"] > 0 and rm["v"] > 0 and rm["k"] == 0
                   and res["visual_output_dim"] == 512 and res["params"]["trainable_fraction"] <= 0.01
                   and all(v["image_bit_identical"] and v["text_bit_identical"] for v in m.values())
                   and all(v["image_changed_after_nonzero_adapter"] > 0 and v["text_changed_after_nonzero_adapter"] > 0
                           for v in m.values()))
    run.finish(res)


def test_zero_sum(run: Run):
    """Amendment 1 S12 (spec line 187): a family with >= 3 genera and uneven species counts, >= 100 real updates."""
    import torch
    from src.data.tol_datamodule import _read_sorted
    from src.data.tol_store import ImageStore, to_model_input
    from src.models.bank import ZERO_SUM_TOL
    from src.models.bioclip_lora import add_qv_lora, load_bioclip1
    from src.models.losses import group_means
    tree, bank = tree_and_bank("cuda")
    seed_ms = bank.m_s.clone()
    fam = None
    for f in range(len(tree.node_keys[4])):
        gen = np.flatnonzero(tree.parent[5] == f)
        counts = tree.n_species[5][gen]
        if len(gen) >= 3 and len(set(counts.tolist())) > 1 and 30 <= counts.sum() <= 120:
            fam = f
            break
    species = np.flatnonzero(tree.anc[:, 4] == fam)
    model, _, _ = load_bioclip1(PATHS.bioclip1_ckpt, device="cuda")
    add_qv_lora(model)
    model = model.cuda().eval()
    store = ImageStore(STORE)
    rng = np.random.default_rng(42)
    n_updates = 120
    for u in range(n_updates):
        sids = np.sort(rng.choice(species, size=min(24, len(species)), replace=False))
        idx, sizes = [], []
        for s in sids:
            n, g = int(tree.N_s[s]), int(tree.g_s[s])
            k = n // g + (rng.random() < (n % g) / g)
            pick = rng.choice(n, size=k, replace=False) + tree.sp_start[s]
            idx.append(pick)
            sizes.append(k)
        idx = np.concatenate(idx)
        u8 = _read_sorted(store, tree.img_shard[idx], tree.img_row[idx])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f, _ = model.encode_image(to_model_input(torch.from_numpy(u8).cuda()))
        v = torch.nn.functional.normalize(f.float(), dim=-1)
        mu = group_means(v, torch.as_tensor(sizes, device="cuda"))
        bank.ema_update(torch.as_tensor(sids, device="cuda"), mu, torch.as_tensor(sizes, device="cuda"))
    moved = float((bank.m_s[torch.as_tensor(species, device="cuda")] - seed_ms[torch.as_tensor(species, device="cuda")]).abs().max())
    sub = bank.zero_sum_check(torch.as_tensor(species, device="cuda"))
    full = bank.zero_sum_check()
    res = {"family": tree.node_keys[4][fam], "n_genera": int(tree.n_children[4][fam]), "n_species": int(len(species)),
           "genus_species_counts": sorted(tree.n_species[5][np.flatnonzero(tree.parent[5] == fam)].tolist()),
           "n_updates": n_updates, "max_species_move_from_seed": moved, "tolerance": ZERO_SUM_TOL,
           "subtree": sub, "full_bank": full}
    res["pass"] = moved > 0 and all(v <= ZERO_SUM_TOL for k, v in sub.items() if k != "zero_sum/max_seen") and \
        all(v <= ZERO_SUM_TOL for k, v in full.items() if k != "zero_sum/max_seen")
    run.p(json.dumps(res, indent=1))
    run.finish(res)


def test_penalty_twice(run: Run):
    """Spec line 188, preflight reading S14: same parameters, bank and batch -> bit-identical L_B; bank and parameters untouched."""
    import torch
    from src.data.tol_store import ImageStore
    torch.use_deterministic_algorithms(True, warn_only=True)
    tree, bank = tree_and_bank("cuda")
    mod = make_module("d", 1.0, "cuda")
    mod.bind(tree, bank, torch.device("cuda"))
    batch = to_dev(plan_batch(tree, ImageStore(STORE), 1000, 1), "cuda")
    snap_bank = [bank.m_s.clone(), [m.clone() for m in bank.m_nodes], bank.root.clone()]
    snap_par = [p.detach().clone() for _, p in mod.trainable]
    outs = []
    for _ in range(2):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = mod.forward_losses(batch, pen_grad=True)
        outs.append({"L_B": o["pen"]["L_B"].detach().clone(), "parts": o["pen"]["parts"].detach().clone(),
                     "mu_hat": o["mu_hat"].detach().clone()})
    same = {k: bool(torch.equal(outs[0][k], outs[1][k])) for k in outs[0]}
    bank_same = (torch.equal(snap_bank[0], bank.m_s) and all(torch.equal(a, b) for a, b in zip(snap_bank[1], bank.m_nodes))
                 and torch.equal(snap_bank[2], bank.root))
    par_same = all(torch.equal(a, p.detach()) for a, (_, p) in zip(snap_par, mod.trainable))
    res = {"L_B": [float(o["L_B"]) for o in outs], "L_B_hex": [float(o["L_B"]).hex() for o in outs],
           "bit_identical": same, "bank_unchanged": bank_same, "params_unchanged": par_same,
           "n_images": int(batch["img"].shape[0]), "n_groups": int(batch["grp_sizes"].shape[0])}
    res["pass"] = all(same.values()) and bank_same and par_same
    run.p(json.dumps(res, indent=1))
    run.finish(res)


def test_penalty_forms(run: Run):
    """Amendment 1 S22: the 1/g_s form and line 107's form agree when every g_s = 1, and differ otherwise."""
    import torch
    from src.data.tol_store import ImageStore
    from src.models.losses import penalty_local
    tree, bank = tree_and_bank("cuda")
    mod = make_module("d", 1.0, "cuda")
    mod.bind(tree, bank, torch.device("cuda"))
    batch = to_dev(plan_batch(tree, ImageStore(STORE), 1000, 1), "cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = mod.forward_losses(batch, pen_grad=False)
    mu, sids, n = out["mu_hat"], batch["grp_species"], int(batch["img"].shape[0])
    g1 = mod.t_g[sids] == 1
    res = {"n_groups": int(len(sids)), "n_groups_g1": int(g1.sum()), "n_groups_g_gt1": int((~g1).sum())}
    for tag, sel in (("all_g1", g1), ("mixed", torch.ones_like(g1))):
        a = penalty_local(mu[sel], sids[sel], bank, mod.t_P, mod.t_held, mod.t_g, n, "inv_groups")["L_B"]
        b = penalty_local(mu[sel], sids[sel], bank, mod.t_P, mod.t_held, mod.t_g, n, "uniform")["L_B"]
        res[tag] = {"inv_groups": float(a), "uniform": float(b), "bit_identical": bool(torch.equal(a, b))}
    res["pass"] = res["all_g1"]["bit_identical"] and res["n_groups_g1"] > 0 and (
        res["n_groups_g_gt1"] == 0 or not res["mixed"]["bit_identical"])
    run.p(json.dumps(res, indent=1))
    run.finish(res)


def test_one_step(run: Run):
    """Spec line 186, preflight reading S35: one step each for (d) and (b) on ~1,000 real-sampler images with lambda = 1."""
    import torch
    from src.data.tol_store import ImageStore
    from src.models.bioclip_lora import clamp_logit_scale
    tree, bank0 = tree_and_bank("cuda")
    store = ImageStore(STORE)
    batch = to_dev(plan_batch(tree, store, 1000, 1), "cuda")
    res = {"n_images": int(batch["img"].shape[0]), "lam": 1.0}
    for arm in ("d", "b"):
        from src.models.bank import Bank
        bank = Bank(tree, 512, "cuda")
        bank.copy_from(bank0)
        mod = make_module(arm, 1.0, "cuda")
        mod.bind(tree, bank, torch.device("cuda"))
        mod.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = mod.forward_losses(batch, pen_grad=True)
        g = mod.separate_grads(out)
        lora = ~torch.tensor([n == "logit_scale" for n, p in mod.trainable for _ in range(p.numel())], device="cuda")
        params = [p for _, p in mod.trainable]
        total = g["con"] + 1.0 * g["pen"]
        i = 0
        for p in params:
            p.grad = total[i: i + p.numel()].view_as(p).clone()
            i += p.numel()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0)
        opt.step()
        clamp_logit_scale(mod.model)
        bank.ema_update(batch["grp_species"], out["mu_hat"].detach().float(), batch["grp_sizes"])
        r = {"L_con": float(out["L_con"]), "L_B": float(out["pen"]["L_B"]),
             "grad_con_norm_lora": float(g["con"][lora].norm()), "grad_pen_norm_lora": float(g["pen"][lora].norm()),
             "params_finite_after_step": all(bool(torch.isfinite(p).all()) for p in params),
             "levels": out["levels"]}
        r["pass"] = (np.isfinite(r["L_con"]) and np.isfinite(r["L_B"]) and r["L_con"] > 0 and r["L_B"] > 0
                     and r["grad_con_norm_lora"] > 0 and r["grad_pen_norm_lora"] > 0 and r["params_finite_after_step"])
        res[arm] = r
        run.p(arm, json.dumps(r))
    res["pass"] = res["d"]["pass"] and res["b"]["pass"]
    run.finish(res)


def test_grad_1v4(run: Run, world_run: int):
    """Preflight reading S17: a fixed 1,000-image batch, 1 GPU vs 4 GPUs; gradient cosine >= 0.999, norms within 1%."""
    import torch
    import torch.distributed as dist
    from src.data.tol_store import ImageStore
    if world_run > 1:
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if world_run > 1 else 0
    dev = torch.device("cuda")
    tree, bank = tree_and_bank(dev)
    store = ImageStore(STORE)
    bins = [rank] if world_run > 1 else [0, 1, 2, 3]
    batch = to_dev(plan_batch(tree, store, 1000, 4, rank_bins=bins), dev)
    mod = make_module("d", 1.0, dev)
    mod.bind(tree, bank, dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = mod.forward_losses(batch, pen_grad=True)
    g = mod.separate_grads(out)
    L_con_global = float(all_sum(out["L_con"].detach()))
    tag = f"world{world_run}"
    shared = REPO_ROOT / "logs" / "smoke" / "grad_1v4" / "grads"
    if rank == 0:
        shared.mkdir(parents=True, exist_ok=True)
        np.savez(shared / f"{tag}.npz", con=g["con"].cpu().numpy(), pen=g["pen"].cpu().numpy(),
                 L_con=L_con_global, job=os.environ.get("SLURM_JOB_ID", "local"))
        run.p(tag, "saved", {k: float(v.norm()) for k, v in g.items()})
    other = shared / ("world1.npz" if world_run > 1 else "world4.npz")
    res = {"this": tag}
    if rank == 0 and other.exists():
        a = np.load(shared / "world1.npz")
        b = np.load(shared / "world4.npz")
        for k in ("con", "pen"):
            x, y = a[k].astype(np.float64), b[k].astype(np.float64)
            res[k] = {"cos": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
                      "norm_1gpu": float(np.linalg.norm(x)), "norm_4gpu": float(np.linalg.norm(y)),
                      "norm_rel_diff": float(abs(np.linalg.norm(x) / np.linalg.norm(y) - 1))}
        res["jobs"] = {"world1": str(a["job"]), "world4": str(b["job"])}
        res["L_con"] = {"world1": float(a["L_con"]), "world4": float(b["L_con"])}
        res["pass"] = all(res[k]["cos"] >= 0.999 and res[k]["norm_rel_diff"] <= 0.01 for k in ("con", "pen"))
        run.p(json.dumps(res, indent=1))
        run.finish(res)
    elif rank == 0:
        run.p("the other world size has not run yet; result.json is written by whichever runs second")
    if world_run > 1:
        dist.barrier()
        dist.destroy_process_group()


def all_sum(x):
    from src.models.losses import all_reduce_sum
    return all_reduce_sum(x.clone())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test", choices=["cache", "adapter", "zero_sum", "penalty_twice", "penalty_forms", "one_step",
                                     "grad_1v4"])
    ap.add_argument("--world-run", type=int, default=1)
    a = ap.parse_args()
    if a.world_run > 1:  # srun: map Slurm's task ids onto torch.distributed's variables
        os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
        os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
        os.environ.setdefault("LOCAL_RANK", os.environ["SLURM_LOCALID"])
    run = Run(a.test)
    {"cache": lambda: test_cache(run), "adapter": lambda: test_adapter(run), "zero_sum": lambda: test_zero_sum(run),
     "penalty_twice": lambda: test_penalty_twice(run), "penalty_forms": lambda: test_penalty_forms(run),
     "one_step": lambda: test_one_step(run),
     "grad_1v4": lambda: test_grad_1v4(run, a.world_run)}[a.test]()


if __name__ == "__main__":
    main()
