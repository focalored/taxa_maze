"""Pilot 1 LightningModule: arms (a)-(d) with manual optimization, hand-averaged gradients, the bank, the monitors, and the per-epoch refresh and ToL-val evaluation.

Usage: `python src/train.py experiment=p1/c_lvl model.lr=1e-4` (configs/experiment/p1/); smoke variants turn `refresh` and `eval_*` off.
"""
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.data.tol_catalog import RANKS
from src.data.tol_store import to_model_input
from src.eval.tolval import tolval_eval
from src.eval.zeroshot import class_text
from src.models.bank import Bank, species_means_pass
from src.models.bioclip_lora import (add_qv_lora, clamp_logit_scale, load_bioclip1, param_counts,
                                     trainable_parameters)
from src.models.losses import (all_reduce_sum, bfl_level_loss_local, gather_nograd, gather_varlen_nograd,
                               gather_with_grad, group_means, level_label_plan, penalty_local, world_info)

ARMS = {"a": ("flat", False), "b": ("flat", True), "c": ("lvl", False), "d": ("lvl", True)}


def _no_autocast():
    return torch.autocast(device_type="cuda", enabled=False)


class P1Module(L.LightningModule):
    def __init__(self, arm: str, lr: float, ckpt_dir: str, bank_seed_file: str, lam: float = 0.0,
                 lora_r: int = 16, lora_alpha: float = 32.0, lora_towers: str = "both", seed: int = 42, epochs: int = 10,
                 warmup_frac: float = 0.01, betas=(0.9, 0.98), eps: float = 1e-6, weight_decay: float = 0.0,
                 grad_clip: float = 1.0, penalty_weighting: str = "inv_groups", monitor_window=(50, 250),
                 monitor_every: int = 50, refresh: bool = True, eval_init: bool = True,
                 eval_each_epoch: bool = True, eval_forms=("photo", "lineage"),
                 drift_floor_file: Optional[str] = None, check_zero_sum: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {sorted(ARMS)}")
        self.loss_kind, self.penalized = ARMS[arm]
        if not self.penalized and lam != 0:
            raise ValueError("control arms keep the penalty out of their loss (Amendment 1 S15): lam must be 0")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        model, _, tokenizer = load_bioclip1(ckpt_dir)
        add_qv_lora(model, r=lora_r, alpha=lora_alpha, seed=seed, towers=lora_towers)
        model.set_grad_checkpointing(True)  # all blocks: 4,096 images per GPU need it (Amendment 2 A2.1)
        self.model, self.tokenizer = model, tokenizer
        self.bank: Optional[Bank] = None
        self._pending: Optional[dict] = None
        self.r_t: Dict[int, float] = {}
        self.decisions: Dict = {"epochs": {}}

    # ------------------------------------------------------------------------------ setup
    def configure_optimizers(self):
        params = [p for _, p in trainable_parameters(self.model)]
        h = self.hparams
        return torch.optim.AdamW(params, lr=h.lr, betas=tuple(h.betas), eps=h.eps, weight_decay=h.weight_decay)

    def on_fit_start(self):
        dm, dev = self.trainer.datamodule, self.device
        tree = dm.tree
        self.bind(tree, Bank(tree, int(self.model.visual.output_dim), dev), dev)
        if self._pending is not None:
            self.bank.load_state_dict(self._pending["bank"])
            self.r_t = {int(k): v for k, v in self._pending["r_t"].items()}
            self.decisions = self._pending["decisions"]
        else:
            seed = np.load(self.hparams.bank_seed_file, mmap_mode="r")
            meta = json.loads(Path(self.hparams.bank_seed_file).with_suffix(".json").read_text())
            if meta["species_keys_sha256"] != tree.species_keys_sha256():
                raise RuntimeError(f"{self.hparams.bank_seed_file} was built for a different species list")
            self.bank.set_species_means(torch.from_numpy(np.ascontiguousarray(seed)).to(dev))
        n_steps = int(tree.N_s.sum()) // dm.B + (int(tree.N_s.sum()) % dm.B > 0)
        if self.hparams.epochs != 10:
            raise ValueError("line 161 and S5: the schedule counts 10 epochs in every run, including the sweep")
        self.total_steps = self.hparams.epochs * n_steps
        self.warmup = math.ceil(self.hparams.warmup_frac * self.total_steps)
        if dm.s9_drop and dm.K == 16 and dm.B == 8192 and (n_steps, self.warmup) != (647, 65):
            raise RuntimeError(f"{n_steps} steps per epoch and {self.warmup} warmup steps; Amendment 1 S9 says 647 and 65")
        floor = self.hparams.drift_floor_file
        if floor and not Path(floor).exists():
            raise FileNotFoundError(f"drift floor {floor} (S25) is missing; run scripts/p1/prepare.sh first")
        self.run_dir = Path(self.trainer.default_root_dir)
        if self.trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            dm.write_stats(self.run_dir)
            info = {"arm": self.hparams.arm, "loss": self.loss_kind, "penalized": self.penalized,
                    "lam": self.hparams.lam, "lr": self.hparams.lr, "s9_drop": dm.s9_drop, "K": dm.K, "B": dm.B, "sampler": dm.sampler,
                    "lora_towers": self.hparams.lora_towers,
                    "world": self.W, "steps_per_epoch": n_steps, "total_steps": self.total_steps,
                    "warmup_steps": self.warmup, "penalty_weighting": self.hparams.penalty_weighting,
                    "params": param_counts(self.model)}
            self.decisions["run"] = info
            (self.run_dir / "run_info.json").write_text(json.dumps(info, indent=1))
        self.model.train()  # load_bioclip1 leaves the towers in eval mode; refresh and evaluations restore the mode they find

    def on_train_start(self):
        logs = {}
        floor = self.hparams.drift_floor_file
        if floor and Path(floor).exists():  # S25 drift floor, measured once by scripts/p1/prepare.py
            logs.update({f"drift_floor/{k}": float(v) for k, v in json.loads(Path(floor).read_text()).items()
                         if isinstance(v, (int, float))})
        if self.hparams.check_zero_sum:
            logs.update(self.bank.assert_zero_sum())
        self._log_direct(logs)
        if self.hparams.eval_init and self._pending is None:
            self._evaluate(epoch=None)

    def _log_direct(self, d: Dict[str, float]) -> None:
        """Write to the loggers outside Lightning's hook rules (used at step 0)."""
        if d and self.trainer.is_global_zero:
            for lg in self.loggers:
                lg.log_metrics(d, step=self.global_step)

    # ------------------------------------------------------------------------------ helpers
    def _lr(self, t: int) -> float:
        """open_clip's cosine_lr, BioCLIP 1's schedule (Amendment 1 S5)."""
        base = self.hparams.lr
        if t < self.warmup:
            return base * (t + 1) / self.warmup
        return 0.5 * (1 + math.cos(math.pi * (t - self.warmup) / (self.total_steps - self.warmup))) * base

    def _is_monitor_step(self, t: int) -> bool:
        lo, hi = self.hparams.monitor_window
        return lo <= t <= hi or (t > hi and t % self.hparams.monitor_every == 0)

    def _flat_grads(self, loss, params, retain: bool) -> torch.Tensor:
        gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, params)])

    def _mean_over_ranks(self, g: torch.Tensor) -> torch.Tensor:
        return all_reduce_sum(g) / self.W

    def _gather_rows_with_grad(self, x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        """All-gather rows whose count differs by rank, keeping gradients (zero-padded to the max)."""
        if self.W == 1:
            return x
        m = max(sizes)
        pad = torch.cat([x, x.new_zeros(m - x.shape[0], *x.shape[1:])]) if x.shape[0] < m else x
        full = gather_with_grad(pad)
        return torch.cat([full[r * m: r * m + sizes[r]] for r in range(self.W)])

    def _contrastive(self, v32, v_all, sp_all, off, n_loc, levels, grad: bool) -> Dict[int, Dict]:
        dev, W, R = self.device, self.W, self.R
        plans, strings = [], []
        for d in levels:
            ids = sp_all if d == 6 else self.t_anc[sp_all, d]
            uniq, inv, shares = level_label_plan(ids, W)
            a, b = shares[R]
            strings += [class_text(self.tree.node_keys[d][i].split("|"), "photo") for i in uniq[a:b].tolist()]
            plans.append((d, uniq, inv, shares))
        with torch.set_grad_enabled(grad):
            if strings:
                z = self.model.encode_text(self.tokenizer(strings).to(dev))
            else:
                z = torch.zeros(0, v32.shape[1], device=dev)
            with _no_autocast():
                z = F.normalize(z.float(), dim=-1)
                sizes = [sum(sh[r][1] - sh[r][0] for _, _, _, sh in plans) for r in range(W)]
                z_flat = self._gather_rows_with_grad(z, sizes)
                starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
                out, within = {}, [0] * W
                scale = self.model.logit_scale.exp()
                for d, uniq, inv, shares in plans:
                    pieces = []
                    for r in range(W):
                        k = shares[r][1] - shares[r][0]
                        pieces.append(z_flat[starts[r] + within[r]: starts[r] + within[r] + k])
                        within[r] += k
                    z_all = torch.cat(pieces)
                    out[d] = bfl_level_loss_local(v32, v_all, z_all, inv[off: off + n_loc], inv, shares[R], scale)
        return out

    # ------------------------------------------------------------------------------ training
    def bind(self, tree, bank, device) -> None:
        """Attach the tree and bank outside a Trainer (smoke tests); on_fit_start does this in training."""
        self.tree, self.bank, (self.W, self.R) = tree, bank, world_info()
        self.t_img_species = torch.as_tensor(tree.img_species, dtype=torch.long, device=device)
        self.t_anc = torch.as_tensor(tree.anc, dtype=torch.long, device=device)
        self.t_P = torch.as_tensor(tree.P_mask, device=device)
        self.t_held = torch.as_tensor(tree.heldout, device=device)
        self.t_g = torch.as_tensor(tree.g_s, dtype=torch.long, device=device)
        self.trainable = trainable_parameters(self.model)
        self.vis_mask = torch.cat([torch.full((p.numel(),), n.startswith("visual."), device=device)
                                   for n, p in self.trainable])

    def forward_losses(self, batch, pen_grad: bool) -> Dict:
        """One forward on this rank's micro-batch: contrastive parts, group means and the penalty."""
        dev, R = self.device, self.R
        img, gsp, gsz = batch["img"], batch["grp_species"], batch["grp_sizes"]
        n_loc = int(img.shape[0])
        feats, _ = self.model.encode_image(to_model_input(batch["u8"]))       # bf16 under Lightning's autocast
        with _no_autocast():
            v32 = F.normalize(feats.float(), dim=-1)
            sizes = gather_nograd(torch.tensor([n_loc], device=dev)).tolist()
            off = sum(sizes[:R])
            sp_all = gather_varlen_nograd(self.t_img_species[img])
            v_all = self._gather_rows_with_grad(v32, sizes)
        levels = list(range(7)) if self.loss_kind == "lvl" else [6]
        con = self._contrastive(v32, v_all, sp_all, off, n_loc, levels, grad=True)
        with _no_autocast():
            L_con = torch.stack([con[d]["i2t"] + con[d]["t2i"] for d in levels]).mean()
            with torch.set_grad_enabled(pen_grad):
                mu_hat = group_means(v32 if pen_grad else v32.detach(), gsz)
                pen = penalty_local(mu_hat, gsp, self.bank, self.t_P, self.t_held, self.t_g, n_loc,
                                    self.hparams.penalty_weighting)
        return {"L_con": L_con, "pen": pen, "con": con, "levels": levels, "mu_hat": mu_hat, "feats": feats,
                "v_all": v_all, "sizes": sizes, "sp_all": sp_all, "off": off, "n_loc": n_loc}

    def separate_grads(self, out: Dict) -> Dict[str, torch.Tensor]:
        """S16: each loss's gradient from its own backward pass on the same forward, averaged over ranks by hand."""
        params = [p for _, p in self.trainable]
        gB = self._mean_over_ranks(self._flat_grads(out["pen"]["L_B"], params, retain=True))
        gC = self._mean_over_ranks(self._flat_grads(self.W * out["L_con"], params, retain=False))
        return {"pen": gB, "con": gC}

    def training_step(self, batch, batch_idx):
        t, h = self.global_step, self.hparams
        t_start = time.perf_counter()
        monitor = self._is_monitor_step(t)
        full_log = monitor or t % h.monitor_every == 0
        out = self.forward_losses(batch, pen_grad=self.penalized or monitor)
        L_con, pen, con, levels = out["L_con"], out["pen"], out["con"], out["levels"]
        params = [p for _, p in self.trainable]
        lam = float(h.lam) if self.penalized else 0.0
        logs: Dict[str, float] = {}
        if monitor:
            g2 = self.separate_grads(out)
            gB, gC = g2["pen"], g2["con"]
            for tag, m in (("all", None), ("image_lora", self.vis_mask)):
                b, c = (gB, gC) if m is None else (gB[m], gC[m])
                nb, nc = float(b.norm()), float(c.norm())
                logs[f"grad/{tag}/norm_pen"], logs[f"grad/{tag}/norm_con"] = nb, nc
                logs[f"grad/{tag}/ratio"] = nb / nc if nc > 0 else float("nan")
                with torch.autocast(device_type=b.device.type, enabled=False):  # the S16 cosine in fp64, not bf16
                    logs[f"grad/{tag}/cos"] = (float(torch.dot(b.double(), c.double()) / (nb * nc))
                                               if nb > 0 and nc > 0 else float("nan"))
            self.r_t[t] = logs["grad/all/ratio"]
            if self.penalized:
                logs["grad/lam_times_ratio"] = lam * logs["grad/all/ratio"]
            g = gC + lam * gB if self.penalized else gC
        else:
            loss = self.W * L_con + (lam * pen["L_B"] if self.penalized else 0.0)
            g = self._mean_over_ranks(self._flat_grads(loss, params, retain=False))

        i = 0
        for p in params:
            p.grad = g[i: i + p.numel()].view_as(p).to(p.dtype).clone()
            i += p.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(params, h.grad_clip)
        opt = self.optimizers()
        lr = self._lr(t)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        clamp_logit_scale(self.model)

        with torch.no_grad():
            batch_g = batch["grp_species"]
            self.bank.ema_update(gather_varlen_nograd(batch_g), gather_varlen_nograd(out["mu_hat"].detach().float()),
                                 gather_varlen_nograd(batch["grp_sizes"]))
            stats = [L_con.detach()]
            for d in levels:
                stats += [con[d]["i2t"].detach(), con[d]["t2i"].detach(), con[d]["t2i_floor"].detach()]
            red = all_reduce_sum(torch.stack(stats).double())
            pen_red = all_reduce_sum(torch.cat([pen["L_B"].detach()[None], pen["parts"].detach(),
                                                pen["cos2_sum"], pen["cos2_n"]]).double())
        L_pen = float(pen_red[0]) / self.W
        logs.update({"loss/con": float(red[0]), "loss/pen": L_pen, "loss/total": float(red[0]) + lam * L_pen,
                     "lr": lr, "logit_scale": float(self.model.logit_scale), "grad/total_norm_preclip": float(gnorm),
                     "zero_sum/bank_updates": float(self.bank.n_updates)})
        for j, d in enumerate(levels):
            logs[f"con/{RANKS[d]}/i2t"], logs[f"con/{RANKS[d]}/t2i"], logs[f"con/{RANKS[d]}/t2i_floor"] = (
                float(red[1 + 3 * j]), float(red[2 + 3 * j]), float(red[3 + 3 * j]))
        for d in range(6):
            logs[f"pen/{RANKS[d]}/term"] = float(pen_red[1 + d]) / self.W
            n_valid = float(pen_red[13 + d])
            logs[f"pen/{RANKS[d]}/cos2_mean"] = float(pen_red[7 + d]) / n_valid if n_valid else float("nan")
        if full_log:
            logs.update(self._full_log(out["v_all"].detach(), out["feats"], out["sizes"], out["sp_all"], out["off"],
                                       out["n_loc"], levels))
        logs["time/step_s"] = time.perf_counter() - t_start
        self.log_dict(logs, on_step=True, on_epoch=False, rank_zero_only=True)
        if self.trainer.is_global_zero:  # exact float64 record; Lightning's logger stores float32
            with open(self.run_dir / "steps.jsonl", "a") as f:
                f.write(json.dumps({"step": t, "epoch": self.current_epoch,
                                    **{k: float(v).hex() for k, v in logs.items()}}) + "\n")

    @torch.no_grad()
    def _full_log(self, v_all, feats, sizes, sp_all, off, n_loc, levels) -> Dict[str, float]:
        out = {}
        if self.loss_kind == "flat":  # preflight reading S42: the six other levels, no gradient
            extra = self._contrastive(v_all[off: off + n_loc], v_all, sp_all, off, n_loc, list(range(6)), grad=False)
            red = all_reduce_sum(torch.stack([extra[d][k] for d in range(6) for k in ("i2t", "t2i", "t2i_floor")]).double())
            for d in range(6):
                for j, k in enumerate(("i2t", "t2i", "t2i_floor")):
                    out[f"con/{RANKS[d]}/{k}"] = float(red[3 * d + j])
        with _no_autocast():  # S41: global batch, normalized embeddings, eigenvalues in fp64
            x = v_all.double()
            xc = x - x.mean(0)
            ev = torch.linalg.eigvalsh(xc.T @ xc / (x.shape[0] - 1))
            out["collapse/participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
            s = x.sum(0)
            out["collapse/mean_pairwise_cos"] = float((s @ s - x.shape[0]) / (x.shape[0] * (x.shape[0] - 1)))
            norms = self._gather_rows_with_grad(feats.detach().float().norm(dim=1, keepdim=True), sizes)
            out["collapse/mean_norm"] = float(norms.mean())
        if self.hparams.check_zero_sum:
            out.update(self.bank.assert_zero_sum())
        out.update(self.bank.monitor())
        out["mem/max_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        chk = torch.stack([self.bank.m_s.sum(), self.bank.root.sum(),
                           torch.cat([p.detach().reshape(-1).double() for _, p in self.trainable]).sum()])
        allc = gather_nograd(chk[None])
        out["sync/max_rank_spread"] = float((allc.max(0).values - allc.min(0).values).abs().max())
        return out

    # ------------------------------------------------------------------------------ epoch end
    def on_train_epoch_end(self):
        epoch = self.current_epoch  # 0-based, like Lightning and the checkpoint file names (Amendment 2 A2.3)
        if self.hparams.refresh:
            t0 = time.time()
            m_s = self._refresh_species_means()
            fresh = Bank(self.tree, self.bank.dim, self.device)
            fresh.set_species_means(m_s)
            logs = self.bank.drift(fresh)
            self.bank.copy_from(fresh)
            logs.update(self.bank.canonical_penalty(self.t_P, self.t_held))
            logs.update({f"refreshed/{k}": v for k, v in self.bank.monitor().items()})
            logs["time/refresh_s"] = time.time() - t0
            self.log_dict(logs, on_step=False, on_epoch=True)
        # Every rank logs identical epoch-end values: ModelCheckpoint's save ends in a barrier, so all
        # ranks must find its monitor.
        if self.hparams.eval_each_epoch:
            self._evaluate(epoch)

    @torch.no_grad()
    def _refresh_species_means(self) -> torch.Tensor:
        loader, a, b, _, _ = self.trainer.datamodule.refresh_loader(self.R, self.W)
        sums = species_means_pass(self.model, loader, self.t_img_species[a:b], self.tree.n_species_total,
                                  self.device, torch.float16)
        all_reduce_sum(sums)
        return sums / self.bank.N_s[:, None]

    def _evaluate(self, epoch: Optional[int]) -> None:
        """ToL-val after training epoch `epoch` (0-based), or of the untouched model when `epoch` is None."""
        t0 = time.time()
        self.model.set_grad_checkpointing(False)
        res = tolval_eval(self.model, self.tokenizer, self.trainer.datamodule, self.device, self.R, self.W,
                          self.hparams.eval_forms)
        self.model.set_grad_checkpointing(True)
        prefix = "val" if epoch is not None else "val_init"
        logs = {}
        for form, per in res.items():
            for r, v in per.items():
                if r == "mean7":
                    logs[f"{prefix}/tol/{form}/mean7"] = v
                else:
                    logs[f"{prefix}/tol/{form}/{r}"] = v["top1"]
                    logs[f"{prefix}/tol/{form}/{r}_seen"] = v["seen_top1"]
        logs[f"time/{prefix}_eval_s"] = time.time() - t0
        if epoch is not None:
            self.log_dict(logs, on_step=False, on_epoch=True)
            res["checkpoint"] = f"epoch_{epoch:02d}.ckpt"
        else:
            self._log_direct(logs)
        self.decisions["epochs"]["init" if epoch is None else str(epoch)] = res
        lo, hi = self.hparams.monitor_window
        window = [v for s, v in self.r_t.items() if lo <= s <= hi]
        if len(window) == hi - lo + 1:
            mean_r = float(np.mean(window))
            self.decisions["lambda_calibration"] = {"steps": [lo, hi], "mean_r": mean_r, "lam_0.2": 0.2 / mean_r}
        self.decisions["r_t"] = {str(k): v for k, v in sorted(self.r_t.items())}
        if self.trainer.is_global_zero:
            (self.run_dir / "decisions.json").write_text(json.dumps(self.decisions, indent=1))

    # ------------------------------------------------------------------------------ checkpoints
    def on_save_checkpoint(self, checkpoint):
        dm = self.trainer.datamodule
        checkpoint["p1"] = {"bank": self.bank.state_dict(), "r_t": self.r_t, "decisions": self.decisions,
                            "lam": float(self.hparams.lam),
                            "heldout_keys": [self.tree.node_keys[6][i] for i in np.flatnonzero(self.tree.heldout)],
                            "dedup_drop": sorted(dm.dedup_uuids) if getattr(dm, "dedup_uuids", None) else None}

    def on_load_checkpoint(self, checkpoint):
        self._pending = checkpoint["p1"]
