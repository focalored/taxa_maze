"""fp64 ToL bank kept outside the nn.Module: deterministic EMA updates, species-uniform ancestors, zero-sum check, drift and monitors (spec section 4; Amendment 1 S12, S24, S25).

Usage: `bank = Bank(tree, device=dev); bank.set_species_means(m_s); bank.ema_update(sids, mu_hat, k_s)`.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.tol_catalog import RANKS
from src.data.tol_store import to_model_input
from src.models.losses import DEN_FLOOR

ZERO_SUM_TOL = 1e-10


class Bank:
    def __init__(self, tree, dim: int = 512, device="cpu"):
        self.tree = tree
        self.dim = dim
        self.device = torch.device(device)
        dev = self.device
        self.S = tree.n_species_total
        self.anc = torch.as_tensor(tree.anc, dtype=torch.long, device=dev)                 # (S, 6)
        self.n_a = [torch.as_tensor(tree.n_species[d], dtype=torch.float64, device=dev) for d in range(6)]
        self.parent = [None] + [torch.as_tensor(tree.parent[d], dtype=torch.long, device=dev) for d in range(1, 7)]
        self.n_children = [torch.as_tensor(tree.n_children[d], dtype=torch.long, device=dev) for d in range(6)]
        self.N_s = torch.as_tensor(tree.N_s, dtype=torch.float64, device=dev)
        self.m_s = torch.zeros(self.S, dim, dtype=torch.float64, device=dev)
        self.m_nodes = [torch.zeros(len(tree.node_keys[d]), dim, dtype=torch.float64, device=dev) for d in range(6)]
        self.root = torch.zeros(dim, dtype=torch.float64, device=dev)
        self.n_updates = 0
        self.zero_sum_max_seen = 0.0
        # segment starts of each depth over the species order (contiguous by construction)
        self._seg_starts = [np.flatnonzero(np.r_[True, np.diff(tree.anc[:, d]) != 0]) for d in range(6)]

    def set_species_means(self, m_s: torch.Tensor) -> None:
        """Seed or refresh: species entries given; ancestors and root as species-uniform means."""
        if m_s.shape != (self.S, self.dim) or m_s.dtype != torch.float64:
            raise ValueError(f"species means must be ({self.S}, {self.dim}) float64")
        ms_np = m_s.detach().cpu().numpy()
        for d in range(6):
            sums = np.add.reduceat(ms_np, self._seg_starts[d], axis=0)
            if sums.shape[0] != self.m_nodes[d].shape[0]:
                raise RuntimeError(f"depth {d}: {sums.shape[0]} segments for {self.m_nodes[d].shape[0]} nodes")
            self.m_nodes[d] = torch.from_numpy(sums / self.tree.n_species[d][:, None]).to(self.device)
        self.root = torch.from_numpy(ms_np.sum(axis=0) / self.S).to(self.device)
        self.m_s = m_s.detach().to(self.device).clone()

    @torch.no_grad()
    def ema_update(self, sids: torch.Tensor, mu_hat: torch.Tensor, k_s: torch.Tensor) -> None:
        """One step's deltas from every rank, applied after the optimizer step (preflight reading S34)."""
        order = torch.argsort(sids)
        sids, mu_hat, k_s = sids[order], mu_hat[order], k_s[order]
        if len(sids) > 1 and not bool((sids[1:] > sids[:-1]).all()):
            raise RuntimeError("a species appears twice in one step's bank update (S3 is violated)")
        n = len(sids)
        beta = k_s.to(torch.float64) / self.N_s[sids]
        delta = beta[:, None] * (mu_hat.to(torch.float64) - self.m_s[sids])
        self.m_s[sids] += delta
        cols = torch.arange(n, device=self.device)
        for d in range(6):
            a = self.anc[sids, d]
            uniq, inv = torch.unique(a, sorted=True, return_inverse=True)
            M = torch.zeros(len(uniq), n, dtype=torch.float64, device=self.device)
            M[inv, cols] = 1.0 / self.n_a[d][a]
            self.m_nodes[d][uniq] += M @ delta
        w = torch.full((1, n), 1.0 / self.S, dtype=torch.float64, device=self.device)
        self.root += (w @ delta)[0]
        self.n_updates += 1

    @torch.no_grad()
    def zero_sum_check(self, species: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """S12 on the whole bank (or on the nodes above `species`): returns the two maxima."""
        dev = self.device
        gap_mean, gap_child = 0.0, 0.0
        for d in range(6):
            nn_ = self.m_nodes[d].shape[0]
            fresh = torch.zeros(nn_, self.dim, dtype=torch.float64, device=dev)
            fresh.index_add_(0, self.anc[:, d], self.m_s)
            fresh /= self.n_a[d][:, None]
            rows = slice(None) if species is None else torch.unique(self.anc[species, d])
            gap_mean = max(gap_mean, float((self.m_nodes[d][rows] - fresh[rows]).abs().max()))
            # children of depth-d nodes: depth d+1 nodes (or species when d == 5)
            if d < 5:
                child_m, child_n, par = self.m_nodes[d + 1], self.n_a[d + 1], self.parent[d + 1]
            else:
                child_m, child_n, par = self.m_s, torch.ones(self.S, dtype=torch.float64, device=dev), self.parent[6]
            acc = torch.zeros(nn_, self.dim, dtype=torch.float64, device=dev)
            acc.index_add_(0, par, child_n[:, None] * (child_m - self.m_nodes[d][par]))
            gap_child = max(gap_child, float((acc[rows].abs().max(dim=1).values / self.n_a[d][rows]).max()))
        root_gap = float((self.root - self.m_s.mean(dim=0)).abs().max())
        out = {"zero_sum/max_mean_gap": max(gap_mean, root_gap), "zero_sum/max_child_sum_over_n": gap_child}
        self.zero_sum_max_seen = max(self.zero_sum_max_seen, out["zero_sum/max_mean_gap"], gap_child)
        out["zero_sum/max_seen"] = self.zero_sum_max_seen
        return out

    def assert_zero_sum(self, species: Optional[torch.Tensor] = None) -> Dict[str, float]:
        out = self.zero_sum_check(species)
        if out["zero_sum/max_mean_gap"] > ZERO_SUM_TOL or out["zero_sum/max_child_sum_over_n"] > ZERO_SUM_TOL:
            raise AssertionError(f"bank zero-sum check failed (tolerance {ZERO_SUM_TOL}): {out}")
        return out

    @torch.no_grad()
    def monitor(self) -> Dict[str, float]:
        """S24: per edge rank (p = 6, genus->species, is the S39 extra), two drop rules x two averages."""
        out = {}
        for p in range(1, 7):
            pd = p - 1
            par = self.parent[p]
            child_m = self.m_nodes[p] if p < 6 else self.m_s
            m_a = self.m_nodes[pd][par]
            step, pos = child_m - m_a, m_a - self.root
            cos2 = (step * pos).sum(1) ** 2 / ((step * step).sum(1) * (pos * pos).sum(1))
            w_sp = (self.n_a[p] if p < 6 else torch.ones(self.S, dtype=torch.float64, device=self.device))
            parent_ok = self.n_children[pd][par] > 1
            child_ok = (self.n_children[p] != 1) if p < 6 else torch.ones_like(parent_ok)
            name = f"{RANKS[pd]}->{RANKS[p]}"
            for rule, keep in (("parent", parent_ok), ("literal", parent_ok & child_ok)):
                keep = keep & torch.isfinite(cos2)
                out[f"bank_monitor/{rule}_edges/{name}"] = float(cos2[keep].mean()) if keep.any() else float("nan")
                out[f"bank_monitor/{rule}_species/{name}"] = (
                    float((cos2[keep] * w_sp[keep]).sum() / w_sp[keep].sum()) if keep.any() else float("nan"))
        return out

    @torch.no_grad()
    def canonical_penalty(self, P_mask: torch.Tensor, heldout: torch.Tensor) -> Dict[str, float]:
        """L* (line 100) on this bank, with mu_s = m_s (preflight readings S43, S23)."""
        out = {}
        terms = torch.zeros(self.S, 6, dtype=torch.float64, device=self.device)
        for d in range(6):
            m_a = self.m_nodes[d][self.anc[:, d]]
            step, pos = self.m_s - m_a, m_a - self.root
            step2, pos2 = (step * step).sum(1).clamp_min(DEN_FLOOR), (pos * pos).sum(1)  # line 112's δ, as in penalty_local
            valid = P_mask[:, d] & (pos2 > 0)
            c2 = (step * pos).sum(1) ** 2 / (step2 * torch.where(valid, pos2, torch.ones_like(pos2)))
            terms[:, d] = torch.where(valid, c2, torch.zeros_like(c2))
        for name, sel in (("penalized", ~heldout), ("heldout", heldout)):
            out[f"L_star/{name}"] = float(terms[sel].sum(1).mean())
            for d in range(6):
                v = P_mask[sel, d]
                out[f"L_star/{name}/cos2_{RANKS[d]}"] = float(terms[sel, d][v].mean()) if v.any() else float("nan")
        return out

    @torch.no_grad()
    def drift(self, fresh: "Bank") -> Dict[str, float]:
        """S40: EMA bank vs refreshed bank, root-centered, before the fresh bank replaces it."""
        out = {}

        def stats(a, b, ra, rb, tag):
            ca, cb = a - ra, b - rb
            dist = (a - b).norm(dim=1)
            one_minus_cos = 1 - (ca * cb).sum(1) / (ca.norm(dim=1) * cb.norm(dim=1))
            for nm, v in (("dist", dist), ("1-cos", one_minus_cos)):
                out[f"drift/{tag}/{nm}_mean"] = float(v.mean())
                out[f"drift/{tag}/{nm}_p95"] = float(torch.quantile(v.float(), 0.95)) if v.numel() > 1 else float(v.mean())

        stats(self.m_s, fresh.m_s, self.root, fresh.root, "species")
        for d in range(6):
            stats(self.m_nodes[d], fresh.m_nodes[d], self.root, fresh.root, RANKS[d])
        return out

    def state_dict(self) -> dict:
        return {"m_s": self.m_s.cpu(), "m_nodes": [m.cpu() for m in self.m_nodes], "root": self.root.cpu(),
                "n_updates": self.n_updates, "zero_sum_max_seen": self.zero_sum_max_seen,
                "species_keys_sha256": self.tree.species_keys_sha256(), "n_species": self.S}

    def load_state_dict(self, sd: dict) -> None:
        if sd["species_keys_sha256"] != self.tree.species_keys_sha256():
            raise RuntimeError("bank state belongs to a different species list")
        self.m_s = sd["m_s"].to(self.device)
        self.m_nodes = [m.to(self.device) for m in sd["m_nodes"]]
        self.root = sd["root"].to(self.device)
        self.n_updates = sd["n_updates"]
        self.zero_sum_max_seen = sd["zero_sum_max_seen"]

    def copy_from(self, other: "Bank") -> None:
        self.m_s = other.m_s.clone()
        self.m_nodes = [m.clone() for m in other.m_nodes]
        self.root = other.root.clone()


@torch.no_grad()
def species_means_pass(model, loader, img_species: torch.Tensor, S: int, device, amp_dtype) -> torch.Tensor:
    """(S, D) fp64 per-species sums of L2-normalized embeddings over a species-sorted store loader (S25)."""
    was_training, was_ckpt = model.training, model.visual.transformer.grad_checkpointing
    model.eval()
    model.set_grad_checkpointing(False)
    sums = torch.zeros(S, model.visual.output_dim, dtype=torch.float64, device=device)
    for batch in loader:
        x = to_model_input(batch["u8"].to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            f, _ = model.encode_image(x)
        v = F.normalize(f.float(), dim=-1).double()  # fp32 before normalizing, fp64 before averaging
        sp = img_species[batch["start"]: batch["stop"]]
        uniq, inv = torch.unique_consecutive(sp, return_inverse=True)
        M = torch.zeros(len(uniq), len(sp), dtype=torch.float64, device=device)
        M[inv, torch.arange(len(sp), device=device)] = 1.0
        sums[uniq] += M @ v
    model.set_grad_checkpointing(was_ckpt)
    model.train(was_training)
    return sums
