"""BFL level-restricted and flat contrastive losses in distributed form, and the cross-level penalty (spec lines 95-120; preflight readings S17, S38; Amendment 1 S21, S22).

Usage: `bfl_level_loss_local(...)` per level on each rank (rank parts sum to the global loss); `penalty_local(mu_hat, sids, bank, ...)`.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEN_FLOOR = 1e-5


def world_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Concatenate equal-shaped tensors from every rank; gradients flow back to their owners."""
    W, _ = world_info()
    if W == 1:
        return x
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(x), dim=0)


def gather_nograd(x: torch.Tensor) -> torch.Tensor:
    W, _ = world_info()
    if W == 1:
        return x
    out = [torch.empty_like(x) for _ in range(W)]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def gather_varlen_nograd(x: torch.Tensor) -> torch.Tensor:
    """All-gather tensors whose first dimension differs across ranks, in rank order."""
    W, _ = world_info()
    if W == 1:
        return x
    n = torch.tensor([x.shape[0]], device=x.device)
    ns = gather_nograd(n).tolist()
    pad = torch.zeros((max(ns),) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    pad[: x.shape[0]] = x
    full = gather_nograd(pad).view(W, max(ns), *x.shape[1:])
    return torch.cat([full[r, : ns[r]] for r in range(W)], dim=0)


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    W, _ = world_info()
    if W > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def level_label_plan(ids_all: torch.Tensor, world: int) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """(sorted unique ids, label index per image, each rank's (start, stop) share) for one level."""
    uniq, inv = torch.unique(ids_all, sorted=True, return_inverse=True)
    K = len(uniq)
    bounds = [(K * r) // world for r in range(world + 1)]
    return uniq, inv, [(bounds[r], bounds[r + 1]) for r in range(world)]


def bfl_level_loss_local(v_local: torch.Tensor, v_all: torch.Tensor, z_all: torch.Tensor,
                         inv_local: torch.Tensor, inv_all: torch.Tensor, share: Tuple[int, int],
                         scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """This rank's I->T rows and T->I label share of one level's BFL loss (Eqs. 1-3); rank parts sum to it."""
    N, K = v_all.shape[0], z_all.shape[0]
    logits_i2t = scale * v_local @ z_all.T                                  # (n, K)
    i2t = F.cross_entropy(logits_i2t, inv_local, reduction="sum") / N
    a, b = share
    if b > a:
        logits_t2i = scale * z_all[a:b] @ v_all.T                           # (k_r, N)
        logp = F.log_softmax(logits_t2i, dim=1)
        pos = (inv_all[None, :] == torch.arange(a, b, device=inv_all.device)[:, None]).to(logp.dtype)
        n_pos = pos.sum(1)
        t2i = (-(pos * logp).sum(1) / n_pos).sum() / K
        floor = torch.log(n_pos).sum() / K
    else:
        t2i = logits_i2t.new_zeros(())
        floor = logits_i2t.new_zeros(())
    return {"i2t": i2t, "t2i": t2i, "t2i_floor": floor}


def level_weights(resid: torch.Tensor, scheme: str, tau: float = 1.0, ref: Optional[torch.Tensor] = None,
                  grad_norms: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-level weights with mean 1 for the level-restricted loss (A15 second-phase probes; line 96 is the uniform mean).
    resid: the global per-level loss above its floor, detached. lse: n * softmax(resid / tau), TaxaWalk's tilted mean, whose
    gradient weights the level with the larger loss more. ratio: resid / ref, each level's loss relative to its own start,
    normalized; ones until ref exists. gradnorm: proportional to 1 / grad_norms, so every level's gradient on the image
    embeddings has the same norm. Usage: `w = level_weights(resid, "lse", tau=1.0)`.
    """
    n = resid.shape[0]
    if scheme == "lse":
        return torch.softmax(resid / tau, 0) * n
    if scheme == "ratio":
        if ref is None:
            return torch.ones_like(resid)
        r = resid / ref
        return r / r.mean()
    if scheme == "gradnorm":
        inv = 1.0 / grad_norms.clamp_min(1e-12)
        return inv / inv.mean()
    raise ValueError(f"unknown level weighting {scheme!r}")


def group_means(v32: torch.Tensor, grp_sizes: torch.Tensor) -> torch.Tensor:
    """muhat per group (groups contiguous in v32), as a one-hot fp32 matrix product (deterministic)."""
    n_g = len(grp_sizes)
    owner = torch.repeat_interleave(torch.arange(n_g, device=v32.device), grp_sizes)
    M = torch.zeros(n_g, v32.shape[0], dtype=v32.dtype, device=v32.device)
    M[owner, torch.arange(v32.shape[0], device=v32.device)] = 1.0
    return (M @ v32) / grp_sizes[:, None].to(v32.dtype)


def penalty_local(mu_hat: torch.Tensor, sids: torch.Tensor, bank, P_mask: torch.Tensor,
                  heldout: torch.Tensor, g_s: torch.Tensor, n_local_images: int,
                  weighting: str = "inv_groups") -> Dict[str, torch.Tensor]:
    """This rank's L_B (line 146 normalizes by the local batch size) and its per-rank parts."""
    if weighting not in ("inv_groups", "uniform"):
        raise ValueError(f"unknown penalty weighting {weighting!r}")
    mu = mu_hat.to(torch.float64)
    keep = ~heldout[sids]
    w = (1.0 / g_s[sids].to(torch.float64)) if weighting == "inv_groups" else torch.ones_like(mu[:, 0])
    w = torch.where(keep, w, torch.zeros_like(w))
    single = g_s[sids] == 1
    m_s = bank.m_s[sids]
    parts, cos2_sum, cos2_n = [], [], []
    for d in range(6):
        m_a = bank.m_nodes[d][bank.anc[sids, d]]
        pos = m_a - bank.root
        step = mu - m_a
        num = (step * pos).sum(1) ** 2
        den = torch.where(single, (step.detach() ** 2).sum(1), ((m_s - m_a) ** 2).sum(1)).clamp_min(DEN_FLOOR)
        pos2 = (pos * pos).sum(1)
        valid = P_mask[sids, d] & keep & (pos2 > 0)
        # safe denominators for skipped terms: torch.where backpropagates NaN from the unselected branch
        term = num / (den * torch.where(valid, pos2, torch.ones_like(pos2)))
        term = torch.where(valid, term, torch.zeros_like(term))
        parts.append((w * term).sum() / n_local_images)
        with torch.no_grad():
            step2 = (step * step).sum(1)
            ok = valid & (step2 > 0)
            c2 = num / (torch.where(ok, step2, torch.ones_like(step2)) * torch.where(ok, pos2, torch.ones_like(pos2)))
            cos2_sum.append(torch.where(ok, c2, torch.zeros_like(c2)).sum())
            cos2_n.append(ok.sum().to(torch.float64))
    L_B = torch.stack(parts).sum()
    return {"L_B": L_B, "parts": torch.stack(parts), "cos2_sum": torch.stack(cos2_sum),
            "cos2_n": torch.stack(cos2_n)}
