"""Epoch batch plan: species groups of at most K images packed into batches of exactly B, at most one group per species per batch (preflight reading S7; Amendment 1 S3); plus a random-batch variant for diagnostics.

Usage: `plan = make_epoch_plan(tree.sp_start, tree.N_s, K=16, B=8192, world=4, seed=42, epoch=e)`; `plan.steps[t][rank]` is (image_idx, group_species, group_sizes). `make_random_epoch_plan(tree.sp_start, tree.N_s, B=8192, world=2, seed=42, epoch=e)` has the same output format.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EpochPlan:
    epoch: int
    world: int
    # steps[t][r] = (image_idx, group_species, group_sizes) for rank r at step t; image_idx
    # indexes the tree's species-sorted image arrays and lists each group's images contiguously.
    steps: List[List[tuple]]

    def __len__(self):
        return len(self.steps)


def make_epoch_plan(sp_start: np.ndarray, N_s: np.ndarray, K: int, B: int, world: int,
                    seed: int, epoch: int) -> EpochPlan:
    if B % world:
        raise ValueError(f"B={B} is not divisible by world={world}")
    rng = np.random.default_rng([seed, epoch])
    S, T = len(N_s), int(N_s.sum())
    g = -(-N_s // K)

    # random order within each species: images are contiguous per species in the tree's arrays
    img_species = np.repeat(np.arange(S), N_s)
    perm = np.lexsort((rng.random(T), img_species))  # species-major, random within species

    # balanced contiguous groups over each species' permuted images
    G = int(g.sum())
    grp_species = np.repeat(np.arange(S), g)
    k_in_sp = np.arange(G) - np.repeat(np.cumsum(g) - g, g)
    base, extra = N_s // g, N_s % g
    grp_size = base[grp_species] + (k_in_sp < extra[grp_species])
    grp_off = np.cumsum(grp_size) - grp_size  # offset into perm

    n_full, rem = T // B, T % B
    n_batches = n_full + (rem > 0)
    per = B // world
    free = np.full((n_batches, world), per, dtype=np.int64)
    if rem:
        free[n_full] = rem // world + (np.arange(world) < rem % world)

    # Dealt round-robin in descending size; groups of multi-group species hold >= 9 images, so the
    # conflict-free 1-image groups dealt last fill every bin exactly.
    order = np.lexsort((rng.random(G), -grp_size))
    multi = g > 1
    taken = [set() for _ in range(n_batches)]       # multi-group species already in each batch
    bins = [[[] for _ in range(world)] for _ in range(n_batches)]
    active = list(rng.permutation(n_batches))
    p = 0
    for gi in order:
        sz, s = int(grp_size[gi]), int(grp_species[gi])
        tried = 0
        while True:
            if not active:
                raise RuntimeError("batch plan: ran out of room before every group was placed")
            p %= len(active)
            b = active[p]
            r = int(np.argmax(free[b]))
            if free[b, r] >= sz and not (multi[s] and s in taken[b]):
                break
            p += 1
            tried += 1
            if tried > len(active):
                raise RuntimeError(f"batch plan: group of {sz} (species {s}) fits no batch")
        bins[b][r].append(gi)
        free[b, r] -= sz
        if multi[s]:
            taken[b].add(s)
        if free[b].sum() == 0:
            active.pop(p)
        else:
            p += 1
    if free.sum() != 0:
        raise RuntimeError(f"batch plan: {int(free.sum())} slots left unfilled")

    batch_order = np.concatenate([rng.permutation(n_full), [n_full] if rem else []]).astype(int)
    steps = []
    for b in batch_order:
        ranks = []
        for r in range(world):
            gis = np.array(bins[b][r], dtype=np.int64)
            idx = np.concatenate([perm[grp_off[i]: grp_off[i] + grp_size[i]] for i in gis]) if len(gis) else np.zeros(0, np.int64)
            ranks.append((idx, grp_species[gis].astype(np.int64), grp_size[gis].astype(np.int64)))
        steps.append(ranks)
    plan = EpochPlan(epoch=epoch, world=world, steps=steps)
    check_plan(plan, N_s, B, sp_start)
    return plan


def make_random_epoch_plan(sp_start: np.ndarray, N_s: np.ndarray, B: int, world: int, seed: int, epoch: int) -> EpochPlan:
    """Diagnostic plan, not the spec's batches (ask A14, 2026-10-05): a uniformly random partition of the epoch's images into
    batches of exactly B; the images of one species that land in the same batch form its one group there, so S3 still holds
    and the losses see the usual group structure. Usage: `make_random_epoch_plan(tree.sp_start, tree.N_s, 8192, 2, 42, e)`.
    """
    if B % world:
        raise ValueError(f"B={B} is not divisible by world={world}")
    rng = np.random.default_rng([seed, epoch])
    S, T = len(N_s), int(N_s.sum())
    img_species = np.repeat(np.arange(S), N_s)
    perm = rng.permutation(T)
    n_full, rem = T // B, T % B
    steps = []
    for b in range(n_full + (rem > 0)):
        idx = np.sort(perm[b * B: (b + 1) * B])  # ascending index = species-major, so each species' images are contiguous
        gsp, first, gsz = np.unique(img_species[idx], return_index=True, return_counts=True)
        n = len(idx)
        free = n // world + (np.arange(world) < n % world)
        bins = [[] for _ in range(world)]
        multi, single = np.flatnonzero(gsz > 1), np.flatnonzero(gsz == 1)
        for gi in multi[np.lexsort((rng.random(len(multi)), -gsz[multi]))]:  # largest first, to the rank with most room
            r = int(np.argmax(free))
            if free[r] < gsz[gi]:
                raise RuntimeError(f"random plan: batch {b}, group of {int(gsz[gi])} fits no rank")
            bins[r].append(int(gi))
            free[r] -= gsz[gi]
        if len(single) != int(free.sum()):
            raise RuntimeError(f"random plan: batch {b}, {len(single)} single images for {int(free.sum())} slots")
        rng.shuffle(single)  # singletons fill the remaining slots exactly
        for r, chunk in enumerate(np.split(single, np.cumsum(free)[:-1])):
            bins[r].extend(chunk.tolist())
        ranks = []
        for r in range(world):
            gis = np.array(bins[r], dtype=np.int64)
            ridx = np.concatenate([idx[first[i]: first[i] + gsz[i]] for i in gis]) if len(gis) else np.zeros(0, np.int64)
            ranks.append((ridx, gsp[gis].astype(np.int64), gsz[gis].astype(np.int64)))
        steps.append(ranks)
    plan = EpochPlan(epoch=epoch, world=world, steps=steps)
    check_plan(plan, N_s, B, sp_start)
    return plan


def check_plan(plan: EpochPlan, N_s: np.ndarray, B: int, sp_start: np.ndarray) -> None:
    """Amendment 1 S3 and preflight reading S7, asserted every epoch."""
    T = int(N_s.sum())
    seen = np.zeros(T, dtype=np.int64)
    per = B // plan.world
    for t, ranks in enumerate(plan.steps):
        sp_all = np.concatenate([r[1] for r in ranks])
        if len(np.unique(sp_all)) != len(sp_all):
            raise AssertionError(f"step {t}: a species has more than one group in the batch")
        sizes = [int(r[2].sum()) for r in ranks]
        if t < len(plan.steps) - 1 or sum(sizes) == B:
            if sizes != [per] * plan.world:
                raise AssertionError(f"step {t}: rank sizes {sizes}, expected {per} each")
        for idx, gs, gz in ranks:
            if len(idx) != int(gz.sum()):
                raise AssertionError(f"step {t}: image count does not match group sizes")
            # each group's images belong to its species
            sp_of_img = np.repeat(gs, gz)
            lo, hi = sp_start[sp_of_img], sp_start[sp_of_img] + N_s[sp_of_img]
            if not np.all((idx >= lo) & (idx < hi)):
                raise AssertionError(f"step {t}: an image sits in another species' group")
            seen[idx] += 1
    if not np.all(seen == 1):
        raise AssertionError(f"plan uses {int((seen == 0).sum())} images zero times, {int((seen > 1).sum())} more than once")
