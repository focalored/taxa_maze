"""Pilot 1 data module: training batches from the epoch plan, plus the bank-refresh and ToL-val loaders, all reading uint8 crops from the store.

Usage: `dm = P1TolDataModule(data_dir); dm.setup(); loader = dm.train_dataloader()`.
"""
import os
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.taxonomy import build_train_tree, build_val_set, save_stats
from src.data.tol_sampler import make_epoch_plan, make_random_epoch_plan
from src.data.tol_store import ImageStore
from src.utils.paths import assert_outside_projects


def _read_sorted(store: ImageStore, shards: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Read crops in storage order (better locality), return them in the requested order."""
    order = np.lexsort((rows, shards))
    out = np.empty((len(rows), 224, 224, 3), dtype=np.uint8)
    out[order] = store.read_many(shards[order], rows[order])
    return out


class PlanDataset(Dataset):
    """Item t = this rank's micro-batch at step t of the current epoch's plan."""

    def __init__(self, tree, store: ImageStore, K: int, B: int, world: int, rank: int, seed: int,
                 sampler: str = "grouped"):
        self.tree, self.store, self.K, self.B, self.sampler = tree, store, K, B, sampler
        self.world, self.rank, self.seed = world, rank, seed
        T = int(tree.N_s.sum())
        self.n_steps = T // B + (T % B > 0)
        self.plan = None
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        if self.epoch != epoch:
            if self.sampler == "random":  # diagnostic only (ask A14); the spec's batches are the grouped plan
                self.plan = make_random_epoch_plan(self.tree.sp_start, self.tree.N_s, self.B, self.world, self.seed, epoch)
            else:
                self.plan = make_epoch_plan(self.tree.sp_start, self.tree.N_s, self.K, self.B,
                                            self.world, self.seed, epoch)
            self.epoch = epoch

    def __len__(self):
        return self.n_steps

    def __getitem__(self, t: int):
        idx, gsp, gsz = self.plan.steps[t][self.rank]
        u8 = _read_sorted(self.store, self.tree.img_shard[idx], self.tree.img_row[idx])
        return {"u8": torch.from_numpy(u8), "img": torch.from_numpy(idx), "grp_species": torch.from_numpy(gsp),
                "grp_sizes": torch.from_numpy(gsz), "step": t, "epoch": self.epoch}


class PlanSampler(Sampler):
    """Yields step indices in order; Lightning calls set_epoch before each epoch's workers start."""

    def __init__(self, ds: PlanDataset):
        self.ds = ds

    def set_epoch(self, epoch: int) -> None:
        self.ds.set_epoch(epoch)

    def __iter__(self):
        if self.ds.plan is None:
            self.ds.set_epoch(0)
        return iter(range(len(self.ds)))

    def __len__(self):
        return len(self.ds)


class ChunkDataset(Dataset):
    """Fixed chunks of (shard, row) pairs, for the refresh and the evaluation passes."""

    def __init__(self, store: ImageStore, shards: np.ndarray, rows: np.ndarray, chunk: int):
        self.store, self.shards, self.rows, self.chunk = store, shards, rows, chunk

    def __len__(self):
        return (len(self.rows) + self.chunk - 1) // self.chunk

    def __getitem__(self, i: int):
        a, b = i * self.chunk, min((i + 1) * self.chunk, len(self.rows))
        u8 = _read_sorted(self.store, self.shards[a:b], self.rows[a:b])
        return {"u8": torch.from_numpy(u8), "start": a, "stop": b}


class P1TolDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, s9_drop: bool = True, K: int = 16, B: int = 8192, seed: int = 42,
                 num_workers: int = 8, chunk: int = 1024, dedup_file: Optional[str] = None,
                 heldout_file: Optional[str] = None, sampler: str = "grouped"):
        super().__init__()
        if sampler not in ("grouped", "random"):
            raise ValueError(f"sampler must be 'grouped' (the spec's batches) or 'random' (diagnostic), got {sampler!r}")
        self.data_dir = Path(data_dir)
        self.store_dir = self.data_dir / "store"
        self.s9_drop, self.K, self.B, self.seed, self.sampler = s9_drop, K, B, seed, sampler
        self.num_workers, self.chunk = num_workers, chunk
        self.dedup_file = Path(dedup_file) if dedup_file else None
        self.heldout_file = Path(heldout_file) if heldout_file else None
        self.target_world_size = None
        self.tree = None
        self.dedup_uuids = None

    # rank and world come from the launcher's environment (srun / Lightning)
    @staticmethod
    def _rank_world():
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))), int(os.environ.get("WORLD_SIZE", 1))

    def setup(self, stage=None):
        if self.tree is not None:
            return
        if not self.s9_drop:
            raise NotImplementedError("the held-out list, dedup list and bank seed under p1/ are built with the S9 "
                                      "flag on; rebuild them with the flag off (scripts/p1/prepare.py) before such a run")
        self.tree = build_train_tree(self.store_dir, self.s9_drop, self.K, self.heldout_file)
        drop = None
        if self.dedup_file is not None:
            if not self.dedup_file.exists():
                raise FileNotFoundError(f"ToL-val dedup list {self.dedup_file} (Amendment 1 S26) is missing")
            drop = set(self.dedup_file.read_text().split())
        self.dedup_uuids = drop
        self.val = build_val_set(self.store_dir, self.tree, drop)
        self.store = ImageStore(self.store_dir)

    def train_dataloader(self):
        rank, world = self._rank_world()
        if self.target_world_size is not None and world != self.target_world_size and world != 1:
            raise RuntimeError(f"world {world} != planned {self.target_world_size}")
        ds = PlanDataset(self.tree, self.store, self.K, self.B, world, rank, self.seed, self.sampler)
        return DataLoader(ds, batch_size=None, sampler=PlanSampler(ds), num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=False, prefetch_factor=2 if self.num_workers else None)

    def refresh_loader(self, rank: int, world: int):
        """This rank's contiguous species range (balanced by images), species-sorted (S25)."""
        cum = np.cumsum(self.tree.N_s)
        T = int(cum[-1])
        cuts = [0] + [int(np.searchsorted(cum, T * r // world, side="left")) + 1 for r in range(1, world)] + [len(cum)]
        s_lo, s_hi = cuts[rank], cuts[rank + 1]
        a = int(self.tree.sp_start[s_lo]) if s_lo < len(cum) else T
        b = int(self.tree.sp_start[s_hi]) if s_hi < len(cum) else T
        ds = ChunkDataset(self.store, self.tree.img_shard[a:b], self.tree.img_row[a:b], self.chunk)
        loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return loader, a, b, s_lo, s_hi

    def val_loader(self, rank: int, world: int):
        n = len(self.val.row)
        a, b = (n * rank) // world, (n * (rank + 1)) // world
        ds = ChunkDataset(self.store, self.val.shard[a:b], self.val.row[a:b], self.chunk)
        return DataLoader(ds, batch_size=None, shuffle=False, num_workers=self.num_workers, pin_memory=True), a, b

    def write_stats(self, out_dir) -> None:
        out = assert_outside_projects(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_stats(out / "data_stats.json", {"train": self.tree.stats, "val": self.val.stats})
