"""Reader for the uint8 image store; `to_model_input` finishes ToTensor and Normalize bit-identically to `preprocess_val`.

Usage: `store = ImageStore(store_dir); x = to_model_input(torch.from_numpy(store.read_many(shards, rows)).cuda())`.
"""
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

CROP = 224
ROW_BYTES = CROP * CROP * 3
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def to_model_input(u8: torch.Tensor) -> torch.Tensor:
    """(B, 224, 224, 3) uint8 -> (B, 3, 224, 224) float32, as ToTensor then Normalize."""
    x = u8.permute(0, 3, 1, 2).to(torch.float32).div(255)
    mean = torch.as_tensor(OPENAI_MEAN, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(OPENAI_STD, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return x.sub(mean).div(std)


class ImageStore:
    """Random access to stored crops by (shard, row) with os.pread, one descriptor per shard and process.

    No memory maps: with every loader worker mapping the whole ~850 GB NFS store, workers stalled
    in the kernel (audit/2026-10-05_lr3e-4_stall.md).
    """

    def __init__(self, store_dir):
        self.dir = Path(store_dir)
        self._fd: Dict[int, int] = {}

    def _open(self, shard: int) -> int:
        fd = self._fd.get(shard)
        if fd is None:
            fd = os.open(self.dir / "shards" / f"image_set_{shard:02d}.u8", os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self._fd[shard] = fd
        return fd

    def _read_into(self, shard: int, row: int, out: np.ndarray) -> None:
        buf, fd, off, got = memoryview(out).cast("B"), self._open(shard), row * ROW_BYTES, 0
        while got < ROW_BYTES:
            n = os.preadv(fd, [buf[got:]], off + got)
            if n == 0:
                raise EOFError(f"store shard {shard} has no row {row}")
            got += n
        os.posix_fadvise(fd, off, ROW_BYTES, os.POSIX_FADV_DONTNEED)  # read once per epoch: keep it out of the page cache

    def read(self, shard: int, row: int) -> np.ndarray:
        out = np.empty((CROP, CROP, 3), dtype=np.uint8)
        self._read_into(int(shard), int(row), out)
        return out

    def read_many(self, shards: np.ndarray, rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), CROP, CROP, 3), dtype=np.uint8)
        for i in np.lexsort((rows, shards)):  # file order; each row lands at its own index
            self._read_into(int(shards[i]), int(rows[i]), out[i])
        return out

    def __getstate__(self):  # spawned workers open their own descriptors; forked ones may share (pread has no file position)
        return {"dir": self.dir, "_fd": {}}
