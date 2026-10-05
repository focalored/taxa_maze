"""Reader for the uint8 image store; `to_model_input` finishes ToTensor and Normalize bit-identically to `preprocess_val`.

Usage: `store = ImageStore(store_dir); x = to_model_input(torch.from_numpy(store.read_many(shards, rows)).cuda())`.
"""
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
    """Random access to stored crops by (shard, row). Memmaps open lazily, once per process."""

    def __init__(self, store_dir):
        self.dir = Path(store_dir)
        self._mm: Dict[int, np.memmap] = {}

    def _shard(self, shard: int) -> np.memmap:
        mm = self._mm.get(shard)
        if mm is None:
            path = self.dir / "shards" / f"image_set_{shard:02d}.u8"
            n = path.stat().st_size // ROW_BYTES
            mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, CROP, CROP, 3))
            self._mm[shard] = mm
        return mm

    def read(self, shard: int, row: int) -> np.ndarray:
        return np.asarray(self._shard(int(shard))[int(row)])

    def read_many(self, shards: np.ndarray, rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), CROP, CROP, 3), dtype=np.uint8)
        for i, (s, r) in enumerate(zip(shards, rows)):
            out[i] = self._shard(int(s))[int(r)]
        return out

    def __getstate__(self):  # DataLoader workers reopen their own memmaps
        return {"dir": self.dir, "_mm": {}}
