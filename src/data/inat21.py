"""iNat21-val images in the row order of tol_embed's bioclip1 cache, plus a cache's labels
(spec lines 54 and 190). The decode chain copies tol_embed scripts/encode_cache.py
INat21ImageDataset: Image.open, convert("RGB"), then the checkpoint's preprocess_val.

Usage:
    ds = INat21Val(preprocess_val)                                    # ds[i] -> (tensor, i)
    codes, vocab = load_inat21_labels(inat21_cache_dir("bioclip1"))
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.utils.paths import load_paths

N_RANKS = 7


def inat21_cache_dir(ckpt):
    """tol_embed's iNat21-val cache folder for ckpt (rule from data_paths.py, cache_root_for)."""
    results = Path(load_paths().tol_embed_dir) / "results"
    root = results / ("probe_cache" if ckpt == "bioclip2" else "probe_cache_b16")
    return root / "inat21-val" / ckpt


def load_inat21_labels(cache_dir):
    """Return (codes, vocab): codes.i32.npy as an int64 (N, 7) array and the parsed vocab.json."""
    cache_dir = Path(cache_dir)
    codes = np.load(cache_dir / "codes.i32.npy").astype(np.int64)
    if codes.ndim != 2 or codes.shape[1] != N_RANKS:
        raise ValueError(f"{cache_dir / 'codes.i32.npy'} has shape {codes.shape}; expected (N, {N_RANKS})")
    vocab = json.loads((cache_dir / "vocab.json").read_text(encoding="utf-8"))
    return codes, vocab


class INat21Val(Dataset):
    """Rows follow cache_dir's ids.txt (default the bioclip1 cache); each item is (tensor, row index)."""

    def __init__(self, transform, cache_dir=None, root=None):
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else inat21_cache_dir("bioclip1")
        # ids.txt lines already start with "val/", so the join root is the dataset directory itself.
        self.root = Path(root) if root is not None else Path(load_paths().inat21_dir)
        self.file_names = (self.cache_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        bad = [i for i, name in enumerate(self.file_names) if not name.startswith("val/")]
        if bad:
            raise ValueError(f"{self.cache_dir / 'ids.txt'}: {len(bad)} rows do not start with 'val/' "
                             f"(first: row {bad[0]}, {self.file_names[bad[0]]!r})")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with Image.open(self.root / self.file_names[idx]) as im:
            img = im.convert("RGB")
        return self.transform(img), idx
