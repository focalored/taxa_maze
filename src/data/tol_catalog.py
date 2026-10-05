"""ToL-10M EOL catalog: the train/val population, lineage keys, and the frozen S9 key list (spec lines 41-53; Amendment 1 S9, S52).

Usage: `df = read_eol_trainval(catalog_csv)`; `keys = load_s9_keys()`.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

import polars as pl

from src.utils.paths import REPO_ROOT

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
SPLITS = ("train", "val")

# Amendment 1 S52: asserted counts (EOL rows; after the lineage drop; after the S9 drop).
N_EOL = {"train": 5_908_775, "val": 310_899}
N_COMPLETE = {"train": 5_372_586, "val": 282_542}
N_AFTER_S9 = {"train": 5_298_907, "val": 278_613}

S9_KEYS_FILE = REPO_ROOT / "specs" / "pilot1_s9_missing_species_keys.txt"
S9_KEYS_SHA256 = "3ac09b97abc94748a3368c4d96992aaad2eb590ef63823e98213232ca31fbf64"
S9_N_KEYS = 8_084


def read_eol_trainval(catalog_csv) -> pl.DataFrame:
    """EOL (non-null eol_content_id) train and val rows with `complete` = all seven ranks non-null; asserts the S52 counts."""
    cols = ["split", "treeoflife_id", "eol_content_id", *RANKS]
    df = pl.read_csv(catalog_csv, columns=cols, infer_schema_length=0)
    df = df.filter(pl.col("eol_content_id").is_not_null() & pl.col("split").is_in(SPLITS))
    df = df.rename({"treeoflife_id": "uuid"}).drop("eol_content_id")
    for r in RANKS:
        # The preflight found only nulls; an empty or padded value would change the lineage rule.
        bad = df.filter(pl.col(r).is_not_null() & (pl.col(r).str.strip_chars() != pl.col(r)) | (pl.col(r) == ""))
        if bad.height:
            raise RuntimeError(f"rank {r}: {bad.height} empty or padded values; the lineage rule assumes none")
    df = df.with_columns(pl.all_horizontal([pl.col(r).is_not_null() for r in RANKS]).alias("complete"))
    for s in SPLITS:
        n = df.filter(pl.col("split") == s).height
        n_c = df.filter((pl.col("split") == s) & pl.col("complete")).height
        if n != N_EOL[s] or n_c != N_COMPLETE[s]:
            raise RuntimeError(f"{s}: {n} EOL rows / {n_c} complete; Amendment 1 S52 asserts {N_EOL[s]} / {N_COMPLETE[s]}")
    if df["uuid"].n_unique() != df.height:
        raise RuntimeError("uuid is not unique across EOL train+val (preflight C3 says it is)")
    return df


def key_expr(depth: int) -> pl.Expr:
    """Polars expression for the lineage key down to rank index `depth` (0 = kingdom, 6 = species)."""
    return pl.concat_str([pl.col(r) for r in RANKS[: depth + 1]], separator="|")


def load_s9_keys(path: Path = S9_KEYS_FILE) -> List[str]:
    """The frozen S9 list (Amendment 1 S9). Checks the sha256 and count before use."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != S9_KEYS_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} != Amendment 1's {S9_KEYS_SHA256}")
    keys = raw.decode("utf-8").splitlines()
    if len(keys) != S9_N_KEYS or len(set(keys)) != S9_N_KEYS:
        raise RuntimeError(f"{path}: {len(keys)} lines ({len(set(keys))} distinct), expected {S9_N_KEYS}")
    return keys
