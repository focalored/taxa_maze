"""Train tree (species, ancestors, n_a, P_s, g_s, held-out 5%) and ToL-val candidate sets, built from the store tables.

Usage: `tree = build_train_tree(store_dir, s9_drop=True, K=16, heldout_file=f)`; `val = build_val_set(store_dir, tree, drop_uuids)`.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from src.data.tol_catalog import N_AFTER_S9, N_COMPLETE, RANKS, key_expr, load_s9_keys

N_DEPTH = len(RANKS)  # 7
HELDOUT_FRACTION = 0.05
HELDOUT_SEED = 42


@dataclass
class TrainTree:
    s9_drop: bool
    K: int
    node_keys: List[List[str]]          # per depth, sorted; node_keys[6] are the species keys
    parent: List[np.ndarray]            # parent[d][i] = id at depth d-1 of node i at depth d (d >= 1)
    n_species: List[np.ndarray]         # species beneath each node, per depth (depth 6: all ones)
    n_children: List[np.ndarray]        # distinct child taxa per node, depths 0..5
    anc: np.ndarray                     # (S, 6) ancestor ids at depths 0..5 for each species
    N_s: np.ndarray                     # (S,) train images per species
    g_s: np.ndarray                     # (S,) groups per epoch
    P_mask: np.ndarray                  # (S, 6) bool, the valid ranks P_s
    heldout: np.ndarray                 # (S,) bool
    # train images, sorted by (species, shard, row)
    img_species: np.ndarray             # (T,) int32
    img_shard: np.ndarray               # (T,) int16
    img_row: np.ndarray                 # (T,) int32
    sp_start: np.ndarray                # (S,) offset of each species' images
    stats: Dict = field(default_factory=dict)

    @property
    def n_species_total(self) -> int:
        return len(self.node_keys[6])

    def image_node_ids(self, depth: int) -> np.ndarray:
        """Node id at `depth` of every train image (in the species-sorted image order)."""
        if depth == 6:
            return self.img_species
        return self.anc[self.img_species, depth]

    def species_keys_sha256(self) -> str:
        return hashlib.sha256("".join(k + "\n" for k in self.node_keys[6]).encode()).hexdigest()

    def vocab_key(self, s: int) -> str:
        """Preflight reading U7: the vocab.json form of a species key, with a "Genus epithet" leaf."""
        parts = self.node_keys[6][s].split("|")
        return "|".join(parts[:6] + [f"{parts[5]} {parts[6]}"])


def _lineage_tuple(key: str):
    return tuple(key.split("|"))


def _train_frame(store_dir: Path, s9_drop: bool):
    cat = pl.read_parquet(store_dir / "catalog.parquet")
    idx = pl.read_parquet(store_dir / "index.parquet", columns=["uuid", "shard", "row"])
    df = cat.filter(pl.col("complete")).join(idx, on="uuid", how="inner")
    df = df.with_columns([key_expr(d).alias(f"k{d}") for d in range(N_DEPTH)])
    n_complete = {s: df.filter(pl.col("split") == s).height for s in ("train", "val")}
    if n_complete != N_COMPLETE:
        raise RuntimeError(f"store-joined complete rows {n_complete} != {N_COMPLETE}")
    s9 = set(load_s9_keys())
    is_s9 = pl.col("k6").is_in(list(s9))
    dropped = df.filter(is_s9)
    drop_stats = {
        "s9_drop": s9_drop,
        "s9_keys_in_list": len(s9),
        "s9_rows_matching": {s: dropped.filter(pl.col("split") == s).height for s in ("train", "val")},
        "s9_keys_matching": {s: dropped.filter(pl.col("split") == s)["k6"].n_unique() for s in ("train", "val")},
        "s9_dropped_keys": {r["k6"]: {"train": r["train"], "val": r["val"]} for r in
                            dropped.group_by("k6").agg((pl.col("split") == "train").sum().alias("train"),
                                                       (pl.col("split") == "val").sum().alias("val")).sort("k6").to_dicts()},
    }
    if s9_drop:
        df = df.filter(~is_s9)
        kept = {s: df.filter(pl.col("split") == s).height for s in ("train", "val")}
        if kept != N_AFTER_S9:
            raise RuntimeError(f"rows after the S9 drop {kept} != Amendment 1 S52's {N_AFTER_S9}")
    drop_stats["rows_kept"] = {s: df.filter(pl.col("split") == s).height for s in ("train", "val")}
    return df, drop_stats


def build_train_tree(store_dir, s9_drop: bool = True, K: int = 16,
                     heldout_file: Optional[Path] = None) -> TrainTree:
    store_dir = Path(store_dir)
    df, stats = _train_frame(store_dir, s9_drop)
    tr = df.filter(pl.col("split") == "train")

    # Sorted by component tuple, not by the joined string ("Ab|x" < "A|y" as strings), so every
    # node's species form one contiguous id range at every depth; the bank's segment sums rely on it.
    node_keys = [sorted(tr[f"k{d}"].unique().to_list(), key=_lineage_tuple) for d in range(N_DEPTH)]
    ids = [{k: i for i, k in enumerate(keys)} for keys in node_keys]
    sp_keys = node_keys[6]
    S = len(sp_keys)
    anc = np.empty((S, 6), dtype=np.int32)
    for s, key in enumerate(sp_keys):
        parts = key.split("|")
        for d in range(6):
            anc[s, d] = ids[d]["|".join(parts[: d + 1])]
    parent = [np.zeros(0, dtype=np.int32)]
    for d in range(1, N_DEPTH):
        parent.append(np.array([ids[d - 1][k.rsplit("|", 1)[0]] for k in node_keys[d]], dtype=np.int32))
    n_species = [np.bincount(anc[:, d], minlength=len(node_keys[d])).astype(np.int64) for d in range(6)]
    n_species.append(np.ones(S, dtype=np.int64))
    n_children = [np.bincount(parent[d + 1], minlength=len(node_keys[d])).astype(np.int64) for d in range(6)]

    # images, sorted by (species, shard, row)
    sp_of = pl.DataFrame({"k6": sp_keys, "sid": np.arange(S, dtype=np.int32)})
    im = tr.select("k6", "shard", "row").join(sp_of, on="k6", how="inner").sort(["sid", "shard", "row"])
    img_species = im["sid"].to_numpy().astype(np.int32)
    N_s = np.bincount(img_species, minlength=S).astype(np.int64)
    sp_start = np.concatenate([[0], np.cumsum(N_s)[:-1]]).astype(np.int64)
    g_s = -(-N_s // K)
    P_mask = np.stack([n_species[d][anc[:, d]] >= 2 for d in range(6)], axis=1)
    if not all(np.all(np.diff(anc[:, d]) >= 0) for d in range(6)):
        raise RuntimeError("ancestor ids are not contiguous over the species order")

    if heldout_file is not None and Path(heldout_file).exists():
        held_keys = Path(heldout_file).read_text().splitlines()
        held_ids = np.array([ids[6][k] for k in held_keys], dtype=np.int64)
    else:
        n_held = int(round(HELDOUT_FRACTION * S))
        held_ids = np.sort(np.random.default_rng(HELDOUT_SEED).choice(S, size=n_held, replace=False))
        if heldout_file is not None:
            Path(heldout_file).write_text("".join(sp_keys[i] + "\n" for i in held_ids))
    heldout = np.zeros(S, dtype=bool)
    heldout[held_ids] = True

    # names used under more than one parent prefix (true homonyms included; not preflight U8's count)
    homonyms = {}
    for d in range(1, 6):
        names = pl.Series([k.rsplit("|", 1)[1] for k in node_keys[d]])
        homonyms[RANKS[d]] = int((names.value_counts()["count"] > 1).sum())
    stats.update({
        "n_species": S, "n_train_images": int(N_s.sum()),
        "nodes_per_rank": {RANKS[d]: len(node_keys[d]) for d in range(N_DEPTH)},
        "single_child_taxa_per_rank": {RANKS[d]: int((n_children[d] == 1).sum()) for d in range(6)},
        "single_species_taxa_per_rank": {RANKS[d]: int((n_species[d] == 1).sum()) for d in range(6)},
        "names_under_several_parent_prefixes": homonyms,
        "penalty_terms_valid": int(P_mask.sum()), "penalty_terms_skipped": int((~P_mask).sum()),
        "K": K, "groups_per_epoch": int(g_s.sum()), "max_groups": int(g_s.max()),
        "largest_species": sp_keys[int(np.argmax(N_s))], "largest_N_s": int(N_s.max()),
        "n_heldout": int(heldout.sum()),
    })
    return TrainTree(
        s9_drop=s9_drop, K=K, node_keys=node_keys, parent=parent, n_species=n_species,
        n_children=n_children, anc=anc, N_s=N_s, g_s=g_s, P_mask=P_mask, heldout=heldout,
        img_species=img_species, img_shard=im["shard"].to_numpy().astype(np.int16),
        img_row=im["row"].to_numpy().astype(np.int32), sp_start=sp_start, stats=stats,
    )


@dataclass
class ValSet:
    """ToL-val images (complete lineage, after S9 if on, after dedup) and their candidate taxa (S4)."""
    uuid: List[str]
    shard: np.ndarray
    row: np.ndarray
    node_keys: List[List[str]]   # per depth: the taxa present in this val set, sorted (the candidates)
    labels: np.ndarray           # (N, 7) candidate index per depth
    seen: np.ndarray             # (N,) species key present in the train tree
    stats: Dict = field(default_factory=dict)


def build_val_set(store_dir, tree: TrainTree, drop_uuids: Optional[set] = None) -> ValSet:
    store_dir = Path(store_dir)
    df, _ = _train_frame(store_dir, tree.s9_drop)
    va = df.filter(pl.col("split") == "val").sort(["shard", "row"])
    n_before = va.height
    if drop_uuids:
        va = va.filter(~pl.col("uuid").is_in(list(drop_uuids)))
    node_keys = [sorted(va[f"k{d}"].unique().to_list(), key=_lineage_tuple) for d in range(N_DEPTH)]
    labels = np.empty((va.height, N_DEPTH), dtype=np.int64)
    for d in range(N_DEPTH):
        m = {k: i for i, k in enumerate(node_keys[d])}
        labels[:, d] = [m[k] for k in va[f"k{d}"].to_list()]
    train_sp = set(tree.node_keys[6])
    seen = np.array([k in train_sp for k in va["k6"].to_list()], dtype=bool)
    stats = {"n_val_before_dedup": n_before, "n_val": va.height, "n_dedup_dropped": n_before - va.height,
             "candidates_per_rank": {RANKS[d]: len(node_keys[d]) for d in range(N_DEPTH)},
             "n_seen_species_images": int(seen.sum())}
    return ValSet(uuid=va["uuid"].to_list(), shard=va["shard"].to_numpy().astype(np.int16),
                  row=va["row"].to_numpy().astype(np.int32), node_keys=node_keys, labels=labels,
                  seen=seen, stats=stats)


def save_stats(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
