#!/usr/bin/env python3
"""
refbeh_dropped_catalog.py -- look up the 215 shard members that the BioCLIP 1
ToL cache lacks (from refbeh_census.json) in catalog.csv.
(audit/2026-10-02_reference_behaviour.md, R1.)

Checks:
  - the dropped uuids are exactly the catalog EOL ids that are absent from the cache's ids.txt;
  - the split of each dropped uuid, counted per shard (spec line 47 counts image_set_60);
  - whether the dropped images have a complete 7-rank lineage.
Reads catalog.csv (2 GB), so it runs under Slurm. Writes audit/preflight/refbeh_dropped_catalog.json.
"""

import json
from pathlib import Path

import polars as pl

TAXA = Path("/projects/bdbk/liv/repos/taxa_maze")
CAT = Path("/u/liv/bdbk/data/tol10m/metadata/catalog.csv")
IDS = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1/ids.txt")
OUT = TAXA / "audit/preflight/refbeh_dropped_catalog.json"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

census = json.loads((TAXA / "audit/preflight/refbeh_census.json").read_text())
dropped = [(d["shard"], d["name"][:-4]) for d in census["summary"]["dropped_list"]]
dd = pl.DataFrame({"shard": [s for s, _ in dropped], "treeoflife_id": [u for _, u in dropped]})

cat = pl.read_csv(CAT, columns=["split", "treeoflife_id", "eol_content_id"] + RANKS, infer_schema_length=0)
eol = cat.filter(pl.col("eol_content_id").is_not_null())
agg = (eol.group_by("treeoflife_id")
          .agg([pl.col("split").unique().sort().alias("splits")] + [pl.col(r).first().alias(r) for r in RANKS])
          .with_columns(pl.when(pl.col("splits").list.contains("val")).then(pl.lit("val"))
                          .when(pl.col("splits").list.contains("train")).then(pl.lit("train"))
                          .otherwise(pl.lit("train_small_only")).alias("split1")))
ids = pl.DataFrame({"treeoflife_id": IDS.read_text().splitlines()})
not_in_cache = agg.join(ids, on="treeoflife_id", how="anti")

j = dd.join(agg, on="treeoflife_id", how="left")
complete = j.select([pl.col(r).is_not_null() & (pl.col(r) != "") for r in RANKS]).to_numpy().all(axis=1)
res = dict(
    n_dropped=dd.height,
    n_dropped_without_catalog_eol_row=int(j["split1"].null_count()),
    dropped_equals_catalog_eol_ids_not_in_cache=set(dd["treeoflife_id"].to_list()) == set(not_in_cache["treeoflife_id"].to_list()),
    n_catalog_eol_ids_not_in_cache=not_in_cache.height,
    dropped_split_counts={r["split1"]: r["len"] for r in j.group_by("split1").len().to_dicts()},
    dropped_split_by_shard_60={r["split1"]: r["len"] for r in j.filter(pl.col("shard") == "image_set_60").group_by("split1").len().to_dicts()},
    dropped_split_by_range={
        "01-59": {r["split1"]: r["len"] for r in j.filter(pl.col("shard") <= "image_set_59").group_by("split1").len().to_dicts()},
        "61-63": {r["split1"]: r["len"] for r in j.filter(pl.col("shard") >= "image_set_61").group_by("split1").len().to_dicts()},
    },
    dropped_with_complete_lineage=int(complete.sum()),
    dropped_complete_lineage_by_split={s: int(complete[(j["split1"] == s).fill_null(False).to_numpy()].sum())
                                      for s in ("train", "val")},
)
OUT.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
