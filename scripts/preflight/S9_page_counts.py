#!/usr/bin/env python
"""S9: train/val row counts for a few EOL pages quoted in the S9 report (catalog.csv, read-only).
Pages: 65270257 (Tenthredo Linnaeus), 65269361 (Macromischa Roger), 104136 (Bombus, DH genus page),
65269955 and 64690618 (Bombus Latreille). Output: audit/preflight/S9_bioclip1/S9_page_counts.json
"""
import json

import polars as pl

PAGES = [65270257, 65269361, 104136, 65269955, 64690618]
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
cat = pl.read_csv("/u/liv/bdbk/data/tol10m/metadata/catalog.csv",
                  columns=["split", "eol_page_id", "common"] + RANKS, infer_schema=False)
cat = cat.with_columns(pl.col("eol_page_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias("page_id"))
d = cat.filter(pl.col("page_id").is_in(PAGES) & pl.col("split").is_in(["train", "val"]))
out = d.group_by("page_id", "split", *RANKS, "common").len().sort("page_id", "split").to_dicts()
with open("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_page_counts.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
for r in out:
    print(r)
