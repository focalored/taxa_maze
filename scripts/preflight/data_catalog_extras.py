#!/usr/bin/env python
"""Follow-up facts for C6/C9 (same definitions as data_catalog_facts.py):
  * train species (complete lineage) whose images are ALL missing from the cache
  * top-30 species by N_s, with a sample of the 'common' column
  * species whose epithet is a placeholder token (sp., spp., xx, ...)
  * catalog rows for the largest species 'Bombus latreille'
Output: /projects/bdbk/liv/repos/taxa_maze/audit/preflight/data_catalog_extras.json
"""
import json
import time

import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
CACHE = "/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1"
OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
PLACEHOLDERS = ["sp", "sp.", "spp", "spp.", "xx", "x", "?", "incertae sedis"]
T0 = time.time()

df = pl.read_csv(CATALOG, columns=["split", "treeoflife_id", "eol_content_id", "common"] + RANKS,
                 infer_schema=False)
blank = lambda c: pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")
df = df.with_columns([pl.when(blank(c)).then(None).otherwise(pl.col(c).str.strip_chars()).alias(c)
                      for c in RANKS])
T = df.filter(~blank("eol_content_id") & (pl.col("split") == "train")
              & pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
T = T.with_columns(pl.concat_str([pl.col(c) for c in RANKS], separator="|").alias("k_species"))
with open(f"{CACHE}/ids.txt") as fh:
    C = pl.DataFrame({"treeoflife_id": [ln.strip() for ln in fh if ln.strip()]})
R = {"n_T": T.height}

T = T.join(C.with_columns(pl.lit(True).alias("in_cache")), on="treeoflife_id", how="left") \
       .with_columns(pl.col("in_cache").fill_null(False))
ns = T.group_by("k_species").agg(pl.len().alias("N_s"), pl.col("in_cache").sum().alias("N_s_in_cache"))
lost = ns.filter(pl.col("N_s_in_cache") == 0)
R["species_with_no_cached_image"] = lost.to_dicts()
R["species_with_some_missing_images"] = ns.filter(pl.col("N_s_in_cache") < pl.col("N_s")).height

top = ns.sort("N_s", descending=True).head(30)
commons = (T.join(top.select("k_species"), on="k_species", how="semi")
           .group_by("k_species").agg(pl.col("common").drop_nulls().unique().sort().head(3).alias("common_sample")))
R["top30"] = top.join(commons, on="k_species", how="left").sort("N_s", descending=True).to_dicts()

ph = ns.with_columns(pl.col("k_species").str.split("|").list.last().str.to_lowercase().alias("ep")) \
       .filter(pl.col("ep").is_in(PLACEHOLDERS))
R["placeholder_epithet_species"] = {"n_species": ph.height, "n_images": int(ph["N_s"].sum()),
                                    "by_token": ph.group_by("ep").agg(pl.len().alias("n_species"),
                                                                      pl.col("N_s").sum().alias("n_images"))
                                    .sort("n_images", descending=True).to_dicts(),
                                    "largest": ph.sort("N_s", descending=True).head(5).to_dicts()}

bl = df.filter((pl.col("genus") == "Bombus") & (pl.col("species") == "latreille"))
R["bombus_latreille_rows_by_split"] = bl.group_by("split").len().sort("split").to_dicts()
R["bombus_latreille_common_values"] = bl.group_by("common").len().sort("len", descending=True).head(5).to_dicts()
R["bombus_species_in_T"] = T.filter(pl.col("genus") == "Bombus").select("species").n_unique()
R["elapsed_s"] = time.time() - T0
with open(f"{OUT}/data_catalog_extras.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
