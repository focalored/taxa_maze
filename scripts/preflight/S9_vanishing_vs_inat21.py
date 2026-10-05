#!/usr/bin/env python
"""S9: which taxa would leave pilot 1's training tree under option (b), and do any of them
occur in the evaluation set (iNat21 val)?

Option (b) = drop EOL train rows (complete lineage) whose species key is flagged in
S9_suspect_keys.csv as an exact placeholder or an author name (run 3 of S9_provenance.py).
A genus (family, order) 'vanishes' when every species key under it is flagged.
iNat21 val lineages come from the 10,000 class folder names in /u/liv/bdbk/data/inat21/val.
The match is on the full lineage prefix (for example 'Animalia|...|Apidae|Bombus'), the same
key style the bank uses (spec line 121).
Output: audit/preflight/S9_bioclip1/S9_vanishing_vs_inat21.json
Run (cpu partition):
  srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=8 --mem=48G --time=00:30:00 \
    bash -c 'source "$HOME/.hpc_env.sh" && conda activate bioclip && \
             python -u /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/S9_vanishing_vs_inat21.py' \
    > /projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_vanishing_vs_inat21.log 2>&1
"""
import json
import os

import polars as pl

EVID = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1"
CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
VAL = "/u/liv/bdbk/data/inat21/val"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

sk = pl.read_csv(f"{EVID}/S9_suspect_keys.csv", infer_schema=False)
flagged = sk.filter((pl.col("is_placeholder") == "true") | (pl.col("is_author") == "true")).select("key") \
    .with_columns(pl.lit(True).alias("s9"))

cat = pl.read_csv(CATALOG, columns=["split", "eol_content_id"] + RANKS, infer_schema=False)
blank = lambda c: pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")  # noqa: E731
cat = cat.filter(~blank("eol_content_id") & (pl.col("split") == "train"))
cat = cat.with_columns([pl.when(blank(c)).then(None).otherwise(pl.col(c)).alias(c) for c in RANKS])
cat = cat.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
cat = cat.with_columns(pl.concat_str([pl.col(c) for c in RANKS], separator="|").alias("key"))
keys = cat.group_by("key").agg(pl.len().alias("N_s"), *[pl.col(c).first() for c in RANKS[:6]])
keys = keys.join(flagged, on="key", how="left").with_columns(pl.col("s9").fill_null(False))

inat = []
for d in sorted(os.listdir(VAL)):
    p = d.split("_")
    if len(p) == 8:
        inat.append({"dir": d, "n_images": len(os.listdir(os.path.join(VAL, d))), **dict(zip(RANKS, p[1:]))})
IN = pl.DataFrame(inat)

out = {"flagged_keys": flagged.height, "inat21_val_classes": IN.height, "inat21_val_images": int(IN["n_images"].sum())}
for k, rank in [(6, "genus"), (5, "family"), (4, "order")]:
    pref = pl.concat_str([pl.col(c) for c in RANKS[:k]], separator="|").alias("prefix")
    nodes = keys.with_columns(pref).group_by("prefix").agg(
        pl.len().alias("n_species_before"), (~pl.col("s9")).sum().alias("n_species_after"),
        pl.col("N_s").sum().alias("train_images"))
    vanish = nodes.filter(pl.col("n_species_after") == 0)
    inat_nodes = IN.with_columns(pref).group_by("prefix").agg(pl.len().alias("inat_classes"),
                                                             pl.col("n_images").sum().alias("inat_images"))
    hit = vanish.join(inat_nodes, on="prefix", how="inner")
    out[rank] = {"nodes_in_train_tree": nodes.height, "nodes_that_vanish": vanish.height,
                 "train_images_under_vanishing_nodes": int(vanish["train_images"].sum()),
                 "vanishing_nodes_that_are_in_inat21_val": hit.height,
                 "inat21_val_classes_under_them": int(hit["inat_classes"].sum()) if hit.height else 0,
                 "inat21_val_images_under_them": int(hit["inat_images"].sum()) if hit.height else 0,
                 "examples": hit.sort("inat_images", descending=True).head(20).to_dicts()}
with open(f"{EVID}/S9_vanishing_vs_inat21.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
