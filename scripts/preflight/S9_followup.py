#!/usr/bin/env python
"""S9 follow-up counts for audit/2026-10-02_S9_bioclip1_provenance.md (reads run-3 outputs).

1. For the S9 option (b) rows (EOL train, complete lineage; keys from S9_suspect_keys.csv with
   is_S9b semantics = exact placeholder or author), how many have a catalog `common` equal to the
   scientific name "Genus epithet"? make_catalog.py:159 read `common` from the training shards'
   common_name.txt, and make_wds.py:256-257 writes the scientific name there when no vernacular
   exists. So such rows show the epithet string inside the shard text itself.
2. How many author rows sit on scraped pages whose CSV row was split at an unquoted comma
   (for example '65269955,Bombus Latreille, 1802' -> field 'Bombus Latreille' + extra ' 1802').
3. Same two counts for val.
Output: audit/preflight/S9_bioclip1/S9_followup.json
Run (cpu partition):
  srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=8 --mem=48G --time=00:30:00 \
    bash -c 'source "$HOME/.hpc_env.sh" && conda activate bioclip && \
             python -u /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/S9_followup.py' \
    > /projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_followup.log 2>&1
"""
import csv
import json
import time

import polars as pl

T0 = time.time()
EVID = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1"
CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
SCRAPED = f"{EVID}/bioclip_fe4bd61/data/eol/scraped_page_ids.csv"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

sk = pl.read_csv(f"{EVID}/S9_suspect_keys.csv", infer_schema=False)
sk = sk.with_columns((pl.col("is_placeholder") == "true").alias("ph"), (pl.col("is_author") == "true").alias("au"),
                     (pl.col("is_placeholder_ext") == "true").alias("phx"))
flag = sk.filter(pl.col("ph") | pl.col("au") | pl.col("phx")).select("key", "ph", "au", "phx")

extra = {}
with open(SCRAPED, newline="") as fh:
    for row in csv.DictReader(fh):
        pid = int(row["page_id"])
        if pid not in extra:
            extra[pid] = ",".join(row.get(None) or [])
SC = pl.DataFrame({"page_id": list(extra.keys()), "csv_text_after_comma": list(extra.values())})

cat = pl.read_csv(CATALOG, columns=["split", "eol_content_id", "eol_page_id", "common"] + RANKS, infer_schema=False)
blank = lambda c: pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")  # noqa: E731
cat = cat.filter(~blank("eol_content_id") & pl.col("split").is_in(["train", "val"]))
cat = cat.with_columns([pl.when(blank(c)).then(None).otherwise(pl.col(c)).alias(c) for c in RANKS])
cat = cat.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
cat = cat.with_columns(pl.concat_str([pl.col(c) for c in RANKS], separator="|").alias("key"),
                       pl.col("eol_page_id").cast(pl.Float64).cast(pl.Int64).alias("page_id"),
                       (pl.col("genus") + " " + pl.col("species")).alias("scientific"))
cat = cat.join(flag, on="key", how="inner").join(SC, on="page_id", how="left")
out = {}
for split in ["train", "val"]:
    d = cat.filter(pl.col("split") == split)
    for name, expr in [("placeholder (exact)", pl.col("ph")), ("author", pl.col("au")),
                       ("S9 option (b) = placeholder (exact) or author", pl.col("ph") | pl.col("au")),
                       ("extended placeholder", pl.col("phx"))]:
        x = d.filter(expr)
        out[f"{split} | {name}"] = {
            "rows": x.height,
            "catalog_common_equals_'Genus epithet'": x.filter(pl.col("common") == pl.col("scientific")).height,
            "rows_on_pages_whose_csv_row_was_split_at_a_comma": x.filter(pl.col("csv_text_after_comma").fill_null("") != "").height,
            "examples_split_rows": x.filter(pl.col("csv_text_after_comma").fill_null("") != "").group_by(
                "key", "page_id", "csv_text_after_comma").len().sort("len", descending=True).head(5).to_dicts(),
            "examples_common_not_scientific": x.filter(pl.col("common") != pl.col("scientific")).group_by(
                "key", "common").len().sort("len", descending=True).head(5).to_dicts(),
        }
out["elapsed_s"] = round(time.time() - T0, 1)
with open(f"{EVID}/S9_followup.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
