"""Preflight check E6 extra: is iNat21 val "outside ToL-10M" by image id? (spec line 54)

Reads (read-only):
  /projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/inat21-val/bioclip1/ids.txt
      (100,000 iNat21 val file names, val/<class folder>/<uuid>.jpg)
  /u/liv/bdbk/data/tol10m/metadata/catalog.csv (2 GB; columns split, treeoflife_id,
      inat21_filename, inat21_cls_name, inat21_cls_num)
Checks whether any val uuid appears as a catalog treeoflife_id, or as the
basename of a catalog inat21_filename, and how catalog inat21_filename values
look. This is an id check only; content-level near-duplicates are not checked.

Writes /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_vs_tol_ids.json
Launched via scripts/preflight/eval_inat_vs_tol_ids.sh (srun, cpu partition).
"""
import json
import os
from pathlib import Path

import polars as pl

IDS = Path("/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/inat21-val/bioclip1/ids.txt")
CATALOG = Path("/u/liv/bdbk/data/tol10m/metadata/catalog.csv")
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_vs_tol_ids.json")


def main():
    files = IDS.read_text().splitlines()
    uuids = [Path(f).stem for f in files]
    val = pl.DataFrame({"file_name": files, "uuid": uuids})
    cat = pl.read_csv(CATALOG, columns=["split", "treeoflife_id", "inat21_filename",
                                        "inat21_cls_name", "inat21_cls_num"],
                      infer_schema_length=0)
    inat_rows = cat.filter(pl.col("inat21_filename").is_not_null())
    inat_rows = inat_rows.with_columns(
        pl.col("inat21_filename").str.split("/").list.last()
        .str.replace(r"\.jpg$", "").alias("inat_stem"))
    res = dict(
        host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        n_val_files=len(files), n_val_uuids_distinct=len(set(uuids)),
        catalog_rows=cat.height,
        catalog_rows_with_inat21_filename=inat_rows.height,
        catalog_inat21_rows_by_split={r["split"]: r["len"] for r in
                                      inat_rows.group_by("split").len().to_dicts()},
        catalog_inat21_filename_examples=inat_rows["inat21_filename"].head(5).to_list(),
        catalog_inat21_filename_prefixes={r["p"]: r["len"] for r in inat_rows.select(
            pl.col("inat21_filename").str.split("/").list.first().alias("p"))
            .group_by("p").len().sort("len", descending=True).head(10).to_dicts()},
        val_uuid_in_catalog_treeoflife_id=val.join(cat, left_on="uuid",
                                                   right_on="treeoflife_id",
                                                   how="semi").height,
        val_uuid_in_catalog_inat21_filename_stem=val.join(inat_rows, left_on="uuid",
                                                          right_on="inat_stem",
                                                          how="semi").height,
        val_file_name_equal_catalog_inat21_filename=val.join(
            inat_rows, left_on="file_name", right_on="inat21_filename", how="semi").height,
    )
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
