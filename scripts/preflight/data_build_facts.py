#!/usr/bin/env python
"""Collect the implementer-facing numbers from the data preflight outputs into
/projects/bdbk/liv/repos/taxa_maze/audit/preflight/data_facts.json.

Inputs (all under audit/preflight/, written by the other data_* scripts):
  data_catalog_facts.json, data_catalog_extras.json, data_cache_norms.json,
  data_tar_image_set_{60,63}.json, data_tar_probe_image_set_{60,63}.json,
  data_tar_probe_image_set_{60,63}_nolimit.json
Light work (small JSON files only); safe on the login node.
"""
import json
import os
import re

P = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
SHARDS = "/u/liv/bdbk/data/tol10m/dataset/EOL"
CACHE = "/projects/bdbk/liv/repos/tol_embed/results/probe_cache_b16/tol-eol/bioclip1"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
L = lambda n: json.load(open(f"{P}/{n}"))

F = L("data_catalog_facts.json")
X = L("data_catalog_extras.json")
N = L("data_cache_norms.json")
TAR = {s: L(f"data_tar_image_set_{s}.json") for s in ("60", "63")}
PROBE = {s: L(f"data_tar_probe_image_set_{s}.json") for s in ("60", "63")}
PROBE_NL = {s: L(f"data_tar_probe_image_set_{s}_nolimit.json") for s in ("60", "63")}
idx = json.load(open(f"{CACHE}/shards_index.json"))
nrows = {e["shard"]: e["n_rows"] for e in idx}


def gap_rows(split):
    return sum(p["rows"] for p in F["C4"]["eol"][split]["patterns"]
               if not re.fullmatch(r"x*-*", p["pattern"]))


sizes = {f"image_set_{i:02d}": os.path.getsize(f"{SHARDS}/image_set_{i:02d}.tar.gz") for i in range(1, 64)}
full = [v for k, v in sizes.items() if k != "image_set_63"]
c9 = F["C9"]["catalog"]
c4e = F["C4"]["eol"]
out = {
    "provenance": {
        "report": "/projects/bdbk/liv/repos/taxa_maze/audit/2026-10-02_preflight_data.md",
        "spec": "/projects/bdbk/liv/repos/taxa_maze/specs/pilot1.md",
        "scripts": "/projects/bdbk/liv/repos/taxa_maze/scripts/preflight/data_*.py|sh",
        "sources": ["data_catalog_facts.json", "data_catalog_extras.json", "data_cache_norms.json",
                    "data_tar_image_set_60.json", "data_tar_image_set_63.json",
                    "data_tar_probe_image_set_{60,63}[_nolimit].json"],
        "definitions": F["definitions"],
    },
    "catalog": {
        "rows_total": F["C1"]["n_rows_total"],
        "rows_by_split_all_sources": {k: v["rows"] for k, v in F["C1"]["all_rows"].items()},
        "rows_by_split_eol": {k: v["rows"] for k, v in F["C1"]["eol_rows"].items()},
        "spec_train_5908775_matches": "EOL rows with split == train, before lineage drop",
        "spec_val_310899_matches": "EOL rows with split == val, before lineage drop",
        "train_small_is_duplicate_subset_of_train": F["C2"]["eol"]["train_small_ids_also_train"]
        == F["C2"]["eol"]["train_small_distinct_ids"],
        "train_small_eol_ids": F["C2"]["eol"]["train_small_distinct_ids"],
        "train_small_lineage_mismatches_vs_train_row": F["C2"]["train_small_vs_train_lineage_or_source_mismatches"],
        "ids_unique_within_eol_train": F["C3"]["eol"]["train"]["ids_with_more_than_one_row"] == 0,
        "ids_unique_within_eol_val": F["C3"]["eol"]["val"]["ids_with_more_than_one_row"] == 0,
        "eol_train_val_shared_ids": F["C3"]["eol"]["train_and_val_shared_ids"],
    },
    "lineage_drop": {
        "eol_train": {"rows": c4e["train"]["rows"], "incomplete": c4e["train"]["incomplete_rows"],
                      "complete": c4e["train"]["complete_rows"], "rows_with_interior_gap": gap_rows("train")},
        "eol_val": {"rows": c4e["val"]["rows"], "incomplete": c4e["val"]["incomplete_rows"],
                    "complete": c4e["val"]["complete_rows"], "rows_with_interior_gap": gap_rows("val")},
        "eol_train_top_patterns": c4e["train"]["patterns"][:8],
        "eol_val_top_patterns": c4e["val"]["patterns"][:8],
        "missing_value_kinds": "all missing rank values are CSV nulls; 0 empty strings, 0 whitespace-only",
        "complete_but_placeholder_epithet_train": X["placeholder_epithet_species"]["n_species"],
        "complete_but_placeholder_epithet_train_images": X["placeholder_epithet_species"]["n_images"],
        "multiword_epithets_train_distinct": F["epithet_notes_train_complete"]["multiword"],
        "multiword_epithet_rows_train": F["epithet_notes_train_complete"]["rows_with_multiword_epithet"],
    },
    "populations_after_drop": {
        "train_complete_catalog": F["populations"]["T_eol_train_complete_rows"],
        "train_complete_in_cache": F["populations"]["T_in_cache"],
        "val_complete_catalog": F["populations"]["V_eol_val_complete_rows"],
        "val_complete_in_cache": F["populations"]["V_in_cache"],
    },
    "shards": {
        "composition_by_cache_ids": {"image_set_01-59": "all train",
                                     "image_set_60": {"train": 8775, "val": 91222},
                                     "image_set_61-63": "all val"},
        "image_set_60_train_block": F["C5"]["shard_60_positions"],
        "per_shard_table_csv": f"{P}/data_shard_composition.csv",
        "totals_cache_ids": F["C5"]["totals"],
        "tar_members": {s: {"n_members": TAR[s]["n_jpg_members"], "n_ids_txt": TAR[s]["n_ids_txt"],
                            "only_in_tar": TAR[s]["n_only_in_tar"],
                            "order_equal_after_dropping_tar_only": TAR[s]["order_equal_after_dropping_tar_only"],
                            "bare_uuid_jpg_only": TAR[s]["n_non_jpg_file_members"] == 0
                            and TAR[s]["n_dir_members"] == 0 and TAR[s]["n_member_names_not_uuid"] == 0,
                            "tar_tzf_wall_s": TAR[s]["tar_wall_s"], "MB_per_s": TAR[s]["read_MB_per_s"],
                            "members_per_s": TAR[s]["members_per_s"]} for s in TAR},
        "image_set_60_by_tar_members": {"train": 8775, "val": 91222 + TAR["60"]["n_only_in_tar"]},
        "bytes": {"image_set_01-62_min": min(full), "image_set_01-62_max": max(full),
                  "image_set_63": sizes["image_set_63"], "total": sum(sizes.values())},
        "old_encoder_img_per_s_from_done_json": "23.29-27.07 (shards 01-62), 5.31 (shard 63)",
    },
    "cache": {
        "shape": N["shape"], "dtype": N["dtype"], "normalized_per_manifest": N["manifest_normalized"],
        "ids_txt_lines": N["ids_txt"]["n_lines"], "ids_txt_distinct": N["ids_txt"]["n_distinct"],
        "ids_sha256_matches_manifest": N["ids_txt"]["manifest_sha_matches"],
        "row_norm_min": N["norms"]["min"], "row_norm_max": N["norms"]["max"],
        "row_norm_median": N["norms"]["quantiles"]["0.5"], "row_norm_mean": N["norms"]["mean"],
        "rows_nan": N["norms"]["n_rows_with_nan"], "rows_inf": N["norms"]["n_rows_with_inf"],
        "rows_zero_norm": N["norms"]["n_zero_norm"],
        "emb_slices_match_shard_raw_sha256": N["all_slices_match_index_sha"],
        "ids_txt_equals_concat_of_shard_ids": F["C5"]["ids_checks"]["concat_shard_ids_equals_cache_ids_txt"],
        "vocab_json_lengths": F["C8"]["vocab_list_lengths"],
        "vocab_json_built_from": "cache ids (EOL train + val), prefixes truncated at first missing rank; "
                                 "species element = 'Genus epithet'",
        "vocab_json_is_train_only": False,
    },
    "coverage": {
        "catalog_eol_train_val_ids": F["C6"]["catalog_eol_train_val_distinct_ids"],
        "cache_ids": F["C6"]["cache_distinct_ids"],
        "missing_from_cache": F["C6"]["catalog_ids_missing_from_cache"],
        "missing_by_split": {r["split"]: r["len"] for r in F["C6"]["missing_by_split"]},
        "missing_by_split_and_complete_lineage": {
            f"{r['split']}_{'complete' if r['pattern'] else 'incomplete'}": r["len"]
            for r in F["C6"]["missing_by_split_and_complete"]},
        "cache_ids_not_in_catalog": F["C6"]["of_which_not_in_catalog"],
        "cache_ids_train_small_only": F["C6"]["of_which_train_small_only"],
        "missing_ids_csv": f"{P}/data_cache_missing_ids.csv",
        "cause_verified_for_4_of_215": "PIL DecompressionBombError (>178,956,970 px); decode fine with "
                                       "Image.MAX_IMAGE_PIXELS=None",
        "per_shard_deficits_sum": {"shards_01_59_vs_100000": sum(100000 - nrows[f"image_set_{i:02d}"]
                                                                  for i in range(1, 60)),
                                   "shards_60_63": (100000 - nrows["image_set_60"]) + (100000 - nrows["image_set_61"])
                                   + (100000 - nrows["image_set_62"]) + (19674 - nrows["image_set_63"])},
        "train_species_with_no_cached_image": X["species_with_no_cached_image"],
        "train_species_with_some_missing_images": X["species_with_some_missing_images"],
        "probe_nolimit": {s: PROBE_NL[s]["found"] for s in PROBE_NL},
    },
    "taxa_train_complete": F["C8"]["train_complete"],
    "taxa_val_complete": F["C8"]["val_complete"],
    "taxa_train_complete_name_only": F["C8"]["train_complete_by_name_only"],
    "species_sizes_train_complete": {k: v for k, v in c9.items() if k != "K"},
    "batch_mining": c9["K"],
    "batch_mining_in_cache_population": F["C9"]["in_cache"]["K"],
    "single_child": F["C10"]["per_rank"],
    "penalty_terms": F["C10"]["penalty_terms"],
    "kept_terms_per_species_hist": F["C10"]["kept_terms_per_species_hist"],
    "bank_monitor_pairs_p1_to_p5": F["C10"]["monitor_pairs"],
    "homonyms_train_complete": {r: {k: v for k, v in d.items() if k != "examples"}
                                for r, d in F["C11"]["train_complete"].items()},
    "val_vs_train": {r: {k: v for k, v in F["C12"][r].items() if k != "examples"} for r in RANKS},
    "largest_species": X["top30"][:5],
    "store_estimate_uint8_224": {
        "bytes_per_image": 224 * 224 * 3,
        "eol_train_val_all_GB": 6219674 * 224 * 224 * 3 / 1e9,
        "complete_lineage_train_val_GB": (F["populations"]["T_eol_train_complete_rows"]
                                          + F["populations"]["V_eol_val_complete_rows"]) * 224 * 224 * 3 / 1e9,
    },
}
with open(f"{P}/data_facts.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
