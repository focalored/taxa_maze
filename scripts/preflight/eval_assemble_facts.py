"""Assemble /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_facts.json.

Inputs (all written earlier by the other eval_* preflight scripts, plus the
three tol_embed result JSONs, read-only):
  audit/preflight/eval_table_checks.json      (E1-E4; eval_table_checks.py)
  audit/preflight/eval_cache_inventory.json   (E5 light; eval_cache_inventory.py)
  audit/preflight/eval_inat_heavy.json        (E5 deep, E6, E7-iNat; Slurm job 245649)
  audit/preflight/eval_anova_tol.json         (E7-ToL; Slurm job 245659)
  audit/preflight/eval_inat_vs_tol_ids.json   (E6 extra; Slurm job 245663)
  audit/preflight/eval_tokenizer_case.json    (E1 note on "A photo" vs "a photo")

Light work; run on the login node:
    source "$HOME/.hpc_env.sh" && conda activate bioclip
    python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_assemble_facts.py
"""
import json
from pathlib import Path

PF = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight")
RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
COLS = {"clip-l14 `photo`": ("photo", "clip-l14-laion2b"),
        "bioclip2 `photo`": ("photo", "bioclip2"),
        "rcme `lineage`": ("lineage", "rcme"),
        "bioclip1 `photo`": ("photo", "bioclip1"),
        "bfl-euc `photo`": ("photo", "bfl-euclidean"),
        "bfl-hyp `photo`": ("photo", "bfl-hyperbolic")}
SOURCE_FILE = {"photo": {"clip-l14-laion2b": "ZEROSHOT_RANKS_inat21-val__forms.json",
                         "bioclip2": "ZEROSHOT_RANKS_inat21-val__forms.json",
                         "rcme": "ZEROSHOT_RANKS_inat21-val__forms.json",
                         "bioclip1": "ZEROSHOT_RANKS_inat21-val__forms.json",
                         "bfl-euclidean": "ZEROSHOT_RANKS_inat21-val__bfl.json",
                         "bfl-hyperbolic": "ZEROSHOT_RANKS_inat21-val__bfl.json"},
               "lineage": {"clip-l14-laion2b": "ZEROSHOT_RANKS_inat21-val__forms.json",
                           "bioclip2": "ZEROSHOT_RANKS_inat21-val__forms.json",
                           "rcme": "ZEROSHOT_RANKS_inat21-val__forms.json",
                           "bioclip1": "ZEROSHOT_RANKS_inat21-val__forms.json",
                           "bfl-euclidean": "ZEROSHOT_RANKS_inat21-val__bfl.json",
                           "bfl-hyperbolic": "ZEROSHOT_RANKS_inat21-val__bfl.json"}}


def load(p):
    return json.loads(Path(p).read_text())


def gate_targets(res_by_file, form, model):
    r = res_by_file[SOURCE_FILE[form][model]]["results"][form][model]
    out = {}
    for k in RANKS:
        c = int(round(r[k]["top1"] * r[k]["n_eval"] / 100.0))
        assert 100.0 * c / r[k]["n_eval"] == r[k]["top1"]
        out[k] = dict(n_classes=r[k]["n_classes"], n_paths=r[k]["n_paths"],
                      n_eval=r[k]["n_eval"], correct=c, top1=r[k]["top1"])
    out["average"] = dict(top1=r["average"]["top1"])
    return out


def main():
    tc = load(PF / "eval_table_checks.json")
    inv = load(PF / "eval_cache_inventory.json")
    heavy = load(PF / "eval_inat_heavy.json")
    tol = load(PF / "eval_anova_tol.json")
    ids = load(PF / "eval_inat_vs_tol_ids.json")
    tok = load(PF / "eval_tokenizer_case.json")
    res_by_file = {p.name: load(p) for p in sorted(RES.glob("ZEROSHOT_RANKS_inat21-val*.json"))}

    facts = dict(
        generated_by="scripts/preflight/eval_assemble_facts.py",
        spec="specs/pilot1.md (taxa_maze HEAD 90a7d38)",
        json_sha256={k: v["sha256"] for k, v in
                     load(PF / "eval_json_cells.json")["meta"].items()},
        column_map=tc["column_map"],
        form_definitions_from_zeroshot_ranks_render={
            "photo": "a photo of <lineage kingdom..rank, space-joined>.",
            "lineage": "<lineage kingdom..rank, space-joined> (no template, no period)",
            "bare": "<rank's own name>",
            "species_leaf": "binomial leaf; the doubled genus is stripped, so species "
                            "renders '... Genus epithet'"},
        tokenizer_case_check=tok["results"],
    )
    # E1
    facts["E1_table"] = dict(summary=tc["e1_summary"], cells=tc["e1"],
                             bold_marks_equal_row_max=all(b["consistent"]
                                                          for b in tc["bold_check"]),
                             best_form=tc["best_form"])
    # E2: gate targets for the spec's six columns, plus the other template
    facts["E2_gate_targets"] = {col: dict(form=f, model=m, **gate_targets(res_by_file, f, m))
                                for col, (f, m) in COLS.items()}
    facts["E2_gate_targets_other_template"] = {
        m: dict(form=("photo" if f == "lineage" else "lineage"),
                **gate_targets(res_by_file, "photo" if f == "lineage" else "lineage", m))
        for _, (f, m) in COLS.items()}
    facts["E2_average_check"] = tc["average_check"]
    # paper rows vs harness rows (spec line 20 says the table is from our harness)
    pub, sr = {}, {}
    for fname, d in res_by_file.items():
        pub.update(d.get("published", {}))
        sr.update(d.get("self_reported", {}))
    paper_cmp = {}
    for col, (f, m) in COLS.items():
        row = pub.get(m) or sr.get(m)
        if row is None:
            continue
        key = lambda k: "cls" if k == "class" else k  # noqa: E731
        paper_cmp[col] = {k: dict(paper=row[key(k)],
                                  harness=facts["E2_gate_targets"][col][k]["top1"])
                          for k in RANKS + ["average"]}
    facts["paper_vs_harness"] = paper_cmp
    # E3, E4
    facts["E3_prose_claims"] = tc["e3"]
    facts["E4_decision_constants"] = tc["e4"]
    # E5
    e5 = {}
    for m, e in inv["caches"].items():
        deep = heavy["e5_e7"][m]
        e5[m] = dict(path=e["path"], is_spec_column=e["is_spec_column"],
                     files=e["files"], subdirs=e["subdirs"],
                     emb_shape=e["emb_header"]["shape"], emb_dtype=e["emb_header"]["dtype"],
                     codes_shape=e["codes_header"]["shape"],
                     codes_dtype=e["codes_header"]["dtype"],
                     ids_count=e["ids_count"], ids_distinct=e["ids_distinct"],
                     rows_equal_json_n_eval=e["emb_rows_equal_json_n_eval_all_ranks"],
                     json_counts_match_cache=all(v["all_equal"] for f in
                                                 e["json_comparison"].values()
                                                 for v in f.values()),
                     vocab_n_per_rank=e["vocab_n_per_rank"],
                     codes_valid_per_rank=e["codes_valid_per_rank"],
                     manifest=dict((k, e["manifest"][k]) for k in
                                   ("model", "dataset", "N", "D", "dtype", "precision",
                                    "normalized", "written_utc", "slurm_job_id")),
                     n_nonfinite=deep["n_nonfinite"],
                     row_norm_full=dict(min=deep["row_norm_min"],
                                        median=deep["row_norm_median"],
                                        max=deep["row_norm_max"]),
                     emb_sha256=deep["emb_sha256"])
    facts["E5_caches"] = dict(per_cache=e5,
                              identical_across_caches=inv["identical_across_caches"],
                              hashes=inv["hashes"])
    # E6
    facts["E6_inat21_val"] = dict(heavy["e6"], id_overlap_with_tol10m_catalog={
        k: ids[k] for k in ("n_val_files", "catalog_rows",
                            "catalog_rows_with_inat21_filename",
                            "catalog_inat21_rows_by_split",
                            "catalog_inat21_filename_examples",
                            "val_uuid_in_catalog_treeoflife_id",
                            "val_uuid_in_catalog_inat21_filename_stem",
                            "val_file_name_equal_catalog_inat21_filename")})
    # E7
    keep = ("N", "S", "T", "W", "B_iw", "B_su", "within_frac_W_over_T",
            "between_frac_1_minus_W_over_T", "B_su_over_T",
            "within_frac_species_uniform_W_over_W_plus_Bsu",
            "between_frac_species_uniform_Bsu_over_W_plus_Bsu", "identity_check")
    facts["E7_anova"] = dict(
        spec_line_131=dict(bioclip1=[48.6, 51.4], **{"bfl-euclidean": [33.0, 67.0]}),
        inat21_val={m: {k: e["anova_l2_normalized_float64"][k] for k in keep}
                    for m, e in heavy["e5_e7"].items()},
        inat21_val_exploratory_raw_unnormalized={
            m: e["exploratory_anova_raw_unnormalized_float64"]["within_frac_W_over_T"]
            for m, e in heavy["e5_e7"].items()},
        tol_eol_train_bioclip1={k: tol["anova_l2_normalized_float64"][k] for k in keep},
        tol_eol_train_bioclip1_exploratory_raw_unnormalized={
            k: tol["anova_exploratory_raw_unnormalized_float64"][k] for k in keep},
        tol_eol_selection={k: tol[k] for k in (
            "ids_count", "catalog_rows", "catalog_split_counts", "catalog_train_rows",
            "catalog_train_complete_rows", "selected_rows", "n_species",
            "images_per_species", "xcheck_grouping_is_bijection",
            "xcheck_vocab_species_path_mismatches", "catalog_ids_in_both_train_and_val",
            "catalog_train_rank_whitespace_padded_values")},
        tol_eol_bfl_euclidean_cache_exists=Path(
            RES / "probe_cache_b16/tol-eol/bfl-euclidean").exists(),
    )
    facts["extras_outside_assigned_area"] = dict(
        tol_eol_bioclip1_cache_shape=tol["emb_shape"], tol_eol_ids=tol["ids_count"],
        tol_eol_row_norm_all_rows=tol["row_norm_all_rows"],
        tol_eol_cache_rows_train_any_lineage=tol["extra_cache_rows_train_any_lineage"],
        tol_eol_cache_rows_val_any_lineage=tol["extra_cache_rows_val_any_lineage"],
        tol_eol_cache_rows_with_all_7_codes=tol["xcheck_cache_rows_with_all_7_codes"],
        tol_eol_train_complete_distinct_per_rank=tol["extra_distinct_per_rank_on_selected"])
    facts["evidence"] = dict(
        scripts="/projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_*.py, eval_*.sh",
        logs=sorted(str(p) for p in PF.glob("eval_*.log")),
        slurm_jobs={"245649": "eval_inat_heavy (E5 deep, E6, E7-iNat), cpu, 55 s",
                    "245659": "eval_anova_tol (E7-ToL), cpu, 84 s",
                    "245663": "eval_inat_vs_tol_ids (E6 extra), cpu, srun"})
    out = PF / "eval_facts.json"
    out.write_text(json.dumps(facts, indent=1))
    print("wrote", out, f"({out.stat().st_size:,} bytes)")
    print("E1:", facts["E1_table"]["summary"])
    for col, g in facts["E2_gate_targets"].items():
        print(col, {k: (v["n_classes"], v["n_eval"], v["correct"]) for k, v in g.items()
                    if k in RANKS})
    print("other template:")
    for m, g in facts["E2_gate_targets_other_template"].items():
        print(m, g["form"], {k: (v["n_classes"], v["n_eval"], v["correct"])
                             for k, v in g.items() if k in RANKS},
              "avg", round(g["average"]["top1"], 4))
    print("paper vs harness:")
    for col, d in paper_cmp.items():
        print(" ", col, {k: (v["paper"], round(v["harness"], 3)) for k, v in d.items()})
    print("E7 iNat W/T:", {m: round(v["within_frac_W_over_T"], 6)
                           for m, v in facts["E7_anova"]["inat21_val"].items()})
    print("E7 ToL bioclip1 W/T:",
          round(facts["E7_anova"]["tol_eol_train_bioclip1"]["within_frac_W_over_T"], 6),
          "species-uniform W/(W+Bsu):",
          round(facts["E7_anova"]["tol_eol_train_bioclip1"]
                ["within_frac_species_uniform_W_over_W_plus_Bsu"], 6))
    print("bfl-euclidean ToL-EOL cache exists:",
          facts["E7_anova"]["tol_eol_bfl_euclidean_cache_exists"])


if __name__ == "__main__":
    main()
