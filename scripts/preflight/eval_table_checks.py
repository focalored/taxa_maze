"""Preflight checks E1-E4: the baseline table in specs/pilot1.md vs the eval JSONs.

Reads (read-only):
  /projects/bdbk/liv/repos/taxa_maze/specs/pilot1.md  lines 26-35 (the table)
  /projects/bdbk/liv/repos/tol_embed/results/ZEROSHOT_RANKS_inat21-val{,__forms,__bfl}.json
Writes:
  /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_table_checks.json

Light work (three ~12 KB JSONs, one 191-line Markdown file); runs on the login node:
    source "$HOME/.hpc_env.sh" && conda activate bioclip
    python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_table_checks.py

Form names come from tol_embed/scripts/zeroshot_ranks.py render():
  photo   -> f"a photo of {' '.join(lineage[:r+1])}."   (lineage from kingdom down)
  lineage -> ' '.join(lineage[:r+1])                    (no template, no period)
  bare    -> lineage[r] alone
"""
import json
import re
from pathlib import Path

import numpy as np

SPEC = Path("/projects/bdbk/liv/repos/taxa_maze/specs/pilot1.md")
RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
FILES = {
    "main": RES / "ZEROSHOT_RANKS_inat21-val.json",
    "forms": RES / "ZEROSHOT_RANKS_inat21-val__forms.json",
    "bfl": RES / "ZEROSHOT_RANKS_inat21-val__bfl.json",
}
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_table_checks.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# spec column header -> (JSON form name, JSON model key)
COLUMNS = {
    "clip-l14 `photo`": ("photo", "clip-l14-laion2b"),
    "bioclip2 `photo`": ("photo", "bioclip2"),
    "rcme `lineage`": ("lineage", "rcme"),
    "bioclip1 `photo`": ("photo", "bioclip1"),
    "bfl-euc `photo`": ("photo", "bfl-euclidean"),
    "bfl-hyp `photo`": ("photo", "bfl-hyperbolic"),
}


def parse_spec_table():
    lines = SPEC.read_text().splitlines()
    block = lines[25:35]                     # 1-indexed lines 26..35
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block]
    header, body = rows[0], rows[2:]
    assert header[0] == "rank", header
    assert re.fullmatch(r"-+", rows[1][0]), rows[1]
    cols = header[1:]
    table, bold = {}, {}
    for r in body:
        rank = r[0].replace("*", "").strip()
        for c, cell in zip(cols, r[1:]):
            table.setdefault(c, {})[rank] = cell.replace("*", "").strip()
            bold.setdefault(c, {})[rank] = cell.startswith("**")
    return cols, table, bold, {i + 26: block[i] for i in range(len(block))}


def load_results():
    """(form, model) -> result dict, plus which files hold that pair."""
    res, where = {}, {}
    for tag, p in FILES.items():
        d = json.loads(p.read_text())
        for form, models in d["results"].items():
            for model, r in models.items():
                key = (form, model)
                if key in res:
                    assert res[key] == r, f"cross-file mismatch for {key}"
                res[key] = r
                where.setdefault(key, []).append(p.name)
    return res, where


def exact_correct(top1, n_eval):
    """Recover the integer correct count from top1 = 100.0 * correct / n_eval."""
    c = int(round(top1 * n_eval / 100.0))
    return c, (100.0 * c / max(n_eval, 1)) == top1


def main():
    cols, spec, bold, raw_lines = parse_spec_table()
    res, where = load_results()
    assert cols == list(COLUMNS), cols
    out = dict(spec_table_lines=raw_lines, column_map={}, e1=[], e2=[], e1_summary={},
               average_check=[], best_form=[], bold_check=[], e3={}, e4={})

    # ---------------- E1 + E2 ----------------
    n_match = n_mis = 0
    for col, (form, model) in COLUMNS.items():
        r = res[(form, model)]
        out["column_map"][col] = dict(form=form, model=model, files=where[(form, model)])
        for rank in RANKS + ["average"]:
            js = r[rank]["top1"]
            js_str = f"{js:.2f}"
            sp = spec[col][rank]
            ok = js_str == sp
            n_match += ok
            n_mis += not ok
            out["e1"].append(dict(column=col, form=form, model=model, rank=rank,
                                  spec=sp, json_top1=js, json_2dp=js_str,
                                  abs_diff=round(abs(js - float(sp)), 6),
                                  status="MATCH" if ok else "MISMATCH"))
            if rank != "average":
                c, exact = exact_correct(js, r[rank]["n_eval"])
                out["e2"].append(dict(column=col, form=form, model=model, rank=rank,
                                      n_classes=r[rank]["n_classes"],
                                      n_paths=r[rank]["n_paths"],
                                      n_eval=r[rank]["n_eval"], correct=c,
                                      correct_recovers_top1_bit_exact=exact,
                                      top1=js))
        accs = [r[k]["top1"] for k in RANKS]
        mean7 = float(np.mean(accs))
        spec_mean = float(np.mean([float(spec[col][k]) for k in RANKS]))
        out["average_check"].append(dict(
            column=col, json_average=r["average"]["top1"], mean_of_7_json=mean7,
            json_average_equals_mean7=(mean7 == r["average"]["top1"]),
            abs_diff_json=abs(mean7 - r["average"]["top1"]),
            spec_average=spec[col]["average"],
            mean_of_7_spec_printed=round(spec_mean, 4),
            mean_of_7_spec_printed_2dp=f"{spec_mean:.2f}"))
    out["e1_summary"] = dict(n_cells=n_match + n_mis, n_match=n_match, n_mismatch=n_mis)

    # ---------------- best form (spec line 22) ----------------
    for col, (form, model) in COLUMNS.items():
        avgs = {f: res[(f, model)]["average"]["top1"]
                for f in ("lineage", "photo", "bare") if (f, model) in res}
        best = max(avgs, key=avgs.get)
        out["best_form"].append(dict(column=col, model=model, spec_form=form,
                                     averages_by_form=avgs, best_form=best,
                                     spec_form_is_best=(best == form)))
    # openclip-b16 is in the JSONs but is not a spec column; record for context
    out["openclip_b16_context"] = {
        f: res[(f, "openclip-b16")]["average"]["top1"]
        for f in ("lineage", "photo", "bare") if (f, "openclip-b16") in res}

    # ---------------- bold markers = row maximum? ----------------
    for rank in RANKS + ["average"]:
        vals = {col: res[COLUMNS[col]][rank]["top1"] for col in COLUMNS}
        argmax = max(vals, key=vals.get)
        bolded = [c for c in COLUMNS if bold[c][rank]]
        out["bold_check"].append(dict(rank=rank, bolded=bolded, json_row_max=argmax,
                                      consistent=(bolded == [argmax])))

    # ---------------- E3: prose claims at spec line 37 ----------------
    coarse = RANKS[:4]
    g = lambda form, model, rank: res[(form, model)][rank]["top1"]  # noqa: E731
    e3 = {}
    for model in ("bfl-euclidean", "bfl-hyperbolic"):
        v = {k: g("photo", model, k) for k in coarse}
        e3[f"{model}_photo_kingdom_to_order"] = dict(
            values=v, mean=float(np.mean(list(v.values()))),
            min=min(v.values()), max=max(v.values()))
    rc_sp = g("lineage", "rcme", "species")
    e3["species_gap_rcme_lineage_minus_bfl_photo"] = {
        m: dict(rcme_lineage=rc_sp, bfl_photo=g("photo", m, "species"),
                gap=rc_sp - g("photo", m, "species"))
        for m in ("bfl-euclidean", "bfl-hyperbolic")}
    # same comparison with both models under the same template, for context only
    e3["species_gap_same_template_context"] = {
        "rcme_photo_minus_bfl-euclidean_photo":
            g("photo", "rcme", "species") - g("photo", "bfl-euclidean", "species"),
        "rcme_lineage_minus_bfl-euclidean_lineage":
            g("lineage", "rcme", "species") - g("lineage", "bfl-euclidean", "species"),
        "rcme_photo_species": g("photo", "rcme", "species"),
        "bfl-euclidean_lineage_species": g("lineage", "bfl-euclidean", "species"),
    }
    out["e3"] = e3

    # ---------------- E4: decision constants (tabulate only) ----------------
    e4 = {"species_floor_bioclip1_photo": dict(
        exact=g("photo", "bioclip1", "species"),
        two_dp=f"{g('photo', 'bioclip1', 'species'):.2f}",
        correct_over_n_eval=exact_correct(g("photo", "bioclip1", "species"),
                                          res[("photo", "bioclip1")]["species"]["n_eval"])[0],
        n_eval=res[("photo", "bioclip1")]["species"]["n_eval"],
        spec_line_174="70.19")}
    e4["coarse_refs"] = {
        f"{form}:{model}": {k: dict(top1=g(form, model, k),
                                    minus1=g(form, model, k) - 1.0,
                                    plus1=g(form, model, k) + 1.0) for k in coarse}
        for form, model in (("lineage", "rcme"), ("photo", "bfl-euclidean"))}
    e4["fine_refs"] = {
        f"photo:{model}": {k: dict(top1=g("photo", model, k),
                                   minus1=g("photo", model, k) - 1.0,
                                   plus1=g("photo", model, k) + 1.0)
                           for k in ("genus", "species")}
        for model in ("bfl-euclidean", "bfl-hyperbolic", "bioclip1")}
    # context only: rcme under photo, the template the pilot calls decisive (line 171)
    e4["context_rcme_photo_coarse"] = {k: g("photo", "rcme", k) for k in coarse}
    out["e4"] = e4

    OUT.write_text(json.dumps(out, indent=1))

    # ---------------- print ----------------
    print("Column map (spec header -> JSON form, model, files):")
    for c, m in out["column_map"].items():
        print(f"  {c:18s} -> {m}")
    print("\nE1: spec vs JSON (2 dp)")
    hdr = "rank".ljust(9) + "".join(c.split()[0].ljust(22) for c in COLUMNS)
    print(hdr)
    for rank in RANKS + ["average"]:
        cells = [e for e in out["e1"] if e["rank"] == rank]
        print(rank.ljust(9) + "".join(
            f"{e['spec']}/{e['json_2dp']} {'ok' if e['status'] == 'MATCH' else 'XX'}".ljust(22)
            for e in cells))
    print("E1 summary:", out["e1_summary"])
    print("\nE2: n_classes / n_paths / n_eval / correct (bit-exact recovery)")
    for e in out["e2"]:
        print(f"  {e['column']:18s} {e['rank']:8s} n_classes={e['n_classes']:6d} "
              f"n_paths={e['n_paths']:6d} n_eval={e['n_eval']:6d} correct={e['correct']:6d} "
              f"exact={e['correct_recovers_top1_bit_exact']} top1={e['top1']!r}")
    print("\naverage check:")
    for a in out["average_check"]:
        print("  ", a)
    print("\nbest form:")
    for b in out["best_form"]:
        print("  ", b)
    print("  openclip-b16 (not a spec column):", out["openclip_b16_context"])
    print("\nbold check:")
    for b in out["bold_check"]:
        print("  ", b)
    print("\nE3:", json.dumps(e3, indent=1))
    print("\nE4:", json.dumps(e4, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
