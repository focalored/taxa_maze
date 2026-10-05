#!/usr/bin/env python3
"""
refbeh_zs_static.py -- light, CPU-only checks of the zero-shot reference JSONs.

Part of the 2026-10-02 reference-behaviour audit
(audit/2026-10-02_reference_behaviour.md). It reads only small files:
  - the three ZEROSHOT_RANKS_inat21-val*.json files (about 11-14 KB each),
  - vocab.json (1.3 MB) and codes.i32.npy (2.8 MB) of the inat21-val caches.

It does NOT import anything from tol_embed. render() below is our own
re-implementation of the class-string rule documented at
tol_embed/scripts/zeroshot_ranks.py:204-222, so that a mismatch here would
show up as a mismatch against the JSONs.

What it checks:
  1. JSON layout: top-level keys, forms, model lists, per-cell fields.
  2. Implied correct counts: correct = top1 * n_eval / 100 must be an integer.
  3. "average" == mean of the seven rank top1 values.
  4. n_paths, n_classes and n_eval recomputed from vocab.json + codes.i32.npy
     with our render(), compared to every JSON cell.
  5. Cells that appear in more than one JSON agree with each other.
  6. Example class strings (fox and two others) for every rank and form.

Output: audit/preflight/refbeh_zs_static.json (machine-readable) and stdout.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
JSONS = [
    "ZEROSHOT_RANKS_inat21-val.json",
    "ZEROSHOT_RANKS_inat21-val__forms.json",
    "ZEROSHOT_RANKS_inat21-val__bfl.json",
]
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_zs_static.json")


def cache_dir(ckpt):
    # Same rule as data_paths.cache_root_for (data_paths.py:411-414),
    # re-stated here rather than imported.
    root = RES / ("probe_cache" if ckpt == "bioclip2" else "probe_cache_b16")
    return root / "inat21-val" / ckpt


def render(path, rank_idx, form, sep="|"):
    """Our re-implementation of the reference class-string rule."""
    parts = path.split(sep)[: rank_idx + 1]
    if rank_idx == len(RANKS) - 1 and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    if form == "bare":
        return parts[-1]
    text = " ".join(parts)
    return f"a photo of {text}." if form == "photo" else text


def main():
    out = {"jsons": {}, "recomputed": {}, "mismatches": [], "cross_json": [], "examples": {}}

    # ---- 1-3: JSON layout, implied counts, averages -------------------------------
    cells = {}  # (json, form, ckpt) -> dict
    for name in JSONS:
        j = json.loads((RES / name).read_text())
        info = {
            "top_keys": list(j),
            "route": j.get("route"),
            "forms": j.get("forms"),
            "limit": j.get("limit"),
            "models_per_form": {f: list(d) for f, d in j["results"].items()},
            "has_self_reported": "self_reported" in j,
            "cell_fields": sorted(next(iter(next(iter(j["results"].values())).values()))["kingdom"]),
        }
        max_int_err = 0.0
        max_avg_err = 0.0
        decimals = set()
        for form, d in j["results"].items():
            for ck, res in d.items():
                row = {}
                for r in RANKS:
                    c = res[r]
                    implied = c["top1"] * c["n_eval"] / 100.0
                    err = abs(implied - round(implied))
                    max_int_err = max(max_int_err, err)
                    s = repr(c["top1"])
                    decimals.add(len(s.split(".")[1]) if "." in s else 0)
                    row[r] = dict(top1=c["top1"], n_classes=c["n_classes"], n_paths=c["n_paths"],
                                  n_eval=c["n_eval"], correct=int(round(implied)))
                mean7 = float(np.mean([res[r]["top1"] for r in RANKS]))
                avg = res["average"]["top1"]
                max_avg_err = max(max_avg_err, abs(mean7 - avg))
                row["average"] = dict(top1=avg, mean_of_7=mean7)
                cells[(name, form, ck)] = row
        info["max_abs_error_correct_integrality"] = max_int_err
        info["max_abs_error_average_vs_mean7"] = max_avg_err
        info["top1_decimal_digits_seen"] = sorted(decimals)
        out["jsons"][name] = info

    # ---- 4: recompute n_paths, n_classes, n_eval from vocab + codes --------------------
    # All seven inat21-val caches hold byte-identical vocab.json and codes.i32.npy
    # (sha256 checked separately); use each model's own files anyway, as the
    # reference does (zeroshot_ranks.py:245-249).
    recomputed = {}
    for (name, form, ck), row in cells.items():
        key = (ck, form)
        if key not in recomputed:
            cdir = cache_dir(ck)
            vocab = json.loads((cdir / "vocab.json").read_text())
            codes = np.load(cdir / "codes.i32.npy", mmap_mode="r")
            sep, V = vocab.get("sep", "|"), vocab["vocab"]
            rec = {}
            for r, rank in enumerate(RANKS):
                paths = V[rank]
                strings = [render(p, r, form, sep) for p in paths]
                uniq = list(dict.fromkeys(strings))
                n_eval = int((np.asarray(codes[:, r]) >= 0).sum())
                rec[rank] = dict(n_paths=len(paths), n_classes=len(uniq), n_eval=n_eval)
            recomputed[key] = rec
        rec = recomputed[key]
        for rank in RANKS:
            for f in ("n_paths", "n_classes", "n_eval"):
                if rec[rank][f] != row[rank][f]:
                    out["mismatches"].append(dict(json=name, form=form, ckpt=ck, rank=rank, field=f,
                                                  json_value=row[rank][f], recomputed=rec[rank][f]))
    out["recomputed"] = {f"{ck}|{form}": rec for (ck, form), rec in recomputed.items()}

    # ---- 5: cross-JSON agreement for cells present in more than one file ------------------
    by = {}
    for (name, form, ck), row in cells.items():
        by.setdefault((form, ck), []).append((name, row))
    for (form, ck), lst in sorted(by.items()):
        if len(lst) < 2:
            continue
        ref_name, ref = lst[0]
        for name, row in lst[1:]:
            diffs = {r: (ref[r]["correct"], row[r]["correct"]) for r in RANKS
                     if ref[r]["correct"] != row[r]["correct"] or ref[r]["n_classes"] != row[r]["n_classes"]}
            out["cross_json"].append(dict(form=form, ckpt=ck, files=[ref_name, name],
                                          identical=not diffs, diffs=diffs))

    # ---- 6: example class strings ---------------------------------------------------------
    vocab = json.loads((cache_dir("bioclip1") / "vocab.json").read_text())
    species = vocab["vocab"]["species"]
    wanted = ["Vulpes vulpes", "Melospiza melodia", "Amanita muscaria"]
    for w in wanted:
        hits = [p for p in species if p.endswith("|" + w)]
        if not hits:
            out["examples"][w] = "not in inat21-val vocab"
            continue
        p = hits[0]
        ex = {"vocab_path": p}
        for form in ("photo", "lineage", "bare"):
            ex[form] = {rank: render(p, r, form) for r, rank in enumerate(RANKS)}
        out["examples"][w] = ex

    # ---- table of correct counts for the six spec columns --------------------------------
    spec_cols = [("clip-l14-laion2b", "photo"), ("bioclip2", "photo"), ("rcme", "lineage"),
                 ("bioclip1", "photo"), ("bfl-euclidean", "photo"), ("bfl-hyperbolic", "photo")]
    six = {}
    for ck, form in spec_cols:
        srcs = [(name, row) for (name, f, c), row in cells.items() if c == ck and f == form]
        six[f"{ck}|{form}"] = {
            "found_in": [n for n, _ in srcs],
            "correct": {r: srcs[0][1][r]["correct"] for r in RANKS} if srcs else None,
            "top1": {r: srcs[0][1][r]["top1"] for r in RANKS} if srcs else None,
            "n_classes": {r: srcs[0][1][r]["n_classes"] for r in RANKS} if srcs else None,
            "n_eval": {r: srcs[0][1][r]["n_eval"] for r in RANKS} if srcs else None,
            "average": srcs[0][1]["average"] if srcs else None,
        }
    out["spec_six_columns"] = six
    out["all_cells"] = {f"{n}|{f}|{c}": row for (n, f, c), row in cells.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    # ---- stdout summary -------------------------------------------------------------------
    for name, info in out["jsons"].items():
        print(f"== {name}")
        for k, v in info.items():
            print(f"   {k}: {v}")
    print(f"\nrecomputed (ckpt|form) combos: {len(recomputed)}; field mismatches vs JSON: {len(out['mismatches'])}")
    for m in out["mismatches"][:20]:
        print("   MISMATCH", m)
    some = recomputed[next(iter(recomputed))]
    for form in ("lineage", "photo", "bare"):
        k = next((k for k in recomputed if k[1] == form), None)
        if k:
            print(f"   n_classes per rank, form={form}: " +
                  ", ".join(f"{r}={recomputed[k][r]['n_classes']}" for r in RANKS) +
                  f"   (n_paths: {', '.join(str(recomputed[k][r]['n_paths']) for r in RANKS)};"
                  f" n_eval: {sorted({recomputed[k][r]['n_eval'] for r in RANKS})})")
    print("\ncross-JSON duplicated cells:")
    for c in out["cross_json"]:
        print("  ", c["form"], c["ckpt"], c["files"], "identical" if c["identical"] else f"DIFF {c['diffs']}")
    print("\nexamples:")
    for w, ex in out["examples"].items():
        print(f"  {w}: {ex if isinstance(ex, str) else ex['vocab_path']}")
        if isinstance(ex, dict):
            for form in ("photo", "lineage", "bare"):
                for rank in ("kingdom", "order", "genus", "species"):
                    print(f"     {form:8s} {rank:8s} {ex[form][rank]!r}")
    print("\nspec six columns (correct counts derived as top1*n_eval/100):")
    for k, v in six.items():
        print(f"   {k:26s} in {v['found_in']}  correct={v['correct']}  avg={v['average']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
