"""Preflight check E1/E2 helper: flatten the three ZEROSHOT_RANKS JSONs.

Reads only the three result JSONs in tol_embed/results (read-only).
Writes audit/preflight/eval_json_cells.json (every cell, every file) and
prints a cross-file consistency check for (form, model) pairs that appear in
more than one file.

Run on the login node (the JSONs are ~11-14 KB each):
    source "$HOME/.hpc_env.sh" && conda activate bioclip
    python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_dump_jsons.py
"""
import hashlib
import json
from pathlib import Path

BASE = Path("/projects/bdbk/liv/repos/tol_embed/results")
FILES = [
    "ZEROSHOT_RANKS_inat21-val.json",
    "ZEROSHOT_RANKS_inat21-val__forms.json",
    "ZEROSHOT_RANKS_inat21-val__bfl.json",
]
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_json_cells.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def main():
    cells = []          # one dict per (file, form, model, rank)
    meta = {}
    for fn in FILES:
        p = BASE / fn
        raw = p.read_bytes()
        d = json.loads(raw)
        meta[fn] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                        route=d.get("route"), forms=d.get("forms"),
                        limit=d.get("limit"), top_keys=list(d.keys()))
        for form, models in d["results"].items():
            for model, res in models.items():
                for rank in RANKS + ["average"]:
                    r = res[rank]
                    cells.append(dict(file=fn, form=form, model=model, rank=rank,
                                      top1=r.get("top1"),
                                      n_classes=r.get("n_classes"),
                                      n_paths=r.get("n_paths"),
                                      n_eval=r.get("n_eval"),
                                      extra_keys=sorted(set(r) - {"top1", "n_classes",
                                                                  "n_paths", "n_eval"})))
    # cross-file consistency for repeated (form, model, rank)
    by_key = {}
    for c in cells:
        by_key.setdefault((c["form"], c["model"], c["rank"]), []).append(c)
    repeats = []
    for k, lst in sorted(by_key.items()):
        if len(lst) > 1:
            fields = ("top1", "n_classes", "n_paths", "n_eval")
            same = all(all(x[f] == lst[0][f] for f in fields) for x in lst[1:])
            repeats.append(dict(form=k[0], model=k[1], rank=k[2],
                                files=[x["file"] for x in lst],
                                top1=[x["top1"] for x in lst],
                                identical=same))
    OUT.write_text(json.dumps(dict(meta=meta, cells=cells, repeats=repeats), indent=1))
    for fn, m in meta.items():
        print(fn, m)
    print()
    n_rep = len(repeats)
    n_bad = sum(not r["identical"] for r in repeats)
    print(f"repeated (form, model, rank) cells across files: {n_rep}; not identical: {n_bad}")
    for r in repeats:
        if not r["identical"]:
            print("  DIFF", r)
    # print the available (form, model) pairs per file
    print()
    for fn in FILES:
        pairs = sorted({(c["form"], c["model"]) for c in cells if c["file"] == fn})
        print(fn, pairs)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
