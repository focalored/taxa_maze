"""Preflight check E5 (light part): inventory of the seven iNat21-val caches.

For each cache directory (read-only):
  - files present (name, bytes), sub-directories and their files
  - emb.f16.npy and codes.i32.npy header: shape, dtype (header only, no data read)
  - ids.txt line count, distinct count, sha256 (vs manifest ids_sha256)
  - codes.i32.npy (100,000 x 7 int32, 2.8 MB): valid (>= 0) count per rank
  - vocab.json: entries per rank; distinct rendered strings under the
    photo and lineage forms (re-implemented here from the documented render()
    rules in tol_embed/scripts/zeroshot_ranks.py; nothing is imported from tol_embed)
  - parents.i32.npz: array names, shapes, dtypes
  - manifest.json: selected provenance fields
Then compares n_eval / n_paths / n_classes to the eval JSONs, and checks that
ids, codes and vocab are byte-identical across caches.

The six spec columns use bioclip2 (probe_cache/) and clip-l14-laion2b, rcme,
bioclip1, bfl-euclidean, bfl-hyperbolic (probe_cache_b16/). openclip-b16 is
inventoried too because the task's input list names it; it is not a spec column.

Light work (small files and npy headers); runs on the login node:
    source "$HOME/.hpc_env.sh" && conda activate bioclip
    python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_cache_inventory.py
Writes /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_cache_inventory.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib import format as npf

RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
CACHES = {
    "bioclip2": RES / "probe_cache/inat21-val/bioclip2",
    "clip-l14-laion2b": RES / "probe_cache_b16/inat21-val/clip-l14-laion2b",
    "rcme": RES / "probe_cache_b16/inat21-val/rcme",
    "bioclip1": RES / "probe_cache_b16/inat21-val/bioclip1",
    "bfl-euclidean": RES / "probe_cache_b16/inat21-val/bfl-euclidean",
    "bfl-hyperbolic": RES / "probe_cache_b16/inat21-val/bfl-hyperbolic",
    "openclip-b16": RES / "probe_cache_b16/inat21-val/openclip-b16",
}
SPEC_COLUMN = {"bioclip2": ("photo", True), "clip-l14-laion2b": ("photo", True),
               "rcme": ("lineage", True), "bioclip1": ("photo", True),
               "bfl-euclidean": ("photo", True), "bfl-hyperbolic": ("photo", True),
               "openclip-b16": (None, False)}
JSONS = [RES / "ZEROSHOT_RANKS_inat21-val.json",
         RES / "ZEROSHOT_RANKS_inat21-val__forms.json",
         RES / "ZEROSHOT_RANKS_inat21-val__bfl.json"]
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_cache_inventory.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
MANIFEST_KEYS = ["dataset", "checkpoint", "model", "N", "D", "dtype", "precision",
                 "normalized", "ids_sha256", "image_root", "git_commit", "slurm_job_id",
                 "written_utc", "weight_sha256", "upstream", "stats"]


def npy_header(p):
    with open(p, "rb") as f:
        v = npf.read_magic(f)
        shape, fortran, dtype = npf._read_array_header(f, v)
    return dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(path, r, form, sep="|"):
    """Same rules as zeroshot_ranks.render (documented there); re-implemented."""
    parts = path.split(sep)[: r + 1]
    if r == len(RANKS) - 1 and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    if form == "bare":
        return parts[-1]
    text = " ".join(parts)
    return f"a photo of {text}." if form == "photo" else text


def json_results():
    res = {}
    for p in JSONS:
        for form, models in json.loads(p.read_text())["results"].items():
            for model, r in models.items():
                res[(form, model)] = r
    return res


def main():
    jres = json_results()
    inv, hashes = {}, {}
    for name, d in CACHES.items():
        e = dict(path=str(d), files={}, subdirs={})
        for p in sorted(d.iterdir()):
            if p.is_dir():
                e["subdirs"][p.name] = {q.name: q.stat().st_size for q in sorted(p.iterdir())}
            else:
                e["files"][p.name] = p.stat().st_size
        e["emb_header"] = npy_header(d / "emb.f16.npy")
        e["codes_header"] = npy_header(d / "codes.i32.npy")
        ids = (d / "ids.txt").read_text().splitlines()
        e["ids_count"] = len(ids)
        e["ids_distinct"] = len(set(ids))
        e["ids_sha256_file"] = sha256(d / "ids.txt")
        e["ids_first"] = ids[0]
        e["ids_class_folders_distinct"] = len({i.split("/")[1] for i in ids})
        e["ids_prefix_val_all"] = all(i.startswith("val/") for i in ids)
        codes = np.load(d / "codes.i32.npy")
        e["codes_valid_per_rank"] = {k: int((codes[:, r] >= 0).sum())
                                     for r, k in enumerate(RANKS)}
        e["codes_distinct_per_rank"] = {k: int(np.unique(codes[:, r][codes[:, r] >= 0]).size)
                                        for r, k in enumerate(RANKS)}
        vocab = json.loads((d / "vocab.json").read_text())
        V, sep = vocab["vocab"], vocab.get("sep", "|")
        e["vocab_keys"] = list(vocab.keys())
        e["vocab_sep"] = sep
        e["vocab_ranks"] = vocab.get("ranks")
        e["vocab_n_per_rank"] = {k: len(V[k]) for k in RANKS}
        e["vocab_species_example"] = V["species"][0]
        e["rendered_distinct"] = {f: {k: len({render(p, r, f, sep) for p in V[k]})
                                      for r, k in enumerate(RANKS)}
                                  for f in ("photo", "lineage", "bare")}
        e["rendered_example_photo"] = {k: render(V["species"][0], r, "photo", sep)
                                       for r, k in enumerate(RANKS)}
        par = np.load(d / "parents.i32.npz")
        e["parents_npz"] = {k: dict(shape=list(par[k].shape), dtype=str(par[k].dtype))
                            for k in par.files}
        man = json.loads((d / "manifest.json").read_text())
        e["manifest"] = {k: man.get(k) for k in MANIFEST_KEYS}
        e["manifest_ids_sha256_matches_file"] = man.get("ids_sha256") == e["ids_sha256_file"]
        cr = json.loads((d / "codes_report.json").read_text())
        e["codes_report"] = {k: cr.get(k) for k in ("dataset", "mode", "n_rows",
                                                     "n_distinct_ids", "n_duplicate_ids",
                                                     "n_ids_with_no_metadata_row",
                                                     "n_rows_truncated_at_gap", "per_rank",
                                                     "source")}
        # compare to the JSON for the spec column's form (and every form present)
        cmp = {}
        for form in ("photo", "lineage", "bare"):
            r = jres.get((form, name))
            if r is None:
                continue
            cmp[form] = {k: dict(json_n_eval=r[k]["n_eval"],
                                 codes_valid=e["codes_valid_per_rank"][k],
                                 json_n_paths=r[k]["n_paths"],
                                 vocab_n=e["vocab_n_per_rank"][k],
                                 json_n_classes=r[k]["n_classes"],
                                 rendered_distinct=e["rendered_distinct"][form][k],
                                 all_equal=(r[k]["n_eval"] == e["codes_valid_per_rank"][k]
                                            and r[k]["n_paths"] == e["vocab_n_per_rank"][k]
                                            and r[k]["n_classes"]
                                            == e["rendered_distinct"][form][k]))
                         for k in RANKS}
        e["json_comparison"] = cmp
        e["emb_rows_equal_json_n_eval_all_ranks"] = all(
            e["emb_header"]["shape"][0] == c["json_n_eval"]
            for f in cmp.values() for c in f.values())
        e["spec_column_form"], e["is_spec_column"] = SPEC_COLUMN[name]
        inv[name] = e
        hashes[name] = dict(ids=e["ids_sha256_file"], codes=sha256(d / "codes.i32.npy"),
                            vocab=sha256(d / "vocab.json"),
                            parents=sha256(d / "parents.i32.npz"))
    same = {k: len({h[k] for h in hashes.values()}) == 1 for k in ("ids", "codes", "vocab",
                                                                   "parents")}
    OUT.write_text(json.dumps(dict(caches=inv, hashes=hashes, identical_across_caches=same),
                              indent=1))
    for name, e in inv.items():
        print(f"== {name}  ({e['path']})")
        print("   files:", e["files"])
        if e["subdirs"]:
            print("   subdirs:", e["subdirs"])
        print("   emb:", e["emb_header"], " codes:", e["codes_header"])
        print("   ids:", e["ids_count"], "distinct", e["ids_distinct"], "class folders",
              e["ids_class_folders_distinct"], "manifest sha match",
              e["manifest_ids_sha256_matches_file"])
        print("   codes valid per rank:", e["codes_valid_per_rank"])
        print("   codes distinct per rank:", e["codes_distinct_per_rank"])
        print("   vocab n per rank:", e["vocab_n_per_rank"], " keys:", e["vocab_keys"])
        print("   rendered distinct photo:", e["rendered_distinct"]["photo"])
        print("   rendered distinct lineage:", e["rendered_distinct"]["lineage"])
        print("   rendered distinct bare:", e["rendered_distinct"]["bare"])
        print("   parents:", e["parents_npz"])
        print("   manifest:", {k: e["manifest"][k] for k in ("model", "N", "D", "dtype",
                                                           "precision", "normalized",
                                                           "written_utc", "slurm_job_id")})
        print("   manifest stats:", e["manifest"]["stats"])
        for f, c in e["json_comparison"].items():
            print(f"   json[{f}] all_equal per rank:",
                  {k: v["all_equal"] for k, v in c.items()})
        print("   emb rows == json n_eval (all ranks, all forms):",
              e["emb_rows_equal_json_n_eval_all_ranks"])
    print("\nidentical across the 7 caches:", same)
    print("example photo strings:", inv["bioclip1"]["rendered_example_photo"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
