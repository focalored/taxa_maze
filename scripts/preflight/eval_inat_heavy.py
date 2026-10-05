"""Preflight checks E5 (deep), E6 and E7-iNat21: runs on a cpu compute node.

E5 deep  For each of the seven iNat21-val caches: load emb.f16.npy in full,
         count non-finite values, full row-norm min/median/max, sha256 of the
         array file (to show the models' arrays are distinct).
E6       Count image files and class folders under /u/liv/bdbk/data/inat21/val;
         compare the on-disk file set with ids.txt and with meta/val.json;
         check every cache label (codes.i32.npy -> vocab.json path) against
         the category lineage in meta/val.json.
E7-iNat  ANOVA split of L2-normalized (float64) image embeddings by species
         (species = codes[:, 6], i.e. the full-lineage species path):
           T  = mean_i ||x_i - xbar||^2                 (total)
           W  = mean_i ||x_i - m_s(i)||^2               (within-species)
           B_iw = T - W                                 (between, image-weighted)
           B_su = mean_s ||m_s - mbar||^2, mbar = mean_s m_s (species-uniform)
         computed directly (two passes), plus identity cross-checks and a few
         labelled exploratory variants (raw / unnormalized rows; unbiased
         denominators). Primary models: bioclip1, bfl-euclidean. The other five
         caches are computed for context.

Reads only allowed artifacts (read-only). Writes
  /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_heavy.json
Launched by scripts/preflight/eval_inat_heavy.sh through srun on the cpu partition.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
CACHES = {
    "bioclip1": RES / "probe_cache_b16/inat21-val/bioclip1",
    "bfl-euclidean": RES / "probe_cache_b16/inat21-val/bfl-euclidean",
    "bfl-hyperbolic": RES / "probe_cache_b16/inat21-val/bfl-hyperbolic",
    "rcme": RES / "probe_cache_b16/inat21-val/rcme",
    "openclip-b16": RES / "probe_cache_b16/inat21-val/openclip-b16",
    "clip-l14-laion2b": RES / "probe_cache_b16/inat21-val/clip-l14-laion2b",
    "bioclip2": RES / "probe_cache/inat21-val/bioclip2",
}
INAT = Path("/u/liv/bdbk/data/inat21")
OUT = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_heavy.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def group_means(X, labels, S):
    """Per-label mean rows and counts (labels in 0..S-1), float64."""
    sums = np.zeros((S, X.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, X)
    cnt = np.bincount(labels, minlength=S).astype(np.float64)
    return sums / cnt[:, None], cnt


def anova(X, labels):
    """Direct two-pass ANOVA traces for rows X (float64) grouped by labels."""
    labels = np.asarray(labels, dtype=np.int64)
    uniq, lab = np.unique(labels, return_inverse=True)
    S, N = uniq.size, X.shape[0]
    xbar = X.mean(axis=0)
    m, cnt = group_means(X, lab, S)
    T = float(np.mean(np.sum((X - xbar) ** 2, axis=1)))
    W = float(np.mean(np.sum((X - m[lab]) ** 2, axis=1)))
    B_iw_direct = float(np.sum(cnt * np.sum((m - xbar) ** 2, axis=1)) / N)
    mbar = m.mean(axis=0)
    B_su = float(np.mean(np.sum((m - mbar) ** 2, axis=1)))
    sq = np.sum(X ** 2, axis=1)
    T_id = float(sq.mean() - xbar @ xbar)
    W_id = float(sq.mean() - np.sum(cnt * np.sum(m ** 2, axis=1)) / N)
    SSW, SSB = W * N, B_iw_direct * N
    W_unb, B_unb = SSW / (N - S), SSB / (S - 1)
    return dict(
        N=int(N), S=int(S), images_per_species_min=int(cnt.min()),
        images_per_species_max=int(cnt.max()),
        T=T, W=W, B_iw=T - W, B_iw_direct=B_iw_direct, B_su=B_su,
        within_frac_W_over_T=W / T, between_frac_1_minus_W_over_T=1 - W / T,
        B_su_over_T=B_su / T,
        within_frac_species_uniform_W_over_W_plus_Bsu=W / (W + B_su),
        between_frac_species_uniform_Bsu_over_W_plus_Bsu=B_su / (W + B_su),
        identity_check=dict(T_identity=T_id, W_identity=W_id,
                            abs_err_T=abs(T_id - T), abs_err_W=abs(W_id - W),
                            abs_err_B=abs((T - W) - B_iw_direct)),
        exploratory_unbiased=dict(W_over_N_minus_S=W_unb, B_over_S_minus_1=B_unb,
                                  within_frac=W_unb / (W_unb + B_unb)),
        mean_row_sq_norm=float(sq.mean()),
    )


def e5_e7():
    out = {}
    codes = np.load(CACHES["bioclip1"] / "codes.i32.npy")
    species = codes[:, 6].astype(np.int64)
    assert (species >= 0).all()
    for name, d in CACHES.items():
        t0 = time.time()
        raw = np.load(d / "emb.f16.npy")
        c = np.load(d / "codes.i32.npy")
        assert np.array_equal(c, codes), f"{name}: codes differ from bioclip1's"
        X = raw.astype(np.float64)
        nonfinite = int((~np.isfinite(X)).sum())
        norms = np.linalg.norm(X, axis=1)
        e = dict(rows=int(X.shape[0]), dim=int(X.shape[1]), dtype=str(raw.dtype),
                 n_nonfinite=nonfinite, n_zero_norm_rows=int((norms == 0).sum()),
                 row_norm_min=float(norms.min()), row_norm_median=float(np.median(norms)),
                 row_norm_max=float(norms.max()), emb_sha256=sha256(d / "emb.f16.npy"))
        Xn = X / norms[:, None]
        e["anova_l2_normalized_float64"] = anova(Xn, species)
        e["exploratory_anova_raw_unnormalized_float64"] = anova(X, species)
        e["seconds"] = time.time() - t0
        out[name] = e
        a = e["anova_l2_normalized_float64"]
        print(f"[E5/E7] {name:16s} rows={e['rows']} dim={e['dim']} nonfinite={nonfinite} "
              f"norm[min/med/max]={e['row_norm_min']:.4f}/{e['row_norm_median']:.4f}/"
              f"{e['row_norm_max']:.4f}  W/T={a['within_frac_W_over_T']:.6f} "
              f"1-W/T={a['between_frac_1_minus_W_over_T']:.6f} "
              f"Bsu/(W+Bsu)={a['between_frac_species_uniform_Bsu_over_W_plus_Bsu']:.6f} "
              f"T={a['T']:.6f} W={a['W']:.6f} Bsu={a['B_su']:.6f}", flush=True)
    return out


def e6():
    t0 = time.time()
    val = INAT / "val"
    folders = sorted(e.name for e in os.scandir(val) if e.is_dir())
    non_dirs = sorted(e.name for e in os.scandir(val) if not e.is_dir())
    per_folder, ext_hist, disk_files = [], {}, set()
    for f in folders:
        names = [e.name for e in os.scandir(val / f) if e.is_file()]
        per_folder.append(len(names))
        for n in names:
            ext = os.path.splitext(n)[1]
            ext_hist[ext] = ext_hist.get(ext, 0) + 1
            disk_files.add(f"val/{f}/{n}")
    prefixes = sorted(int(f.split("_", 1)[0]) for f in folders)
    meta_dir = sorted((p.name, p.stat().st_size) for p in (INAT / "meta").iterdir())

    vj = json.loads((INAT / "meta/val.json").read_text())
    imgs, anns, cats = vj["images"], vj["annotations"], vj["categories"]
    file_names = {im["file_name"] for im in imgs}
    ids = (CACHES["bioclip1"] / "ids.txt").read_text().splitlines()
    ids_set = set(ids)

    # label check: cache codes -> vocab path vs val.json category lineage
    img2file = {im["id"]: im["file_name"] for im in imgs}
    file2cat = {img2file[a["image_id"]]: a["category_id"] for a in anns}
    cat_by_id = {c["id"]: c for c in cats}
    vocab = json.loads((CACHES["bioclip1"] / "vocab.json").read_text())
    V, sep = vocab["vocab"], vocab["sep"]
    codes = np.load(CACHES["bioclip1"] / "codes.i32.npy")
    n_bad = {k: 0 for k in RANKS}
    bad_examples = []
    for i, fid in enumerate(ids):
        c = cat_by_id[file2cat[fid]]
        lin = [c["kingdom"], c["phylum"], c["class"], c["order"], c["family"], c["genus"],
               f"{c['genus']} {c['specific_epithet']}"]
        for r, k in enumerate(RANKS):
            want = sep.join(lin[: r + 1])
            got = V[k][codes[i, r]]
            if want != got:
                n_bad[k] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(dict(row=i, rank=k, want=want, got=got))
    # does the folder name encode the same lineage as val.json?
    folder_mismatch = 0
    for c in cats:
        parts = c["image_dir_name"].split("_")
        if parts[1:] != [c["kingdom"], c["phylum"], c["class"], c["order"], c["family"],
                         c["genus"], c["specific_epithet"]]:
            folder_mismatch += 1
    res = dict(
        val_dir_exists=val.is_dir(),
        n_class_folders=len(folders), n_non_dir_entries_in_val=len(non_dirs),
        n_images_on_disk=sum(per_folder), images_per_folder_min=min(per_folder),
        images_per_folder_max=max(per_folder), extension_histogram=ext_hist,
        folder_prefix_min=prefixes[0], folder_prefix_max=prefixes[-1],
        folder_prefixes_are_0_to_9999=(prefixes == list(range(10000))),
        meta_dir_listing=meta_dir,
        val_json=dict(n_images=len(imgs), n_annotations=len(anns), n_categories=len(cats),
                      n_distinct_file_names=len(file_names), top_keys=list(vj.keys())),
        disk_equals_val_json_file_names=(disk_files == file_names),
        disk_equals_cache_ids=(disk_files == ids_set),
        n_disk_not_in_ids=len(disk_files - ids_set), n_ids_not_on_disk=len(ids_set - disk_files),
        label_mismatches_cache_vs_val_json=n_bad, label_mismatch_examples=bad_examples,
        category_folder_name_vs_fields_mismatches=folder_mismatch,
        seconds=time.time() - t0,
    )
    print("[E6]", json.dumps({k: v for k, v in res.items() if k != "meta_dir_listing"}),
          flush=True)
    print("[E6] meta/:", meta_dir, flush=True)
    return res


def main():
    t0 = time.time()
    print("host", os.uname().nodename, "SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID"),
          flush=True)
    res = dict(host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    res["e6"] = e6()
    res["e5_e7"] = e5_e7()
    res["seconds_total"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, "in", round(res["seconds_total"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
