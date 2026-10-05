#!/usr/bin/env python3
"""
refbeh_cpu.py -- CPU-only checks for the 2026-10-02 reference-behaviour audit
(audit/2026-10-02_reference_behaviour.md). Run through Slurm (refbeh_cpu.sh),
never on the login node: it loads catalog.csv (2 GB), val.json (44 MB),
the ToL ids.txt (230 MB) and codes (174 MB), and streams tar.gz shards.

Nothing here imports tol_embed code. Reads only:
  - /u/liv/bdbk/data/inat21/meta/val.json and image headers under /u/liv/bdbk/data/inat21
  - /u/liv/bdbk/data/tol10m/dataset/EOL/image_set_*.tar.gz (streamed, read-only)
  - /u/liv/bdbk/data/tol10m/metadata/catalog.csv
  - cache artifacts (ids.txt, codes.i32.npy, vocab.json, shards/*.ids.txt,
    shards_index.json) under tol_embed/results/probe_cache*/
  - checkpoint weight files under /u/liv/bdbk/ckpt and the HF cache (hashing only)
Writes only: audit/preflight/refbeh_cpu.json

Parts:
  A  iNat21-val: ids.txt order vs val.json, label/code alignment, file existence,
     image modes and EXIF orientation on a sample.
  B  ToL shards: gzip magic, tar member names/types, ids.txt order vs tar order,
     raw stream rate, decode and transform cost per image, image modes and EXIF.
  C  catalog.csv: join cache uuids on treeoflife_id, split per shard, codes vs
     catalog lineage.
  D  sha256 of checkpoint weight files vs SHA256SUMS and cache manifests.
"""

import hashlib
import io
import json
import os
import platform
import re
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TAXA = Path("/projects/bdbk/liv/repos/taxa_maze")
sys.path.insert(0, str(TAXA))  # for src.open_clip (the copied package)

OUT = TAXA / "audit/preflight/refbeh_cpu.json"
RES = Path("/projects/bdbk/liv/repos/tol_embed/results")
INAT = Path("/u/liv/bdbk/data/inat21")
EOL = Path("/u/liv/bdbk/data/tol10m/dataset/EOL")
CATALOG = Path("/u/liv/bdbk/data/tol10m/metadata/catalog.csv")
TOLC = RES / "probe_cache_b16/tol-eol/bioclip1"
INATC = RES / "probe_cache_b16/inat21-val/bioclip1"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
UUID_JPG = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg")

out = {"env": {}, "A_inat21": {}, "B_tol_shards": {}, "C_catalog": {}, "D_weights": {}}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")


def wds_key(name):
    """webdataset 0.2.86 base_plus_ext (tariterators.py:25-39): key, lowercased suffix."""
    m = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", name)
    return (None, None) if not m else (m.group(1), m.group(2).lower())


def env():
    import PIL, torch, torchvision, webdataset, polars
    out["env"] = dict(python=platform.python_version(), host=platform.node(),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      numpy=np.__version__, torch=torch.__version__, torchvision=torchvision.__version__,
                      pillow=PIL.__version__, webdataset=webdataset.__version__, polars=polars.__version__,
                      cpus=len(os.sched_getaffinity(0)))
    log("env", out["env"])


# ---------------------------------------------------------------------------
# A. iNat21-val
# ---------------------------------------------------------------------------
def part_a():
    from PIL import Image
    A = out["A_inat21"]
    t0 = time.time()
    meta = json.loads((INAT / "meta/val.json").read_text())
    A["val_json_keys"] = sorted(meta)
    images, anns, cats = meta["images"], meta["annotations"], meta["categories"]
    A["n_images"], A["n_annotations"], A["n_categories"] = len(images), len(anns), len(cats)
    A["image_fields"] = sorted(images[0])
    A["category_fields"] = sorted(cats[0])
    file_names = [im["file_name"] for im in images]
    ids = (INATC / "ids.txt").read_text(encoding="utf-8").splitlines()
    A["ids_txt_n"] = len(ids)
    A["ids_txt_equals_val_json_images_order"] = ids == file_names
    A["first_file_names"] = file_names[:3]
    cat_of = {a["image_id"]: a["category_id"] for a in anns}
    A["annotation_image_ids_unique"] = len(cat_of) == len(anns)
    A["every_image_has_annotation"] = all(im["id"] in cat_of for im in images)
    A["images_order_equals_annotations_order"] = [im["id"] for im in images] == [a["image_id"] for a in anns]
    catd = {c["id"]: c for c in cats}
    A["category_ids_are_0_to_n_minus_1"] = sorted(catd) == list(range(len(cats)))

    vocab = json.loads((INATC / "vocab.json").read_text())
    V = vocab["vocab"]
    codes = np.load(INATC / "codes.i32.npy")
    mism, examples = 0, []
    species_code_eq_cat = True
    dir_matches_cat = 0
    for row, im in enumerate(images):
        cid = cat_of[im["id"]]
        c = catd[cid]
        comps = [c["kingdom"], c["phylum"], c["class"], c["order"], c["family"], c["genus"],
                 f'{c["genus"]} {c["specific_epithet"]}']
        for r in range(7):
            path = "|".join(comps[: r + 1])
            got = V[RANKS[r]][codes[row, r]]
            if got != path:
                mism += 1
                if len(examples) < 5:
                    examples.append(dict(row=row, rank=RANKS[r], vocab=got, from_json=path))
        if int(codes[row, 6]) != int(cid):
            species_code_eq_cat = False
        if im["file_name"].split("/")[1] == c.get("image_dir_name"):
            dir_matches_cat += 1
    A["codes_vs_val_json_lineage_mismatches"] = mism
    A["codes_vs_val_json_lineage_mismatch_examples"] = examples
    A["species_code_equals_category_id_for_all_rows"] = species_code_eq_cat
    A["file_name_dir_equals_category_image_dir_name"] = dir_matches_cat
    # vocab species order vs category-id order
    sp_by_cat = ["|".join([catd[k]["kingdom"], catd[k]["phylum"], catd[k]["class"], catd[k]["order"],
                           catd[k]["family"], catd[k]["genus"],
                           f'{catd[k]["genus"]} {catd[k]["specific_epithet"]}']) for k in range(len(cats))]
    A["vocab_species_order_equals_category_id_order"] = sp_by_cat == V["species"]
    log("A: val.json/ids/codes checks done in %.1fs" % (time.time() - t0))

    # every file exists under the join root /u/liv/bdbk/data/inat21 (not .../inat21/val)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as pool:
        exists = list(pool.map(lambda n: (INAT / n).is_file(), file_names))
    A["n_files_exist_under_inat21_root"] = int(sum(exists))
    A["n_files_exist_under_inat21_val_root"] = int(sum((INAT / "val" / n).is_file() for n in file_names[:200]))
    A["exist_check_seconds"] = time.time() - t0
    log("A: existence", A["n_files_exist_under_inat21_root"], "of", len(file_names))

    # modes / EXIF on a seeded sample of 2,000 (header-only opens)
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(len(file_names), size=2000, replace=False).tolist())
    modes, orient, fmts = Counter(), Counter(), Counter()
    for i in sample:
        with Image.open(INAT / file_names[i]) as im:
            modes[im.mode] += 1
            fmts[im.format] += 1
            try:
                orient[int(im.getexif().get(274, 1))] += 1
            except Exception:
                orient["unreadable"] += 1
    A["sample2000_modes"] = dict(modes)
    A["sample2000_formats"] = dict(fmts)
    A["sample2000_exif_orientation"] = {str(k): v for k, v in orient.items()}
    save()


# ---------------------------------------------------------------------------
# B. ToL shards
# ---------------------------------------------------------------------------
def stream_members(path, max_members=None, keep_first_bytes=0):
    """Stream a .tar.gz with tarfile 'r|gz' (sequential, like webdataset's 'r|*')."""
    t0 = time.time()
    names, types, nbytes, kept = [], Counter(), 0, []
    with open(path, "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for ti in tf:
                types["reg" if ti.isreg() else ("dir" if ti.isdir() else f"type_{ti.type!r}")] += 1
                names.append(ti.name)
                if ti.isreg():
                    data = tf.extractfile(ti).read()
                    nbytes += len(data)
                    if len(kept) < keep_first_bytes:
                        kept.append((ti.name, data))
                if max_members and len(names) >= max_members:
                    break
            comp = raw.tell()
    el = time.time() - t0
    return dict(names=names, types=dict(types), member_bytes=nbytes, compressed_bytes_read=comp,
                seconds=el, members_per_s=len(names) / max(el, 1e-9),
                compressed_mb_per_s=comp / 1e6 / max(el, 1e-9)), kept


def name_report(names):
    regs = [n for n in names]
    bad = [n for n in regs if not UUID_JPG.fullmatch(n)]
    exts = Counter(n.rsplit(".", 1)[-1] if "." in n else "<none>" for n in regs)
    return dict(n=len(regs), n_not_bare_uuid_jpg=len(bad), examples_not_bare_uuid_jpg=bad[:10],
                n_with_slash=sum("/" in n for n in regs), extension_counts=dict(exts),
                n_distinct=len(set(regs)))


def part_b():
    from PIL import Image
    import torch
    B = out["B_tol_shards"]
    shards = sorted(EOL.glob("image_set_*.tar.gz"))
    B["n_shards"] = len(shards)
    B["shard_names_first_last"] = [shards[0].name, shards[-1].name]
    B["all_names_match_image_set_NN"] = all(re.fullmatch(r"image_set_\d\d\.tar\.gz", p.name) for p in shards)
    sizes = {p.name: p.stat().st_size for p in shards}
    B["shard_bytes_min_median_max"] = [min(sizes.values()), int(np.median(list(sizes.values()))), max(sizes.values())]
    B["shard_bytes_image_set_63"] = sizes.get("image_set_63.tar.gz")
    B["total_bytes"] = sum(sizes.values())
    magic = {}
    for p in shards:
        with open(p, "rb") as fh:
            magic[p.name] = fh.read(3).hex()
    B["gzip_magic_all_1f8b08"] = all(m == "1f8b08" for m in magic.values())
    log("B: shard sizes/magic done")
    save()

    # B1. full listing of the runt shard image_set_63, compared to its cached ids
    res63, _ = stream_members(EOL / "image_set_63.tar.gz")
    ids63 = (TOLC / "shards/image_set_63.ids.txt").read_text().splitlines()
    keys63 = [wds_key(n)[0] for n in res63["names"]]
    B["image_set_63_full"] = {k: v for k, v in res63.items() if k != "names"}
    B["image_set_63_full"]["names"] = name_report(res63["names"])
    B["image_set_63_full"]["cached_ids_n"] = len(ids63)
    B["image_set_63_full"]["wds_keys_equal_cached_ids_in_order"] = keys63 == ids63
    B["image_set_63_full"]["wds_keys_equal_cached_ids_as_set"] = set(keys63) == set(ids63)
    B["image_set_63_full"]["first_names"] = res63["names"][:3]
    log("B: image_set_63 full listing", B["image_set_63_full"]["seconds"], "s",
        B["image_set_63_full"]["names"], B["image_set_63_full"]["wds_keys_equal_cached_ids_in_order"])
    save()

    # B2. first 3,000 members of image_set_01 and image_set_60
    kept01 = []
    for stem in ("image_set_01", "image_set_60"):
        res, kept = stream_members(EOL / f"{stem}.tar.gz", max_members=3000,
                                   keep_first_bytes=3000 if stem == "image_set_01" else 0)
        if stem == "image_set_01":
            kept01 = kept
        ids = (TOLC / f"shards/{stem}.ids.txt").read_text().splitlines()
        keys = [wds_key(n)[0] for n in res["names"]]
        d = {k: v for k, v in res.items() if k != "names"}
        d["names"] = name_report(res["names"])
        d["wds_keys_equal_first_cached_ids_in_order"] = keys == ids[: len(keys)]
        d["first_names"] = res["names"][:3]
        B[f"{stem}_first3000"] = d
        log(f"B: {stem} first 3000:", d["seconds"], "s", d["members_per_s"], "members/s",
            d["compressed_mb_per_s"], "MB/s", d["wds_keys_equal_first_cached_ids_in_order"])
        save()

    # B3. decode + transform cost per image on the first 500 of image_set_01 (one process)
    from src.open_clip.transform import PreprocessCfg, image_transform_v2
    from src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    tf = image_transform_v2(PreprocessCfg(size=(224, 224), mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD,
                                          interpolation="bicubic", resize_mode="shortest"), is_train=False)
    B["transform_repr"] = repr(tf)
    torch.set_num_threads(1)
    t_dec, t_tf, sizes_wh, modes, orient, fmts, bytes_ = [], [], [], Counter(), Counter(), Counter(), []
    for i, (name, data) in enumerate(kept01):
        with Image.open(io.BytesIO(data)) as im0:
            modes[im0.mode] += 1
            fmts[im0.format] += 1
            try:
                orient[int(im0.getexif().get(274, 1))] += 1
            except Exception:
                orient["unreadable"] += 1
            sizes_wh.append(im0.size)
        bytes_.append(len(data))
        if i < 500:
            t0 = time.perf_counter()
            im = Image.open(io.BytesIO(data)); im.load(); im = im.convert("RGB")
            t1 = time.perf_counter()
            _ = tf(im)
            t2 = time.perf_counter()
            t_dec.append(t1 - t0); t_tf.append(t2 - t1)
    px = np.array([w * h for w, h in sizes_wh], dtype=np.float64)
    B["image_set_01_first3000_image_stats"] = dict(
        modes=dict(modes), formats=dict(fmts), exif_orientation={str(k): v for k, v in orient.items()},
        megapixels_median=float(np.median(px) / 1e6), megapixels_p95=float(np.percentile(px, 95) / 1e6),
        jpeg_kb_median=float(np.median(bytes_) / 1e3), jpeg_kb_mean=float(np.mean(bytes_) / 1e3),
        short_side_min=int(min(min(s) for s in sizes_wh)))
    B["decode_transform_first500_one_thread"] = dict(
        decode_ms_median=1e3 * float(np.median(t_dec)), transform_ms_median=1e3 * float(np.median(t_tf)),
        decode_ms_mean=1e3 * float(np.mean(t_dec)), transform_ms_mean=1e3 * float(np.mean(t_tf)),
        implied_img_per_s_decode_plus_transform=1.0 / (float(np.mean(t_dec)) + float(np.mean(t_tf))))
    log("B: decode/transform", B["decode_transform_first500_one_thread"], B["image_set_01_first3000_image_stats"])
    save()


# ---------------------------------------------------------------------------
# C. catalog.csv join and codes
# ---------------------------------------------------------------------------
def part_c():
    import polars as pl
    C = out["C_catalog"]
    t0 = time.time()
    cols = ["split", "treeoflife_id", "eol_content_id"] + RANKS
    cat = pl.read_csv(CATALOG, columns=cols, infer_schema_length=0)  # all columns as strings
    C["catalog_rows"] = cat.height
    C["catalog_distinct_ids"] = cat["treeoflife_id"].n_unique()
    C["catalog_load_seconds"] = time.time() - t0
    eol = cat.filter(pl.col("eol_content_id").is_not_null())
    C["eol_rows"] = eol.height
    C["eol_distinct_ids"] = eol["treeoflife_id"].n_unique()
    log("C: catalog loaded", C["catalog_rows"], C["catalog_distinct_ids"], C["eol_rows"], C["eol_distinct_ids"])

    # one row per id: split priority val > train > train_small; ranks must agree across duplicates
    agg = (eol.group_by("treeoflife_id")
              .agg([pl.col("split").unique().sort().alias("splits"), pl.len().alias("n_rows")]
                   + [pl.col(r).n_unique().alias(f"nu_{r}") for r in RANKS]
                   + [pl.col(r).first().alias(r) for r in RANKS]))
    agg = agg.with_columns(
        pl.when(pl.col("splits").list.contains("val")).then(pl.lit("val"))
          .when(pl.col("splits").list.contains("train")).then(pl.lit("train"))
          .otherwise(pl.lit("train_small_only")).alias("split1"))
    C["eol_ids_split_counts"] = {r["split1"]: r["len"] for r in agg.group_by("split1").len().to_dicts()}
    C["eol_ids_with_rank_disagreement_across_duplicates"] = int(agg.filter(
        pl.any_horizontal([pl.col(f"nu_{r}") > 1 for r in RANKS])).height)

    ids = (TOLC / "ids.txt").read_text().splitlines()
    N = len(ids)
    idx = json.loads((TOLC / "shards_index.json").read_text())
    shard_of_row = np.empty(N, dtype=np.int16)
    for k, e in enumerate(idx):
        shard_of_row[e["start_row"]: e["start_row"] + e["n_rows"]] = k
    df = pl.DataFrame({"treeoflife_id": ids, "row": np.arange(N, dtype=np.int64),
                       "shard": [idx[k]["shard"] for k in shard_of_row.tolist()]})
    j = df.join(agg, on="treeoflife_id", how="left").sort("row")
    C["cache_rows"] = N
    C["cache_rows_without_catalog_eol_row"] = int(j["split1"].null_count())
    C["cache_split_counts"] = {r["split1"]: r["len"] for r in j.group_by("split1").len().to_dicts()}
    per = (j.group_by(["shard", "split1"]).len().sort(["shard", "split1"]))
    per_shard = {}
    for r in per.to_dicts():
        per_shard.setdefault(r["shard"], {})[r["split1"]] = r["len"]
    C["per_shard_split_counts"] = per_shard
    C["shards_all_train"] = [s for s, d in per_shard.items() if set(d) == {"train"}]
    C["shards_all_val"] = [s for s, d in per_shard.items() if set(d) == {"val"}]
    C["shards_mixed"] = {s: d for s, d in per_shard.items() if len(d) > 1}
    missing_ids = agg.join(df.select("treeoflife_id"), on="treeoflife_id", how="anti")
    C["catalog_eol_ids_not_in_cache"] = {r["split1"]: r["len"] for r in missing_ids.group_by("split1").len().to_dicts()}
    log("C: join done", C["cache_rows_without_catalog_eol_row"], C["cache_split_counts"], C["shards_mixed"])
    save()

    # codes vs catalog lineage (prefix truncated at the first missing rank; species leaf = binomial)
    vocab = json.loads((TOLC / "vocab.json").read_text())
    V = vocab["vocab"]
    codes = np.load(TOLC / "codes.i32.npy")
    C["codes_shape_dtype"] = [list(codes.shape), str(codes.dtype)]
    C["vocab_sizes"] = {r: len(V[r]) for r in RANKS}
    # Raw catalog strings, no stripping; an empty string counts as missing.
    comp = j.select(RANKS).with_columns(
        [pl.when(pl.col(r) == "").then(None).otherwise(pl.col(r)).alias(r) for r in RANKS])
    comp = comp.with_columns(
        pl.when(pl.col("genus").is_not_null() & pl.col("species").is_not_null())
          .then(pl.col("genus") + pl.lit(" ") + pl.col("species")).otherwise(None).alias("species"))
    # present-prefix mask: rank r is usable only if ranks 0..r are all present
    pres = np.stack([comp[r].is_not_null().to_numpy() for r in RANKS], axis=1)
    prefix_ok = np.cumprod(pres, axis=1).astype(bool)
    code_valid = codes >= 0
    C["rows_code_valid_equals_catalog_prefix_mask"] = int((code_valid == prefix_ok).all(axis=1).sum())
    C["rows_complete_lineage_codes"] = int(code_valid.all(axis=1).sum())
    C["rows_complete_lineage_by_split"] = {
        s: int(code_valid.all(axis=1)[(j["split1"] == s).fill_null(False).to_numpy()].sum())
        for s in ("train", "val")}
    # path equality on every valid code
    n_bad = 0
    bad_ex = []
    prefix_series = None
    for r, rank in enumerate(RANKS):
        part = comp[rank].fill_null("")
        prefix_series = part if r == 0 else prefix_series + "|" + part
        valid = code_valid[:, r]
        vocab_paths = np.asarray(V[rank], dtype=object)
        got = vocab_paths[codes[valid, r]]
        exp = prefix_series.to_numpy()[valid]
        neq = got != exp
        n_bad += int(neq.sum())
        if neq.any() and len(bad_ex) < 5:
            w = np.flatnonzero(neq)[:3]
            bad_ex += [dict(rank=rank, vocab=str(got[i]), catalog=str(exp[i])) for i in w]
    C["codes_vs_catalog_path_mismatches"] = n_bad
    C["codes_vs_catalog_path_mismatch_examples"] = bad_ex
    log("C: codes vs catalog", C["rows_code_valid_equals_catalog_prefix_mask"], "of", N,
        "path mismatches:", n_bad, bad_ex[:2])
    save()


# ---------------------------------------------------------------------------
# D. weight hashes
# ---------------------------------------------------------------------------
def sha256(p, block=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def part_d():
    D = out["D_weights"]
    ck = Path("/u/liv/bdbk/ckpt")
    items = {
        "bioclip1": (ck / "bioclip1/open_clip_pytorch_model.bin", RES / "probe_cache_b16/inat21-val/bioclip1/manifest.json"),
        "rcme": (ck / "rcme/open_clip_model.safetensors", RES / "probe_cache_b16/inat21-val/rcme/manifest.json"),
        "openclip-b16": (ck / "openclip-b16/open_clip_model.safetensors", RES / "probe_cache_b16/inat21-val/openclip-b16/manifest.json"),
        "clip-l14-laion2b": (ck / "clip-l14-laion2b/open_clip_pytorch_model.bin", RES / "probe_cache_b16/inat21-val/clip-l14-laion2b/manifest.json"),
        "bfl-euclidean": (ck / "bfl-euclidean/open_clip_model.safetensors", RES / "probe_cache_b16/inat21-val/bfl-euclidean/manifest.json"),
        "bfl-hyperbolic": (ck / "bfl-hyperbolic-vanilla/open_clip_model.safetensors", RES / "probe_cache_b16/inat21-val/bfl-hyperbolic/manifest.json"),
    }
    for key, (w, man) in items.items():
        t0 = time.time()
        h = sha256(w)
        m = json.loads(man.read_text())
        sums = w.parent / "SHA256SUMS"
        listed = None
        if sums.is_file():
            for line in sums.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1] == w.name:
                    listed = parts[0]
        D[key] = dict(file=str(w), sha256=h, manifest_weight_sha256=m.get("weight_sha256"),
                      equals_manifest=(h == m.get("weight_sha256")), sha256sums=listed,
                      equals_sha256sums=(None if listed is None else h == listed), seconds=time.time() - t0)
        log("D:", key, D[key]["equals_manifest"], D[key]["equals_sha256sums"])
    snap = Path("/projects/bdbk/liv/cache/hf/hub/models--imageomics--bioclip-2/snapshots")
    for s in sorted(snap.iterdir()):
        f = s / "open_clip_model.safetensors"
        if f.exists():
            h = sha256(f)
            D["bioclip2"] = dict(file=str(f), resolved=str(f.resolve()), sha256=h,
                                 equals_lfs_blob_name=(f.resolve().name == h), snapshot=s.name,
                                 manifest_weight_sha256=None)
            log("D: bioclip2", D["bioclip2"]["equals_lfs_blob_name"])
    save()


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "D"]
    env()
    save()
    for p in which:
        t0 = time.time()
        try:
            {"A": part_a, "B": part_b, "C": part_c, "D": part_d}[p]()
        except Exception as e:  # keep going so one failure does not hide the other parts
            import traceback
            traceback.print_exc()
            out.setdefault("errors", {})[p] = repr(e)
            save()
        log(f"part {p} finished in {time.time() - t0:.1f}s")
    save()
    log("wrote", OUT)
