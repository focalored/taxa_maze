#!/usr/bin/env python
"""Pilot-1 eval-harness gate (spec line 173 and section 9 bullet 5; preflight S11, S48, U2, U4, U5).
Part A requires src/eval/zeroshot.py to give the reference JSONs' exact n_classes, n_paths, n_eval
and correct for the six table checkpoints' cached embeddings under photo and lineage (84 cells);
Part B re-encodes iNat21-val with each checkpoint (fp32 weights, fp16 autocast) and requires every
row at cosine >= 0.9999 against that checkpoint's cache. It writes result.json and gate.log to
logs/smoke/eval_gate/<stamp>_j<job>/ and baseline_counts.json to logs/smoke/eval_gate/, and exits 1
if any check fails.

Usage (inside a GPU allocation; scripts/smoke/eval_gate.sh wraps this):
    python scripts/smoke/eval_gate.py [--part A|B|AB] [--limit N] [--models bioclip1,rcme]
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import ImageFile  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import src.open_clip  # noqa: E402
from src.data.inat21 import INat21Val, load_inat21_labels  # noqa: E402
from src.eval.checkpoints import CHECKPOINTS, load_checkpoint  # noqa: E402
from src.eval.zeroshot import RANKS, evaluate_ranks, render_vocab_path  # noqa: E402
from src.utils.paths import assert_outside_projects, load_paths  # noqa: E402

GATE_DIR = REPO / "logs" / "smoke" / "eval_gate"
BASELINE_PATH = GATE_DIR / "baseline_counts.json"
RESULTS_DIR = Path(load_paths().tol_embed_dir) / "results"
JSON_NAMES = (
    "ZEROSHOT_RANKS_inat21-val.json",
    "ZEROSHOT_RANKS_inat21-val__forms.json",
    "ZEROSHOT_RANKS_inat21-val__bfl.json",
)
# Spec table column -> registry/JSON model key and form (audit/2026-10-02_preflight_eval.md, E1).
SPEC_COLUMNS = (
    ("clip-l14 photo", "clip-l14-laion2b", "photo"),
    ("bioclip2 photo", "bioclip2", "photo"),
    ("rcme lineage", "rcme", "lineage"),
    ("bioclip1 photo", "bioclip1", "photo"),
    ("bfl-euc photo", "bfl-euclidean", "photo"),
    ("bfl-hyp photo", "bfl-hyperbolic", "photo"),
)
MODELS = tuple(model for _, model, _ in SPEC_COLUMNS)
GATE_FORMS = ("photo", "lineage")
CELL_FIELDS = ("n_classes", "n_paths", "n_eval", "correct")
N_VAL = 100_000
COS_BAR = 0.9999
ROW_ORDER_CKPT = "bioclip1"
LABEL_FILES = ("ids.txt", "codes.i32.npy", "vocab.json")


class Log:
    """Print a line and append it to gate.log."""

    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def allocated_cpus():
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(env) if env else len(os.sched_getaffinity(0))


def environment():
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(dict(index=i, name=p.name, total_memory_gib=round(p.total_memory / 2**30, 2)))
    return dict(
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=np.__version__, pillow=PIL.__version__,
        gpus=gpus, open_clip_file=src.open_clip.__file__,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cuda_matmul_allow_fp16_reduced_precision_reduction=(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        HF_HOME=os.environ.get("HF_HOME"), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"), allocated_cpus=allocated_cpus(),
    )


def load_targets(log):
    """Return (targets, sources, problems); targets[(model, form)] holds the JSON row per rank.

    correct = round(top1 * n_eval / 100), and 100 * correct / n_eval must give back top1 exactly.
    A row that appears in several JSONs must be identical in all of them.
    """
    targets, sources, problems = {}, {}, []
    for name in JSON_NAMES:
        path = RESULTS_DIR / name
        data = json.loads(path.read_text(encoding="utf-8"))
        sources[name] = dict(path=str(path), sha256=sha256(path), route=data.get("route"),
                             forms=data.get("forms"), limit=data.get("limit"))
        if data.get("route") != "inat21-val" or data.get("limit") is not None:
            problems.append(f"{name}: route={data.get('route')!r} limit={data.get('limit')!r}")
        for form, by_model in data["results"].items():
            if form not in GATE_FORMS:
                continue
            for model, row in by_model.items():
                if model not in MODELS:
                    continue
                ranks = {}
                for rank in RANKS:
                    cell = row[rank]
                    top1, n_eval = cell["top1"], cell["n_eval"]
                    correct = round(top1 * n_eval / 100)
                    if 100.0 * correct / n_eval != top1:
                        problems.append(f"{name} {form} {model} {rank}: top1 {top1!r} is not "
                                        f"100 * k / {n_eval} for an integer k")
                    ranks[rank] = dict(top1=top1, n_classes=cell["n_classes"],
                                       n_paths=cell["n_paths"], n_eval=n_eval, correct=correct)
                entry = dict(ranks=ranks, average=row["average"]["top1"], files=[name])
                old = targets.get((model, form))
                if old is None:
                    targets[(model, form)] = entry
                    continue
                if old["ranks"] != entry["ranks"] or old["average"] != entry["average"]:
                    problems.append(f"{model} {form}: {name} disagrees with {old['files']}")
                old["files"].append(name)
    problems += [f"no JSON holds {m} {f}" for m in MODELS for f in GATE_FORMS if (m, f) not in targets]
    log(f"targets: {len(targets)} (model, form) rows from {len(sources)} JSON files; "
        f"{len(problems)} problems")
    for p in problems:
        log(f"  PROBLEM {p}")
    return targets, sources, problems


def tokenizer_case_check(tokenizer, vocab):
    """S48: "A photo of ..." and "a photo of ..." must tokenize identically for every class string."""
    sep, paths_by_rank = vocab.get("sep", "|"), vocab["vocab"]
    n_strings, n_differ, example = 0, 0, None
    for r, rank in enumerate(RANKS):
        lower = [render_vocab_path(p, r, "photo", sep) for p in paths_by_rank[rank]]
        differ = (tokenizer(lower) != tokenizer(["A" + s[1:] for s in lower])).any(dim=1)
        n_strings += len(lower)
        n_differ += int(differ.sum())
        if example is None and bool(differ.any()):
            example = lower[int(differ.nonzero()[0, 0])]
    return {"n_strings": n_strings, "n_differ": n_differ, "first_differing": example,
            "pass": n_differ == 0}


def run_part_a(models, targets, n, full_rows, device, log):
    log("")
    log("=" * 100)
    log(f"PART A: evaluate_ranks on the cached embeddings, rows 0..{n - 1}, forms {GATE_FORMS}")
    log("A cell passes when n_classes, n_paths, n_eval and correct all equal the JSON's.")
    if not full_rows:
        log(f"--limit {n}: the JSON counts cover all {N_VAL:,} rows, so cells are NOT compared.")
    log("=" * 100)
    cells, per_model = [], {}
    for name in models:
        spec = CHECKPOINTS[name]
        cache = Path(spec["cache_dir"])
        info = per_model[name] = {"cache_dir": str(cache), "model": spec["model"]}
        t0 = time.time()
        emb = np.load(cache / "emb.f16.npy")
        codes, vocab = load_inat21_labels(cache)
        if emb.dtype != np.float16 or emb.shape != (N_VAL, spec["dim"]) or codes.shape != (N_VAL, 7):
            raise RuntimeError(f"{name}: emb {emb.dtype} {emb.shape}, codes {codes.shape}; expected "
                               f"float16 ({N_VAL}, {spec['dim']}) and ({N_VAL}, 7)")
        info["emb_sha256"] = sha256(cache / "emb.f16.npy")
        model, _, tokenizer = load_checkpoint(name, device)
        info["load_s"] = round(time.time() - t0, 2)
        info["tokenizer_case_S48"] = tokenizer_case_check(tokenizer, vocab)
        log(f"\n[{name}] cache {cache}")
        log(f"  loaded {spec['model']} in {info['load_s']} s; S48: 'A photo' and 'a photo' token "
            f"ids differ for {info['tokenizer_case_S48']['n_differ']} of "
            f"{info['tokenizer_case_S48']['n_strings']:,} class strings")
        info["forms"] = {}
        for form in GATE_FORMS:
            t1 = time.time()
            got = evaluate_ranks(emb[:n], codes[:n], vocab, model, tokenizer, form, device)
            want = targets[(name, form)]
            n_pass = 0
            log(f"  form {form} ({time.time() - t1:.1f} s); want from {', '.join(want['files'])}")
            log(f"    {'rank':<8} {'n_classes got/want':>19} {'n_paths got/want':>17} "
                f"{'n_eval got/want':>17} {'correct got/want':>17}  pass")
            for rank in RANKS:
                g, w = got[rank], want["ranks"][rank]
                ok = all(g[f] == w[f] for f in CELL_FIELDS) if full_rows else None
                n_pass += bool(ok)
                cell = {"model": name, "form": form, "rank": rank, "pass": ok}
                cell.update({f: {"got": g[f], "want": w[f]} for f in CELL_FIELDS + ("top1",)})
                cells.append(cell)
                flag = "-" if ok is None else ("PASS" if ok else "FAIL")
                log(f"    {rank:<8} {g['n_classes']:>9}/{w['n_classes']:<9} {g['n_paths']:>8}/"
                    f"{w['n_paths']:<8} {g['n_eval']:>8}/{w['n_eval']:<8} {g['correct']:>8}/"
                    f"{w['correct']:<8}  {flag}")
            info["forms"][form] = dict(
                seconds=round(time.time() - t1, 2), n_pass=n_pass,
                average_top1_got=got["average"]["top1"], average_top1_want=want["average"],
                average_equal=(got["average"]["top1"] == want["average"]) if full_rows else None)
        del model, tokenizer
        torch.cuda.empty_cache()

    n_cells = len(cells)
    n_pass = sum(1 for c in cells if c["pass"])
    s48_ok = all(per_model[m]["tokenizer_case_S48"]["pass"] for m in models)
    passed = (n_pass == n_cells and s48_ok) if full_rows else (None if s48_ok else False)
    log(f"\nPART A: {n_pass} of {n_cells} cells pass" + ("" if full_rows else " (not compared: --limit)")
        + f"; S48 tokenizer check {'PASS' if s48_ok else 'FAIL'}")
    for c in cells:
        if c["pass"] is False:
            log("  FAILED " + c["model"] + " " + c["form"] + " " + c["rank"] + ": " + ", ".join(
                f"{f} {c[f]['got']}/{c[f]['want']}" for f in CELL_FIELDS if c[f]["got"] != c[f]["want"]))
    return dict(rows=n, full_rows=full_rows, n_cells=n_cells, n_pass=n_pass,
                tokenizer_case_S48_pass=s48_ok, pass_=passed, cells=cells, per_model=per_model)


def write_baseline(targets, sources, part_a, run_dir, log):
    by_cell = {(c["model"], c["form"], c["rank"]): c for c in part_a["cells"]}
    columns = {}
    for column, model, form in SPEC_COLUMNS:
        want = targets[(model, form)]["ranks"]
        columns[column] = dict(
            model=model, form=form,
            correct={rank: want[rank]["correct"] for rank in RANKS},
            n_eval={rank: want[rank]["n_eval"] for rank in RANKS},
            reproduced_by_gate=all(by_cell[(model, form, rank)]["pass"] for rank in RANKS))
    out = dict(
        description=("Integer correct counts per rank for the six columns of the pilot-1 baseline "
                     "table (specs/pilot1.md lines 26-34), keyed by column name, for Amendment 1 "
                     "S31 and S32 (the S32 floor is 'bioclip1 photo' species). Each count is "
                     "round(top1 * n_eval / 100) from tol_embed's ZEROSHOT_RANKS JSONs; "
                     "reproduced_by_gate says src/eval/zeroshot.py gave exactly these counts."),
        written_by="scripts/smoke/eval_gate.py", gate_run=str(run_dir),
        written_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        ranks=list(RANKS), sources=sources, columns=columns,
        all_reproduced=all(c["reproduced_by_gate"] for c in columns.values()))
    assert_outside_projects(BASELINE_PATH)
    BASELINE_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {BASELINE_PATH} (all_reproduced={out['all_reproduced']})")


def row_cosines(a, b, device, chunk=8192):
    out = np.empty(a.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, a.shape[0], chunk):
            x = torch.from_numpy(np.asarray(a[s:s + chunk], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.asarray(b[s:s + chunk], dtype=np.float32)).to(device)
            out[s:s + chunk] = (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1).cpu().numpy()
    return out


def run_part_b(models, n, full_rows, device, workers, batch, log, ids, cudnn_benchmark=False):
    # The caches ran cudnn.benchmark=True, which picks the conv algorithm by timing, so their exact
    # bits are not reproducible run to run; benchmark=False makes this check deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    backend = {"cudnn.deterministic": True, "cudnn.benchmark": cudnn_benchmark,
               "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
               "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "PIL.ImageFile.LOAD_TRUNCATED_IMAGES": False}
    log("")
    log("=" * 100)
    log(f"PART B: re-encode rows 0..{n - 1} with each checkpoint (fp32 weights, fp16 autocast), "
        f"batch {batch}, {workers} loader workers; every row needs cosine >= {COS_BAR} against "
        f"its own cache")
    log("=" * 100)
    per_model, loaded, groups = {}, {}, {}
    for name in models:
        spec = CHECKPOINTS[name]
        t0 = time.time()
        loaded[name] = load_checkpoint(name, device)
        manifest = json.loads((Path(spec["cache_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        rep = repr(loaded[name][1])
        same_as_manifest = (re.sub(r" at 0x[0-9a-f]+", "", rep)
                            == re.sub(r" at 0x[0-9a-f]+", "", manifest.get("preprocess_val_repr", "")))
        per_model[name] = dict(
            model=spec["model"], cache_dir=str(spec["cache_dir"]), load_s=round(time.time() - t0, 2),
            preprocess_val_repr=rep, preprocess_matches_cache_manifest=same_as_manifest,
            cache_manifest={k: manifest.get(k) for k in
                            ("model", "precision", "batch_size", "backend", "slurm_job_id")})
        groups.setdefault(rep, []).append(name)
        log(f"[{name}] loaded in {per_model[name]['load_s']} s; preprocess_val equals the cache "
            f"manifest's: {same_as_manifest}")

    decode_passes = []
    for gi, (rep, names) in enumerate(groups.items(), 1):
        dataset = INat21Val(loaded[names[0]][1], cache_dir=CHECKPOINTS[ROW_ORDER_CKPT]["cache_dir"])
        if len(dataset) != N_VAL:
            raise RuntimeError(f"INat21Val has {len(dataset)} rows; expected {N_VAL}")
        if n < len(dataset):
            dataset = Subset(dataset, range(n))
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=True, drop_last=False)
        feats = {nm: np.zeros((n, CHECKPOINTS[nm]["dim"]), dtype=np.float32) for nm in names}
        seen = np.zeros(n, dtype=bool)
        log(f"\n[decode pass {gi}/{len(groups)}] {', '.join(names)}")
        t0 = time.time()
        with torch.no_grad():
            for bi, (images, idx) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                idx_np = idx.numpy()
                for nm in names:
                    with torch.autocast("cuda", dtype=torch.float16):
                        features, extra = loaded[nm][0].encode_image(images)
                    if extra is not None or tuple(features.shape) != (len(idx_np), CHECKPOINTS[nm]["dim"]):
                        raise RuntimeError(f"{nm}: encode_image gave shape {tuple(features.shape)}, "
                                           f"second output {type(extra).__name__}")
                    feats[nm][idx_np] = features.float().cpu().numpy()
                seen[idx_np] = True
                if bi % 25 == 0 or seen.all():
                    done = int(seen.sum())
                    log(f"  {done:>7,}/{n:,} rows  {done / max(time.time() - t0, 1e-9):6.0f} img/s")
        elapsed = time.time() - t0
        if not seen.all():
            raise RuntimeError(f"decode pass {gi}: {int((~seen).sum())} rows never came back")
        decode_passes.append(dict(models=names, preprocess_val_repr=rep, rows=n,
                                  seconds=round(elapsed, 1), img_per_s=round(n / elapsed, 1)))

        for nm in names:
            info, (model, _, tokenizer) = per_model[nm], loaded.pop(nm)
            cache = Path(CHECKPOINTS[nm]["cache_dir"])
            cached = np.load(cache / "emb.f16.npy")[:n]
            cos = row_cosines(feats[nm], cached, device)
            finite = np.isfinite(cos)
            below = int((~finite).sum() + (cos[finite] < COS_BAR).sum())
            worst = np.argsort(np.where(finite, cos, -np.inf))[:5]
            # autocast outputs are fp16 values, so this cast is exact and compares bit patterns.
            identical = int((feats[nm].astype(np.float16) == cached).all(axis=1).sum())
            info["cosine"] = dict(
                rows=n, min=float(np.nanmin(cos)), median=float(np.nanmedian(cos)),
                mean=float(np.nanmean(cos)), max=float(np.nanmax(cos)), n_nonfinite=int((~finite).sum()),
                n_below_bar=below, bar=COS_BAR, rows_bit_identical_fp16=identical,
                worst_rows=[dict(row=int(i), id=ids[i], cos=float(cos[i])) for i in worst])
            info["pass"] = below == 0
            codes, vocab = load_inat21_labels(cache)
            info["counts"] = {}
            for form in (("photo", "lineage") if nm == "rcme" else ("photo",)):
                re_res = evaluate_ranks(feats[nm], codes[:n], vocab, model, tokenizer, form, device)
                ca_res = evaluate_ranks(cached, codes[:n], vocab, model, tokenizer, form, device)
                info["counts"][form] = {rank: dict(reencoded=re_res[rank]["correct"],
                                                   cached=ca_res[rank]["correct"],
                                                   diff=re_res[rank]["correct"] - ca_res[rank]["correct"],
                                                   n_eval=re_res[rank]["n_eval"]) for rank in RANKS}
            c = info["cosine"]
            log(f"  [{nm}] cosine min {c['min']:.7f}  median {c['median']:.7f}  below {COS_BAR}: "
                f"{c['n_below_bar']}  bit-identical rows (fp16): {identical:,}/{n:,}  -> "
                f"{'PASS' if info['pass'] else 'FAIL'}")
            for form, counts in info["counts"].items():
                log(f"    correct, {form} (re-encoded/cached): " + "  ".join(
                    f"{rank} {counts[rank]['reencoded']}/{counts[rank]['cached']}" for rank in RANKS))
            del model, tokenizer
        del feats
        torch.cuda.empty_cache()

    passed = all(per_model[m]["pass"] for m in models)
    log(f"\nPART B: {sum(per_model[m]['pass'] for m in models)} of {len(models)} checkpoints have "
        f"every row at cosine >= {COS_BAR}")
    return dict(rows=n, full_rows=full_rows, bar=COS_BAR, batch=batch, workers=workers,
                weights="fp32", autocast="cuda float16", backend=backend,
                decode_passes=decode_passes, models=per_model, pass_=passed)


def parse_args():
    ap = argparse.ArgumentParser(description="pilot-1 eval-harness gate (spec line 173, section 9)")
    ap.add_argument("--part", choices=("A", "B", "AB"), default="AB")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N iNat21-val rows (a quick try, not a gate run)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma list of checkpoints (default: all six table columns)")
    ap.add_argument("--batch", type=int, default=512, help="Part B images per batch")
    ap.add_argument("--workers", type=int, default=None,
                    help="Part B DataLoader workers (default: allocated CPUs - 2)")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="Part B with cudnn.benchmark=True, as the caches were encoded (not reproducible)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def summary(result, log):
    log("")
    log("=" * 100)
    log("SUMMARY")
    a, b = result.get("part_a"), result.get("part_b")
    if a:
        log(f"  Part A: {a['n_pass']} of {a['n_cells']} cells pass "
            f"(rows {a['rows']:,}{'' if a['full_rows'] else ', NOT compared'}); "
            f"S48 tokenizer check {'PASS' if a['tokenizer_case_S48_pass'] else 'FAIL'}")
        log("  table columns, correct got/want (got = this harness on the cache; want = JSON):")
        log(f"    {'rank':<8}" + "".join(f"{col:>18}" for col, _, _ in SPEC_COLUMNS))
        by_cell = {(c["model"], c["form"], c["rank"]): c for c in a["cells"]}
        for rank in RANKS:
            row = f"    {rank:<8}"
            for _, model, form in SPEC_COLUMNS:
                c = by_cell.get((model, form, rank))
                row += f"{'' if c is None else str(c['correct']['got']) + '/' + str(c['correct']['want']):>18}"
            log(row)
    if b:
        log(f"  Part B: rows {b['rows']:,}; cosine bar {b['bar']}")
        log(f"    {'checkpoint':<18}{'min':>11}{'median':>11}{'below':>7}{'bit-identical':>15}  pass")
        for name, info in b["models"].items():
            c = info.get("cosine")
            if c:
                log(f"    {name:<18}{c['min']:>11.7f}{c['median']:>11.7f}{c['n_below_bar']:>7}"
                    f"{c['rows_bit_identical_fp16']:>15,}  {'PASS' if info['pass'] else 'FAIL'}")
    log(f"  STATUS: {result['status']}")
    log("=" * 100)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown checkpoints {unknown}; choose from {MODELS}")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device: run inside a GPU allocation (see scripts/smoke/eval_gate.sh)")
    n = N_VAL if args.limit is None else max(1, min(args.limit, N_VAL))
    full_rows = n == N_VAL
    workers = args.workers if args.workers is not None else max(1, allocated_cpus() - 2)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    run_dir = GATE_DIR / f"{stamp}_j{job}"
    assert_outside_projects(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    log = Log(run_dir / "gate.log")
    t_start = time.time()
    result = dict(
        gate="pilot-1 eval-harness gate: spec line 173 and section 9 bullet 5 (preflight S11, S48, U2, U4, U5)",
        status=None, argv=sys.argv, part=args.part, limit=args.limit, rows=n, models=models,
        run_dir=str(run_dir), slurm_job_id=job, host=socket.gethostname(),
        started=datetime.datetime.now().isoformat(timespec="seconds"), env=environment(),
        checkpoints={m: dict(model=CHECKPOINTS[m]["model"], dim=CHECKPOINTS[m]["dim"],
                             cache_dir=str(CHECKPOINTS[m]["cache_dir"])) for m in models},
        timings_s={}, errors=[])
    log(f"eval gate  run dir {run_dir}")
    log(f"host {result['host']}  job {job}  torch {torch.__version__}  GPU "
        f"{', '.join(g['name'] for g in result['env']['gpus'])}")
    log(f"part {args.part}  rows {n:,}  models {', '.join(models)}")
    log(f"fp32 matmul precision {torch.get_float32_matmul_precision()!r}, "
        f"allow_tf32 {torch.backends.cuda.matmul.allow_tf32}; open_clip {src.open_clip.__file__}")
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        result["errors"].append("fp32 matmul precision is not 'highest' with TF32 off (U4)")

    try:
        targets, sources, problems = load_targets(log)
        result["json_sources"] = sources
        result["json_problems"] = problems
        result["errors"].extend(f"JSON: {p}" for p in problems)
        # Both parts assume one row order and one label set: every cache's label files must be
        # byte-identical to those of the cache whose ids.txt INat21Val follows.
        row_cache = Path(CHECKPOINTS[ROW_ORDER_CKPT]["cache_dir"])
        ref_hashes = {f: sha256(row_cache / f) for f in LABEL_FILES}
        result["row_order_cache"] = dict(cache_dir=str(row_cache), sha256=ref_hashes)
        result["label_files_identical"] = {}
        for m in models:
            cache = Path(CHECKPOINTS[m]["cache_dir"])
            same = all(sha256(cache / f) == ref_hashes[f] for f in LABEL_FILES)
            result["label_files_identical"][m] = same
            if not same:
                result["errors"].append(f"{m}: label files differ from the {ROW_ORDER_CKPT} cache's")
        log(f"label files (ids.txt, codes.i32.npy, vocab.json) identical to the {ROW_ORDER_CKPT} "
            f"cache's: {result['label_files_identical']}")
        ids = (row_cache / "ids.txt").read_text(encoding="utf-8").splitlines()

        if "A" in args.part:
            t0 = time.time()
            try:
                result["part_a"] = run_part_a(models, targets, n, full_rows, args.device, log)
                if full_rows and set(models) == set(MODELS):
                    write_baseline(targets, sources, result["part_a"], run_dir, log)
                    result["baseline_counts"] = str(BASELINE_PATH)
            except Exception:
                result["errors"].append("Part A: " + traceback.format_exc())
                log(traceback.format_exc())
            result["timings_s"]["part_a"] = round(time.time() - t0, 1)
        if "B" in args.part:
            t0 = time.time()
            try:
                result["part_b"] = run_part_b(models, n, full_rows, args.device, workers,
                                              args.batch, log, ids, args.cudnn_benchmark)
            except Exception:
                result["errors"].append("Part B: " + traceback.format_exc())
                log(traceback.format_exc())
            result["timings_s"]["part_b"] = round(time.time() - t0, 1)
    except Exception:
        result["errors"].append(traceback.format_exc())
        log(traceback.format_exc())

    for key in ("part_a", "part_b"):
        if key in result:
            result[key]["pass"] = result[key].pop("pass_")
    checks = [result[k]["pass"] for k in ("part_a", "part_b") if k in result]
    failed = (bool(result["errors"]) or any(c is False for c in checks)
              or ("A" in args.part and "part_a" not in result)
              or ("B" in args.part and "part_b" not in result))
    partial = (not full_rows or set(models) != set(MODELS) or args.part != "AB"
               or args.batch != 512 or args.cudnn_benchmark)
    result["status"] = "FAIL" if failed else ("PARTIAL" if partial else "PASS")
    result["timings_s"]["total"] = round(time.time() - t_start, 1)
    result["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    summary(result, log)
    for e in result["errors"]:
        lines = [x for x in e.splitlines() if x.strip()] or ["(empty)"]
        log("ERROR " + lines[0] + ("" if len(lines) == 1 else " ... " + lines[-1]))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {run_dir / 'result.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
