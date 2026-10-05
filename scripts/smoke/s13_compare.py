#!/usr/bin/env python
"""Amendment 1 S13: compare the λ = 0 runs with their controls step by step from each run's exact steps.jsonl, and list the operations that warned under deterministic mode.

Usage: `python scripts/smoke/s13_compare.py` after the six runs under logs/smoke/lam0/ finish; writes logs/smoke/lam0/s13_result.json.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.tol_catalog import RANKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "logs" / "smoke" / "lam0"
PAIRS = [("c_run1", "c_run2", "d_lam0"), ("a_run1", "a_run2", "b_lam0")]
N_STEPS = 100  # Amendment 1 S13
WARN_RE = re.compile(r"([A-Za-z_0-9:.]+) does not have a deterministic implementation|"
                     r"(Flash Attention|Memory Efficient attention) defaults to a non-deterministic algorithm")


def latest_run(tag: str) -> Path:
    runs = sorted(p for p in (ROOT / tag).iterdir() if (p / "steps.jsonl").exists())
    if not runs:
        raise FileNotFoundError(f"no finished run under {ROOT / tag}")
    return runs[-1]


def load(tag: str):
    d = latest_run(tag)
    rows = {}
    for line in (d / "steps.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["step"]] = {k: float.fromhex(v) for k, v in r.items() if k not in ("step", "epoch")}
    if len(rows) != N_STEPS:
        raise FileNotFoundError(f"{d} has {len(rows)} steps, expected {N_STEPS}")
    job = d.name.rsplit("_j", 1)[1]
    log = next(iter(Path(ROOT.parents[1], "slurm").glob(f"*_{job}.log")), None)
    warns = sorted({m.group(1) or m.group(2) for m in WARN_RE.finditer(log.read_text())}) if log else ["(no slurm log found)"]
    info = json.loads((d / "run_info.json").read_text())
    # The seed is only in the Hydra config; each rank opens its own Hydra dir, all with the same seed.
    cfg = d / ".hydra" / "config.yaml"
    if not cfg.exists():
        cfg = next(iter(sorted(d.parent.glob(f"*_j{job}/.hydra/config.yaml"))), None)
    m = re.search(r"^seed:\s*(\S+)", cfg.read_text(), re.M) if cfg else None
    info["seed"] = m.group(1) if m else None
    return {"dir": str(d), "job": job, "rows": rows, "warnings": warns, "info": info}


def loss_keys(rows) -> list:
    keys = ["loss/con", "loss/total"]
    keys += [f"con/{r}/{k}" for r in RANKS for k in ("i2t", "t2i")]
    return [k for k in keys if any(k in rows[s] for s in rows)]


def gaps(a, b, keys):
    """Per key: max over shared steps of |a - b|, and how many steps had the key in both runs."""
    out = {}
    for k in keys:
        steps = [s for s in a if s in b and k in a[s] and k in b[s]]
        out[k] = {"max_gap": max((abs(a[s][k] - b[s][k]) for s in steps), default=float("nan")), "n_steps": len(steps),
                  "bit_identical": all(a[s][k] == b[s][k] for s in steps)}
    return out


def main():
    result = {"rule": "If the two controls are bit-identical, the lambda=0 run must be too; otherwise its per-step gap "
                      "(total and each level) must stay within the largest gap between the two controls (Amendment 1 S13).",
              "comparisons": {}}
    all_pass = True
    for c1, c2, lam0 in PAIRS:
        try:
            A, B, D = load(c1), load(c2), load(lam0)
        except FileNotFoundError as e:
            result["comparisons"][f"{lam0}_vs_{c1}"] = {"status": f"incomplete: {e}"}
            all_pass = False
            continue
        same_setup = A["info"]["seed"] is not None and all(A["info"][k] == B["info"][k] == D["info"][k]
                         for k in ("seed", "K", "B", "world", "s9_drop", "steps_per_epoch", "warmup_steps", "lr"))
        keys = loss_keys(A["rows"])
        cc, dc, dc2 = gaps(A["rows"], B["rows"], keys), gaps(A["rows"], D["rows"], keys), gaps(B["rows"], D["rows"], keys)
        controls_identical = all(v["bit_identical"] for v in cc.values())
        per_key = {}
        ok = same_setup
        for k in keys:
            if controls_identical:
                p = dc[k]["bit_identical"]
            else:
                p = dc[k]["max_gap"] <= cc[k]["max_gap"] or dc[k]["bit_identical"]
            per_key[k] = {"control_gap": cc[k]["max_gap"], "lam0_gap_vs_run1": dc[k]["max_gap"],
                          "lam0_gap_vs_run2": dc2[k]["max_gap"], "n_steps": dc[k]["n_steps"], "pass": bool(p)}
            ok = ok and p
        all_pass = all_pass and ok
        result["comparisons"][f"{lam0}_vs_{c1}"] = {
            "runs": {t: {"dir": X["dir"], "job": X["job"], "n_steps": len(X["rows"]), "deterministic_warnings": X["warnings"]}
                     for t, X in ((c1, A), (c2, B), (lam0, D))},
            "same_setup": same_setup, "controls_bit_identical": controls_identical,
            "lam0_bit_identical_to_run1": all(v["bit_identical"] for v in dc.values()),
            "per_key": per_key, "pass": bool(ok)}
        print(f"{lam0} vs {c1}: controls identical={controls_identical}  pass={ok}")
        for k, v in per_key.items():
            print(f"  {k:<24} control gap {v['control_gap']:.3e}  lam0 gap {v['lam0_gap_vs_run1']:.3e}  steps {v['n_steps']}  {'ok' if v['pass'] else 'FAIL'}")
    result["pass"] = all_pass
    (ROOT / "s13_result.json").write_text(json.dumps(result, indent=1))
    print("PASS" if all_pass else "FAIL", "->", ROOT / "s13_result.json")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
