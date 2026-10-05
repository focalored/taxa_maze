"""Configured paths for scripts that run outside Hydra, and the Amendment 1 S1 rule that every write resolves outside /projects/bdbk.

Usage: `p = load_paths(); out = assert_outside_projects(Path(p.p1_data_dir) / "store")`.
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS_YAML = REPO_ROOT / "configs" / "paths" / "default.yaml"
FORBIDDEN_WRITE_PREFIX = "/projects/bdbk"


def load_paths():
    """Return the paths config. Only plain keys resolve outside Hydra (no `${hydra:...}`)."""
    return OmegaConf.load(PATHS_YAML)


def assert_outside_projects(path) -> Path:
    """Amendment 1 S1: every pilot-1 write must resolve (`readlink -f`) outside /projects/bdbk."""
    resolved = os.path.realpath(str(path))
    if resolved == FORBIDDEN_WRITE_PREFIX or resolved.startswith(FORBIDDEN_WRITE_PREFIX + "/"):
        raise RuntimeError(
            f"refusing to write to {path}: it resolves to {resolved}, under {FORBIDDEN_WRITE_PREFIX}, "
            f"which Amendment 1 S1 forbids (the allocation is over its file quota)."
        )
    return Path(resolved)
