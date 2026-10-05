"""The six checkpoints of pilot 1's baseline table (spec lines 26-34) with their tol_embed
iNat21-val caches, copied from tol_embed scripts/data_paths.py CHECKPOINTS, and a loader that
matches tol_embed's zeroshot_ranks.py (fp32, then .to(device).eval()).

Usage:
    model, preprocess_val, tokenizer = load_checkpoint("bioclip1", "cuda")
"""
from pathlib import Path

from src.data.inat21 import inat21_cache_dir
from src.open_clip import create_model_and_transforms, get_tokenizer
from src.utils.paths import load_paths

PROJ_CKPT = Path("/u/liv/bdbk/ckpt")  # tol_embed scripts/data_paths.py, PROJ_CKPT

# tol_embed loads hf-hub:imageomics/bioclip-2 through HF_HOME=/u/liv/bdbk/cache/hf. Pilot-1 jobs
# use another HF_HOME and must not write under /projects (Amendment 1, S1), so bioclip2 reads that
# cache's snapshot in place: the same open_clip_config.json and open_clip_model.safetensors.
BIOCLIP2_SNAPSHOT = Path(
    "/u/liv/bdbk/cache/hf/hub/models--imageomics--bioclip-2/snapshots/"
    "2957b322090f9cb17ae72c71981c7218a28d81e0"
)

CHECKPOINTS = {
    "bioclip1": dict(
        model=f"local-dir:{load_paths().bioclip1_ckpt}", pretrained=None, dim=512,
        cache_dir=inat21_cache_dir("bioclip1"), upstream="imageomics/bioclip"),
    "rcme": dict(
        model=f"local-dir:{PROJ_CKPT / 'rcme'}", pretrained=None, dim=512,
        cache_dir=inat21_cache_dir("rcme"), upstream="MVRL/rcme-tol-vit-base-patch16"),
    "bioclip2": dict(
        model=f"local-dir:{BIOCLIP2_SNAPSHOT}", pretrained=None, dim=768,
        cache_dir=inat21_cache_dir("bioclip2"), upstream="imageomics/bioclip-2"),
    "clip-l14-laion2b": dict(
        model=f"local-dir:{PROJ_CKPT / 'clip-l14-laion2b'}", pretrained=None, dim=768,
        cache_dir=inat21_cache_dir("clip-l14-laion2b"),
        upstream="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
    "bfl-euclidean": dict(
        model=f"local-dir:{PROJ_CKPT / 'bfl-euclidean'}", pretrained=None, dim=512,
        cache_dir=inat21_cache_dir("bfl-euclidean"), upstream="imageomics/bioclip-hc-euclidean"),
    "bfl-hyperbolic": dict(
        model=f"local-dir:{PROJ_CKPT / 'bfl-hyperbolic-vanilla'}", pretrained=None, dim=512,
        cache_dir=inat21_cache_dir("bfl-hyperbolic"), upstream="imageomics/bioclip-hc-hyperbolic"),
}


def load_checkpoint(name, device):
    """Return (model, preprocess_val, tokenizer); raise if visual.output_dim is not the entry's dim."""
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; known: {sorted(CHECKPOINTS)}")
    spec = CHECKPOINTS[name]
    model, _, preprocess_val = create_model_and_transforms(spec["model"], pretrained=spec.get("pretrained"))
    model = model.to(device).eval()
    output_dim = getattr(getattr(model, "visual", None), "output_dim", None)
    if output_dim is None or int(output_dim) != spec["dim"]:
        raise RuntimeError(f"checkpoint {name!r} ({spec['model']}) loaded with visual.output_dim="
                           f"{output_dim}, but the registry says dim={spec['dim']}")
    return model, preprocess_val, get_tokenizer(spec["model"])
