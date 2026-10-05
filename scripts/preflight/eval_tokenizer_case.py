"""Preflight check E1 helper: does "A photo of ..." tokenize the same as "a photo of ..."?

The spec (specs/pilot1.md line 23) writes the photo template with a capital
"A"; tol_embed/scripts/zeroshot_ranks.py render() writes a lowercase "a". This
builds each checkpoint's tokenizer with the taxa_maze copy of open_clip (no
model weights are loaded) and compares token ids for both spellings.

Run from the taxa_maze repo root on the login node (tokenizer only):
    source "$HOME/.hpc_env.sh" && conda activate bioclip
    cd /projects/bdbk/liv/repos/taxa_maze
    HF_HUB_OFFLINE=1 python -m scripts.preflight.eval_tokenizer_case
"""
import json
import sys
from pathlib import Path

ROOT = Path("/projects/bdbk/liv/repos/taxa_maze")
# Do not write __pycache__ into src/open_clip (outside the preflight write area).
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from src.open_clip import get_tokenizer  # noqa: E402  (taxa_maze copy)
import src.open_clip as oc  # noqa: E402

CKPT = Path("/u/liv/bdbk/ckpt")
MODELS = {
    "bioclip2": "hf-hub:imageomics/bioclip-2",
    "bioclip1": f"local-dir:{CKPT / 'bioclip1'}",
    "rcme": f"local-dir:{CKPT / 'rcme'}",
    "openclip-b16": f"local-dir:{CKPT / 'openclip-b16'}",
    "clip-l14-laion2b": f"local-dir:{CKPT / 'clip-l14-laion2b'}",
    "bfl-euclidean": f"local-dir:{CKPT / 'bfl-euclidean'}",
    "bfl-hyperbolic": f"local-dir:{CKPT / 'bfl-hyperbolic-vanilla'}",
}
PAIRS = [
    ("A photo of Animalia.", "a photo of Animalia."),
    ("A photo of Animalia Chordata Mammalia Carnivora.",
     "a photo of Animalia Chordata Mammalia Carnivora."),
    ("A photo of Animalia Chordata Aves Passeriformes Passerellidae Melospiza melodia.",
     "a photo of Animalia Chordata Aves Passeriformes Passerellidae Melospiza melodia."),
]
OUT = ROOT / "audit/preflight/eval_tokenizer_case.json"


def main():
    print("open_clip imported from:", Path(oc.__file__).resolve())
    res = {}
    for name, ident in MODELS.items():
        tok = get_tokenizer(ident)
        same = []
        for upper, lower in PAIRS:
            a, b = tok([upper]), tok([lower])
            same.append(bool((a == b).all()))
        res[name] = dict(tokenizer_class=type(tok).__name__,
                         clean_fn=getattr(getattr(tok, "clean_fn", None), "__name__", None),
                         all_pairs_identical=all(same), per_pair=same)
        print(f"{name:18s} {res[name]}")
    OUT.write_text(json.dumps(dict(open_clip_file=str(Path(oc.__file__).resolve()),
                                   pairs=PAIRS, results=res), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
