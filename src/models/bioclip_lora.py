"""BioCLIP 1 with LoRA (r=16, alpha=32) on the q and v rows of all 24 packed in_proj_weights, added through parametrize so the attention fast path sees it too (spec section 3; preflight reading S50).

Usage: `model, preprocess_val, tokenizer = load_bioclip1(ckpt_dir); add_qv_lora(model)`.
"""
import math
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

LOGIT_SCALE_MAX = math.log(100.0)
N_PARAMS_BIOCLIP1 = 149_620_737  # ViT-B/16 CLIP; spec line 60 says "~149.6M"


class QVLoRA(nn.Module):
    """Parametrization W -> W + scale * [B_q A_q; 0; B_v A_v] for a packed (3d, d) in_proj_weight."""

    def __init__(self, d: int, r: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.d, self.r, self.scale = d, r, alpha / r
        self.A_q = nn.Parameter(torch.empty(r, d))
        self.B_q = nn.Parameter(torch.zeros(d, r))
        self.A_v = nn.Parameter(torch.empty(r, d))
        self.B_v = nn.Parameter(torch.zeros(d, r))
        # loralib / peft default for A: kaiming_uniform_(a=sqrt(5)), i.e. U(-1/sqrt(d), 1/sqrt(d))
        bound = 1.0 / math.sqrt(d)
        with torch.no_grad():
            for A in (self.A_q, self.A_v):
                A.copy_(torch.rand(A.shape, generator=generator) * 2 * bound - bound)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        d = self.d
        # The update is formed in fp32 even under autocast; the matmul that consumes the
        # adapted weight then casts it as autocast dictates.
        with torch.autocast(device_type=W.device.type, enabled=False):
            dq = (self.B_q.float() @ self.A_q.float()) * self.scale
            dv = (self.B_v.float() @ self.A_v.float()) * self.scale
        return torch.cat([W[:d] + dq.to(W.dtype), W[d: 2 * d], W[2 * d:] + dv.to(W.dtype)], dim=0)


def load_bioclip1(ckpt_dir: str, device="cpu"):
    """Return (model, preprocess_val, tokenizer) with every base weight frozen, fp32."""
    from src.open_clip import create_model_and_transforms, get_tokenizer

    name = f"local-dir:{ckpt_dir}"
    model, _, preprocess_val = create_model_and_transforms(name, pretrained=None, device=device)
    tokenizer = get_tokenizer(name)
    if int(model.visual.output_dim) != 512:
        raise RuntimeError(f"visual.output_dim = {model.visual.output_dim}, spec line 60 expects 512")
    n = sum(p.numel() for p in model.parameters())
    if abs(n - N_PARAMS_BIOCLIP1) > 100_000:
        raise RuntimeError(f"{n:,} parameters; spec line 60 expects ~149.6M")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, preprocess_val, tokenizer


def attention_modules(model) -> List[Tuple[str, nn.MultiheadAttention]]:
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
    names = [n for n, _ in mods]
    vis = [n for n in names if n.startswith("visual.transformer.resblocks.")]
    txt = [n for n in names if n.startswith("transformer.resblocks.")]
    if len(vis) != 12 or len(txt) != 12 or len(mods) != 24:
        raise RuntimeError(f"expected 12 + 12 attention blocks, found {names}")
    return mods


def add_qv_lora(model, r: int = 16, alpha: float = 32.0, seed: int = 42) -> None:
    """Attach the adapters to all 24 blocks and make logit_scale trainable."""
    g = torch.Generator().manual_seed(seed)
    for name, mha in attention_modules(model):
        if not mha._qkv_same_embed_dim:
            raise RuntimeError(f"{name}: q, k, v are not packed into in_proj_weight")
        d = mha.embed_dim
        assert mha.in_proj_weight.shape == (3 * d, d)
        lora = QVLoRA(d, r, alpha, g).to(mha.in_proj_weight.device)
        parametrize.register_parametrization(mha, "in_proj_weight", lora, unsafe=False)
    model.logit_scale.requires_grad_(True)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_tr / n_all > 0.01:
        raise RuntimeError(f"trainable fraction {n_tr / n_all:.4%} exceeds spec line 62's ~1%")


def lora_modules(model) -> Iterator[Tuple[str, QVLoRA]]:
    for n, m in model.named_modules():
        if isinstance(m, QVLoRA):
            yield n, m


def trainable_parameters(model) -> List[Tuple[str, nn.Parameter]]:
    """Adapters (both towers) plus logit_scale, in a fixed order."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def clamp_logit_scale(model) -> None:
    with torch.no_grad():
        model.logit_scale.clamp_(0.0, LOGIT_SCALE_MAX)


def param_counts(model) -> dict:
    n_all = sum(p.numel() for p in model.parameters())
    tr = trainable_parameters(model)
    n_tr = sum(p.numel() for _, p in tr)
    n_vis = sum(p.numel() for n, p in tr if n.startswith("visual."))
    return {"total": n_all, "trainable": n_tr, "trainable_fraction": n_tr / n_all,
            "trainable_image_lora": n_vis, "trainable_text_lora": n_tr - n_vis - 1}
