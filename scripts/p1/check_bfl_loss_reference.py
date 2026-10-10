"""Independent check of src/models/losses.py:bfl_level_loss_local against BFL's Eqs. (1)-(2) as transcribed in
agent/litreview/bfl-level-restricted-loss.md section 3 (image-to-text: single positive over the unique labels; text-to-image:
mean over each label's positives of the log-softmax over all images, averaged over unique labels; floor sum(log|P_k|)/|K|), on a
random fp64 batch with repeated and singleton labels, plus the two-rank decomposition. Usage: `python scripts/p1/check_bfl_loss_reference.py`.
"""
import math

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.losses import bfl_level_loss_local, level_label_plan  # noqa: E402


def reference(v, z, inv):
    """Eqs. (1)-(2) written directly: logits s = scale * v z^T, one row per image, one column per unique label."""
    N, K = v.shape[0], z.shape[0]
    s = 100.0 * v @ z.T
    i2t = -(s[torch.arange(N), inv] - torch.logsumexp(s, 1)).mean()
    t2i, floor = 0.0, 0.0
    for k in range(K):
        P = (inv == k).nonzero().flatten()
        logp = s[:, k] - torch.logsumexp(s[:, k], 0)
        t2i += -logp[P].mean()
        floor += math.log(len(P))
    return i2t, t2i / K, floor / K


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    N, D = 96, 16
    v = F.normalize(torch.randn(N, D), dim=-1)
    ids = torch.cat([torch.randint(0, 12, (N - 3,)), torch.tensor([50, 51, 52])])  # 12 repeated labels, three singletons
    uniq, inv, shares = level_label_plan(ids, 1)
    z = F.normalize(torch.randn(len(uniq), D), dim=-1)
    ours = bfl_level_loss_local(v, v, z, inv, inv, shares[0], torch.tensor(100.0))
    ref = reference(v, z, inv)
    diffs = [abs(float(ours[k]) - float(r)) for k, r in zip(("i2t", "t2i", "t2i_floor"), ref)]
    uniq2, inv2, shares2 = level_label_plan(ids, 2)
    parts = [bfl_level_loss_local(v[:40], v, z, inv[:40], inv, shares2[0], torch.tensor(100.0)),
             bfl_level_loss_local(v[40:], v, z, inv[40:], inv, shares2[1], torch.tensor(100.0))]
    split = [abs(float(parts[0][k] + parts[1][k]) - float(ours[k])) for k in ("i2t", "t2i", "t2i_floor")]
    print(f"unique labels {len(uniq)} of {N} images; |ours - reference| = {diffs}; |two-rank sum - single rank| = {split}")
    print("PASS" if max(diffs) < 1e-9 and max(split) < 1e-9 else "FAIL")


if __name__ == "__main__":
    main()
