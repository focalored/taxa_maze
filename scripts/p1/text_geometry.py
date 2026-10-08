"""Text-space geometry of BioCLIP 1's taxonomic labels (A15 plan step 0): (a) the zero-mean parent test, how close each internal
node's truncated-label embedding is to the species-uniform mean of its species' label embeddings, and (b) the penalty's cos² terms on
the text space, with the truncated labels as the internal nodes ("label form") and with the species-uniform means as the nodes ("mean
form", the bank's construction), each against a shuffled null. Usage: `python scripts/p1/text_geometry.py --out audit/2026-10-08_text_geometry`.
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.taxonomy import build_train_tree  # noqa: E402
from src.eval.zeroshot import RANKS, class_text, encode_texts  # noqa: E402
from src.models.bank import Bank  # noqa: E402
from src.models.bioclip_lora import load_bioclip1  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

STORE = Path("/u/liv/data/taxa_maze/store")
P1 = Path("/u/liv/data/taxa_maze/p1")


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def cos2(step, pos):
    return (step * pos).sum(1) ** 2 / ((step * step).sum(1) * (pos * pos).sum(1))


def summarize(x, w=None):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    x = x[ok]
    if w is not None:
        w = np.asarray(w, dtype=np.float64)[ok]
        return {"mean": float((x * w).sum() / w.sum()), "median": float(np.median(x)), "n": int(ok.sum())}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "n": int(ok.sum())}


def analyse(emb, tree, rng, form):
    """emb[d]: fp64 L2-normalized label embeddings of the nodes at depth d (6 = species)."""
    S = tree.n_species_total
    bank = Bank(tree, emb[6].shape[1], device="cpu")
    bank.set_species_means(torch.from_numpy(emb[6]))
    means = [bank.m_nodes[d].numpy() for d in range(6)]        # species-uniform means of species labels
    root = bank.root.numpy()
    perm = rng.permutation(S)
    bank_null = Bank(tree, emb[6].shape[1], device="cpu")
    bank_null.set_species_means(torch.from_numpy(emb[6][perm]))  # species shuffled across the tree's slots
    means_null = [bank_null.m_nodes[d].numpy() for d in range(6)]
    out = {"form": form, "a_zero_mean_parent": {}, "b_label_form": {}, "b_mean_form": {}}

    # (a) truncated label t_a against the species-uniform mean m_a of its species' labels
    for d in range(6):
        t, m, mn = emb[d], means[d], means_null[d]
        rel_gap = np.linalg.norm(t - m, axis=1) / np.linalg.norm(m, axis=1)
        other = rng.permutation(len(t))                        # a random other node at the same depth
        out["a_zero_mean_parent"][RANKS[d]] = {
            "cos_label_vs_mean": summarize(cos(t, m)), "cos_label_vs_shuffled_mean": summarize(cos(t, mn)),
            "cos_label_vs_other_nodes_mean": summarize(cos(t, m[other])), "relative_gap": summarize(rel_gap),
            "norm_of_mean": summarize(np.linalg.norm(m, axis=1)), "nodes": int(len(t)),
            "multi_species_nodes": int((tree.n_species[d] > 1).sum())}

    # (b1) label form: nodes are the truncated-label embeddings; root is the species-uniform mean of species labels
    t_s = emb[6]
    valid_all = tree.P_mask
    for d in range(6):
        a = tree.anc[:, d]
        t_a = emb[d][a]
        c2 = cos2(t_s - t_a, t_a - root)
        a_null = a[rng.permutation(S)]
        t_an = emb[d][a_null]
        c2n = cos2(t_s - t_an, t_an - root)
        valid = valid_all[:, d] & np.isfinite(c2)
        parent_rule = valid & (tree.n_children[d][a] > 1)
        out["b_label_form"][RANKS[d]] = {
            "cos2_species_mean": summarize(c2[valid]), "cos2_parent_rule_mean": summarize(c2[parent_rule]),
            "cos2_shuffled_ancestor": summarize(c2n[valid]), "n_valid": int(valid.sum()), "n_parent_rule": int(parent_rule.sum())}
    # (b2) mean form: the bank's construction (species-uniform means), the S24 monitor and L*
    P_mask = torch.from_numpy(tree.P_mask)
    heldout = torch.from_numpy(tree.heldout)
    out["b_mean_form"] = {"monitor": bank.monitor(), "monitor_shuffled": bank_null.monitor(),
                          "L_star": bank.canonical_penalty(P_mask, heldout), "L_star_shuffled": bank_null.canonical_penalty(P_mask, heldout)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("audit/2026-10-08_text_geometry"))
    ap.add_argument("--forms", default="photo,lineage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tree = build_train_tree(STORE, True, 16, P1 / "heldout_species_s9.txt")
    model, _, tokenizer = load_bioclip1(load_paths().bioclip1_ckpt, device)
    print(f"tree: {tree.n_species_total} species, nodes per depth {[len(k) for k in tree.node_keys]}; model loaded; {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(args.seed)
    result = {"purpose": "A15 plan step 0: text-space geometry of untouched BioCLIP 1's taxonomic labels", "seed": args.seed,
              "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "n_species": tree.n_species_total,
              "nodes_per_depth": [len(k) for k in tree.node_keys], "forms": {}}
    for form in args.forms.split(","):
        t1 = time.time()
        emb = []
        for d in range(7):
            strings = [class_text(k.split("|"), form) for k in tree.node_keys[d]]
            with torch.no_grad():
                z = encode_texts(model, tokenizer, strings, device)
            emb.append(F.normalize(z.float(), dim=-1).double().cpu().numpy())
        t_enc = time.time() - t1
        print(f"[{form}] encoded {sum(len(e) for e in emb)} labels in {t_enc:.0f} s", flush=True)
        res = analyse(emb, tree, rng, form)
        res["encode_s"] = round(t_enc, 1)
        result["forms"][form] = res
        for d in range(6):
            a = res["a_zero_mean_parent"][RANKS[d]]; b = res["b_label_form"][RANKS[d]]
            print(f"  {RANKS[d]:8s} (a) cos(label, mean of its species) {a['cos_label_vs_mean']['mean']:.3f} | shuffled {a['cos_label_vs_shuffled_mean']['mean']:.3f} | other nodes {a['cos_label_vs_other_nodes_mean']['mean']:.3f} | rel. gap {a['relative_gap']['mean']:.3f}"
                  f"   (b1) cos2 label form {b['cos2_species_mean']['mean']:.4f} (parent rule {b['cos2_parent_rule_mean']['mean']:.4f}) | shuffled {b['cos2_shuffled_ancestor']['mean']:.4f}", flush=True)
        mon = res["b_mean_form"]["monitor"]; mn = res["b_mean_form"]["monitor_shuffled"]
        print("  (b2) mean form, bank monitor parent rule, species-weighted: " + ", ".join(f"{k.split('/')[-1]} {v:.4f} (shuffled {mn[k]:.4f})" for k, v in mon.items() if "parent_species" in k), flush=True)
        print("  (b2) L*: " + ", ".join(f"{k.split('/',1)[1]} {v:.4f}" for k, v in res["b_mean_form"]["L_star"].items() if k.count("/") == 1), flush=True)
    result["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    result["total_s"] = round(time.time() - t0, 1)
    (args.out / "text_geometry.json").write_text(json.dumps(result, indent=1))
    print("wrote", args.out / "text_geometry.json", f"in {result['total_s']} s")


if __name__ == "__main__":
    main()
