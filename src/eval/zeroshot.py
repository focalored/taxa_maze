"""Per-rank zero-shot top-1 accuracy, kingdom to species, copied operation by operation from
tol_embed's scripts/zeroshot_ranks.py so that integer correct counts match its JSONs exactly
(spec section 7; gated by scripts/smoke/eval_gate.py). Text and scoring run in fp32 with TF32 off.

Usage:
    model, preprocess_val, tokenizer = load_checkpoint("bioclip1", "cuda")   # src.eval.checkpoints
    codes, vocab = load_inat21_labels(inat21_cache_dir("bioclip1"))          # src.data.inat21
    res = evaluate_ranks(img_emb, codes, vocab, model, tokenizer, "photo", "cuda")
"""
import numpy as np
import torch
import torch.nn.functional as F

# The reference ran fp32 matmuls with TF32 off (the PyTorch default); src/train.py sets "medium".
torch.set_float32_matmul_precision("highest")

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
FORMS = ("photo", "lineage", "bare")
_SPECIES = len(RANKS) - 1


def _check_fp32_matmul():
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            f"fp32 matmul precision is {torch.get_float32_matmul_precision()!r} "
            f"(allow_tf32={torch.backends.cuda.matmul.allow_tf32}); the eval needs 'highest'")


def class_text(parts, form="photo"):
    """Class string for lineage components kingdom..rank; at species parts[-1] is the bare epithet."""
    if form not in FORMS:
        raise ValueError(f"unknown text form {form!r}; expected one of {FORMS}")
    if form == "bare":
        return parts[-1]
    text = " ".join(parts)
    return f"a photo of {text}." if form == "photo" else text


def render_vocab_path(path, rank_idx, form, sep="|"):
    """Class string for a vocab.json path cut at rank index rank_idx (0 = kingdom).

    vocab.json's species leaf is the binomial, so "...|Melospiza|Melospiza melodia" renders as
    "... Melospiza melodia", without the genus twice.
    """
    parts = path.split(sep)[: rank_idx + 1]
    if rank_idx == _SPECIES and len(parts) >= 2:
        genus, leaf = parts[-2], parts[-1]
        if leaf.lower().startswith(genus.lower() + " "):
            parts[-1] = leaf[len(genus) + 1:]
    return class_text(parts, form)


def merge_classes(strings):
    """Return (uniq, index): distinct strings in first-occurrence order, and int64 index[i] into uniq."""
    uniq, first = [], {}
    index = np.empty(len(strings), dtype=np.int64)
    for i, s in enumerate(strings):
        if s not in first:
            first[s] = len(uniq)
            uniq.append(s)
        index[i] = first[s]
    return uniq, index


def encode_texts(model, tokenizer, texts, device, batch=256):
    """Return the L2-normalized fp32 (C, D) CPU embeddings of texts, encoded in batches of `batch`."""
    _check_fp32_matmul()
    texts = list(texts)
    if not texts:
        raise ValueError("encode_texts got no strings")
    was_training = model.training
    model.eval()
    out = []
    try:
        # A caller's autocast (Lightning's bf16-mixed) must not reach the eval (Amendment 1, S4).
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for i in range(0, len(texts), batch):
                t = tokenizer(texts[i:i + batch]).to(device)
                e = model.encode_text(t)  # a bare tensor in this fork; encode_image returns a 2-tuple
                out.append(F.normalize(e.float(), dim=-1).cpu())
    finally:
        model.train(was_training)
    return torch.cat(out)


def top1_correct(img_emb, txt, gold, device, chunk=8192):
    """Return (correct, n_valid, pred) for (N, D) image embeddings against (C, D) class embeddings.

    gold holds class indices, -1 = no label; pred is every image's argmax class. Images are cast
    to fp32 and L2-normalized in chunks of `chunk` rows, in the reference's order: the matmul
    shapes decide cuBLAS's kernels, so the chunk size is part of exact reproduction.
    """
    _check_fp32_matmul()
    gold = gold.detach().cpu().numpy() if torch.is_tensor(gold) else np.asarray(gold)
    gold = gold.astype(np.int64, copy=False)
    n = int(img_emb.shape[0])
    if gold.shape != (n,):
        raise ValueError(f"gold has shape {gold.shape}; expected ({n},) to match img_emb")
    valid = gold >= 0
    pred = np.empty(n, dtype=np.int64)
    correct = 0
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        T = txt.to(device=device, dtype=torch.float32)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            if torch.is_tensor(img_emb):
                x = img_emb[a:b].detach().to(device=device, dtype=torch.float32)
            else:
                x = torch.from_numpy(np.asarray(img_emb[a:b], dtype=np.float32)).to(device)
            x = F.normalize(x, dim=-1)
            p = (x @ T.T).argmax(dim=1).cpu().numpy()
            pred[a:b] = p
            m = valid[a:b]
            correct += int((p[m] == gold[a:b][m]).sum())
        del T
    return correct, int(valid.sum()), pred


def evaluate_ranks(img_emb, codes, vocab, model, tokenizer, form, device):
    """Top-1 at all seven ranks for one text form.

    codes is (N, 7), column r indexing vocab[RANKS[r]] (-1 = no label); vocab is the parsed
    vocab.json ({"sep", "vocab": {rank: [path, ...]}}) or a bare {rank: [path, ...]}. Returns
    {rank: {correct, n_eval, n_classes, n_paths, top1}} plus "average": {top1}; top1 is in %.
    """
    if form not in FORMS:
        raise ValueError(f"unknown text form {form!r}; expected one of {FORMS}")
    if isinstance(vocab.get("vocab"), dict):
        sep, paths_by_rank = vocab.get("sep", "|"), vocab["vocab"]
    else:
        sep, paths_by_rank = "|", vocab
    codes = codes.detach().cpu().numpy() if torch.is_tensor(codes) else np.asarray(codes)
    n = int(img_emb.shape[0])
    if codes.shape != (n, len(RANKS)):
        raise ValueError(f"codes has shape {codes.shape}; expected ({n}, {len(RANKS)})")

    res = {}
    for r, rank in enumerate(RANKS):
        paths = paths_by_rank[rank]
        uniq, path2class = merge_classes([render_vocab_path(p, r, form, sep) for p in paths])
        txt = encode_texts(model, tokenizer, uniq, device)
        gold_path = np.asarray(codes[:, r], dtype=np.int64)
        valid = gold_path >= 0
        gold = np.where(valid, path2class[np.clip(gold_path, 0, None)], -1)
        correct, n_valid, _ = top1_correct(img_emb, txt, gold, device)
        res[rank] = dict(correct=correct, n_eval=n_valid, n_classes=len(uniq), n_paths=len(paths),
                         top1=100.0 * correct / max(n_valid, 1))
    res["average"] = dict(top1=float(np.mean([res[rank]["top1"] for rank in RANKS])))
    return res
