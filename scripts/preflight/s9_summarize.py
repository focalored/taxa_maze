#!/usr/bin/env python
"""Summaries for the S9 report, read from s9_epithet_census.json, plus a membership check against BioCLIP's
released label list (TreeOfLife-10M embeddings/txt_emb_species.json at HF commit 4e71907). Writes s9_summary.json.
Run: srun <cpu flags> --output=<repo>/audit/preflight/s9_summary_%j.log bash s9_run.sh s9_summarize.py
"""
import collections
import json
import re
import urllib.request

from scipy.stats import beta

OUT = "/projects/bdbk/liv/repos/taxa_maze/audit/preflight"
LABELS = ("https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/"
          "4e719078e4cab104a0b631c0fd724ba9ee02411b/embeddings/txt_emb_species.json")
R = json.load(open(f"{OUT}/s9_epithet_census.json"))
nc = R["non_clean_keys"]
S = {}


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [k, n, k / n, lo, hi]


def tag(r):
    if r["sub"] in ("author_only_single", "author_only_multi"):
        return "A_author_only"
    if r["sub"] == "epithet_plus_author":
        return "A_epithet_plus_author"
    return r["cat"]


def p_kind(s):
    if re.match(r"^(sp|spp)\.? ?\S", s) and s not in ("sp.", "sp", "spp.", "spp"):
        return "sp_plus_label"
    if s.startswith(("cf", "aff", "nr")):
        return "cf_aff_nr_plus_epithet"
    if "?" in s:
        return "question_mark"
    if s.startswith(("subg.", "sect.", "ser.", "subsect.")):
        return "subgenus_section_series"
    if s.startswith(("undescribed", "undefined")) or s in ("nov.", "sp.nov", "sp.nov.", "nov"):
        return "undescribed_or_nov"
    if s in ("sp.", "sp", "spp.", "spp", "species", "xx", "xxx", "x", "complex", "hybrid", "ss.", "cv", "cv."):
        return "bare_genus_catch_all"
    return "other"


def agg(rows, keyf):
    d = collections.defaultdict(lambda: {"keys": 0, "train_images": 0, "val_images": 0})
    for r in rows:
        k = keyf(r)
        d[k]["keys"] += 1
        d[k]["train_images"] += r["n_tr"]
        d[k]["val_images"] += r["n_va"]
    return dict(sorted(d.items(), key=lambda kv: -kv[1]["train_images"]))


S["P_by_kind"] = agg([r for r in nc if r["cat"] == "P"], lambda r: p_kind(r["species"]))
S["A_by_subtype_and_reason"] = agg([r for r in nc if r["cat"] == "A"], lambda r: f"{r['sub']}|{r['reason']}")
S["A_by_tag"] = agg([r for r in nc if r["cat"] == "A"], tag)
seven = {"sp.", "sp", "xx", "spp.", "x", "?", "spp"}
S["preflight_seven_tokens"] = agg([r for r in nc if r["cat"] == "P" and r["species"] in seven], lambda r: "all")
single = [r for r in nc if r["sub"] == "author_only_single"]
words = agg(single, lambda r: r["species"])
S["author_only_single_distinct_words"] = len(words)
S["author_only_single_top25_words"] = dict(list(words.items())[:25])
S["named_authors"] = {w: words.get(w) for w in ["latreille", "linnaeus", "fabricius", "lamarck", "meigen", "walker",
                                                  "schiner", "hübner", "förster", "stål", "guenée", "gray", "say",
                                                  "mill.", "fr.", "dc."]}

ch = R["gbif"]["checks"]
a = ch["A_single_all"]
S["gbif_A_single_all"] = {
    "checked": len(a), "binomial_is_gbif_species": cp(sum(r["gbif_binomial_is_species"] for r in a), len(a)),
    "epithet_in_gbif_genus_authorship": sum(r["epithet_in_gbif_genus_authorship"] for r in a),
    "genus_exact_in_gbif": sum(1 for r in a if (r["gbif_genus"] or {}).get("matchType") == "EXACT"),
    "binomial_match_types": dict(collections.Counter((r["gbif_binomial"] or {}).get("matchType") for r in a)),
    "reversed_to_clean": [(r["genus"], r["species"], r["n_tr_pair"], (r["gbif_binomial"] or {}).get("scientificName"))
                          for r in a if r["gbif_binomial_is_species"]]}
w = ch["C_weak_author_candidates"]
S["gbif_C_weak_candidates"] = {"checked": len(w), "binomial_is_gbif_species": sum(r["gbif_binomial_is_species"] for r in w),
                               "promoted_to_A": sum((not r["gbif_binomial_is_species"]) and r["epithet_in_gbif_genus_authorship"] for r in w)}
c = ch["C_random_single_word"]
S["gbif_C_random_sample"] = {"checked": len(c), "gbif_species": cp(sum(r["gbif_binomial_is_species"] for r in c), len(c)),
                             "not_species": [(r["genus"], r["species"], (r["gbif_binomial"] or {}).get("matchType"))
                                             for r in c if not r["gbif_binomial_is_species"]]}
s = ch["S_top100_plus_random100"]
S["gbif_S_sample"] = {"checked": len(s), "full_name_is_gbif_infraspecific": sum(r["full_is_infraspecific_name"] for r in s),
                      "first_word_is_gbif_species": sum(r["first_word_is_species"] for r in s),
                      "not_confirmed": [(r["genus"], r["species"]) for r in s if not r["full_is_infraspecific_name"]]}
a1 = ch["A_epithet_plus_author_top100_plus_random100"]
S["gbif_A_epithet_plus_author_sample"] = {
    "checked": len(a1), "epithet_part_is_gbif_species": sum(r["epithet_part_is_species"] for r in a1),
    "not_confirmed": [(r["genus"], r["species"], (r["gbif_epithet_part"] or {}).get("matchType")) for r in a1
                      if not r["epithet_part_is_species"]]}

with urllib.request.urlopen(LABELS, timeout=120) as fh:
    labels = json.loads(fh.read().decode())
lab = {"|".join(x[0]) for x in labels}
inl = agg([r for r in nc if r["key"] in lab], tag)
S["bioclip_label_list"] = {"url": LABELS, "entries": len(labels), "distinct": len(lab),
                           "non_clean_keys_in_list": inl, "non_clean_keys_total": agg(nc, tag),
                           "bombus_latreille_in_list": "Animalia|Arthropoda|Insecta|Hymenoptera|Apidae|Bombus|latreille" in lab}
print(json.dumps(S, indent=1, ensure_ascii=False))
with open(f"{OUT}/s9_summary.json", "w") as fh:
    json.dump(S, fh, indent=1, ensure_ascii=False)
