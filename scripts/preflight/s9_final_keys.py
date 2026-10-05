"""S9 final drop set (2026-10-04): rows whose species field holds no real species epithet.

Rule confirmed by the user on 2026-10-04: drop a row when its species field contains no real species
epithet; keep fields that contain a real epithet plus a placeholder, tag or author name. Starts from the
census categories (audit/preflight/s9_epithet_census.json, non_clean_keys) and applies these rules:
  single word : drop census author-only words (per key) and pure placeholders. Keep words that hold a real
                epithet: "epithet?" forms ("gracilis?", "10-punctata?"), "a_or_b" forms, a quoted known epithet
                ("'blakeana'"), and census "abbreviations" that are really an epithet with a stray period
                ("deliciosa."): the stem is a known epithet, or has a Latin ending and is never an author word.
  several words: drop author-only strings (unless a word forms a known species of that genus),
                subgenus/section/series labels, "sp." plus a tag or locality, "morph ...", "cv ...",
                "- cultivar", cultivar names wholly in quotes, and machine tags with nothing else;
                keep everything else (cf./aff./nr. forms, epithet + author, subspecies, hybrids, "sp. <epithet>").
Writes the key list to specs/pilot1_s9_missing_species_keys.txt and the reasons to audit/preflight/.
"""
import json, re, math, hashlib, os
from collections import defaultdict, Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PRE = os.path.join(REPO, "audit/preflight")
nk = json.load(open(f"{PRE}/s9_epithet_census.json"))["non_clean_keys"]
voc = json.load(open(f"{PRE}/S9_bioclip1/S9_vocab.json"))
gbif = json.load(open(f"{PRE}/s9_gbif_cache.json"))
EPI = {w: voc["EPI_SPECIES_COUNT"].get(w, 0) + voc["EPI_INFRA_COUNT"].get(w, 0)
       for w in set(voc["EPI_SPECIES_COUNT"]) | set(voc["EPI_INFRA_COUNT"])}
AUTH = voc["AUTH_COUNT"]
# known species per genus: clean census-free labels from BioCLIP's released list, plus GBIF exact species matches
binom = set()
non_clean = {e["key"] for e in nk}
for ranks, _ in json.load(open(f"{PRE}/S9_bioclip1/txt_emb_species.json")):
    if "|".join(ranks) not in non_clean and ranks[6] and " " not in ranks[6]:
        binom.add((ranks[5].lower(), ranks[6].lower()))
for q, r in gbif.items():
    if isinstance(r, dict) and r.get("matchType") == "EXACT" and r.get("rank") in ("SPECIES", "SUBSPECIES", "VARIETY") and r.get("canonicalName"):
        w = r["canonicalName"].lower().split()
        if len(w) >= 2: binom.add((w[0], w[1]))

PH = {"sp", "spp", "ssp", "species", "spec", "xx", "xxx", "x", "nov", "n", "indet", "undet", "ined", "unknown",
      "unidentified", "undescribed", "undefined", "cf", "aff", "nr", "near", "complex", "hybrid", "group", "agg",
      "prob", "cv", "morph", "cultivar"}
RANKM = {"subg", "subgen", "sect", "ser", "subsect", "section", "subgenus", "series"}
LATIN = re.compile(r".{2,}(a|ae|i|ii|um|us|is|es|ensis|ense|oides|ata|atus|ica|icus|ina|inus|osa|osus|ella|ellus|ana|anus|iae|orum|arum|ior|ius|ia|ea|eus|ens|ans|alis|ale|ilis)$")

def words(s):
    return [t for t in (w.strip(".,;:?!*\"'()[]=-").lstrip("×") for w in s.lower().replace("_or_", " ").split()) if t]

def epithet_like(t):
    """A plain Latin-looking word that is not a placeholder and not mainly an author name."""
    t = re.sub(r"^\d+-", "", t)                                # numeral epithets: "10-punctata"
    if t in PH or len(t) < 3 or not re.fullmatch(r"[a-zà-ÿ][a-zà-ÿ\-]*", t):
        return False
    a, e = AUTH.get(t, 0), EPI.get(t, 0)
    return not (a >= 1 and (e == 0 or a >= 5 * (e + 1)))

def reason_to_drop(e):
    s, sub, cat = e["species"].strip(), e["sub"], e["cat"]
    genus = e["key"].split("|")[5].lower()
    w = words(s)
    multi = " " in s
    if sub == "author_only_single":
        stem = s.rstrip(".")
        if s.endswith(".") and epithet_like(stem) and (EPI.get(stem, 0) >= 1 or
                                                       (AUTH.get(stem, 0) == 0 and len(stem) >= 5 and LATIN.match(stem))):
            return None                                       # "deliciosa.": an epithet with a stray period
        return "single-word author name"
    if sub == "machine_tag":
        rest = re.sub(r"taxonomy:\S+|geo:\S+", " ", s)
        return None if any(epithet_like(t) for t in words(rest)) else "machine tag only"
    if not multi:
        if cat == "P":
            if sub == "either_or" and any(epithet_like(t) for t in w):
                return None                                   # "elegans_or_baileyi"
            if "?" in s and w and all(epithet_like(t) for t in w):
                return None                                   # "gracilis?", "10-punctata?", "ochropus?."
            if s.startswith("'") and w and EPI.get(w[0], 0) >= 1 and epithet_like(w[0]):
                return None                                   # "'blakeana'"
            return "single-word placeholder"
        return None
    head = s.split()[0].strip(".").lower()
    if head in RANKM:
        return "subgenus, section or series label"
    if sub == "author_only_multi":
        return None if any((genus, t) in binom for t in w) else "author-only string"
    if cat == "P":
        if head in {"cf", "aff", "nr", "near"}:
            return None
        if head in {"sp", "spp"}:
            return None if any(EPI.get(t, 0) and epithet_like(t) and LATIN.match(t) for t in w[1:]) else "sp. plus a tag or locality"
        if head in {"morph", "cv"} or s.startswith("- ") or (s.startswith("'") and not re.search(r"[a-z]{3,}\s*$", s.split("'")[0])):
            return "morph label, cultivar marker or cultivar name"
        return None
    return None

drop, why = [], {}
for e in nk:
    r = reason_to_drop(e)
    if r:
        drop.append(e); why[e["key"]] = r

T, V, B = 5372586, 282542, 8192
tr, va = sum(e["n_tr"] for e in drop), sum(e["n_va"] for e in drop)
left = T - tr
steps = math.ceil(left / B)
agg = defaultdict(lambda: [0, 0, 0, []])
for e in drop:
    g = agg[why[e["key"]]]; g[0] += 1; g[1] += e["n_tr"]; g[2] += e["n_va"]; g[3].append(e["species"])
summary = {"keys": len(drop), "train_dropped": tr, "val_dropped": va, "train_left": left, "val_left": V - va,
           "steps_per_epoch_ceil": steps, "steps_per_epoch_floor": left // B, "warmup_1pct": math.ceil(0.01 * 10 * steps),
           "by_reason": {r: {"keys": n, "train": a, "val": b, "examples": sorted(set(x))[:10]} for r, (n, a, b, x) in agg.items()}}
out = os.path.join(REPO, "specs/pilot1_s9_missing_species_keys.txt")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("".join(k + "\n" for k in sorted(why)))
summary["key_file"] = {"path": "specs/pilot1_s9_missing_species_keys.txt", "lines": len(why),
                       "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
with open(f"{PRE}/s9_final_keys_reasons.tsv", "w", encoding="utf-8") as f:
    f.write("key\tspecies\treason\tn_train\tn_val\n")
    for e in sorted(drop, key=lambda e: e["key"]):
        f.write(f"{e['key']}\t{e['species']}\t{why[e['key']]}\t{e['n_tr']}\t{e['n_va']}\n")
json.dump(summary, open(f"{PRE}/s9_final_keys_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
