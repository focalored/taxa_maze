#!/usr/bin/env python
"""S9: do pilot 1's evaluation classes (iNat21 val, 10,000 class folders) contain any of the
S9 epithet patterns? Light job (directory listing only); runs on the login node.

Checks every class folder name '<idx>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<epithet>':
  * folder names that do not split into exactly 8 parts (multi-word epithets);
  * epithets in the exact or extended placeholder sets of S9_provenance.py;
  * epithets equal to an author epithet flagged in EOL train (S9_provenance.json, all author keys).
Output: audit/preflight/S9_bioclip1/S9_inat21_classes.json
"""
import csv
import json
import os


ROOT = "/projects/bdbk/liv/repos/taxa_maze"
EVID = f"{ROOT}/audit/preflight/S9_bioclip1"
VAL = "/u/liv/bdbk/data/inat21/val"
PLACEHOLDERS_EXT = ["sp.", "sp", "spp.", "spp", "xx", "x", "?", "??", "???", "spec.", "spec", "species", "species?",
                    "species101", "nov.", "sp.nov", "cf.", "sp.b", "sp.'", "xxx", "xxxx", "xxxxx", "indet", "undet",
                    "undetermined", "unidentified", "hybrid", "complex", "group", "cultivar", "cv", "sedis",
                    "undescribed_sandiego", "undefined-1", "undefined-2", "undefined-5", "1", "2"]  # = S9_provenance.py

author_epithets = set()
with open(f"{EVID}/S9_suspect_keys.csv", newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["is_author"] == "true":
            author_epithets.add(row["key"].split("|")[-1])

dirs = sorted(d for d in os.listdir(VAL) if os.path.isdir(os.path.join(VAL, d)))
bad_parts, placeholder, author = [], [], []
for d in dirs:
    parts = d.split("_")
    if len(parts) != 8:
        bad_parts.append(d)
        continue
    ep = parts[7]
    if ep in PLACEHOLDERS_EXT:
        placeholder.append(d)
    if ep in author_epithets:
        author.append(d)
out = {"n_class_dirs": len(dirs), "n_author_epithets_checked": len(author_epithets),
       "dirs_not_8_parts": bad_parts, "placeholder_epithets": placeholder, "author_epithets": author}
with open(f"{EVID}/S9_inat21_classes.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:10])) for k, v in out.items()}, ensure_ascii=False))
