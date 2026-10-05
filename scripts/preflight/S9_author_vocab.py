#!/usr/bin/env python
"""S9 helper: word counts from EOL DH taxon.tab, used to tell author names from epithets.

Run 2 of S9_provenance.py dropped any candidate author word that occurs anywhere in the DH
epithet vocabulary. That was too strict: 'latreille', 'linnaeus', 'fabricius' and 'gray' occur
as words of a few DH species names, so Bombus|latreille was no longer flagged. This script
counts how often each word is used as an author word versus as an epithet word, so that the
rule can compare the two counts instead.

Outputs (audit/preflight/S9_bioclip1/):
  S9_vocab.json            AUTH_COUNT, EPI_SPECIES_COUNT, EPI_INFRA_COUNT, dh_genus_names (with eolID),
                           genus_auth (genus -> author words), dh_eol_ids count
  S9_candidate_words.csv   every single-token epithet flagged by the run-2 'B naive' or 'A raw' rule
                           (from S9_suspect_keys_run2.csv), with its author and epithet counts
  S9_emulation_check.json  re-run of the released 'low quality names' path with Taxon built twice,
                           as make_metadata.py does (EolNameLookup.taxon -> NameUpgrader.upgrade)
Run (cpu partition):
  srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=4 --mem=32G --time=00:30:00 \
    bash -c 'source "$HOME/.hpc_env.sh" && conda activate bioclip && \
             python -u /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/S9_author_vocab.py' \
    > /projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_author_vocab.log 2>&1
"""
import collections
import csv
import json
import re
import sys
import time

T0 = time.time()
ROOT = "/projects/bdbk/liv/repos/taxa_maze"
EVID = f"{ROOT}/audit/preflight/S9_bioclip1"
SNAP = f"{EVID}/bioclip_fe4bd61"
TAXON_TAB = "/u/liv/bdbk/data/tol10m/metadata/taxon.tab"
LOOKUP = "/u/liv/bdbk/data/tol10m/metadata/naming/eol_name_lookup.json"
SCRAPED = f"{SNAP}/data/eol/scraped_page_ids.csv"
AUTHOR_STOP = {"and", "ex", "in", "et", "von", "van", "der", "den", "de", "la", "le", "du", "da",
               "di", "del", "des", "dos", "das", "ter", "ten", "non", "nom", "nov", "auct", "sensu",
               "emend", "fide", "comb", "stat", "nud", "ined", "al", "sp", "spp", "var", "f", "fil",
               "subsp", "ssp", "nec", "pro", "parte", "pp", "sensulato", "sl", "ss", "orth", "corr"}
sys.path.insert(0, f"{SNAP}/src")
from imageomics import naming  # noqa: E402


def norm(s):
    return re.sub(r"[\W\d_]+", "", (s or "").lower())


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


csv.field_size_limit(sys.maxsize)
AUTH = collections.Counter()
EPI_SP = collections.Counter()
EPI_INFRA = collections.Counter()
dh_eol_ids, dh_genus_names = set(), set()
genus_auth = collections.defaultdict(set)
with open(TAXON_TAB, newline="") as fh:  # parsed as eol.py:42-63 does (csv.DictReader, tab)
    for row in csv.DictReader(fh, delimiter="\t"):
        rank = (row.get("taxonRank") or "").lower()
        words = {norm(w) for w in (row.get("authority") or "").split()}
        words = {w for w in words if len(w) >= 2 and w not in AUTHOR_STOP}
        AUTH.update(words)
        cname = (row.get("canonicalName") or "").strip()
        if row.get("eolID"):
            dh_eol_ids.add(int(row["eolID"]))
            if rank == "genus" and cname:
                dh_genus_names.add(cname.lower())
        if rank == "genus" and cname:
            genus_auth[cname.lower()] |= words
        if cname and rank in ("species", "subspecies", "variety", "form", "infraspecies"):
            target = EPI_SP if rank == "species" else EPI_INFRA
            for w in cname.split()[1:]:
                n = norm(w)
                if n:
                    target[n] += 1
log(f"taxon.tab parsed: {len(AUTH)} author words, {len(EPI_SP)} species-epithet words, "
    f"{len(EPI_INFRA)} infraspecific words, {len(dh_genus_names)} DH genus names")
with open(f"{EVID}/S9_vocab.json", "w") as fh:
    json.dump({"AUTH_COUNT": AUTH, "EPI_SPECIES_COUNT": EPI_SP, "EPI_INFRA_COUNT": EPI_INFRA,
               "dh_genus_names": sorted(dh_genus_names), "n_dh_eol_ids": len(dh_eol_ids),
               "genus_auth": {g: sorted(v) for g, v in genus_auth.items()}}, fh, ensure_ascii=False)

# Candidate words from run 2.
rows = []
with open(f"{EVID}/S9_suspect_keys_run2.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ep = r["key"].split("|")[-1]
        if " " in ep or not (r["B_naive"] == "true" or r["is_authorA"] == "true"):
            continue
        rows.append((ep, int(r["flagged_rows_train"]), r["B_naive"] == "true", r["key"]))
agg = collections.defaultdict(lambda: [0, 0, 0])
for ep, n, b, key in rows:
    a = agg[ep]
    a[0] += n
    a[1] += 1
    a[2] += int(b)
with open(f"{EVID}/S9_candidate_words.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["epithet", "norm", "flagged_train_images", "keys", "keys_from_B_naive", "author_count",
                "species_epithet_count", "infra_epithet_count"])
    for ep, (n, k, kb) in sorted(agg.items(), key=lambda x: -x[1][0]):
        nn = norm(ep)
        w.writerow([ep, nn, n, k, kb, AUTH.get(nn, 0), EPI_SP.get(nn, 0), EPI_INFRA.get(nn, 0)])
log(f"candidate words: {len(agg)}")

# Emulation with Taxon built twice (EolNameLookup.taxon builds it; NameUpgrader.upgrade rebuilds it).
scraped = {}
with open(SCRAPED, newline="") as fh:
    for r in csv.DictReader(fh):
        scraped.setdefault(int(r["page_id"]), r["scientific_name"])
lk = json.load(open(LOOKUP))
n = agree_sp = agree_g = 0
bad = []
for pid_s, (taxa, common, classes) in lk.items():
    pid = int(pid_s)
    if pid in dh_eol_ids or pid not in scraped:
        continue
    n += 1
    name = naming.clean_name(scraped[pid])
    parts = name.split()
    if len(parts) == 2:
        g = parts[0] if parts[0] in dh_genus_names else ""
        t = naming.Taxon("", "", "", "", "", g, parts[1])
    else:
        t = naming.Taxon("", "", "", "", "", "", name)
    t = naming.Taxon(**{r: getattr(t, r) for r in naming.taxon_ranks})  # second build, as in upgrade()
    ok_sp = t.species == taxa[6].lower()
    ok_g = t.genus == taxa[5].lower()
    agree_sp += ok_sp
    agree_g += ok_g
    if not (ok_sp and ok_g) and len(bad) < 20:
        bad.append({"page_id": pid, "scraped": scraped[pid], "clean": name, "emu": [t.genus, t.species],
                    "lookup": [taxa[5], taxa[6]]})
out = {"scraped_only_pages_in_lookup": n, "species_agree": agree_sp, "genus_agree": agree_g, "disagree_examples": bad}
with open(f"{EVID}/S9_emulation_check.json", "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
log("emulation (Taxon built twice): " + json.dumps({k: v for k, v in out.items() if k != "disagree_examples"}))
