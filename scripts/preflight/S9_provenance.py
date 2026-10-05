#!/usr/bin/env python
"""S9 provenance check (audit/2026-10-02_S9_bioclip1_provenance.md). Run 3 (final).

Question: did BioCLIP 1 train on TreeOfLife-10M rows whose species field holds a
placeholder (sp., spp., x, ?), an author name, or a multi-word string, and with what
text? Did it train on rows with missing ranks?

Method. BioCLIP's pipeline builds every caption from a per-source name lookup
(make_wds.py:93-113: taxon, common, classes = name_lookup[page_id]; make_txt(taxon, common)),
and its catalog is written from the finished webdataset (make_catalog.py:80-116, which reads
"__key__" and "common_name.txt" from the shards). This script
  1. re-runs BioCLIP's own Taxon class and make_txt() (copied unchanged from
     github.com/Imageomics/bioclip @ fe4bd61 into audit/preflight/S9_bioclip1/bioclip_fe4bd61)
     on the released eol_name_lookup.json, page by page;
  2. checks catalog.csv's seven rank columns and its `common` column against that, row by row;
  3. re-runs BioCLIP's "low quality names" path (eol.py:90-126) on the released
     scraped_page_ids.csv to see whether the released code reproduces the released lookup;
  4. counts suspect epithets (placeholders, author names, multi-word strings) and rows with
     missing ranks, and prints the exact five training captions for examples;
  5. accounts for EOL images that the pipeline dropped (mapping.sqlite vs lookup vs catalog);
  6. measures what option (b) of S9 would remove from pilot 1's tree;
  7. checks the BioCLIP release's own species list (embeddings/txt_emb_species.json).

Run 1 (log S9_provenance_run1_246194.log) used a naive author detector that also flagged
real epithets written with a capital letter (e.g. "Phascolarctos Cinereus"). Run 2
(S9_provenance_run2_246206.log) dropped every word found anywhere in the DH epithet vocabulary,
which also dropped 'latreille', 'linnaeus' and 'gray' (each occurs in a few DH species names).
Run 3 compares counts instead (S9_author_vocab.py; S9_candidate_words.csv):
  author word  :=  AUTH[w] >= 1  and  (EPI[w] == 0  or  AUTH[w] >= 5 * (EPI[w] + 1))
where AUTH[w] counts DH rows whose `authority` contains w, and EPI[w] counts DH species and
infraspecific canonical names that use w as an epithet word. Example: latreille 1338 vs 3 (author);
luna 251 vs 71 (epithet: Actias luna); cinereus 0 vs many (epithet).

All inputs are opened read-only. Outputs:
  audit/preflight/S9_bioclip1/S9_provenance.json
  audit/preflight/S9_bioclip1/S9_suspect_keys.csv
  audit/preflight/S9_bioclip1/S9_review_samples.json   (random samples for a by-eye precision check)
Run (cpu partition):
  srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
    bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/S9_provenance.sh \
    > /projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_provenance.log 2>&1
"""
import ast
import collections
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time

import polars as pl

T0 = time.time()
ROOT = "/projects/bdbk/liv/repos/taxa_maze"
EVID = f"{ROOT}/audit/preflight/S9_bioclip1"
SNAP = f"{EVID}/bioclip_fe4bd61"
META = "/u/liv/bdbk/data/tol10m/metadata"
CATALOG = f"{META}/catalog.csv"
LOOKUP = f"{META}/naming/eol_name_lookup.json"
TAXON_TAB = f"{META}/taxon.tab"
MAPPING = f"{META}/mapping.sqlite"
SCRAPED = f"{SNAP}/data/eol/scraped_page_ids.csv"
SEEN = f"{SNAP}/data/rarespecies/seen_in_training.json"
UNSEEN = f"{SNAP}/data/rarespecies/unseen_in_training.json"
TXT_EMB = f"{EVID}/txt_emb_species.json"
HF_TREE = [f"{EVID}/hf_tree_metadata_91debffb.json", f"{EVID}/hf_tree_embeddings_91debffb.json"]
OUT_JSON = f"{EVID}/S9_provenance.json"
OUT_KEYS = f"{EVID}/S9_suspect_keys.csv"
OUT_REVIEW = f"{EVID}/S9_review_samples.json"

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
# PLACEHOLDERS: exact-match set used in audit/preflight/data_catalog_extras.json.
PLACEHOLDERS = ["sp.", "sp", "spp.", "spp", "xx", "x", "?"]
# PLACEHOLDERS_EXT: adds the other placeholder-like single tokens found by two regex surveys of
# eol_name_lookup.json species values (report, section 4.1). Scraping debris ("undefined-1",
# "1") is listed too. Real epithets that look odd ("io", "nova", "unidentata") are not.
PLACEHOLDERS_EXT = PLACEHOLDERS + ["??", "???", "spec.", "spec", "species", "species?", "species101", "nov.",
                                   "sp.nov", "cf.", "sp.b", "sp.'", "xxx", "xxxx", "xxxxx", "indet", "undet",
                                   "undetermined", "unidentified", "hybrid", "complex", "group", "cultivar",
                                   "cv", "sedis", "undescribed_sandiego", "undefined-1", "undefined-2",
                                   "undefined-5", "1", "2"]
CONNECTORS = {"&", "ex", "et", "and", "in", "de", "von", "van", "der", "den", "la", "le", "du", "da",
              "di", "del", "des", "dos", "das", "f", "fil", "filius"}
AUTHOR_RATIO = 5
# Captions BioCLIP 1 sampled from under --text_type random (src/training/data.py:390, train.py:92-96).
TRAIN_KEYS = ["sci.txt", "com.txt", "taxon.txt", "sci_com.txt", "taxon_com.txt"]
AUTHOR_STOP = {"and", "ex", "in", "et", "von", "van", "der", "den", "de", "la", "le", "du", "da",
               "di", "del", "des", "dos", "das", "ter", "ten", "non", "nom", "nov", "auct", "sensu",
               "emend", "fide", "comb", "stat", "nud", "ined", "al", "sp", "spp", "var", "f", "fil",
               "subsp", "ssp", "nec", "pro", "parte", "pp", "sensulato", "sl", "ss", "orth", "corr"}
NA_LIKE = ["nan", "na", "null", "none", "n/a", "nil", "nat"]
INFRA_RANKS = {"species", "subspecies", "variety", "form", "forma", "infraspecies", "subvariety",
               "subform", "infraspecific name", "cultivar"}

R = {"meta": {"script": __file__, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "placeholders": PLACEHOLDERS, "placeholders_ext": PLACEHOLDERS_EXT}}
REVIEW = {}
random.seed(0)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256(path, bufsize=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm(s):
    """Letters only, lower case: 'Mill.' -> 'mill', "L'Hér." -> 'lhér'."""
    return re.sub(r"[\W\d_]+", "", (s or "").lower())


# --------------------------------------------------------------------------------------
# 0. BioCLIP's own code (unchanged copies) and file identities against Hugging Face.
# --------------------------------------------------------------------------------------
sys.path.insert(0, f"{SNAP}/src")
from imageomics import naming  # noqa: E402  (BioCLIP's naming.py, commit fe4bd61)
from imageomics import naming_reproduce  # noqa: E402


def load_function(path, name, namespace):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
    return namespace[name], fn.lineno, fn.end_lineno


make_txt, l0, l1 = load_function(f"{SNAP}/scripts/evobio10m/make_wds.py", "make_txt", {})
make_txt_rep, r0, r1 = load_function(f"{SNAP}/scripts/evobio10m/make_wds_reproduce.py", "make_txt",
                                     {"naming_reproduce": naming_reproduce})
R["meta"]["make_txt_source"] = f"scripts/evobio10m/make_wds.py:{l0}-{l1} @ fe4bd61"
R["meta"]["make_txt_reproduce_source"] = f"scripts/evobio10m/make_wds_reproduce.py:{r0}-{r1} @ fe4bd61"

hf_oid = {}
for p in HF_TREE:
    for f in json.load(open(p)):
        if f.get("lfs"):
            hf_oid[f["path"]] = f["lfs"]["oid"]
checks = {}
for local, hfpath in [(CATALOG, "metadata/catalog.csv"), (LOOKUP, "metadata/naming/eol_name_lookup.json"),
                      (MAPPING, "metadata/mapping.sqlite"), (TAXON_TAB, "metadata/taxon.tab"),
                      (TXT_EMB, "embeddings/txt_emb_species.json")]:
    d = sha256(local)
    checks[hfpath] = {"local": local, "sha256": d, "hf_lfs_oid_91debffb": hf_oid.get(hfpath),
                      "match": d == hf_oid.get(hfpath)}
    log(f"sha256 {hfpath}: match={checks[hfpath]['match']}")
R["meta"]["file_identity"] = checks

# --------------------------------------------------------------------------------------
# 1. Page-level table: BioCLIP's Taxon + make_txt applied to every lookup entry.
# --------------------------------------------------------------------------------------
with open(LOOKUP) as fh:
    raw_lookup = json.load(fh)
log(f"lookup pages: {len(raw_lookup)}")
cols = collections.defaultdict(list)
n_reclean_changed, n_empty, rep_mismatch = 0, 0, 0
for key, (taxa, common, classes) in raw_lookup.items():
    t = naming.Taxon(*taxa)
    tt = t.to_tuple()
    if tuple(taxa) != tt:
        n_reclean_changed += 1
    try:
        txt = make_txt(t, common)
    except AssertionError:
        n_empty += 1
        txt = {k: None for k in ["scientific_name.txt", "taxonomic_name.txt", "common_name.txt"] + TRAIN_KEYS}
    else:
        if txt != make_txt_rep(naming_reproduce.Taxon(*taxa), common):
            rep_mismatch += 1
    cols["page_id"].append(int(key))
    for r, v in zip(RANKS, taxa):
        cols[f"raw_{r}"].append(v)
    cols["raw_common"].append(common)
    cols["cls_species_idx"].append(classes[-1] if classes else None)
    for r, v in zip(RANKS, tt):
        cols[f"lk_{r}"].append(v)
    for k in ["scientific_name.txt", "taxonomic_name.txt", "common_name.txt"] + TRAIN_KEYS:
        cols[k].append(txt[k])
P = pl.DataFrame(dict(cols))
del cols
R["lookup"] = {"n_pages": P.height, "n_pages_where_Taxon_reclean_changes_raw": n_reclean_changed,
               "n_pages_empty_after_reclean": n_empty,
               "n_pages_make_wds_vs_make_wds_reproduce_text_differs": rep_mismatch,
               "na_like_species_pages": P.filter(pl.col("raw_species").str.to_lowercase().is_in(NA_LIKE))
               .select("page_id", "raw_genus", "raw_species").to_dicts()}
log(f"page table built: {P.height} rows; reclean changed {n_reclean_changed}; empty {n_empty}")

# --------------------------------------------------------------------------------------
# 2. Catalog rows and agreement with the lookup.
# --------------------------------------------------------------------------------------
cat = pl.read_csv(CATALOG, columns=["split", "treeoflife_id", "eol_content_id", "eol_page_id",
                                    "bioscan_filename", "inat21_filename"] + RANKS + ["common"],
                  infer_schema=False)
blank = lambda c: pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")  # noqa: E731
cat = cat.with_columns(
    [pl.when(blank(c)).then(None).otherwise(pl.col(c)).alias(c) for c in RANKS + ["common"]]
    + [pl.when(~blank("eol_content_id")).then(pl.lit("EOL"))
       .when(~blank("bioscan_filename")).then(pl.lit("BIOSCAN"))
       .when(~blank("inat21_filename")).then(pl.lit("iNat21")).otherwise(pl.lit("none")).alias("source")])
cat = cat.with_columns(
    pl.col("eol_page_id").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias("page_id"),
    pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]).alias("complete"),
    pl.concat_str([pl.when(pl.col(c).is_null()).then(pl.lit("-")).otherwise(pl.lit("x")) for c in RANKS])
    .alias("pattern"),
    pl.concat_str([pl.col(c).fill_null("") for c in RANKS], separator="|").alias("key"),
)
R["catalog"] = {"rows_by_split_source": cat.group_by("split", "source").len().sort("split", "source").to_dicts()}
log(f"catalog loaded: {cat.height} rows")

E = cat.filter((pl.col("source") == "EOL") & pl.col("split").is_in(["train", "val"]))
E = E.join(P, on="page_id", how="left")
agree = {"eol_rows_train_val": E.height,
         "rows_whose_page_is_not_in_lookup": E.filter(pl.col("lk_kingdom").is_null()).height}
EJ = E.filter(pl.col("lk_kingdom").is_not_null())
for r in RANKS:
    agree[f"mismatch_{r}"] = EJ.filter(pl.col(r).fill_null("") != pl.col(f"lk_{r}")).height
any_rank = pl.any_horizontal([pl.col(r).fill_null("") != pl.col(f"lk_{r}") for r in RANKS])
agree["rank_mismatch_examples"] = EJ.filter(any_rank).select(
    ["treeoflife_id", "split", "page_id", "key"] + [f"lk_{r}" for r in RANKS]).head(10).to_dicts()
# 2b. The `common` column was read from the training shards (make_catalog.py:159). Classify it.
cm = pl.col("common").fill_null("")
EJ = EJ.with_columns(
    pl.when(cm == pl.col("common_name.txt")).then(pl.lit("equals released make_txt common"))
    .when(cm == pl.col("scientific_name.txt")).then(pl.lit("equals scientific name (shard fell back to it)"))
    .when(cm == pl.col("taxonomic_name.txt")).then(pl.lit("equals taxonomic name (shard fell back to it)"))
    .when(cm.str.to_lowercase() == pl.col("common_name.txt").str.to_lowercase()).then(pl.lit("case-only difference"))
    .otherwise(pl.lit("other vernacular")).alias("common_class"))
agree["common_column_classes"] = EJ.group_by("common_class").len().sort("len", descending=True).to_dicts()
agree["common_examples_by_class"] = {c: EJ.filter(pl.col("common_class") == c).select(
    "key", "page_id", "common", "raw_common", "common_name.txt").head(4).to_dicts()
    for c in ["equals scientific name (shard fell back to it)", "case-only difference", "other vernacular"]}
agree["rows_where_catalog_common_equals_scientific_name"] = EJ.filter(cm == pl.col("scientific_name.txt")).height
R["catalog_vs_lookup"] = agree
E = E.join(EJ.select("treeoflife_id", "common_class"), on="treeoflife_id", how="left")
log("catalog vs lookup: " + json.dumps({k: v for k, v in agree.items() if not k.endswith("examples")
                                         and k != "common_examples_by_class"}, ensure_ascii=False))

# --------------------------------------------------------------------------------------
# 3. mapping.sqlite: split agreement and images BioCLIP's pipeline dropped.
# --------------------------------------------------------------------------------------
con = sqlite3.connect(f"file:{MAPPING}?mode=ro&immutable=1", uri=True)
M = pl.DataFrame(con.execute("SELECT evobio10m_id, content_id, page_id FROM eol").fetchall(),
                 schema=["treeoflife_id", "content_id", "page_id"], orient="row")
S = pl.DataFrame(con.execute("SELECT evobio10m_id, is_val, is_train_small FROM split").fetchall(),
                 schema=["treeoflife_id", "is_val", "is_train_small"], orient="row")
con.close()
M = M.join(S, on="treeoflife_id", how="left")
M = M.join(P.select("page_id", pl.lit(True).alias("page_in_lookup")), on="page_id", how="left") \
     .with_columns(pl.col("page_in_lookup").fill_null(False))
cat_eol_ids = cat.filter(pl.col("source") == "EOL").select("treeoflife_id", "split") \
    .group_by("treeoflife_id").agg(pl.col("split").sort().str.join(",").alias("catalog_splits"))
M = M.join(cat_eol_ids, on="treeoflife_id", how="left")
blk = set()
for d in (json.load(open(SEEN)), json.load(open(UNSEEN))):
    for imgs in d.values():
        blk |= {os.path.basename(i) for i in imgs}
M = M.with_columns(
    (pl.col("content_id").cast(pl.Utf8) + "_" + pl.col("page_id").cast(pl.Utf8) + "_eol-full-size-copy.jpg")
    .is_in(list(blk)).alias("in_rare_species_image_blacklist"))
drops = {}
for is_val, name in [(0, "train(is_val=0)"), (1, "val(is_val=1)")]:
    m = M.filter(pl.col("is_val") == is_val)
    drops[name] = {
        "mapping_images": m.height,
        "page_not_in_lookup (dropped, make_wds.py:93-94)": m.filter(~pl.col("page_in_lookup")).height,
        "distinct_pages_not_in_lookup": m.filter(~pl.col("page_in_lookup")).select("page_id").n_unique(),
        "page_in_lookup_and_in_catalog": m.filter(pl.col("page_in_lookup") & pl.col("catalog_splits").is_not_null()).height,
        "page_in_lookup_not_in_catalog": m.filter(pl.col("page_in_lookup") & pl.col("catalog_splits").is_null()).height,
        "page_in_lookup_not_in_catalog_and_in_rare_species_image_blacklist": m.filter(
            pl.col("page_in_lookup") & pl.col("catalog_splits").is_null() & pl.col("in_rare_species_image_blacklist")).height,
        "page_not_in_lookup_but_in_catalog": m.filter(~pl.col("page_in_lookup") & pl.col("catalog_splits").is_not_null()).height,
    }
drops["split_table_rows_missing_for_eol_ids"] = M.filter(pl.col("is_val").is_null()).height
chk = M.filter(pl.col("catalog_splits").is_not_null()).with_columns(
    pl.col("catalog_splits").str.contains("val").alias("cat_val"))
drops["catalog_split_vs_is_val_disagree"] = chk.filter(pl.col("cat_val") != (pl.col("is_val") == 1)).height
drops["catalog_eol_ids_not_in_mapping"] = cat_eol_ids.join(M.select("treeoflife_id"), on="treeoflife_id", how="anti").height
scraped, scraped_extra = {}, {}
with open(SCRAPED, newline="") as fh:  # read as eol.py:110 does (helpers.csvreader = csv.DictReader)
    for row in csv.DictReader(fh):
        pid = int(row["page_id"])
        if pid not in scraped:
            scraped[pid] = row["scientific_name"]
            scraped_extra[pid] = ",".join(row.get(None) or [])  # text after an unquoted comma
R["pipeline_drops_eol"] = drops
log("pipeline drops: " + json.dumps(drops))

# --------------------------------------------------------------------------------------
# 4. Vocabularies from EOL DH taxon.tab (parsed as eol.py:42-63 does) and author detectors.
# --------------------------------------------------------------------------------------
csv.field_size_limit(sys.maxsize)
dh_eol_ids, dh_genus_names = set(), set()
genus_auth = collections.defaultdict(set)
genus_auth_raw = collections.defaultdict(set)
AUTH = collections.Counter()   # word -> number of DH rows whose `authority` contains it
EPI = collections.Counter()    # word -> number of DH species/infraspecific names using it as an epithet word
rank_counter = collections.Counter()
with open(TAXON_TAB, newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        rank = (row.get("taxonRank") or "").lower()
        rank_counter[rank] += 1
        auth = row.get("authority") or ""
        words = {norm(w) for w in auth.split()}
        words = {w for w in words if len(w) >= 2 and w not in AUTHOR_STOP}
        AUTH.update(words)
        cname = (row.get("canonicalName") or "").strip()
        if row.get("eolID"):
            dh_eol_ids.add(int(row["eolID"]))
            if rank == "genus" and cname:
                dh_genus_names.add(cname.lower())  # eol.py:96-100 genus_lookup keys
        if rank == "genus" and cname:
            genus_auth[cname.lower()] |= words
            if auth:
                genus_auth_raw[cname.lower()].add(auth)
        if cname and rank in ("species", "subspecies", "variety", "form", "infraspecies"):
            for w in cname.split()[1:]:
                n = norm(w)
                if n:
                    EPI[n] += 1
AUTHOR_WORDS = {w for w, a in AUTH.items() if a >= 1 and (EPI.get(w, 0) == 0 or a >= AUTHOR_RATIO * (EPI.get(w, 0) + 1))}
log(f"taxon.tab: {len(dh_eol_ids)} DH page ids, {len(dh_genus_names)} DH genus names, "
    f"{len(AUTH)} author words, {len(EPI)} epithet words, {len(AUTHOR_WORDS)} words pass the author rule")
probe = ["latreille", "linnaeus", "fabricius", "walker", "gray", "hübner", "mill", "pers", "fr", "dana", "simon",
         "smith", "cuvier", "luna", "ascanius", "cinereus", "typus", "leonina", "indri", "macrocephalus", "bicolor"]
R["vocab"] = {"taxonRank_counts_top": rank_counter.most_common(25), "n_author_words": len(AUTH),
              "n_epithet_words": len(EPI), "n_words_passing_author_rule": len(AUTHOR_WORDS),
              "author_rule": f"AUTH[w] >= 1 and (EPI[w] == 0 or AUTH[w] >= {AUTHOR_RATIO} * (EPI[w] + 1))",
              "probe": {w: {"AUTH": AUTH.get(w, 0), "EPI": EPI.get(w, 0), "author_word": w in AUTHOR_WORDS} for w in probe}}
log("probe: " + json.dumps(R["vocab"]["probe"], ensure_ascii=False))

# 4a. Does the released code reproduce the released lookup for scraped-only pages?
# eol.py:110-126 + guess_value (eol.py:82-88); Taxon is built twice, as in make_metadata.py:87-90
# (EolNameLookup.taxon builds it, NameUpgrader.upgrade rebuilds it at naming.py:517-519).
emu = []
for pid_s, (taxa, common, classes) in raw_lookup.items():
    pid = int(pid_s)
    if pid in dh_eol_ids or pid not in scraped:
        continue
    name = naming.clean_name(scraped[pid])
    parts = name.split()
    if len(parts) == 2:
        g = parts[0] if parts[0] in dh_genus_names else ""
        t = naming.Taxon("", "", "", "", "", g, parts[1])
    else:
        t = naming.Taxon("", "", "", "", "", "", name)
    t = naming.Taxon(**{r: getattr(t, r) for r in naming.taxon_ranks})
    emu.append((pid, scraped[pid], scraped_extra[pid], name, t.genus, t.species, taxa[5].lower(), taxa[6].lower()))
EM_ = pl.DataFrame(emu, schema=["page_id", "scraped_name_csv_field", "csv_text_after_comma", "clean_name",
                                "emu_genus", "emu_species", "lookup_genus", "lookup_species"], orient="row")
EM_ = EM_.join(E.filter(pl.col("split") == "train").group_by("page_id").len().rename({"len": "train_images"}),
               on="page_id", how="left").with_columns(pl.col("train_images").fill_null(0))
bad_sp = EM_.filter(pl.col("emu_species") != pl.col("lookup_species"))
bad_g = EM_.filter(pl.col("emu_genus") != pl.col("lookup_genus"))
comma = EM_.filter(pl.col("csv_text_after_comma") != "")
R["released_code_vs_released_lookup_scraped_pages"] = {
    "scraped_only_pages_in_lookup": EM_.height,
    "species_agrees": EM_.height - bad_sp.height, "species_disagrees": bad_sp.height,
    "genus_agrees": EM_.height - bad_g.height, "genus_disagrees": bad_g.height,
    "species_disagree_examples": bad_sp.sort("train_images", descending=True).head(10).to_dicts(),
    "pages_whose_csv_row_has_text_after_an_unquoted_comma": comma.height,
    "train_images_on_those_pages": int(comma["train_images"].sum()),
    "comma_pages_largest": comma.sort("train_images", descending=True).head(8).to_dicts(),
}
log("released code vs lookup: " + json.dumps({k: v for k, v in R["released_code_vs_released_lookup_scraped_pages"].items()
                                               if not (k.endswith("examples") or k.endswith("largest"))}))

# 4b. Flags on catalog rows.
DH = pl.DataFrame({"page_id": sorted(dh_eol_ids)}, schema={"page_id": pl.Int64}).with_columns(pl.lit(True).alias("page_in_DH"))
GA = pl.DataFrame({"g_lower": list(genus_auth.keys()), "genus_author_words": [sorted(v) for v in genus_auth.values()]})
AW = pl.DataFrame({"tok": sorted(AUTHOR_WORDS)}).with_columns(pl.lit(True).alias("is_author_word"))
EVC = pl.DataFrame({"tok": list(EPI.keys()), "epi_count": list(EPI.values())})
AUTHOR_WORDS_LIST = sorted(AUTHOR_WORDS)
CONN_LIST = sorted(CONNECTORS)


def word_info(name):
    w = (name or "").split()
    w2 = w[1] if len(w) >= 2 else ""
    wl = w[-1] if len(w) >= 3 else ""
    return {"w2_cap": bool(w2[:1].isupper()), "w2_norm": norm(w2), "wlast_cap": bool(wl[:1].isupper()), "wlast_norm": norm(wl)}


SC = pl.DataFrame({"page_id": list(scraped.keys()), "scraped_name": list(scraped.values()),
                   "csv_text_after_comma": [scraped_extra[p] for p in scraped.keys()]})
SC = SC.with_columns(pl.col("scraped_name").map_elements(word_info, return_dtype=pl.Struct(
    {"w2_cap": pl.Boolean, "w2_norm": pl.Utf8, "wlast_cap": pl.Boolean, "wlast_norm": pl.Utf8})).alias("wi")).unnest("wi")
words_norm = pl.col("species").str.split(" ").list.eval(
    pl.element().str.to_lowercase().str.replace_all(r"[\W\d_]+", "")).list.eval(
    pl.element().filter((pl.element() != "") & ~pl.element().is_in(CONN_LIST)))
E = E.with_columns(pl.col("genus").str.to_lowercase().alias("g_lower"),
                   pl.col("species").str.to_lowercase().str.replace_all(r"[\W\d_]+", "").alias("e_norm"),
                   pl.col("species").str.split(" ").list.first().str.to_lowercase().str.replace_all(r"[\W\d_]+", "").alias("efirst_norm"),
                   pl.col("species").str.split(" ").list.last().str.to_lowercase().str.replace_all(r"[\W\d_]+", "").alias("elast_norm"),
                   words_norm.alias("e_words"))
E = E.with_columns(pl.col("e_words").list.eval(pl.element().is_in(AUTHOR_WORDS_LIST)).list.all().alias("all_words_author"),
                   pl.col("e_words").list.len().alias("n_e_words"))
E = E.join(DH, on="page_id", how="left").with_columns(pl.col("page_in_DH").fill_null(False))
E = E.join(GA, on="g_lower", how="left").join(SC, on="page_id", how="left")
E = E.join(AW.rename({"tok": "e_norm"}), on="e_norm", how="left") \
     .join(AW.rename({"tok": "elast_norm", "is_author_word": "elast_is_author_word"}), on="elast_norm", how="left") \
     .join(EVC.rename({"tok": "e_norm"}), on="e_norm", how="left")
E = E.with_columns([pl.col(c).fill_null(False) for c in ["is_author_word", "elast_is_author_word", "w2_cap", "wlast_cap",
                                                         "all_words_author"]]
                   + [pl.col("epi_count").fill_null(0)])
single = pl.col("species").is_not_null() & ~pl.col("species").str.contains(" ").fill_null(False)
multi = pl.col("species").str.contains(" ").fill_null(False)
scraped_only = pl.col("scraped_name").is_not_null() & ~pl.col("page_in_DH")
E = E.with_columns(
    pl.col("species").is_in(PLACEHOLDERS).fill_null(False).alias("is_placeholder"),
    pl.col("species").is_in(PLACEHOLDERS_EXT).fill_null(False).alias("is_placeholder_ext"),
    multi.alias("is_multiword"),
    (single & pl.col("genus_author_words").list.contains(pl.col("e_norm")).fill_null(False)).alias("A_raw"),
    (single & scraped_only & pl.col("w2_cap") & (pl.col("w2_norm") == pl.col("e_norm"))).fill_null(False).alias("B_naive"),
)
E = E.with_columns(
    (pl.col("A_raw") & pl.col("is_author_word")).alias("is_authorA"),
    (pl.col("B_naive") & pl.col("is_author_word")).alias("is_authorB"),
    # Multi-word string made only of author words, from a scraped 'Genus Author1 & Author2' page.
    (multi & scraped_only & pl.col("w2_cap") & (pl.col("w2_norm") == pl.col("efirst_norm"))
     & (pl.col("n_e_words") >= 1) & pl.col("all_words_author")).fill_null(False).alias("is_authorBmulti"),
)
E = E.with_columns(
    # Multi-word epithet whose last word is an author carried over from the scraped name.
    (multi & ~pl.col("is_authorBmulti") & scraped_only & pl.col("wlast_cap") & (pl.col("wlast_norm") == pl.col("elast_norm"))
     & pl.col("elast_is_author_word")).fill_null(False).alias("is_author_remnant"),
    (pl.col("is_authorA") | pl.col("is_authorB") | pl.col("is_authorBmulti")).alias("is_author"),
)
E = E.with_columns((pl.col("is_placeholder") | pl.col("is_author")).alias("is_S9b"),
                   (pl.col("is_placeholder_ext") | pl.col("is_author")).alias("is_S9b_ext"))
mw = pl.col("species")
E = E.with_columns(
    pl.when(~pl.col("is_multiword")).then(None)
    .when(pl.col("is_authorBmulti")).then(pl.lit("author-only string"))
    .when(mw.str.starts_with("sp.") | mw.str.starts_with("sp ")).then(pl.lit("sp. <tag>"))
    .when(mw.str.contains(r"(^|\s)cf\.?(\s|$)")).then(pl.lit("cf."))
    .when(mw.str.contains(r"(^|\s)aff\.?(\s|$)")).then(pl.lit("aff."))
    .when(mw.str.contains(r"(^|\s)(x|×)(\s|$)")).then(pl.lit("hybrid x"))
    .when(mw.str.contains(r"(^|\s)subg\.?(\s|$)")).then(pl.lit("subg."))
    .when(pl.col("is_author_remnant")).then(pl.lit("epithet + author remnant"))
    .otherwise(pl.lit("other multi-word")).alias("multiword_kind"))
E = E.with_columns(pl.concat_str([pl.col(c) for c in RANKS[:6]], separator="|").alias("gkey"),
                   pl.col("species").str.split(" ").list.first().alias("first_word"))
sib = E.filter(pl.col("complete") & ~pl.col("is_multiword")).select("gkey", pl.col("species").alias("first_word")).unique() \
       .with_columns(pl.lit(True).alias("first_word_is_sibling_epithet"))
E = E.join(sib, on=["gkey", "first_word"], how="left").with_columns(
    (pl.col("first_word_is_sibling_epithet").fill_null(False) & pl.col("is_multiword")).alias("first_word_is_sibling_epithet"))
log("flags done")

FLAGS = [("all rows", pl.lit(True)),
         ("placeholder (exact set)", pl.col("is_placeholder")),
         ("placeholder (extended set)", pl.col("is_placeholder_ext")),
         ("author A (word of the genus authority; passes author rule)", pl.col("is_authorA")),
         ("author B (scraped 'Genus Author' page; passes author rule)", pl.col("is_authorB")),
         ("author B multi-word (author-only string)", pl.col("is_authorBmulti")),
         ("author (A or B or B multi-word)", pl.col("is_author")),
         ("author A and B", pl.col("is_authorA") & pl.col("is_authorB")),
         ("sensitivity: A without author rule", pl.col("A_raw")),
         ("sensitivity: B naive (run 1 rule)", pl.col("B_naive")),
         ("sensitivity: B naive rejected by author rule", pl.col("B_naive") & ~pl.col("is_author_word")),
         ("S9 option (b) set = placeholder (exact) or author", pl.col("is_S9b")),
         ("S9 option (b) set, extended placeholders", pl.col("is_S9b_ext")),
         ("multi-word", pl.col("is_multiword")),
         ("multi-word, epithet + author remnant", pl.col("is_author_remnant")),
         ("multi-word, first word is a sibling epithet", pl.col("first_word_is_sibling_epithet"))]


def cat_counts(df):
    out = {}
    for name, expr in FLAGS:
        d = df.filter(expr)
        out[name] = {"images": d.height, "keys": d.select("key").n_unique()}
    out["multi-word by kind"] = df.filter(pl.col("is_multiword")).group_by("multiword_kind").agg(
        pl.len().alias("images"), pl.col("key").n_unique().alias("keys")).sort("images", descending=True).to_dicts()
    out["placeholder (extended) by token"] = df.filter(pl.col("is_placeholder_ext")).group_by("species").agg(
        pl.len().alias("images"), pl.col("key").n_unique().alias("keys")).sort("images", descending=True).to_dicts()
    return out


cats = {}
for split in ["train", "val"]:
    d = E.filter(pl.col("split") == split)
    cats[f"EOL {split}, complete lineage"] = cat_counts(d.filter(pl.col("complete")))
    cats[f"EOL {split}, incomplete lineage"] = cat_counts(d.filter(~pl.col("complete")))
R["categories"] = cats
TC = E.filter((pl.col("split") == "train") & pl.col("complete"))
R["author_top_epithets_train_complete"] = TC.filter(pl.col("is_author")).group_by("species").agg(
    pl.len().alias("images"), pl.col("key").n_unique().alias("keys"),
    pl.col("is_authorA").sum().alias("images_A"), pl.col("is_authorB").sum().alias("images_B"),
    pl.col("is_authorBmulti").sum().alias("images_Bmulti")) \
    .sort("images", descending=True).head(50).to_dicts()
prov = {}
for name, expr in [("placeholder (exact)", pl.col("is_placeholder")), ("author", pl.col("is_author")),
                   ("multi-word", pl.col("is_multiword")), ("all complete", pl.lit(True))]:
    d = TC.filter(expr)
    prov[name] = {"images": d.height, "page_in_DH": d.filter(pl.col("page_in_DH")).height,
                  "page_in_scraped_only": d.filter(~pl.col("page_in_DH") & pl.col("scraped_name").is_not_null()).height,
                  "page_in_neither": d.filter(~pl.col("page_in_DH") & pl.col("scraped_name").is_null()).height}
R["provenance_of_page_names_train_complete"] = prov
# Shard text check: rows whose catalog `common` (read from the shards) is the scientific name.
shard = {}
for name, expr in [("placeholder (exact)", pl.col("is_placeholder")), ("author", pl.col("is_author")),
                   ("multi-word", pl.col("is_multiword")), ("all complete", pl.lit(True))]:
    d = TC.filter(expr)
    shard[name] = d.group_by("common_class").len().sort("len", descending=True).to_dicts()
R["catalog_common_class_by_category_train_complete"] = shard
log("categories done")

# Review samples for a by-eye precision check.
def sample_keys(df, n):
    keys = df.select("key").unique().sort("key")["key"].to_list()
    random.shuffle(keys)
    pick = set(keys[:n])
    return df.filter(pl.col("key").is_in(list(pick))).group_by("key").agg(
        pl.len().alias("images"), pl.col("scraped_name").drop_nulls().unique().head(2).alias("scraped_names"),
        pl.col("genus_author_words").first().alias("genus_author_words")).sort("key").to_dicts()


REVIEW["author_A_and_B"] = sample_keys(TC.filter(pl.col("is_authorA") & pl.col("is_authorB")), 30)
REVIEW["author_B_only"] = sample_keys(TC.filter(pl.col("is_authorB") & ~pl.col("is_authorA")), 40)
REVIEW["author_A_only"] = sample_keys(TC.filter(pl.col("is_authorA") & ~pl.col("is_authorB")), 30)
REVIEW["author_B_multiword"] = sample_keys(TC.filter(pl.col("is_authorBmulti")), 30)
REVIEW["B_naive_rejected_by_author_rule"] = sample_keys(TC.filter(pl.col("B_naive") & ~pl.col("is_author_word")), 60)
REVIEW["author_remnant"] = sample_keys(TC.filter(pl.col("is_author_remnant")), 30)
REVIEW["other_multiword"] = sample_keys(TC.filter(pl.col("multiword_kind") == "other multi-word"), 40)
REVIEW["placeholder_ext_not_exact"] = sample_keys(TC.filter(pl.col("is_placeholder_ext") & ~pl.col("is_placeholder")), 30)
odd = TC.filter(~pl.col("is_multiword") & ~pl.col("is_placeholder_ext") & ~pl.col("is_author")
                & ((pl.col("species").str.len_chars() <= 2)
                   | ~pl.col("species").str.contains(r"^[^\W\d_](?:[^\W\d_]|-)*[^\W\d_]$")))
R["odd_single_token_epithets_not_flagged_train_complete"] = {
    "images": odd.height, "keys": odd.select("key").n_unique(),
    "top": odd.group_by("species").agg(pl.len().alias("images"), pl.col("key").n_unique().alias("keys"))
    .sort("images", descending=True).head(60).to_dicts()}

# --------------------------------------------------------------------------------------
# 5. Exact captions for examples (the five BioCLIP 1 sampled from).
# --------------------------------------------------------------------------------------
cap_cols = ["key", "page_id", "page_in_DH", "scraped_name", "csv_text_after_comma", "raw_common", "common",
            "common_name.txt"] + TRAIN_KEYS


def examples(df, n):
    g = df.group_by(["key", "page_id"]).agg(pl.len().alias("images_on_page")).sort("images_on_page", descending=True).head(n)
    return g.join(df.unique(subset=["key", "page_id"]).select(cap_cols), on=["key", "page_id"], how="left") \
        .sort("images_on_page", descending=True).to_dicts()


ex = {}
ex["Bombus latreille (all pages)"] = examples(E.filter((pl.col("genus") == "Bombus") & (pl.col("species") == "latreille")), 10)
ex["placeholder, largest keys"] = examples(TC.filter(pl.col("is_placeholder")), 6)
ex["author, largest keys (excluding Bombus)"] = examples(TC.filter(pl.col("is_author") & (pl.col("genus") != "Bombus")), 6)
ex["author abbreviation (botanical)"] = examples(TC.filter(pl.col("is_author") & pl.col("species").str.ends_with(".")), 4)
ex["multi-word, largest keys"] = examples(TC.filter(pl.col("is_multiword")), 6)
ex["multi-word 'sp. <tag>'"] = examples(TC.filter(pl.col("multiword_kind") == "sp. <tag>"), 4)
ex["multi-word hybrid"] = examples(TC.filter(pl.col("multiword_kind") == "hybrid x"), 3)
ex["multi-word epithet + author remnant"] = examples(TC.filter(pl.col("is_author_remnant")), 4)
ex["named in S9 audit: fasciatus roseatus pilsbry / woodsii backeb. / cf. garbeanus leme"] = examples(
    E.filter(pl.col("species").is_in(["fasciatus roseatus pilsbry", "woodsii backeb.", "cf. garbeanus leme"])), 6)
ex["incomplete rows whose species is an author (genus not found)"] = examples(
    E.filter(~pl.col("complete") & pl.col("is_authorB")), 6)
ex["na-like epithet"] = examples(E.filter(pl.col("raw_species").str.to_lowercase().is_in(NA_LIKE)), 4)
R["caption_examples"] = ex

# --------------------------------------------------------------------------------------
# 6. Bombus: how BioCLIP labelled genus-level Bombus images.
# --------------------------------------------------------------------------------------
B = E.filter(pl.col("genus") == "Bombus")
R["bombus"] = {
    "rows_by_split_and_pattern": B.group_by("split", "pattern").len().sort("split", "pattern").to_dicts(),
    "latreille_rows_by_split_and_page": B.filter(pl.col("species") == "latreille").group_by(
        "split", "page_id", "scraped_name", "csv_text_after_comma").len().sort("split", "page_id").to_dicts(),
    "genus_only_rows_examples": examples(B.filter(pl.col("species").is_null()), 5),
    "author_keys_in_Bombus": B.filter(pl.col("is_author")).group_by("key", "split").len().sort("key", "split").to_dicts(),
    "n_distinct_Bombus_epithets_train_complete": TC.filter(pl.col("genus") == "Bombus").select("species").n_unique(),
    "genus_authority_in_DH": sorted(genus_auth_raw.get("bombus", [])),
    "species_class_index_of_latreille_pages": B.filter(pl.col("species") == "latreille").select("page_id", "cls_species_idx").unique().to_dicts(),
    "make_txt_full_output_page_65269955": make_txt(naming.Taxon(*raw_lookup["65269955"][0]), raw_lookup["65269955"][1]),
    "lookup_raw_entry_page_65269955": raw_lookup["65269955"],
    "scraped_csv_field_page_65269955": scraped.get(65269955),
    "scraped_csv_text_after_comma_page_65269955": scraped_extra.get(65269955),
    "page_65269955_in_DH": 65269955 in dh_eol_ids,
    "released_clean_name_of_csv_field": naming.clean_name(scraped.get(65269955, "")),
    "released_clean_name_of_full_line_text": naming.clean_name("Bombus Latreille, 1802"),
}

# --------------------------------------------------------------------------------------
# 7. Rows with missing ranks: patterns and the captions BioCLIP trained on.
# --------------------------------------------------------------------------------------
part = {}
for split in ["train", "val"]:
    d = E.filter(pl.col("split") == split)
    part[split] = {"rows": d.height, "incomplete_rows": d.filter(~pl.col("complete")).height,
                   "interior_gap_rows": d.filter(~pl.col("complete") & pl.col("pattern").str.contains(r"-x")).height,
                   "patterns": d.filter(~pl.col("complete")).group_by("pattern").len().sort("len", descending=True).head(12).to_dicts()}
inc = E.filter((pl.col("split") == "train") & ~pl.col("complete"))
part["caption_examples_by_pattern"] = {pat: examples(inc.filter(pl.col("pattern") == pat), 3)
                                       for pat in [r["pattern"] for r in part["train"]["patterns"][:8]]}
part["species_only_rows_top_values"] = inc.filter(pl.col("pattern") == "------x").group_by("species").len() \
    .sort("len", descending=True).head(20).to_dicts()
R["partial_lineage"] = part

# --------------------------------------------------------------------------------------
# 8. Effect of option (b) on the pilot's tree (EOL train, complete rows), key level.
# --------------------------------------------------------------------------------------
pref = lambda k: pl.concat_str([pl.col(c) for c in RANKS[:k]], separator="|")  # noqa: E731


def tree_effect(flag):
    keys_all = TC.group_by("key").agg(pl.len().alias("N_s"), pl.col(flag).any().alias("s9"),
                                      pl.col(flag).all().alias("s9_all_rows"),
                                      *[pref(k).first().alias(f"a{k}") for k in range(1, 7)])
    out = {"keys_with_mixed_row_flags": keys_all.filter(pl.col("s9") & ~pl.col("s9_all_rows")).height}
    for k, rank in zip(range(1, 7), RANKS[:6]):
        a = keys_all.group_by(f"a{k}").agg(pl.len().alias("n_before"), (~pl.col("s9")).sum().alias("n_after"),
                                           pl.col("N_s").sum().alias("img_before"))
        out[rank] = {"nodes": a.height,
                     "nodes_that_vanish": a.filter(pl.col("n_after") == 0).height,
                     "images_under_vanishing_nodes": int(a.filter(pl.col("n_after") == 0)["img_before"].sum()),
                     "nodes_from_>=2_to_1_species": a.filter((pl.col("n_before") >= 2) & (pl.col("n_after") == 1)).height,
                     "nodes_from_1_to_0_species": a.filter((pl.col("n_before") == 1) & (pl.col("n_after") == 0)).height}
    out["species_keys_before"] = keys_all.height
    out["species_keys_removed"] = keys_all.filter(pl.col("s9")).height
    out["images_before"] = int(keys_all["N_s"].sum())
    out["images_removed"] = int(keys_all.filter(pl.col("s9"))["N_s"].sum())
    out["largest_N_s_before"] = keys_all.sort("N_s", descending=True).head(1).select("key", "N_s").to_dicts()
    out["largest_N_s_after"] = keys_all.filter(~pl.col("s9")).sort("N_s", descending=True).head(1).select("key", "N_s").to_dicts()
    out["largest_removed_keys"] = keys_all.filter(pl.col("s9")).sort("N_s", descending=True).head(15).select("key", "N_s").to_dicts()
    out["mixed_flag_keys_examples"] = keys_all.filter(pl.col("s9") & ~pl.col("s9_all_rows")).sort("N_s", descending=True).head(10).select("key", "N_s").to_dicts()
    out["vanishing_genera_largest"] = keys_all.group_by("a6").agg(
        (~pl.col("s9")).sum().alias("n_after"), pl.len().alias("n_before"), pl.col("N_s").sum().alias("images"),
        pl.col("key").str.split("|").list.last().alias("epithets")) \
        .filter(pl.col("n_after") == 0).sort("images", descending=True).head(12).to_dicts()
    return out


R["option_b_effect_on_tree_train"] = {"exact placeholders + authors": tree_effect("is_S9b"),
                                      "extended placeholders + authors": tree_effect("is_S9b_ext")}
SK = E.filter(pl.col("complete") & (pl.col("is_S9b_ext") | pl.col("is_multiword") | pl.col("B_naive")))
key_n = E.filter(pl.col("complete")).group_by("key").agg((pl.col("split") == "train").sum().alias("key_rows_train"),
                                                         (pl.col("split") == "val").sum().alias("key_rows_val"))
sk = SK.group_by("key").agg(
    (pl.col("split") == "train").sum().alias("flagged_rows_train"), (pl.col("split") == "val").sum().alias("flagged_rows_val"),
    pl.col("is_placeholder").any(), pl.col("is_placeholder_ext").any(), pl.col("is_authorA").any(), pl.col("is_authorB").any(),
    pl.col("is_authorBmulti").any(), pl.col("is_author").any(), pl.col("B_naive").any(),
    pl.col("is_author_word").first(), pl.col("epi_count").first(), pl.col("multiword_kind").first(),
    pl.col("is_author_remnant").any(), pl.col("first_word_is_sibling_epithet").first(),
    pl.col("page_id").unique().sort().cast(pl.Utf8).str.join(";").alias("page_ids"),
    pl.col("scraped_name").drop_nulls().unique().sort().str.join(" ; ").alias("scraped_names"),
    pl.col("taxon.txt").first(), pl.col("com.txt").first()).join(key_n, on="key", how="left") \
    .sort("flagged_rows_train", descending=True)
sk.write_csv(OUT_KEYS)
R["suspect_key_table"] = {"path": OUT_KEYS, "rows": sk.height}

# --------------------------------------------------------------------------------------
# 9. BioCLIP release's species list (embeddings/txt_emb_species.json).
# --------------------------------------------------------------------------------------
emb = json.load(open(TXT_EMB))
etup = [tuple(x[0]) for x in emb]
EMB = pl.DataFrame({"key": ["|".join(t) for t in etup], "species": [t[6] for t in etup], "genus": [t[5] for t in etup],
                    "common": [x[1] for x in emb], "has_empty_rank": [any(v == "" for v in t) for t in etup]})
EMB = EMB.with_columns(pl.col("genus").str.to_lowercase().alias("g_lower"),
                       pl.col("species").str.to_lowercase().str.replace_all(r"[\W\d_]+", "").alias("e_norm"))
EMB = EMB.join(GA, on="g_lower", how="left").join(AW.rename({"tok": "e_norm"}), on="e_norm", how="left") \
         .with_columns(pl.col("is_author_word").fill_null(False))
ck = cat.filter(pl.col("complete")).select("key").unique()
ekeys = EMB.select("key").unique()
s9keys = TC.filter(pl.col("is_S9b")).select("key").unique()
R["txt_emb_species"] = {
    "entries": EMB.height, "distinct_keys": ekeys.height,
    "entries_with_an_empty_rank": int(EMB["has_empty_rank"].sum()),
    "Bombus latreille entry": EMB.filter(pl.col("key") == "Animalia|Arthropoda|Insecta|Hymenoptera|Apidae|Bombus|latreille").to_dicts(),
    "placeholder (exact) entries": EMB.filter(pl.col("species").is_in(PLACEHOLDERS)).height,
    "placeholder (extended) entries": EMB.filter(pl.col("species").is_in(PLACEHOLDERS_EXT)).height,
    "multi-word entries": EMB.filter(pl.col("species").str.contains(" ")).height,
    "author-A entries (genus authority word, passes author rule)": EMB.filter(
        ~pl.col("species").str.contains(" ") & pl.col("genus_author_words").list.contains(pl.col("e_norm")).fill_null(False)
        & pl.col("is_author_word")).height,
    "keys_in_emb_and_catalog_complete(any split/source)": ekeys.join(ck, on="key", how="semi").height,
    "keys_in_emb_not_in_catalog_complete": ekeys.join(ck, on="key", how="anti").height,
    "keys_in_catalog_complete_not_in_emb": ck.join(ekeys, on="key", how="anti").height,
    "EOL_train_complete_keys": TC.select("key").n_unique(),
    "EOL_train_complete_keys_in_emb": TC.select("key").unique().join(ekeys, on="key", how="semi").height,
    "S9b_train_keys": s9keys.height,
    "S9b_train_keys_in_emb": s9keys.join(ekeys, on="key", how="semi").height,
}

# --------------------------------------------------------------------------------------
# 10. Other sources (in BioCLIP 1's training set, not in pilot 1's local data).
# --------------------------------------------------------------------------------------
oth = {}
for src in ["BIOSCAN", "iNat21"]:
    d = cat.filter((pl.col("source") == src) & (pl.col("split") == "train"))
    oth[src] = {"train_rows": d.height, "complete": d.filter(pl.col("complete")).height,
                "placeholder_exact_rows": d.filter(pl.col("species").is_in(PLACEHOLDERS)).height,
                "multiword_epithet_rows": d.filter(pl.col("species").str.contains(" ").fill_null(False)).height}
R["other_sources_train"] = oth

R["meta"]["elapsed_s"] = round(time.time() - T0, 1)
with open(OUT_JSON, "w") as fh:
    json.dump(R, fh, indent=1, default=str, ensure_ascii=False)
with open(OUT_REVIEW, "w") as fh:
    json.dump(REVIEW, fh, indent=1, default=str, ensure_ascii=False)
log(f"wrote {OUT_JSON}, {OUT_KEYS}, {OUT_REVIEW}")
print(json.dumps(R, indent=1, default=str, ensure_ascii=False))
