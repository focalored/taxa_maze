#!/usr/bin/env python
"""S9 exploration: raw species strings of complete-lineage ToL-10M EOL train/val rows, and the
EOL taxon.tab columns that can tell an author name from an epithet. Prints everything to stdout.
Run: srun <cpu flags> --output=<repo>/audit/preflight/s9_explore_%j.log bash s9_run.sh s9_explore.py
"""
import collections
import json
import time

import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
TAXON = "/u/liv/bdbk/data/tol10m/metadata/taxon.tab"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
T0 = time.time()
pl.Config.set_tbl_rows(400)
pl.Config.set_fmt_str_lengths(120)
pl.Config.set_tbl_width_chars(250)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


df = pl.read_csv(CATALOG, columns=["split", "treeoflife_id", "eol_content_id", "eol_page_id"] + RANKS + ["common"],
                 infer_schema=False)
log("catalog", df.shape)
E = df.filter(pl.col("eol_content_id").is_not_null() & pl.col("split").is_in(["train", "val"]))
C = E.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
log("EOL train+val", E.height, "complete", C.height, C.group_by("split").len().sort("split").to_dicts())

sp = pl.col("species")
raw = C.select(
    (sp != sp.str.strip_chars()).sum().alias("padded"),
    sp.str.contains(r"^\p{Lu}").sum().alias("starts_upper"),
    sp.str.contains(r"\p{Lu}").sum().alias("any_upper"),
    sp.str.contains(r"[^\x00-\x7F]").sum().alias("non_ascii"),
    sp.str.contains(r"\s\s").sum().alias("double_space"),
    sp.str.contains(r"\t|\n|\r").sum().alias("tab_or_newline"),
)
log("raw species checks (rows)", raw.to_dicts())
for c in RANKS[:-1]:
    log(c, "padded", int((C[c] != C[c].str.strip_chars()).sum()),
        "starts_lower", int(C[c].str.contains(r"^\p{Ll}").sum()))

ep = (C.group_by("species").agg((pl.col("split") == "train").sum().alias("n_tr"),
                                (pl.col("split") == "val").sum().alias("n_va"),
                                pl.col("genus").n_unique().alias("n_genera"))
      .with_columns(sp.str.split(" ").list.len().alias("n_words"))
      .sort("n_tr", descending=True))
log("distinct epithets", ep.height, "word-count hist",
    ep.group_by("n_words").agg(pl.len(), pl.col("n_tr").sum(), pl.col("n_va").sum()).sort("n_words").to_dicts())

chars = collections.Counter()
for s, n in zip(ep["species"].to_list(), (ep["n_tr"] + ep["n_va"]).to_list()):
    for ch in set(s):
        if not ("a" <= ch <= "z"):
            chars[ch] += 1
log("non a-z characters, by number of distinct epithets containing them:",
    sorted(chars.items(), key=lambda kv: -kv[1])[:80])

short = ep.filter(sp.str.len_chars() <= 3).sort("n_tr", descending=True)
log("epithets with <= 3 characters", short.height)
print(short.head(150))
odd = ep.filter(~sp.str.contains(r"^[a-z]+(-[a-z]+)*$")).sort("n_tr", descending=True)
log("epithets that are not one lowercase a-z word (hyphens allowed)", odd.height)
print(odd.head(300))

mw = ep.filter(pl.col("n_words") >= 2)
tok = (mw.with_columns(sp.str.split(" ").alias("tok"))
       .with_columns(pl.col("tok").list.first().alias("first"), pl.col("tok").list.last().alias("last")))
log("first tokens of multi-word epithets (distinct epithets, train rows)")
print(tok.group_by("first").agg(pl.len().alias("n_ep"), pl.col("n_tr").sum()).sort("n_ep", descending=True).head(120))
log("last tokens of multi-word epithets")
print(tok.group_by("last").agg(pl.len().alias("n_ep"), pl.col("n_tr").sum()).sort("n_ep", descending=True).head(250))
allt = tok.select("tok", "n_tr").explode("tok")
log("tokens with a dot, digit, or other punctuation")
print(allt.filter(pl.col("tok").str.contains(r"[^a-z\-]")).group_by("tok")
      .agg(pl.len().alias("n_ep"), pl.col("n_tr").sum()).sort("n_ep", descending=True).head(200))
log("top 150 multi-word epithets by train rows")
print(mw.head(150))

bl = C.filter((pl.col("genus") == "Bombus") & (sp == "latreille"))
log("Bombus latreille page ids", bl.group_by("eol_page_id").len().sort("len", descending=True).head(10).to_dicts())

tx = pl.read_csv(TAXON, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True)
log("taxon.tab", tx.shape, tx.columns)
log("taxonRank", tx.group_by("taxonRank").len().sort("len", descending=True).head(40).to_dicts())
log("taxonomicStatus", tx.group_by("taxonomicStatus").len().to_dicts())
log("eolID non-null", int(tx["eolID"].is_not_null().sum()), "authority non-null", int(tx["authority"].is_not_null().sum()))
print(tx.filter(pl.col("authority").is_not_null()).select("scientificName", "canonicalName", "authority", "taxonRank")
      .sample(40, seed=0))
print(tx.filter(pl.col("canonicalName").is_in(["Bombus", "Bombus latreille"]) |
                pl.col("scientificName").str.starts_with("Bombus Latreille"))
      .select("taxonID", "scientificName", "taxonRank", "taxonomicStatus", "canonicalName", "authority", "eolID"))
pages = (C.select(pl.col("eol_page_id").cast(pl.Float64).cast(pl.Int64).alias("page"), "genus", "species", "split")
         .with_columns(pl.concat_str([pl.col("genus"), pl.col("species")], separator=" ").alias("bin")))
txp = (tx.filter(pl.col("eolID").is_not_null())
       .select(pl.col("eolID").cast(pl.Int64, strict=False).alias("page"), "taxonRank", "canonicalName", "authority")
       .unique(subset="page", keep="first"))
j = pages.join(txp, on="page", how="left")
log("complete EOL rows whose page id is in taxon.tab", int(j["taxonRank"].is_not_null().sum() + j.filter(
    pl.col("taxonRank").is_null() & pl.col("canonicalName").is_not_null()).height), "of", j.height)
log("rows by page taxonRank", j.group_by("taxonRank").len().sort("len", descending=True).head(20).to_dicts())
bj = j.filter((pl.col("genus") == "Bombus") & (pl.col("species") == "latreille"))
log("Bombus latreille rows joined to taxon.tab", bj.group_by(["taxonRank", "canonicalName", "authority"]).len().to_dicts())
log("done")
