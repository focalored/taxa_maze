#!/usr/bin/env python
"""S9 exploration, part 2: examples of the ambiguous epithet families seen in s9_explore_246185.log
(lead tokens such as morph/group/subg., odd punctuation, short tokens, hybrids), printed to stdout.
Run: srun <cpu flags> --output=<repo>/audit/preflight/s9_explore2_%j.log bash s9_run.sh s9_explore2.py
"""
import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
pl.Config.set_tbl_rows(200)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(220)

df = pl.read_csv(CATALOG, columns=["split", "eol_content_id"] + RANKS, infer_schema=False)
C = df.filter(pl.col("eol_content_id").is_not_null() & pl.col("split").is_in(["train", "val"])
              & pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
ep = (C.group_by(["genus", "species"]).agg((pl.col("split") == "train").sum().alias("n_tr"),
                                         (pl.col("split") == "val").sum().alias("n_va"))
      .sort("n_tr", descending=True))
sp = pl.col("species")
families = {
    "lead morph/group/complex/agg/prov/nom/cv/form/var/f.": r"^(morph|group|complex|agg\.?|prov\.?|nom\.?|cv\.?|form|forma|var\.?|f\.|ab\.?|hort\.?|prob\.?|sect\.|ser\.|subg\.|subgen\.)( |$)",
    "contains group/complex/agg./prov./nom./cv./hort./prob./ab.": r"(^| )(group|complex|agg\.|prov\.|nom\.|cv\.|hort\.|prob\.|ab\.)( |$)",
    "contains ' or quote or = or : or _ or # or { or *": r"['’\"=:_#{}*!<>\[\]]",
    "contains ? or digit": r"[?0-9]",
    "contains x or × token, or ×-prefix": r"(^| )(x|×)( |$)|×",
    "lead cf/aff/nr/near/sp/spp": r"^(cf\.?|aff\.?|nr\.?|near|sp\.?|spp\.?|ssp\.?)( |$)",
    "single token ending with - or starting with -": r"^-|-$",
    "contains ' and ' or ' in ' or ' et ' or ' ex '": r" (and|in|et|ex|von|van|de|der|la|le|du|da|di|del|y) ",
    "ends with ' ex' or ' et' or ' in' or ' de'": r" (ex|et|in|de|al\.)$",
}
for name, rx in families.items():
    sub = ep.filter(sp.str.contains(rx))
    print(f"\n=== {name}: {sub.height} genus+epithet pairs, train {int(sub['n_tr'].sum())}, val {int(sub['n_va'].sum())}")
    print(sub.head(60))
one = ep.filter(sp.str.contains(r"^[a-z]+$") & (sp.str.len_chars() <= 4))
print(f"\n=== single a-z token of <= 4 letters: {one.height} pairs")
print(one.head(80))
