"""S9 rule check (2026-10-04): size the user's S9 rule under its two readings.

The user's S9 answer drops rows whose species field *exactly* matches an author name or placeholder
"in your list" (whole field, case-insensitive), and keeps every multi-word epithet. The only saved
lists are the census's (audit/preflight/s9_epithet_census.json, non_clean_keys). Two readings:
  per-key : drop the census keys that are author-only single words, or single-token placeholders (P).
  string  : drop any key, in any genus, whose whole species field is one of those keys' strings.
The string reading can also catch real species whose epithet equals an author word used elsewhere.
Population: EOL rows, split train or val, all seven ranks non-null (as the census).
"""
import json, math, os, socket, sys, datetime
import polars as pl

CATALOG = "/u/liv/bdbk/data/tol10m/metadata/catalog.csv"
CENSUS = os.path.join(os.path.dirname(__file__), "../../audit/preflight/s9_epithet_census.json")
OUT = os.path.join(os.path.dirname(__file__), "../../audit/preflight/s9_rule_check.json")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
B = 8192

R = {"meta": {"script": os.path.abspath(__file__), "host": socket.gethostname(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "start_utc": datetime.datetime.utcnow().isoformat()}}

df = pl.read_csv(CATALOG, columns=["split", "eol_content_id"] + RANKS, infer_schema=False)
df = df.filter(pl.col("eol_content_id").is_not_null() & pl.col("split").is_in(["train", "val"])
               & pl.all_horizontal([pl.col(c).is_not_null() for c in RANKS]))
df = df.with_columns(pl.concat_str(RANKS, separator="|").alias("key"))
cnt = (df.group_by(["key", "species"]).agg((pl.col("split") == "train").sum().alias("n_tr"),
                                           (pl.col("split") == "val").sum().alias("n_va")))
R["population"] = {"train_rows": int(cnt["n_tr"].sum()), "val_rows": int(cnt["n_va"].sum()), "keys": cnt.height}
print("population", R["population"], flush=True)

census = json.load(open(CENSUS))["non_clean_keys"]
cat = {e["key"]: (e["cat"], e["sub"]) for e in census}
auth_keys = {e["key"] for e in census if e["sub"] == "author_only_single"}
ph_keys = {e["key"] for e in census if e["cat"] == "P" and " " not in e["species"]}
exact7 = {"sp.", "sp", "spp.", "spp", "xx", "x", "?"}
per_key = auth_keys | ph_keys
strings = {e["species"].lower() for e in census if e["key"] in per_key}
R["lists"] = {"author_only_single_keys": len(auth_keys), "single_token_placeholder_keys": len(ph_keys),
              "per_key_total": len(per_key), "distinct_strings": len(strings),
              "author_only_multi_keys_kept_as_multiword": sum(1 for e in census if e["sub"] == "author_only_multi")}

def summarize(name, mask):
    kept = cnt.filter(~mask)
    drop = cnt.filter(mask)
    n_tr_left = int(kept["n_tr"].sum())
    top = kept.sort("n_tr", descending=True).head(3)
    nmax = int(top["n_tr"][0])
    steps = n_tr_left // B
    res = {"keys_dropped": drop.height, "train_dropped": int(drop["n_tr"].sum()), "val_dropped": int(drop["n_va"].sum()),
           "train_left": n_tr_left, "val_left": int(kept["n_va"].sum()), "species_keys_left_with_train": int((kept["n_tr"] > 0).sum()),
           "steps_per_epoch_floor": steps, "warmup_1pct_steps_10ep": math.ceil(0.01 * 10 * steps),
           "largest": [{"key": k, "N_s": int(n)} for k, n in zip(top["key"], top["n_tr"])],
           "K16_max_groups": math.ceil(nmax / 16), "K8_max_groups": math.ceil(nmax / 8)}
    R[name] = res
    print(name, {k: v for k, v in res.items() if k != "largest"}, res["largest"][0], flush=True)
    return drop

summarize("none", pl.col("n_tr") < 0)  # drops nothing: the current spec
d_key = summarize("per_key", pl.col("key").is_in(list(per_key)))
d_str = summarize("string", pl.col("species").str.to_lowercase().is_in(list(strings)))
summarize("per_key_exact7_placeholders_only", pl.col("key").is_in(list(auth_keys)) | pl.col("species").is_in(list(exact7)))

extra = d_str.filter(~pl.col("key").is_in(list(per_key))).sort("n_tr", descending=True)
extra = extra.with_columns(pl.col("key").map_elements(lambda k: "/".join(map(str, cat.get(k, ("C", None)))),
                                                      return_dtype=pl.String).alias("census_category"))
R["string_minus_per_key"] = {"keys": extra.height, "train": int(extra["n_tr"].sum()), "val": int(extra["n_va"].sum()),
                             "by_census_category": extra.group_by("census_category").agg(pl.len().alias("keys"), pl.col("n_tr").sum().alias("train")).sort("train", descending=True).to_dicts(),
                             "top30": extra.head(30).select("key", "n_tr", "n_va", "census_category").to_dicts()}
print("string_minus_per_key", {k: v for k, v in R["string_minus_per_key"].items() if k != "top30"}, flush=True)
for r in R["string_minus_per_key"]["top30"][:15]:
    print("  ", r, flush=True)
R["meta"]["end_utc"] = datetime.datetime.utcnow().isoformat()
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", os.path.abspath(OUT))
