#!/bin/bash
# Re-run data_tar_compare.py for shards 60 and 63 after data_catalog_facts.py has
# written audit/preflight/data_cache_missing_ids.csv (timing reused from the first run).
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
for NN in 60 63; do
  python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/data_tar_compare.py --shard "$NN"
done
echo "end=$(date -Is)"
