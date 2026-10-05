#!/bin/bash
# Submit the image-store build (scripts/store/build_store.py): catalog -> 63 shard tasks -> merge.
# Usage: bash scripts/store/build_store.sh [--skip-catalog]
# Logs go to logs/store/. Output: /u/liv/data/taxa_maze/store/ (paths.p1_data_dir).
set -euo pipefail
cd /u/liv/repos/taxa_maze
mkdir -p logs/store
common=(--account=bdbk-tgirails --partition=cpu --parsable)
run='source scripts/env.sh && python scripts/store/build_store.py'
dep=()
if [[ "${1:-}" != "--skip-catalog" ]]; then
  j1=$(sbatch "${common[@]}" --job-name=p1store-cat --cpus-per-task=8 --mem=64G --time=00:30:00 \
       -o logs/store/catalog_%j.log --wrap "$run catalog")
  dep=(--dependency=afterok:$j1); echo "catalog job $j1"
fi
j2=$(sbatch "${common[@]}" "${dep[@]}" --job-name=p1store-shard --array=1-63 --cpus-per-task=16 --mem=48G \
     --time=02:00:00 -o logs/store/shard_%A_%a.log --wrap "$run shard")
echo "shard array $j2"
j3=$(sbatch "${common[@]}" --dependency=afterok:$j2 --job-name=p1store-merge --cpus-per-task=8 --mem=96G \
     --time=00:30:00 -o logs/store/merge_%j.log --wrap "$run merge")
echo "merge job $j3"
