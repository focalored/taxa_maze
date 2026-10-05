#!/bin/bash
# Catalog lookup of the 215 dropped ToL members (audit/2026-10-02_reference_behaviour.md). Run with:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#     bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/refbeh_dropped_catalog.sh
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
python -u scripts/preflight/refbeh_dropped_catalog.py
echo "end $(date -Is)"
