#!/bin/bash
source "$HOME/.hpc_env.sh" && conda activate bioclip
# Slurm wrapper for the S9 epithet-census helpers. Example:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#        --output=<repo>/audit/preflight/s9_census_%j.log bash s9_run.sh s9_epithet_census.py
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
echo "python=$(command -v python)"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$(dirname "$0")"
/usr/bin/time -v python "$@"
echo "end=$(date -Is)"
