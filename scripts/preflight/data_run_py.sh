#!/bin/bash
# Slurm wrapper for the data preflight Python scripts.
# Usage: srun|sbatch <flags> data_run_py.sh <script.py> [args...]
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
echo "python=$(command -v python)"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
/usr/bin/time -v python "$@"
echo "end=$(date -Is)"
