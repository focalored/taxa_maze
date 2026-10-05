#!/bin/bash
# S9 provenance check (audit/2026-10-02_S9_bioclip1_provenance.md). Run with:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#     bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/S9_provenance.sh \
#     > /projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1/S9_provenance.log 2>&1
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  start $(date -Is)"
echo "python $(command -v python)  polars $(python -c 'import polars; print(polars.__version__)')"
/usr/bin/time -v python -u scripts/preflight/S9_provenance.py
echo "end $(date -Is)"
