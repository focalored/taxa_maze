#!/bin/bash
# Full shard census for audit/2026-10-02_reference_behaviour.md. Submit with:
#   sbatch --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#     --job-name=refbeh_census \
#     --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_census_%j.log \
#     /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/refbeh_census.sh [NPROC]
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
python -u scripts/preflight/refbeh_census.py "${1:-12}"
echo "end $(date -Is)"
