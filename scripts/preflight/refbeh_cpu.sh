#!/bin/bash
# CPU checks for audit/2026-10-02_reference_behaviour.md. Submit with:
#   sbatch --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#     --job-name=refbeh_cpu \
#     --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_cpu_%j.log \
#     /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/refbeh_cpu.sh
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
python -u scripts/preflight/refbeh_cpu.py "$@"
echo "end $(date -Is)"
