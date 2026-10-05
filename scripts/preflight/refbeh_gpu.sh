#!/bin/bash
# GPU checks for audit/2026-10-02_reference_behaviour.md. Submit with:
#   sbatch --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=8 \
#     --mem=60G --time=00:30:00 --job-name=refbeh_gpu \
#     --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_gpu_%j.log \
#     /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/refbeh_gpu.sh [E] [F] [G]
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
export HF_HUB_OFFLINE=1   # bioclip2 loads from the HF cache under $HF_HOME; no network fetch
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u scripts/preflight/refbeh_gpu.py "$@"
echo "end $(date -Is)"
