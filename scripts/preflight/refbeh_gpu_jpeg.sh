#!/bin/bash
# JPEG-sensitivity follow-up for audit/2026-10-02_reference_behaviour.md. Submit with:
#   sbatch --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=8 \
#     --mem=60G --time=00:30:00 --job-name=refbeh_jpeg \
#     --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/refbeh_gpu_jpeg_%j.log \
#     /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/refbeh_gpu_jpeg.sh
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
python -u scripts/preflight/refbeh_gpu_jpeg.py
echo "end $(date -Is)"
