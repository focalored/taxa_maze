#!/bin/bash
#SBATCH --job-name=p1-eval-gate
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/u/liv/repos/taxa_maze/logs/smoke/eval_gate/slurm_%j.log
# Slurm wrapper for scripts/smoke/eval_gate.py, the pilot-1 eval-harness gate; arguments pass through.
# Usage: sbatch scripts/smoke/eval_gate.sh        (the full gate; logs/smoke/eval_gate/ must exist first)
#        srun --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=00:30:00 bash scripts/smoke/eval_gate.sh --part B --limit 2048
source /u/liv/repos/taxa_maze/scripts/env.sh
set -eo pipefail
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
/u/liv/envs/bioclip/bin/python -u scripts/smoke/eval_gate.py "$@"
