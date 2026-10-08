#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=p1-inat21-diag
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# Diagnostic iNat21-val evaluation of a run's epoch checkpoints (Amendment 3 A3.1), one GPU. Usage: sbatch --dependency=afterany:<train_job> scripts/p1/eval_inat21.sh --run-dir logs/p1/<run_name>; validation: sbatch scripts/p1/eval_inat21.sh --untouched
source /u/liv/repos/taxa_maze/scripts/env.sh
cd /u/liv/repos/taxa_maze && python -u scripts/p1/eval_inat21.py "$@"
