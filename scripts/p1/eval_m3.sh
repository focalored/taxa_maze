#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=p1-eval-m3
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# M3 per-epoch diagnostics of one run on one GPU: iNat21-val then Rare Species for every epoch checkpoint not yet evaluated
# (A2.4's encode setting; `<run>/inat21_diag/epoch_NN.json` and `<run>/rare_species_diag/epoch_NN.json`).
# Usage: sbatch --dependency=afterany:<train_job> scripts/p1/eval_m3.sh <run_name> [<run_name> ...]
source /u/liv/repos/taxa_maze/scripts/env.sh
cd /u/liv/repos/taxa_maze
for r in "$@"; do
  python -u scripts/p1/eval_inat21.py --run-dir logs/p1/$r || echo "inat21 $r failed"
  python -u scripts/p1/eval_inat21.py --dataset rare_species --run-dir logs/p1/$r || echo "rare_species $r failed"
done
