#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=p1-rare-species
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# Rare Species zero-shot evaluation (A16) on one GPU. Usage: `sbatch scripts/p1/eval_rare_species.sh baselines` (the six registered
# baselines, untouched BioCLIP 1, and the sweep and control runs) or `sbatch scripts/p1/eval_rare_species.sh runs <run_name> ...`.
source /u/liv/repos/taxa_maze/scripts/env.sh
cd /u/liv/repos/taxa_maze
mode=$1; shift
if [[ "$mode" == "baselines" ]]; then
  for b in bioclip1 bfl-euclidean bfl-hyperbolic rcme bioclip2 clip-l14-laion2b; do python -u scripts/p1/eval_inat21.py --dataset rare_species --baseline $b || echo "baseline $b failed"; done
  python -u scripts/p1/eval_inat21.py --dataset rare_species --untouched || echo "untouched failed"
  set -- p1c_lvl_lr1e-4_s42_v1 p1c_lvl_lr3e-4_s42_v2 p1c_lvl_lr1e-4_s42_rand_v1
fi
for r in "$@"; do python -u scripts/p1/eval_inat21.py --dataset rare_species --run-dir logs/p1/$r || echo "run $r failed"; done
