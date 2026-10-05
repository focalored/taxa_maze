#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=p1-train
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# Pilot 1 training on 1 node x 2 A100s (Amendment 2 A2.1). Usage: sbatch scripts/p1/train.sh experiment=p1/c_lvl model.lr=1e-4 trainer.max_epochs=3
source /u/liv/repos/taxa_maze/scripts/env.sh
srun python src/train.py "$@"
