#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=p1-text-geometry
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# Text-space geometry probe (A15 plan step 0), one GPU. Usage: sbatch scripts/p1/text_geometry.sh [--out audit/<dir>]
source /u/liv/repos/taxa_maze/scripts/env.sh
cd /u/liv/repos/taxa_maze && python -u scripts/p1/text_geometry.py "$@"
