#!/bin/bash
# Submit pilot 1's one-time prep (scripts/p1/prepare.py): oversize+seed+dedup on 1 GPU, the drift floor on 4 GPUs, then the merge.
# Usage: bash scripts/p1/prepare.sh   (logs: logs/p1_prep/)
set -euo pipefail
cd /u/liv/repos/taxa_maze
mkdir -p logs/p1_prep
run='source scripts/env.sh && python scripts/p1/prepare.py'
g=(--account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:1 --parsable)
j1=$(sbatch "${g[@]}" --job-name=p1prep-seed --cpus-per-task=16 --mem=160G --time=01:00:00 -o logs/p1_prep/seed_%j.log \
     --wrap "$run tree && $run oversize && $run seed && $run dedup")
j2=$(sbatch "${g[@]}" --dependency=afterok:$j1 --job-name=p1prep-floor --array=0-3 --cpus-per-task=16 --mem=96G \
     --time=02:00:00 -o logs/p1_prep/floor_%A_%a.log --wrap "$run floor --part \$SLURM_ARRAY_TASK_ID --parts 4")
j3=$(sbatch --account=bdbk-tgirails --partition=cpu --parsable --dependency=afterok:$j2 --job-name=p1prep-fmerge \
     --cpus-per-task=8 --mem=96G --time=00:30:00 -o logs/p1_prep/floor_merge_%j.log --wrap "$run floor-merge --parts 4")
echo "seed/dedup $j1  floor $j2  merge $j3"
