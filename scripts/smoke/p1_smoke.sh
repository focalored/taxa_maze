#!/bin/bash
# Run one pilot-1 smoke test inside a Slurm allocation. Usage: srun ... bash scripts/smoke/p1_smoke.sh <test> [--world-run 4]
source /u/liv/repos/taxa_maze/scripts/env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
python scripts/smoke/p1_smoke.py "$@"
