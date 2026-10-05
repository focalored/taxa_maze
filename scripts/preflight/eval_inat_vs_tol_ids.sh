source "$HOME/.hpc_env.sh" && conda activate bioclip
# Preflight E6 extra: iNat21 val vs ToL-10M catalog, by image id. Launch from the login node:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G \
#        --time=01:00:00 bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_inat_vs_tol_ids.sh \
#        2>&1 | tee /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_vs_tol_ids.log
set -euo pipefail
export POLARS_MAX_THREADS=16
python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_inat_vs_tol_ids.py
