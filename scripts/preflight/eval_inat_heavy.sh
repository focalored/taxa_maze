source "$HOME/.hpc_env.sh" && conda activate bioclip
# Preflight E5 (deep) + E6 + E7 on iNat21-val. Launch from the login node with:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G \
#        --time=01:00:00 bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_inat_heavy.sh \
#        2>&1 | tee /projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_inat_heavy.log
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python -c "import sys, numpy; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_inat_heavy.py
