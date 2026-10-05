source "$HOME/.hpc_env.sh" && conda activate bioclip
# Preflight E7 on ToL-EOL train (bioclip1). Submit from the login node with:
#   sbatch --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G \
#          --time=01:00:00 --job-name=pf_eval_tol \
#          --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/eval_anova_tol_%j.log \
#          --wrap="bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_anova_tol.sh"
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 POLARS_MAX_THREADS=16
python -c "import sys, numpy, polars, torch; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'polars', polars.__version__, 'torch', torch.__version__)"
python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/eval_anova_tol.py
