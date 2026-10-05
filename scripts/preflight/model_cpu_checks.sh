source "$HOME/.hpc_env.sh" && conda activate bioclip
# Preflight M1-M5 (CPU). Submit from the login node:
#   sbatch --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#          --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/model_cpu_%j.log \
#          /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/model_cpu_checks.sh
set -eo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
export PYTHONPATH=/projects/bdbk/liv/repos/taxa_maze
export PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
which python; python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
python -u scripts/preflight/model_cpu_checks.py
echo "end=$(date -Is)"
