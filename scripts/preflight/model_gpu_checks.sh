source "$HOME/.hpc_env.sh" && conda activate bioclip
# Preflight M6-M7 (one A100). Submit from the login node:
#   sbatch --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=60G \
#          --time=00:30:00 --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/model_gpu_%j.log \
#          /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/model_gpu_checks.sh
set -eo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
export PYTHONPATH=/projects/bdbk/liv/repos/taxa_maze
export PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,power.limit,clocks.max.sm,driver_version --format=csv
which python; python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
python -u scripts/preflight/model_gpu_checks.py
echo "end=$(date -Is)"
