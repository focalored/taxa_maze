source "$HOME/.hpc_env.sh" && conda activate bioclip
# Follow-up GPU probes (one A100). Submit from the login node:
#   sbatch --account=bdbk-tgirails --partition=gpu_a100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=60G \
#          --time=00:30:00 --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/model_gpu_followup_%j.log \
#          --wrap="bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/model_gpu_followup.sh"
set -eo pipefail
cd /projects/bdbk/liv/repos/taxa_maze
export PYTHONPATH=/projects/bdbk/liv/repos/taxa_maze
export PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,power.limit,clocks.max.sm,driver_version --format=csv
python -u scripts/preflight/model_gpu_followup.py
echo "end=$(date -Is)"
