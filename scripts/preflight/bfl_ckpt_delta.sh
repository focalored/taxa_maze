source "$HOME/.hpc_env.sh" && conda activate bioclip
# Slurm wrapper for bfl_ckpt_delta.py (CPU only; reads three ~600 MB checkpoints).
# Run from the login node with:
#   srun --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G --time=01:00:00 \
#        bash /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/bfl_ckpt_delta.sh
set -euo pipefail
LOG="/projects/bdbk/liv/repos/taxa_maze/audit/preflight/bfl_ckpt_delta_${SLURM_JOB_ID:-nojob}.log"
exec > >(tee -a "$LOG") 2>&1
echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}  start=$(date -Is)"
python -c "import torch, numpy, safetensors; print('torch', torch.__version__, 'numpy', numpy.__version__, 'safetensors', safetensors.__version__)"
python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/bfl_ckpt_delta.py
echo "[end] $(date -Is)"
