# Sourced by every pilot-1 Slurm job: the bioclip env, plus HF_HOME and WANDB_DIR moved off
# /projects/bdbk (~/.hpc_env.sh points both there; Amendment 1 S1, agent/USAGE.md Gotchas).
source "$HOME/.hpc_env.sh"
conda activate bioclip
export REPO=/u/liv/repos/taxa_maze
export HF_HOME=/u/liv/.cache/huggingface
export WANDB_DIR=$REPO/logs/wandb
export PYTHONPATH=$REPO
mkdir -p "$HF_HOME" "$WANDB_DIR"
cd "$REPO"
