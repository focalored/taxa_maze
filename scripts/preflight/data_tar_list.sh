#!/bin/bash
# Preflight check C5 (specs/pilot1.md lines 43-50): list the members of one
# ToL-10M EOL shard with a full `tar -tzf` pass, time the pass, and compare the
# member uuids against the BioCLIP 1 cache's shards/image_set_NN.ids.txt.
#
# Usage (cpu partition, via sbatch):
#   sbatch --account=bdbk-tgirails --partition=cpu --cpus-per-task=16 --mem=80G \
#     --time=01:00:00 --job-name=data_tar_NN \
#     --output=/projects/bdbk/liv/repos/taxa_maze/audit/preflight/data_tar_NN_%j.log \
#     /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/data_tar_list.sh NN
#
# Writes: audit/preflight/data_tar_members_image_set_NN.txt  (raw tar -t output)
#         audit/preflight/data_tar_image_set_NN.json         (comparison + timing)
source "$HOME/.hpc_env.sh" && conda activate bioclip
set -euo pipefail

NN="$1"
TAR="/u/liv/bdbk/data/tol10m/dataset/EOL/image_set_${NN}.tar.gz"
OUT=/projects/bdbk/liv/repos/taxa_maze/audit/preflight
LIST="$OUT/data_tar_members_image_set_${NN}.txt"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
stat -c 'file=%n bytes=%s' "$TAR"

TIMEFORMAT='tar_time real_s=%R user_s=%U sys_s=%S'
T0=$(date +%s.%N)
time tar -tzf "$TAR" > "$LIST"
T1=$(date +%s.%N)
echo "wall_t0=$T0 wall_t1=$T1"
wc -l "$LIST"

python /projects/bdbk/liv/repos/taxa_maze/scripts/preflight/data_tar_compare.py \
    --shard "$NN" --t0 "$T0" --t1 "$T1"
echo "end=$(date -Is)"
