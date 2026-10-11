#!/bin/bash
#SBATCH --account=bdbk-tgirails
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=tol10m-download
#SBATCH --output=/u/liv/repos/taxa_maze/logs/slurm/%x_%j.log
# Downloads the two parts of TreeOfLife-10M that are not on this cluster (Amendment 4 A4.7): the iNat21 train images (visipedia S3)
# and BIOSCAN-1M cropped_256 (Zenodo 8030065), plus iNat21's train.json, into /u/liv/data/taxa_maze/raw/{inat21,bioscan}/ (each archive
# in its own directory; they are untrusted data until verified). Resumable (curl -C -), skips a file whose size already matches, and
# verifies every file against the published md5. Usage: sbatch scripts/store/download_tol10m_parts.sh
set -u
RAW=/u/liv/data/taxa_maze/raw
mkdir -p "$RAW/inat21" "$RAW/bioscan"
get() {  # url out expected_bytes expected_md5
  local url=$1 out=$2 bytes=$3 md5=$4
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$bytes" ]; then echo "$(date '+%F %T') $out already has $bytes bytes; verifying only"
  else
    echo "$(date '+%F %T') downloading $out"
    curl -L -C - --retry 20 --retry-delay 30 -sS -o "$out" "$url"; rc=$?; [ $rc -eq 0 ] || { echo "$(date "+%F %T") curl failed for $out (exit $rc)"; return 1; }
    echo "$(date '+%F %T') downloaded $(stat -c %s "$out") bytes (expected $bytes)"
  fi
  local got; got=$(md5sum "$out" | cut -d' ' -f1)
  if [ "$got" = "$md5" ]; then echo "$(date '+%F %T') MD5 OK $out"; else echo "$(date '+%F %T') MD5 MISMATCH $out: got $got expected $md5"; return 1; fi
}
status=0
get "https://zenodo.org/records/8030065/files/cropped_256.zip?download=1" "$RAW/bioscan/cropped_256.zip" 7171212982 fe1175815742db14f7372d505345284a || status=1
get "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz" "$RAW/inat21/train.json.tar.gz" 230885383 38a7bb733f7a09214d44293460ec0021 || status=1
get "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz" "$RAW/inat21/train.tar.gz" 239909164970 e0526d53c7f7b2e3167b2b43bb2690ed || status=1
echo "$(date '+%F %T') done, status $status"; exit $status
