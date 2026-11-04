#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M

set -euo pipefail
ulimit -c 0
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "${KLNN_REPO_ROOT}/klnn_paths.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a kl-nn checkout: ${KLNN_REPO_ROOT} (submit with submit.sh)" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/klnn_paths.sh"

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${KLNN_CONDA_ENV}"
SAMPLE="${SAMPLE:-samples_shear_response_simv3_cosi_5k}"
DATASET="${DATASET:-shear_response_simv3_cosi_5k}"
NBASE="${NBASE:-1000}"
NFOLDERS=100
NSHARDS=10
TOTAL=$((5 * NBASE))
if (( TOTAL % NFOLDERS != 0 )); then
  echo "5*NBASE must be divisible by ${NFOLDERS}; got ${TOTAL}" >&2
  exit 2
fi
CHUNK=$((TOTAL / NFOLDERS))

cd "${KLNN_REPO_ROOT}/data_generate"
python make_database.py -s="${SAMPLE}" -d="${DATASET}" \
  -N="${NFOLDERS}" -n="${CHUNK}" --nspec=5 \
  --num_shards="${NSHARDS}" --merge
