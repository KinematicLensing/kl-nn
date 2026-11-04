#!/bin/bash
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --array=1-100

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
SAMPFILE="${SAMPFILE:-samples_shear_response_simv3_cosi_5k.csv}"
DATASET="${DATASET:-shear_response_simv3_cosi_5k}"
NBASE="${NBASE:-1000}"
NTASKS=100
TOTAL=$((5 * NBASE))
if (( TOTAL % NTASKS != 0 )); then
  echo "5*NBASE must be divisible by ${NTASKS}; got ${TOTAL}" >&2
  exit 2
fi
CHUNK=$((TOTAL / NTASKS))
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
START=$((INDEX * CHUNK))
STOP=$((START + CHUNK))
SAMPLE_PATH="${KLNN_SAMPLES}/${SAMPFILE}"

if [[ ! -f "${SAMPLE_PATH}" ]]; then
  echo "Missing shear-response sample table: ${SAMPLE_PATH}" >&2
  exit 2
fi

mkdir -p "${KLNN_FITS}/${DATASET}"
cd "${KLNN_REPO_ROOT}/data_generate"
python generate_fits_wrapper.py -i="${START}" -j="${STOP}" \
  -n="${SLURM_ARRAY_TASK_ID}" -s="${SAMPFILE}" -d="${DATASET}"
