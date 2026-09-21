#!/bin/bash
#SBATCH --job-name=kl_fisher_fits
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_fisher_stencil_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --array=1-95

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/shared_job_scratch.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch fisher-stencil-fits

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

SAMPFILE="${SAMPFILE:?set SAMPFILE to the sample CSV filename}"
DATASET="${DATASET:?set DATASET to the FITS dataset name}"
NROWS="${NROWS:?set NROWS to the sample table length}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
SAMPLE_PATH="/ocean/projects/phy250048p/shared/samples/${SAMPFILE}"

if (( NROWS <= 0 || CHUNK_SIZE <= 0 )); then
    echo "NROWS and CHUNK_SIZE must be positive" >&2
    exit 2
fi
TASK_COUNT="$(((NROWS + CHUNK_SIZE - 1) / CHUNK_SIZE))"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-${TASK_COUNT}}"
ARRAY_TASK_MIN="${SLURM_ARRAY_TASK_MIN:-1}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${TASK_COUNT}}"
ALLOW_PARTIAL_ARRAY="${ALLOW_PARTIAL_ARRAY:-1}"
TASK_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
if [[ "${ALLOW_PARTIAL_ARRAY}" != "0" && "${ALLOW_PARTIAL_ARRAY}" != "1" ]]; then
    echo "ALLOW_PARTIAL_ARRAY must be 0 or 1" >&2
    exit 2
fi
if (( ARRAY_TASK_COUNT != TASK_COUNT )) && [[ "${ALLOW_PARTIAL_ARRAY}" != "1" ]]; then
    echo "NROWS=${NROWS} and CHUNK_SIZE=${CHUNK_SIZE} require ${TASK_COUNT} array tasks, but this submission has ${ARRAY_TASK_COUNT}; use --array=1-${TASK_COUNT}, or set ALLOW_PARTIAL_ARRAY=1" >&2
    exit 2
fi
if (( TASK_INDEX < 0 || TASK_INDEX >= TASK_COUNT )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside 1-${TASK_COUNT}" >&2
    exit 0
fi
if [[ ! -f "${SAMPLE_PATH}" ]]; then
    echo "Missing Fisher sample table: ${SAMPLE_PATH}" >&2
    exit 2
fi

START="$((TASK_INDEX * CHUNK_SIZE))"
STOP="$((START + CHUNK_SIZE))"
if (( STOP > NROWS )); then
    STOP="${NROWS}"
fi
if (( START >= NROWS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has empty row range" >&2
    exit 0
fi

mkdir -p "/ocean/projects/phy250048p/shared/fits/${DATASET}"
cd "${KLNN_REPO_ROOT}/data_generate"
python generate_fits_wrapper.py \
    -i="${START}" \
    -j="${STOP}" \
    -n="${SLURM_ARRAY_TASK_ID}" \
    -s="${SAMPFILE}" \
    -d="${DATASET}" \
    --skip-existing
