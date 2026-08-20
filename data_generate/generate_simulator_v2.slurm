#!/bin/bash
#SBATCH --job-name=kl_simv2_fits
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_simulator_v2_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

module load anaconda3
source ~/.bashrc
conda activate kl-nn
set -u

SAMPLE_NAME="${SAMPLE_NAME:-small_1m_simv2_galaxyaxis_halpha}"
DATASET_NAME="${DATASET_NAME:-small_1m_simv2_galaxyaxis_halpha}"
TOTAL="${TOTAL:-10000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
SAMPLE_FILE="samples_${SAMPLE_NAME}.csv"
SAMPLE_PATH="/ocean/projects/phy250048p/shared/samples/${SAMPLE_FILE}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 )); then
    echo "TOTAL and CHUNK_SIZE must be positive" >&2
    exit 2
fi
TASK_COUNT="$(((TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE))"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-50}"
TASK_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
if (( TASK_COUNT != ARRAY_TASK_COUNT )); then
    echo "TOTAL=${TOTAL} and CHUNK_SIZE=${CHUNK_SIZE} require ${TASK_COUNT} array tasks, but this job has ${ARRAY_TASK_COUNT};" \
        "submit with --array=1-${TASK_COUNT}" >&2
    exit 2
fi
if (( TASK_INDEX < 0 || TASK_INDEX >= TASK_COUNT )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside the required range 1-${TASK_COUNT}" >&2
    exit 2
fi
if [[ ! -f "${SAMPLE_PATH}" ]]; then
    echo "Missing sample table: ${SAMPLE_PATH}" >&2
    exit 2
fi

START="$((TASK_INDEX * CHUNK_SIZE))"
STOP="$((START + CHUNK_SIZE))"
if (( STOP > TOTAL )); then
    STOP="${TOTAL}"
fi

cd /jet/home/xwang30/kl-nn/data_generate
python generate_fits_wrapper.py \
    -i="${START}" \
    -j="${STOP}" \
    -n="${SLURM_ARRAY_TASK_ID}" \
    -s="${SAMPLE_FILE}" \
    -d="${DATASET_NAME}"
