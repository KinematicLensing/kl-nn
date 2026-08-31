#!/bin/bash
#SBATCH --job-name=kl_simv3_fits
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=164,266-289,429-484
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

source /jet/home/xwang30/kl-nn/shared_job_scratch.sh
setup_shared_job_scratch simulator-v3-fits

module load anaconda3
source ~/.bashrc
conda activate kl-nn
set -u

echo "FITS array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} raw_job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown}"
echo "FITS log=/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv3}"
DATASET_NAME="${DATASET_NAME:-train_1m_simv3}"
TOTAL="${TOTAL:-1000000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
SAMPLE_FILE="${SAMPLE_NAME}.csv"
SAMPLE_PATH="/ocean/projects/phy250048p/shared/samples/${SAMPLE_FILE}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 )); then
    echo "TOTAL and CHUNK_SIZE must be positive" >&2
    exit 2
fi
TASK_COUNT="$(((TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE))"
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
    echo "TOTAL=${TOTAL} and CHUNK_SIZE=${CHUNK_SIZE} require ${TASK_COUNT} array tasks, but this submission has ${ARRAY_TASK_COUNT}; use --array=1-${TASK_COUNT}, or set ALLOW_PARTIAL_ARRAY=1 for an intentional sparse resume" >&2
    exit 2
fi
if (( ARRAY_TASK_MIN < 1 || ARRAY_TASK_MAX > TASK_COUNT )); then
    echo "Array task range ${ARRAY_TASK_MIN}-${ARRAY_TASK_MAX} is outside 1-${TASK_COUNT}" >&2
    exit 2
fi
if (( TASK_INDEX < 0 || TASK_INDEX >= TASK_COUNT )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside the required range 1-${TASK_COUNT}" >&2
    exit 2
fi
if [[ ! -f "${SAMPLE_PATH}" ]]; then
    echo "Missing simulator-v3 sample table: ${SAMPLE_PATH}" >&2
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
    -d="${DATASET_NAME}" \
    --skip-existing
