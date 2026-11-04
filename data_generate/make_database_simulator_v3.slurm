#!/bin/bash
#SBATCH --job-name=kl_simv3_db
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "${KLNN_REPO_ROOT}/klnn_paths.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a kl-nn checkout: ${KLNN_REPO_ROOT} (submit with submit.sh)" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/klnn_paths.sh"
if [[ ! -f "${KLNN_REPO_ROOT}/shared_job_scratch.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch simulator-v3-db

module load anaconda3
source ~/.bashrc
conda activate "${KLNN_CONDA_ENV}"
set -u

SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv3_cosi}"
DATASET_NAME="${DATASET_NAME:-train_1m_simv3_cosi}"
TOTAL="${TOTAL:-1000000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 || TOTAL % CHUNK_SIZE != 0 )); then
    echo "TOTAL must be positive and exactly divisible by CHUNK_SIZE" >&2
    exit 2
fi
PART_COUNT="$((TOTAL / CHUNK_SIZE))"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-500}"
SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
if (( PART_COUNT != ARRAY_TASK_COUNT )); then
    echo "TOTAL=${TOTAL} and CHUNK_SIZE=${CHUNK_SIZE} require ${PART_COUNT} array tasks, but this job has ${ARRAY_TASK_COUNT}; submit with --array=1-${PART_COUNT}" >&2
    exit 2
fi
if (( SHARD_INDEX < 0 || SHARD_INDEX >= PART_COUNT )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside the required range 1-${PART_COUNT}" >&2
    exit 2
fi

cd "${KLNN_REPO_ROOT}/data_generate"
python make_database.py \
    -s="${SAMPLE_NAME}" \
    -d="${DATASET_NAME}" \
    -N="${PART_COUNT}" \
    -n="${CHUNK_SIZE}" \
    --nspec=5 \
    --shard_idx="${SHARD_INDEX}" \
    --num_shards="${PART_COUNT}"
