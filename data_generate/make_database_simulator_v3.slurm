#!/bin/bash
#SBATCH --job-name=kl_simv3_db
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_database_simulator_v3_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -euo pipefail
ulimit -c 0

module load anaconda3
source ~/.bashrc
conda activate kl-nn

SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv3_galaxyaxis_central_halpha}"
DATASET_NAME="${DATASET_NAME:-train_1m_simv3_galaxyaxis_central_halpha}"
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

cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py \
    -s="${SAMPLE_NAME}" \
    -d="${DATASET_NAME}" \
    -N="${PART_COUNT}" \
    -n="${CHUNK_SIZE}" \
    --nspec=5 \
    --shard_idx="${SHARD_INDEX}" \
    --num_shards="${PART_COUNT}"
