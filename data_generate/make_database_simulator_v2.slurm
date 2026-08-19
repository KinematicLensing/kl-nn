#!/bin/bash
#SBATCH --job-name=kl_simv2_db
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_database_simulator_v2_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-50
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

module load anaconda3
source ~/.bashrc
conda activate kl-nn
set -u

SAMPLE_NAME="${SAMPLE_NAME:-valid_1m_simv2_galaxyaxis}"
DATASET_NAME="${DATASET_NAME:-valid_1m_simv2_galaxyaxis}"
TOTAL="${TOTAL:-100000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 || TOTAL % CHUNK_SIZE != 0 )); then
    echo "TOTAL must be positive and exactly divisible by CHUNK_SIZE" >&2
    exit 2
fi
PART_COUNT="$((TOTAL / CHUNK_SIZE))"
SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
if (( SHARD_INDEX >= PART_COUNT )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is beyond ${PART_COUNT}; nothing to do."
    exit 0
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
