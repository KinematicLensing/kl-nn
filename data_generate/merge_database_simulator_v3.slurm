#!/bin/bash
#SBATCH --job-name=kl_simv3_merge
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/merge_database_simulator_v3_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
###SBATCH --cpus-per-task=8
###SBATCH --mem=15G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

source /jet/home/xwang30/kl-nn/shared_job_scratch.sh
setup_shared_job_scratch simulator-v3-merge

module load anaconda3
source ~/.bashrc
conda activate kl-nn
set -u

SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv3}"
DATASET_NAME="${DATASET_NAME:-train_1m_simv3}"
TOTAL="${TOTAL:-1000000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 || TOTAL % CHUNK_SIZE != 0 )); then
    echo "TOTAL must be positive and exactly divisible by CHUNK_SIZE" >&2
    exit 2
fi
PART_COUNT="$((TOTAL / CHUNK_SIZE))"

cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py \
    -s="${SAMPLE_NAME}" \
    -d="${DATASET_NAME}" \
    -N="${PART_COUNT}" \
    -n="${CHUNK_SIZE}" \
    --nspec=5 \
    --num_shards="${PART_COUNT}" \
    --merge
