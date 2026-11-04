#!/bin/bash
#SBATCH --job-name=kl_simv2_merge
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=19G
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

module load anaconda3
source ~/.bashrc
conda activate "${KLNN_CONDA_ENV}"
set -u

SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv2_galaxyaxis_halpha}"
DATASET_NAME="${DATASET_NAME:-train_1m_simv2_galaxyaxis_halpha}"
TOTAL="${TOTAL:-1000000}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

if (( TOTAL <= 0 || CHUNK_SIZE <= 0 || TOTAL % CHUNK_SIZE != 0 )); then
    echo "TOTAL must be positive and exactly divisible by CHUNK_SIZE" >&2
    exit 2
fi
PART_COUNT="$((TOTAL / CHUNK_SIZE))"

cd "${KLNN_REPO_ROOT}/data_generate"
python make_database.py \
    -s="${SAMPLE_NAME}" \
    -d="${DATASET_NAME}" \
    -N="${PART_COUNT}" \
    -n="${CHUNK_SIZE}" \
    --nspec=5 \
    --num_shards="${PART_COUNT}" \
    --merge
