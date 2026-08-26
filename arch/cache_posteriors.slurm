#!/bin/bash
#SBATCH --job-name=klnn_cache
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/cache_posteriors_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -euo pipefail
ulimit -c 0

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

: "${MODEL_NAME:?Set MODEL_NAME to the completed NPE run}"
: "${DATASET:?Set DATASET to an independent held-out dataset not used for training or checkpoint selection}"
EPOCH="${EPOCH:-}"
NPARTS="${NPARTS:-10}"
NGALS="${NGALS:-1000}"
NSAMPLES="${NSAMPLES:-10000}"
SEED="${SEED:-42}"
CACHE_TAG="${CACHE_TAG:-${MODEL_NAME}_posterior_candidates}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
DATA_ROOT="${DATA_ROOT:-/ocean/projects/phy250048p/shared/datasets}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
PART_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-10}"

if (( NPARTS != ARRAY_TASK_COUNT )); then
    echo "NPARTS=${NPARTS}, but this job has ${ARRAY_TASK_COUNT} array tasks; submit with --array=1-${NPARTS}" >&2
    exit 2
fi
if (( PART_INDEX < 0 || PART_INDEX >= NPARTS )); then
    echo "Array index ${PART_INDEX} is outside NPARTS=${NPARTS}" >&2
    exit 2
fi
if (( NSAMPLES <= 0 || NSAMPLES % 2 != 0 )); then
    echo "NSAMPLES must be positive and even for the identity/R90 pair" >&2
    exit 2
fi
SHARED_ROOT="$(dirname "${MODEL_ROOT%/}")"
if [[ ! -f "${SHARED_ROOT}/configs/cfg_${MODEL_NAME}.json" ]]; then
    echo "Missing current-schema model config for ${MODEL_NAME}" >&2
    exit 2
fi
if [[ ! -d "${DATA_ROOT}/${DATASET}" ]]; then
    echo "Missing dataset: ${DATA_ROOT}/${DATASET}" >&2
    exit 2
fi

ARGS=(
    --partition-index "${PART_INDEX}"
    --nparts "${NPARTS}"
    --ngals "${NGALS}"
    --nsamples "${NSAMPLES}"
    --model-name "${MODEL_NAME}"
    --dataset "${DATASET}"
    --model-root "${MODEL_ROOT}"
    --data-root "${DATA_ROOT}"
    --cache-root "${CACHE_ROOT}"
    --cache-tag "${CACHE_TAG}"
    --matched-group-size 1
    --seed "${SEED}"
    --device cuda
)
if [[ -n "${EPOCH}" ]]; then
    ARGS+=(--epoch "${EPOCH}")
fi

echo "model=${MODEL_NAME} epoch=${EPOCH:-best}"
echo "dataset=${DATASET} partition=${PART_INDEX}/${NPARTS} ngals=${NGALS} nsamples=${NSAMPLES}"
echo "cache_tag=${CACHE_TAG}; saving base candidates, posterior TF weights, and raw population TF log ratios"

cd /jet/home/xwang30/kl-nn/arch
python cache_posteriors.py "${ARGS[@]}"
