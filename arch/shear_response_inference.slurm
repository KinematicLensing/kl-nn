#!/bin/bash
#SBATCH --job-name=klnn_response_cache
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/shear_response_inference_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mem=8G

set -euo pipefail
ulimit -c 0

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

: "${MODEL_NAME:?Set MODEL_NAME to the completed NPE run}"
EPOCH="${EPOCH:-}"
DATASET="${DATASET:-shear_response_simv2_galaxyaxis_halpha_5k}"
NBASE="${NBASE:-1000}"
NPARTS="${NPARTS:-10}"
NSAMPLES="${NSAMPLES:-5000}"
SEED="${SEED:-42}"
CACHE_TAG="${CACHE_TAG:-${MODEL_NAME}_shear_response_candidates}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
DATA_ROOT="${DATA_ROOT:-/ocean/projects/phy250048p/shared/datasets}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
TOTAL="$((5 * NBASE))"
PART_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-10}"

if (( NPARTS != ARRAY_TASK_COUNT )); then
    echo "NPARTS=${NPARTS}, but this job has ${ARRAY_TASK_COUNT} array tasks; submit with --array=1-${NPARTS}" >&2
    exit 2
fi
if (( TOTAL % NPARTS != 0 )); then
    echo "5*NBASE must be divisible by NPARTS; got ${TOTAL}/${NPARTS}" >&2
    exit 2
fi
NGALS="$((TOTAL / NPARTS))"
if (( NGALS % 5 != 0 )); then
    echo "Each partition must contain complete five-state groups; got NGALS=${NGALS}" >&2
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
    --matched-group-size 5
    --seed "${SEED}"
    --device cuda
)
if [[ -n "${EPOCH}" ]]; then
    ARGS+=(--epoch "${EPOCH}")
fi

echo "model=${MODEL_NAME} epoch=${EPOCH:-best}"
echo "response_dataset=${DATASET} partition=${PART_INDEX}/${NPARTS} matched_group_size=5"
echo "ngals=${NGALS} nsamples=${NSAMPLES} cache_tag=${CACHE_TAG}"

cd /jet/home/xwang30/kl-nn/arch
python cache_posteriors.py "${ARGS[@]}"
