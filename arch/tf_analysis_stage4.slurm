#!/bin/bash
#SBATCH --job-name=kl_tf_stage4
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_analysis_stage4_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

MODEL_NAME="${MODEL_NAME:-CNN-SetAttn-D4-flow_tf_stage4_s42_43435707}"
EPOCH="${EPOCH:-19}"

SEED="${SEED:-42}"
DATASET="${DATASET:-small_1m}"
SAMPLE_SET="${SAMPLE_SET:-samples_small_1m.csv}"
NGALS="${NGALS:-1000}"
NSAMPLES="${NSAMPLES:-5000}"
NPARTS="${NPARTS:-10}"
CACHE_TAG="${CACHE_TAG:-d4_exact}"
PART_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"

if (( NSAMPLES <= 0 || NSAMPLES % 8 != 0 )); then
    echo "NSAMPLES must be positive and divisible by 8 for balanced D4 sampling" >&2
    exit 2
fi
MODEL_CHECKPOINT="/ocean/projects/phy250048p/shared/models/${MODEL_NAME}/${MODEL_NAME}${EPOCH}"
MODEL_CONFIG="/ocean/projects/phy250048p/shared/configs/cfg_${MODEL_NAME}.json"
if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "Missing NPE checkpoint: ${MODEL_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "Missing NPE config: ${MODEL_CONFIG}" >&2
    exit 2
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME} epoch=${EPOCH}"
echo "posterior=d4 mode=2 cancel_add_noise=false"
echo "dataset=${DATASET} sample_set=${SAMPLE_SET}"
echo "partition=${PART_INDEX}/${NPARTS} ngals=${NGALS} nsamples=${NSAMPLES} seed=${SEED}"

cd /jet/home/xwang30/kl-nn/arch
python '[scr]_tf_analysis.py' \
    -i "${PART_INDEX}" \
    --nparts "${NPARTS}" \
    --ngals "${NGALS}" \
    --nsamples "${NSAMPLES}" \
    --stem "${MODEL_NAME}" \
    --epoch "${EPOCH}" \
    --dataset "${DATASET}" \
    --sample-set "${SAMPLE_SET}" \
    --mode 2 \
    --cache-tag "${CACHE_TAG}" \
    --seed "${SEED}" \
    --conform-to-tf \
    --no-cancel-add-noise \
    --no-compile \
    --no-amp \
    --channels-last \
    --inference-mode \
    --profile
