#!/bin/bash
#SBATCH --job-name=kl_tf_s4_m1_10k
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_analysis_stage4_mode1_10k_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

# Paired raw-flow comparison for the completed Stage 4 mode-2 run.
#
# With the defaults, these ten tasks repeat all ten completed mode-2
# partitions: 1,000 galaxies per task and 10,000 galaxies total. Each task
# uses the same base/partition seed, TF-conformed SNR draws, and injected
# image/spectral noise as its mode-2 counterpart. Only inference mode changes
# from 2 (TF replacement) to 1 (raw flow).
# The checkpoint was trained with TF weighting; "raw" here refers only to
# disabling inference-time TF replacement.
#
# Submit from the repository root with:
#   sbatch arch/tf_analysis_stage4_mode1_10k.slurm

set -eo pipefail
ulimit -c 0

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

MODEL_NAME="${MODEL_NAME:-CNN-SetAttn-D4-circular-flow_tf_stage5_s42_43518153}"
EPOCH="${EPOCH:-19}"

SEED="${SEED:-42}"
NETWORK_SOURCE="${NETWORK_SOURCE:-current}"
DATASET="${DATASET:-small_1m}"
SAMPLE_SET="${SAMPLE_SET:-samples_small_1m.csv}"
NGALS="${NGALS:-1000}"
NSAMPLES="${NSAMPLES:-5000}"
NPARTS="${NPARTS:-10}"
CACHE_TAG="${CACHE_TAG:-d4_exact_mode1_raw_10k_s${SEED}}"
PART_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"

if (( NPARTS != 10 || PART_INDEX < 0 || PART_INDEX >= NPARTS )); then
    echo "This paired comparison requires NPARTS=10 and array tasks 1-10" >&2
    exit 2
fi
if (( NGALS != 1000 )); then
    echo "This paired comparison requires NGALS=1000 per partition (10,000 total)" >&2
    exit 2
fi
if (( NSAMPLES <= 0 || NSAMPLES % 8 != 0 )); then
    echo "NSAMPLES must be positive and divisible by 8 for balanced D4 sampling" >&2
    exit 2
fi
if [[ "${NETWORK_SOURCE}" != "archived" && "${NETWORK_SOURCE}" != "current" ]]; then
    echo "NETWORK_SOURCE must be archived or current, got ${NETWORK_SOURCE}" >&2
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
echo "network_source=${NETWORK_SOURCE}"
echo "posterior=d4 mode=1(raw-flow) cancel_add_noise=false"
echo "dataset=${DATASET} sample_set=${SAMPLE_SET}"
echo "partition=${PART_INDEX}/${NPARTS} ngals=${NGALS} nsamples=${NSAMPLES} seed=${SEED}"
echo "paired_mode2_reference=small_1m_tf_conformed_d4_exact/part${PART_INDEX}of10"
echo "output_dataset=${DATASET}_tf_conformed_${CACHE_TAG}"

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
    --mode 1 \
    --cache-tag "${CACHE_TAG}" \
    --seed "${SEED}" \
    --network-source "${NETWORK_SOURCE}" \
    --conform-to-tf \
    --no-cancel-add-noise \
    --no-compile \
    --no-amp \
    --channels-last \
    --inference-mode \
    --profile
