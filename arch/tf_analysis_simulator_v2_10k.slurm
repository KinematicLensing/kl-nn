#!/bin/bash
#SBATCH --job-name=kl_tf_simv2_10k
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_analysis_simulator_v2_10k_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

# Ten-part, 10,000-galaxy analysis for the simulator-v2 mode-1 baseline.
# TF_INFERENCE=prior_replacement replaces the broad training prior using the
# noisy catalog magnitude and its observed-flux uncertainty. Set
# TF_INFERENCE=none for the otherwise identical broad-prior posterior.
#
# Example:
#   MODEL_NAME=... EPOCH=19 sbatch arch/tf_analysis_simulator_v2_10k.slurm

set -eo pipefail
ulimit -c 0

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

MODEL_NAME="${MODEL_NAME:-CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568}"
EPOCH="${EPOCH:-19}"
SEED="${SEED:-42}"
NETWORK_SOURCE="${NETWORK_SOURCE:-archived}"
DATASET="${DATASET:-small_1m_simv2_galaxyaxis}"
SAMPLE_SET="${SAMPLE_SET:-samples_small_1m_simv2_galaxyaxis.csv}"
TF_INFERENCE="${TF_INFERENCE:-prior_replacement}"
NGALS="${NGALS:-1000}"
NSAMPLES="${NSAMPLES:-5000}"
NPARTS="${NPARTS:-10}"
CACHE_TAG="${CACHE_TAG:-simv2_${TF_INFERENCE}_10k_s${SEED}}"
PART_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"

if (( NPARTS != 10 || PART_INDEX < 0 || PART_INDEX >= NPARTS )); then
    echo "This 10k launcher requires NPARTS=10 and array tasks 1-10" >&2
    exit 2
fi
if (( NGALS != 1000 )); then
    echo "This 10k launcher requires NGALS=1000 per partition" >&2
    exit 2
fi
if (( NSAMPLES <= 0 || NSAMPLES % 8 != 0 )); then
    echo "NSAMPLES must be positive and divisible by 8 for exact D4 sampling" >&2
    exit 2
fi
if [[ "${TF_INFERENCE}" != "none" && "${TF_INFERENCE}" != "prior_replacement" ]]; then
    echo "TF_INFERENCE must be none or prior_replacement" >&2
    exit 2
fi
if [[ "${NETWORK_SOURCE}" != "archived" && "${NETWORK_SOURCE}" != "current" ]]; then
    echo "NETWORK_SOURCE must be archived or current" >&2
    exit 2
fi

MODEL_CHECKPOINT="/ocean/projects/phy250048p/shared/models/${MODEL_NAME}/${MODEL_NAME}${EPOCH}"
MODEL_CONFIG="/ocean/projects/phy250048p/shared/configs/cfg_${MODEL_NAME}.json"
DATASET_PATH="/ocean/projects/phy250048p/shared/datasets/${DATASET}"
SAMPLE_SET_PATH="/ocean/projects/phy250048p/shared/samples/${SAMPLE_SET}"
if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "Missing NPE checkpoint: ${MODEL_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "Missing NPE config: ${MODEL_CONFIG}" >&2
    exit 2
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Missing simulator-v2 dataset: ${DATASET_PATH}" >&2
    exit 2
fi
if [[ ! -f "${SAMPLE_SET_PATH}" ]]; then
    echo "Missing sample-set CSV: ${SAMPLE_SET_PATH}" >&2
    exit 2
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME} epoch=${EPOCH}"
echo "observation=v2 fiber_layout=galaxy_axis mode=1 tf_inference=${TF_INFERENCE}"
echo "dataset=${DATASET} sample_set=${SAMPLE_SET}"
echo "partition=${PART_INDEX}/${NPARTS} ngals=${NGALS} nsamples=${NSAMPLES} seed=${SEED}"
echo "output_cache_tag=${CACHE_TAG}"

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
    --tf-inference "${TF_INFERENCE}" \
    --cache-tag "${CACHE_TAG}" \
    --seed "${SEED}" \
    --network-source "${NETWORK_SOURCE}" \
    --matched-group-size 1 \
    --no-conform-to-tf \
    --no-cancel-add-noise \
    --no-compile \
    --no-amp \
    --channels-last \
    --inference-mode \
    --profile
