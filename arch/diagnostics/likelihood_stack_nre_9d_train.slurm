#!/bin/bash
#SBATCH --job-name=kl_nre9d_train
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre9d_train_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre9d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_train.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
if [[ "${NRE_NAME}" == "${PARENT_NPE}" ]]; then
    echo "NRE_NAME must differ from PARENT_NPE" >&2
    exit 2
fi

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

ARGS=(
    --parent-npe "${PARENT_NPE}"
    --nre-name "${NRE_NAME}"
    --nre-mode 9d
    --model-root "${MODEL_ROOT}"
    --train-data "${TRAIN_DATA}"
    --valid-data "${VALID_DATA}"
    --device "${DEVICE}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --learning-rate "${LR}"
    --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nre_train.py "${ARGS[@]}"
