#!/bin/bash
#SBATCH --job-name=kl_nre_train
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre_train_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-}"
if [[ -z "${KLNN_REPO_ROOT}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/arch/diagnostics/likelihood_stack_nre_train.py" ]]; then
        KLNN_REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        KLNN_REPO_ROOT="/jet/home/xwang30/kl-nn"
    fi
fi
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_train.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre2d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-1}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
STATUS="${REPORT_ROOT}/STATUS.txt"

if [[ "${NRE_NAME}" == "${PARENT_NPE}" ]]; then
    echo "NRE_NAME must differ from PARENT_NPE" >&2
    exit 2
fi
if [[ "${MODEL_ROOT}/${NRE_NAME}" -ef "${MODEL_ROOT}/${PARENT_NPE}" ]]; then
    echo "refusing to write NRE weights into the parent NPE directory" >&2
    exit 2
fi

mkdir -p "${MODEL_ROOT}/${NRE_NAME}" "${REPORT_ROOT}/03_nre" "$(dirname "${STATUS}")"
{
    echo
    echo "NRE_TRAIN_JOB=${SLURM_JOB_ID:-unknown}  started $(date -R)"
    echo "03_nre train  frozen-encoder 2D NRE head on valid_100k; new nre2d dir only"
} >> "${STATUS}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "parent=${PARENT_NPE}"
echo "nre_name=${NRE_NAME}"
echo "train_data=${TRAIN_DATA}"
echo "valid_data=${VALID_DATA}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} seed=${SEED}"

ARGS=(
    --parent-npe "${PARENT_NPE}"
    --nre-name "${NRE_NAME}"
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
