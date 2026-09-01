#!/bin/bash
#SBATCH --job-name=klnn_ccl
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/pretrain_ccl_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -euo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/train_model.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-100}"
CCL_SIGMA_LABEL="${CCL_SIGMA_LABEL:-0.15}"
CCL_D_CUTOFF="${CCL_D_CUTOFF:-0.40}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/train_1m_simv3_cosi/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi/}"
TRAIN_SIZE="${TRAIN_SIZE:-1000000}"
VALID_SIZE="${VALID_SIZE:-100000}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_NAME="${MODEL_NAME:-CNN-CNN-Meta-CCL-simv3-cosi-r90_1m_s${SEED}_${JOB_TAG}}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"

for DATA_DIR in "${TRAIN_DATA}" "${VALID_DATA}"; do
    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "Missing current nine-target dataset: ${DATA_DIR}" >&2
        echo "Build and merge it with the simulator-v3 database launchers before training." >&2
        exit 2
    fi
done

if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$((12000 + ${SLURM_JOB_ID:-356} % 20000))"
fi
export MASTER_PORT PYTHONHASHSEED="${SEED}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME} stage=pretrain views=identity+r90"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"

cd "${KLNN_REPO_ROOT}/arch"
python train_model.py \
    --stage pretrain \
    --train-data "${TRAIN_DATA}" \
    --valid-data "${VALID_DATA}" \
    --train-size "${TRAIN_SIZE}" \
    --valid-size "${VALID_SIZE}" \
    --model-name "${MODEL_NAME}" \
    --model-root "${MODEL_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --ccl-sigma-label "${CCL_SIGMA_LABEL}" \
    --ccl-d-cutoff "${CCL_D_CUTOFF}" \
    --seed "${SEED}" \
    --deterministic \
    --fixed-validation-streams \
    --amp \
    --amp-dtype float16 \
    --compile

echo "Pretraining complete."
echo "PRETRAINED_NAME=${MODEL_NAME} PRETRAIN_FROM=best sbatch arch/train_npe.slurm"
