#!/bin/bash
#SBATCH --job-name=kl_npe_simv2
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_npe_simulator_v2_affine_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=2:00:00
#SBATCH --nodes=1
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

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
INITIAL_LR="${INITIAL_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_PREFIX="CNN-SetAttn-D4-affine_simv2_galaxyaxis"
MODEL_NAME="${MODEL_NAME:-${MODEL_PREFIX}_s${SEED}_${JOB_TAG}}"
PRETRAINED_NAME="${PRETRAINED_NAME:-CNN-SetAttn-D4_CCL_simv2_galaxyaxis_s42_43861749}"
PRETRAIN_FROM="${PRETRAIN_FROM:-19}"
CONFIG_PATH="${CONFIG_PATH:-/ocean/projects/phy250048p/shared/configs/cfg_${PRETRAINED_NAME}.json}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m_simv2_galaxyaxis/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m_simv2_galaxyaxis/}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VALID_SIZE="${VALID_SIZE:-10000}"
CHECKPOINT_SUFFIX="$((EPOCHS - 1))"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing pretraining config: ${CONFIG_PATH}" >&2
    exit 2
fi
PRETRAIN_CHECKPOINT="/ocean/projects/phy250048p/shared/models/${PRETRAINED_NAME}/${PRETRAINED_NAME}${PRETRAIN_FROM}"
if [[ ! -f "${PRETRAIN_CHECKPOINT}" ]]; then
    echo "Missing pretraining checkpoint: ${PRETRAIN_CHECKPOINT}" >&2
    exit 2
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        MASTER_PORT="$((12000 + SLURM_JOB_ID % 20000))"
    else
        MASTER_PORT=12356
    fi
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MASTER_PORT
export TORCH_FR_BUFFER_SIZE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "model=${MODEL_NAME}"
echo "controlled_baseline=galaxy_axis 12-layer affine MAF exact-D4 mode1"
echo "pretrained=${PRETRAINED_NAME} checkpoint=${PRETRAIN_FROM}"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"

cd /jet/home/xwang30/kl-nn/arch
python '[scr]_train_model.py' \
    --config "${CONFIG_PATH}" \
    --model KLNPE \
    --trainer NPETrainer \
    --train-type train \
    --backbone-type stage4_d4 \
    --posterior-symmetry d4 \
    --no-rot90-counterpart \
    --mode 1 \
    --flow-type affine \
    --flow-num-layers 12 \
    --train-data "${TRAIN_DATA}" \
    --valid-data "${VALID_DATA}" \
    --train-size "${TRAIN_SIZE}" \
    --valid-size "${VALID_SIZE}" \
    --model-name "${MODEL_NAME}" \
    --pretrained-name "${PRETRAINED_NAME}" \
    --pretrain-from "${PRETRAIN_FROM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --initial-learning-rate "${INITIAL_LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --observation-model-version 2 \
    --fiber-layout galaxy_axis \
    --rmag-min 15 \
    --rmag-max 23.4 \
    --halpha-flux-min 1.2e-16 \
    --halpha-flux-max 301.43e-16 \
    --halpha-flux-distribution uniform \
    --image-band r \
    --image-depth-5sigma-mag 23.4 \
    --spectral-quality-min 3 \
    --spectral-quality-max 100 \
    --spectral-quality-distribution log_uniform \
    --spectral-units counts \
    --center-fiber-index 2 \
    --center-exposure-s 180 \
    --offset-exposure-s 600 \
    --seed "${SEED}" \
    --deterministic \
    --fixed-validation-streams \
    --no-compile

status=$?
if [[ ${status} -eq 0 ]]; then
    echo "Simulator-v2 affine NPE training complete."
    echo "Use the named best checkpoint when available; final numeric suffix is ${CHECKPOINT_SUFFIX}."
fi
exit ${status}
