#!/bin/bash
#SBATCH --job-name=kl_npe_stage5_circ
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_npe_stage5_circular_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=1:30:00
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

# Stage 5 changes only the posterior flow: theta_int is represented by a
# periodic rational-quadratic spline. Keep the Stage 4 backbone and all other
# controlled-run settings fixed so its result is directly comparable.
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MODE="${MODE:-2}"
FLOW_TYPE="${FLOW_TYPE:-circular_rqs}"
FLOW_NUM_BINS="${FLOW_NUM_BINS:-8}"
INITIAL_LR="${INITIAL_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_PREFIX="CNN-SetAttn-D4-circular-flow_tf_stage5"
MODEL_NAME="${MODEL_NAME:-${MODEL_PREFIX}_s${SEED}_${JOB_TAG}}"
PRETRAINED_NAME="${PRETRAINED_NAME:-CNN-SetAttn-D4_CCL_stage4_fixedgeo_s42_43420237}"
PRETRAIN_FROM="${PRETRAIN_FROM:-14}"
CONFIG_PATH="${CONFIG_PATH:-/ocean/projects/phy250048p/shared/configs/cfg_${PRETRAINED_NAME}.json}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m/}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VALID_SIZE="${VALID_SIZE:-10000}"
CHECKPOINT_SUFFIX="$((EPOCHS - 1))"

if [[ "${FLOW_TYPE}" != "circular_rqs" ]]; then
    echo "Stage 5 requires FLOW_TYPE=circular_rqs, got ${FLOW_TYPE}" >&2
    exit 2
fi
if (( FLOW_NUM_BINS < 2 )); then
    echo "FLOW_NUM_BINS must be at least 2, got ${FLOW_NUM_BINS}" >&2
    exit 2
fi
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
echo "pretrained=${PRETRAINED_NAME} checkpoint=${PRETRAIN_FROM}"
echo "backbone=stage4_d4 posterior_symmetry=d4 rot90_counterpart=false"
echo "flow_type=${FLOW_TYPE} flow_num_bins=${FLOW_NUM_BINS}"
echo "mode=${MODE} seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "initial_lr=${INITIAL_LR} weight_decay=${WEIGHT_DECAY}"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"
echo "gpu_count=${SLURM_GPUS_ON_NODE:-4}"
echo "master_port=${MASTER_PORT}"

cd /jet/home/xwang30/kl-nn/arch
python '[scr]_train_model.py' \
    --config "${CONFIG_PATH}" \
    --model KLNPE \
    --trainer NPETrainer \
    --train-type train \
    --backbone-type stage4_d4 \
    --posterior-symmetry d4 \
    --no-rot90-counterpart \
    --flow-type "${FLOW_TYPE}" \
    --flow-num-bins "${FLOW_NUM_BINS}" \
    --mode "${MODE}" \
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
    --seed "${SEED}" \
    --deterministic \
    --no-compile

echo "Stage 5 circular-theta posterior training complete. Evaluate raw Mode 1 with:"
echo "MODEL_NAME=${MODEL_NAME} EPOCH=${CHECKPOINT_SUFFIX} sbatch arch/tf_analysis_stage4_mode1_10k.slurm"
