#!/bin/bash
#SBATCH --job-name=kl_npe_stage6_bounded
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_npe_stage6_bounded_hybrid_%j.out
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

# Controlled Stage 6 run: retain the frozen Stage 4 D4 context encoder and
# replace the seven-dimensional affine marginal with a compact, identity-
# initialized RQS marginal. theta_int remains a correlated circular factor
# conditioned on both the context and the sampled bounded parameters.
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MODE="${MODE:-2}"
FLOW_TYPE="${FLOW_TYPE:-bounded_hybrid_circular}"
FLOW_NUM_LAYERS="${FLOW_NUM_LAYERS:-4}"
FLOW_NUM_BINS="${FLOW_NUM_BINS:-8}"
BOUNDED_LOGIT_LIMIT="${BOUNDED_LOGIT_LIMIT:-10}"
THETA_NUM_LAYERS="${THETA_NUM_LAYERS:-1}"
THETA_LOGIT_LIMIT="${THETA_LOGIT_LIMIT:-10}"
INITIAL_LR="${INITIAL_LR:-0.0003}"
# This legacy option name controls the complete non-theta branch for both
# affine and bounded hybrid flows.
AFFINE_LR="${AFFINE_LR:-0.0003}"
THETA_LR="${THETA_LR:-0.0001}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-warmup_cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
MIN_LR="${MIN_LR:-0.00001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_PREFIX="CNN-SetAttn-D4-bounded-hybrid-circular-flow_tf_stage6_fixedfiber"
MODEL_NAME="${MODEL_NAME:-${MODEL_PREFIX}_s${SEED}_${JOB_TAG}}"
PRETRAINED_NAME="${PRETRAINED_NAME:-CNN-SetAttn-D4_CCL_stage4_fixedgeo_s42_43717396}"
PRETRAIN_FROM="${PRETRAIN_FROM:-14}"
CONFIG_PATH="${CONFIG_PATH:-/ocean/projects/phy250048p/shared/configs/cfg_${PRETRAINED_NAME}.json}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m_fixedfiber/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m_fixedfiber/}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VALID_SIZE="${VALID_SIZE:-10000}"

if [[ "${FLOW_TYPE}" != "bounded_hybrid_circular" ]]; then
    echo "Stage 6 bounded run requires FLOW_TYPE=bounded_hybrid_circular, got ${FLOW_TYPE}" >&2
    exit 2
fi
if [[ ! "${FLOW_NUM_LAYERS}" =~ ^[0-9]+$ ]] || (( FLOW_NUM_LAYERS < 1 )); then
    echo "FLOW_NUM_LAYERS must be a positive integer, got ${FLOW_NUM_LAYERS}" >&2
    exit 2
fi
if [[ ! "${FLOW_NUM_BINS}" =~ ^[0-9]+$ ]] || (( FLOW_NUM_BINS < 2 )); then
    echo "FLOW_NUM_BINS must be an integer of at least 2, got ${FLOW_NUM_BINS}" >&2
    exit 2
fi
if [[ ! "${THETA_NUM_LAYERS}" =~ ^[0-9]+$ ]] || (( THETA_NUM_LAYERS < 1 )); then
    echo "THETA_NUM_LAYERS must be a positive integer, got ${THETA_NUM_LAYERS}" >&2
    exit 2
fi
if ! python -c 'import math, sys; value = float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' "${BOUNDED_LOGIT_LIMIT}"; then
    echo "BOUNDED_LOGIT_LIMIT must be positive and finite, got ${BOUNDED_LOGIT_LIMIT}" >&2
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
echo "flow_type=${FLOW_TYPE} layers=${FLOW_NUM_LAYERS} bins=${FLOW_NUM_BINS} bounded_logit_limit=${BOUNDED_LOGIT_LIMIT}"
echo "theta_layers=${THETA_NUM_LAYERS} theta_logit_limit=${THETA_LOGIT_LIMIT}"
echo "mode=${MODE} seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "initial_lr=${INITIAL_LR} non_theta_lr=${AFFINE_LR} theta_lr=${THETA_LR} weight_decay=${WEIGHT_DECAY}"
echo "scheduler=${SCHEDULER_TYPE} warmup_epochs=${WARMUP_EPOCHS} min_lr=${MIN_LR}"
echo "fixed_validation=true context_norm_trainable=false early_stop=${EARLY_STOPPING_PATIENCE} min_delta=${EARLY_STOPPING_MIN_DELTA} gradient_clip=${GRADIENT_CLIP_NORM}"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"
echo "gpu_count=${SLURM_GPUS_ON_NODE:-4} master_port=${MASTER_PORT}"

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
    --flow-num-layers "${FLOW_NUM_LAYERS}" \
    --flow-num-bins "${FLOW_NUM_BINS}" \
    --bounded-logit-limit "${BOUNDED_LOGIT_LIMIT}" \
    --theta-num-layers "${THETA_NUM_LAYERS}" \
    --theta-logit-limit "${THETA_LOGIT_LIMIT}" \
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
    --affine-learning-rate "${AFFINE_LR}" \
    --theta-learning-rate "${THETA_LR}" \
    --scheduler-type "${SCHEDULER_TYPE}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --min-learning-rate "${MIN_LR}" \
    --fixed-validation-streams \
    --no-context-norm-trainable \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    --deterministic \
    --no-compile

echo "Stage 6 bounded hybrid circular posterior training complete."
echo "Use the saved best-checkpoint metadata for downstream Mode 1 analysis."
