#!/bin/bash
#SBATCH --job-name=klnn_cmp_npe
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_comparison_npe_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --exclude=v005
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -euo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/train_model.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${KLNN_REPO_ROOT}/arch/comparison_arch.py" ]]; then
    echo "comparison architecture source is missing: ${KLNN_REPO_ROOT}/arch/comparison_arch.py" >&2
    exit 2
fi

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-100}"
FLOW_NUM_LAYERS="${FLOW_NUM_LAYERS:-4}"
FLOW_NUM_BINS="${FLOW_NUM_BINS:-8}"
BOUNDED_LOGIT_LIMIT="${BOUNDED_LOGIT_LIMIT:-10}"
THETA_NUM_LAYERS="${THETA_NUM_LAYERS:-1}"
THETA_LOGIT_LIMIT="${THETA_LOGIT_LIMIT:-10}"
INITIAL_LR="${INITIAL_LR:-0.0003}"
NON_THETA_LR="${NON_THETA_LR:-0.0003}"
THETA_LR="${THETA_LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-warmup_cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
MIN_LR="${MIN_LR:-0.00001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi/}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VALID_SIZE="${VALID_SIZE:-10000}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_STEM="${MODEL_STEM:-position-query-factorized-simv3-cosi-r90_valid100k_comparison}"
MODEL_NAME="${MODEL_NAME:-${MODEL_STEM}_s${SEED}_${JOB_TAG}}"

if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$((12000 + ${SLURM_JOB_ID:-356} % 20000))"
fi
export MASTER_PORT PYTHONHASHSEED="${SEED}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME} stage=npe arch=comparison views=random(identity,r90)"
echo "pretrained=none (from-scratch encoder; no CCL)"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"
echo "freeze_feature_extractor=--no-freeze-feature-extractor"

cd "${KLNN_REPO_ROOT}/arch"
python train_model.py \
    --stage npe \
    --arch comparison \
    --train-data "${TRAIN_DATA}" \
    --valid-data "${VALID_DATA}" \
    --train-size "${TRAIN_SIZE}" \
    --valid-size "${VALID_SIZE}" \
    --model-name "${MODEL_NAME}" \
    --model-root "${MODEL_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --flow-num-layers "${FLOW_NUM_LAYERS}" \
    --flow-num-bins "${FLOW_NUM_BINS}" \
    --bounded-logit-limit "${BOUNDED_LOGIT_LIMIT}" \
    --theta-num-layers "${THETA_NUM_LAYERS}" \
    --theta-logit-limit "${THETA_LOGIT_LIMIT}" \
    --initial-learning-rate "${INITIAL_LR}" \
    --non-theta-learning-rate "${NON_THETA_LR}" \
    --theta-learning-rate "${THETA_LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --scheduler-type "${SCHEDULER_TYPE}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --min-learning-rate "${MIN_LR}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
    --no-freeze-feature-extractor \
    --seed "${SEED}" \
    --deterministic \
    --fixed-validation-streams \
    --feature-norm-trainable \
    --amp \
    --amp-dtype float16 \
    --compile

echo "Comparison NPE training complete. Use the saved best-checkpoint metadata for inference."
