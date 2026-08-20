#!/bin/bash
#SBATCH --job-name=kl_ccl_mlp_probe
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/ccl_mlp_probe_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

: "${MODEL_NAME:?Set MODEL_NAME to the completed CCL pretraining run}"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-best}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
VALID_SAMPLES="${VALID_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512 256}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME} checkpoint_suffix=${CHECKPOINT_SUFFIX}"
echo "train_samples=${TRAIN_SAMPLES} valid_samples=${VALID_SAMPLES}"
echo "batch_size=${BATCH_SIZE} probe_epochs=${PROBE_EPOCHS} seed=${SEED}"
echo "hidden_dims=${HIDDEN_DIMS} lr=${LEARNING_RATE} weight_decay=${WEIGHT_DECAY}"

ARGS=(
    --model-name "${MODEL_NAME}"
    --model-root "${MODEL_ROOT}"
    --checkpoint-suffix "${CHECKPOINT_SUFFIX}"
    --train-samples "${TRAIN_SAMPLES}"
    --valid-samples "${VALID_SAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --epochs "${PROBE_EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --device cuda
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

read -r -a HIDDEN_DIM_ARGS <<< "${HIDDEN_DIMS}"
ARGS+=(--hidden-dims "${HIDDEN_DIM_ARGS[@]}")

cd /jet/home/xwang30/kl-nn/arch
python diagnostics/ccl_mlp_probe.py "${ARGS[@]}"
