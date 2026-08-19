#!/bin/bash
#SBATCH --job-name=kl_ccl_stage4
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/pretrain_ccl_stage4_%j.out
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
# Preserve the Stage 2/3 CCL global-batch geometry. The eight-view orbit raises
# activation memory; all requested GPUs are 32 GB. Override to 50 only if a
# production GPU reports OOM.
BATCH_SIZE="${BATCH_SIZE:-100}"
CCL_SIGMA_LABEL="${CCL_SIGMA_LABEL:-0.15}"
CCL_D_CUTOFF="${CCL_D_CUTOFF:-0.40}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_PREFIX="CNN-SetAttn-D4_CCL_stage4_simv2_galaxyaxis"
MODEL_NAME="${MODEL_NAME:-${MODEL_PREFIX}_s${SEED}_${JOB_TAG}}"
PRETRAIN_FROM="$((EPOCHS - 1))"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m_simv2_galaxyaxis/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m_simv2_galaxyaxis/}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
VALID_SIZE="${VALID_SIZE:-10000}"
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
echo "backbone=stage4_d4 orbit_views=8 seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "ccl_sigma_label=${CCL_SIGMA_LABEL} ccl_d_cutoff=${CCL_D_CUTOFF}"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"
echo "gpu_count=${SLURM_GPUS_ON_NODE:-4}"
echo "master_port=${MASTER_PORT}"
# Exact repetition also requires the same GPU count/layout, immutable LMDBs, and
# the same Python, PyTorch, CUDA, cuDNN, NCCL, and driver versions.
cd /jet/home/xwang30/kl-nn/arch
python '[scr]_train_model.py' \
    --model CCLPretrain \
    --trainer FETrainer \
    --backbone-type stage4_d4 \
    --no-rot90-counterpart \
    --train-type pretrain \
    --train-data "${TRAIN_DATA}" \
    --valid-data "${VALID_DATA}" \
    --model-name "${MODEL_NAME}" \
    --pretrained-name "${MODEL_NAME}" \
    --pretrain-from "${PRETRAIN_FROM}" \
    --train-size "${TRAIN_SIZE}" \
    --valid-size "${VALID_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --ccl-sigma-label "${CCL_SIGMA_LABEL}" \
    --ccl-d-cutoff "${CCL_D_CUTOFF}" \
    --seed "${SEED}" \
    --deterministic \
    --no-compile

status=$?
if [[ ${status} -eq 0 ]]; then
    echo "Stage 4 training complete. Probe with:"
    echo "MODEL_NAME=${MODEL_NAME} CHECKPOINT_SUFFIX=${PRETRAIN_FROM} sbatch arch/ccl_mlp_probe.slurm"
fi
exit ${status}
