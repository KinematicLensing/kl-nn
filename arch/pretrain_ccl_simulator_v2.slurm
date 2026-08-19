#!/bin/bash
#SBATCH --job-name=kl_ccl_simv2
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/pretrain_ccl_simulator_v2_%j.out
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
BATCH_SIZE="${BATCH_SIZE:-100}"
CCL_SIGMA_LABEL="${CCL_SIGMA_LABEL:-0.15}"
CCL_D_CUTOFF="${CCL_D_CUTOFF:-0.40}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
MODEL_PREFIX="CNN-SetAttn-D4_CCL_simv2_galaxyaxis"
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
echo "observation=v2 fiber_layout=galaxy_axis rmag=15:23.4 halpha_flux=1.2e-16:301.43e-16_uniform image_depth_r=23.4 spec_quality=3:100"
echo "train=${TRAIN_DATA} size=${TRAIN_SIZE}"
echo "valid=${VALID_DATA} size=${VALID_SIZE}"

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
    echo "Simulator-v2 CCL training complete."
    echo "PRETRAINED_NAME=${MODEL_NAME} PRETRAIN_FROM=${PRETRAIN_FROM} sbatch arch/train_npe_simulator_v2_affine.slurm"
fi
exit ${status}
