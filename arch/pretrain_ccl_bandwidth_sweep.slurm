#!/bin/bash
#SBATCH --job-name=kl_ccl_bw
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/pretrain_ccl_bw_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -euo pipefail

SIGMAS=(0.125 0.150 0.175)
SIGMA="${SIGMAS[${SLURM_ARRAY_TASK_ID}]}"
SIGMA_TAG="${SIGMA/./p}"

export CCL_SIGMA_LABEL="${SIGMA}"
export CCL_D_CUTOFF="${CCL_D_CUTOFF:-0.40}"
export CCL_OBJECTIVE="ccl"
export EPOCHS="${EPOCHS:-15}"
export SEED="${SEED:-42}"
export MODEL_NAME="${MODEL_NAME_PREFIX:-CNN-CNN_CCL_fixedgeo_rot90_bw}_sig${SIGMA_TAG}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "bandwidth_sweep sigma=${CCL_SIGMA_LABEL} cutoff=${CCL_D_CUTOFF}"
exec bash /jet/home/xwang30/kl-nn/arch/pretrain_ccl_stage1.slurm
