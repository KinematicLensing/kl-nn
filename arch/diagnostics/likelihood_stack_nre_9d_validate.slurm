#!/bin/bash
#SBATCH --job-name=kl_nre9d_validate
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre9d_validate_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
OUTPUT="${OUTPUT:-${REPORT_ROOT}/03_nre_9d/factorized_validation.json}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

mkdir -p "$(dirname "${OUTPUT}")"
cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nre_9d_validate.py \
    --model-root "${MODEL_ROOT}" \
    --valid-data "${VALID_DATA}" \
    --max-samples "${MAX_SAMPLES}" \
    --output "${OUTPUT}" \
    --device cuda
