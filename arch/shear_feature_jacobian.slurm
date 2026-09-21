#!/bin/bash
#SBATCH --job-name=kl_feat_jac
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/shear_feature_jacobian_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/shear_feature_jacobian.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/fiber-gauge/06_feature_jacobian}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "report_dir=${REPORT_DIR}"
echo "model_root=${MODEL_ROOT}"
echo "device=${DEVICE}"

ARGS=(
    --model-root "${MODEL_ROOT}"
    --report-dir "${REPORT_DIR}"
    --device "${DEVICE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/shear_feature_jacobian.py "${ARGS[@]}"
