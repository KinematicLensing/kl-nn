#!/bin/bash
#SBATCH --job-name=kl_hop_npe
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/hop_boundary_npe_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-}"
if [[ -z "${KLNN_REPO_ROOT}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/arch/diagnostics/hop_boundary_npe.py" ]]; then
        KLNN_REPO_ROOT="${SLURM_SUBMIT_DIR}"
    elif [[ -f "${SLURM_SUBMIT_DIR:-}/diagnostics/hop_boundary_npe.py" ]]; then
        KLNN_REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
        KLNN_REPO_ROOT="/jet/home/xwang30/kl-nn"
    fi
fi
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/hop_boundary_npe.py" ]]; then
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
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi_g02}"
MODEL_NAME="${MODEL_NAME:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_train100k_g02_s42_45802965}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/fiber-gauge/07_hop_boundary}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-2048}"
N_REALIZATIONS="${N_REALIZATIONS:-4}"
SEED="${SEED:-42}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME}"
echo "data_dir=${DATA_DIR}"
echo "report_dir=${REPORT_DIR}"
echo "device=${DEVICE} batch_size=${BATCH_SIZE}"
echo "n_samples=${N_SAMPLES} n_realizations=${N_REALIZATIONS} seed=${SEED}"

ARGS=(
    --model-name "${MODEL_NAME}"
    --model-root "${MODEL_ROOT}"
    --data-dir "${DATA_DIR}"
    --report-dir "${REPORT_DIR}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --n-samples "${N_SAMPLES}"
    --n-realizations "${N_REALIZATIONS}"
    --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/hop_boundary_npe.py "${ARGS[@]}"
