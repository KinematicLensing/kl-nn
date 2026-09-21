#!/bin/bash
#SBATCH --job-name=kl_like_stack
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-}"
if [[ -z "${KLNN_REPO_ROOT}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/arch/diagnostics/likelihood_stack.py" ]]; then
        KLNN_REPO_ROOT="${SLURM_SUBMIT_DIR}"
    elif [[ -f "${SLURM_SUBMIT_DIR:-}/diagnostics/likelihood_stack.py" ]]; then
        KLNN_REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
        KLNN_REPO_ROOT="/jet/home/xwang30/kl-nn"
    fi
fi
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack.py" ]]; then
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
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"
MODEL_NAME="${MODEL_NAME:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
GALAXIES="${GALAXIES:-${REPORT_ROOT}/00_select/galaxies.npz}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/01_stack}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-1}"
REUSE_DRAWS="${REUSE_DRAWS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-2048}"
N_REALIZATIONS="${N_REALIZATIONS:-16}"
PROPOSAL_PER_REALIZATION="${PROPOSAL_PER_REALIZATION:-256}"
SEED="${SEED:-42}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "model=${MODEL_NAME}"
echo "data_dir=${DATA_DIR}"
echo "galaxies=${GALAXIES}"
echo "report_dir=${REPORT_DIR}"
echo "device=${DEVICE} batch_size=${BATCH_SIZE}"
echo "n_samples=${N_SAMPLES} n_realizations=${N_REALIZATIONS}"
echo "proposal_per_realization=${PROPOSAL_PER_REALIZATION} seed=${SEED}"

ARGS=(
    --model-name "${MODEL_NAME}"
    --model-root "${MODEL_ROOT}"
    --data-dir "${DATA_DIR}"
    --galaxies "${GALAXIES}"
    --report-dir "${REPORT_DIR}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --n-samples "${N_SAMPLES}"
    --n-realizations "${N_REALIZATIONS}"
    --proposal-per-realization "${PROPOSAL_PER_REALIZATION}"
    --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${REUSE_DRAWS}" == "1" ]]; then
    ARGS+=(--reuse-draws)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack.py "${ARGS[@]}"
python diagnostics/likelihood_stack_write_index.py --report-root "${REPORT_ROOT}"
