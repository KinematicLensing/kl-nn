#!/bin/bash
#SBATCH --job-name=kl_nre_infer
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre_infer_%j.out
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
    if [[ -f "${SLURM_SUBMIT_DIR:-}/arch/diagnostics/likelihood_stack_nre_infer.py" ]]; then
        KLNN_REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        KLNN_REPO_ROOT="/jet/home/xwang30/kl-nn"
    fi
fi
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_infer.py" ]]; then
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
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre2d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
CASE="${CASE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/03_nre}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRID_N="${GRID_N:-21}"
SEED="${SEED:-42}"
STATUS="${REPORT_ROOT}/STATUS.txt"

if [[ "${NRE_NAME}" == "${PARENT_NPE}" ]]; then
    echo "NRE_NAME must differ from PARENT_NPE" >&2
    exit 2
fi

mkdir -p "${REPORT_DIR}" "$(dirname "${STATUS}")"
{
    echo
    echo "NRE_INFER_JOB=${SLURM_JOB_ID:-unknown}  started $(date -R)"
    echo "03_nre infer  xu3 21x21 MAP, coverage, Bernstein vs cached NPE Means"
} >> "${STATUS}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "parent=${PARENT_NPE}"
echo "nre_name=${NRE_NAME}"
echo "data_dir=${DATA_DIR}"
echo "report_dir=${REPORT_DIR}"
echo "grid_n=${GRID_N} batch_size=${BATCH_SIZE} seed=${SEED}"

ARGS=(
    --parent-npe "${PARENT_NPE}"
    --nre-name "${NRE_NAME}"
    --model-root "${MODEL_ROOT}"
    --data-dir "${DATA_DIR}"
    --cache-root "${CACHE_ROOT}"
    --case "${CASE}"
    --report-dir "${REPORT_DIR}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --grid-n "${GRID_N}"
    --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nre_infer.py "${ARGS[@]}"
python diagnostics/likelihood_stack_write_index.py --report-root "${REPORT_ROOT}"
