#!/bin/bash
#SBATCH --job-name=kl_nre9d_nautilus
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre9d_nautilus_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre9d-factorized-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
CHAIN_ROOT="${CHAIN_ROOT:-/ocean/projects/phy250048p/shared/cache}"
RUN_NAME="${RUN_NAME:-nautilus_factorized_n100_s42}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/03_nre_9d/${RUN_NAME}}"
N_GALAXIES="${N_GALAXIES:-100}"
MAX_GALAXIES="${MAX_GALAXIES:-}"
SELECTION_SEED="${SELECTION_SEED:-42}"
SEED="${SEED:-42}"
N_LIVE="${N_LIVE:-256}"
N_BATCH="${N_BATCH:-256}"
N_EFF="${N_EFF:-5000}"
N_LIKE_MAX="${N_LIKE_MAX:-200000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_9d_nautilus.py" ]]; then
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

ARGS=(
    --parent-npe "${PARENT_NPE}"
    --nre-name "${NRE_NAME}"
    --model-root "${MODEL_ROOT}"
    --data-dir "${DATA_DIR}"
    --report-dir "${REPORT_DIR}"
    --cache-root "${CHAIN_ROOT}"
    --run-name "${RUN_NAME}"
    --n-galaxies "${N_GALAXIES}"
    --selection-seed "${SELECTION_SEED}"
    --n-live "${N_LIVE}"
    --n-batch "${N_BATCH}"
    --n-eff "${N_EFF}"
    --n-like-max "${N_LIKE_MAX}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --device cuda
)
if [[ -n "${MAX_GALAXIES}" ]]; then
    ARGS+=(--max-galaxies "${MAX_GALAXIES}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
    ARGS+=(--resume)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nre_9d_nautilus.py "${ARGS[@]}"
python diagnostics/likelihood_stack_nre_9d_nautilus_report.py \
    --parent-npe "${PARENT_NPE}" \
    --nre-name "${NRE_NAME}" \
    --cache-root "${CHAIN_ROOT}" \
    --run-name "${RUN_NAME}" \
    --report-dir "${REPORT_DIR}" \
    --overwrite
python diagnostics/likelihood_stack_write_index.py \
    --report-root "${REPORT_ROOT}" \
    --nre9d-report "${REPORT_DIR}"
