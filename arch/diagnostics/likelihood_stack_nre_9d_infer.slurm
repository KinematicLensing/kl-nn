#!/bin/bash
#SBATCH --job-name=kl_nre9d_infer
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre9d_infer_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
MODEL_ROOT="${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre9d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
CHAIN_ROOT="${CHAIN_ROOT:-/ocean/projects/phy250048p/shared/cache}"
RUN_NAME="${RUN_NAME:-emcee_w48_b2500_p6000_s42}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/03_nre_9d/${RUN_NAME}}"
GALAXIES="${GALAXIES:-${REPORT_ROOT}/00_select/galaxies.npz}"
STACK_DRAWS="${STACK_DRAWS:-${REPORT_ROOT}/01_stack/stack.npz}"
DEVICE="${DEVICE:-cuda}"
N_WALKERS="${N_WALKERS:-48}"
BURNIN="${BURNIN:-2500}"
PRODUCTION="${PRODUCTION:-6000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_GALAXIES="${MAX_GALAXIES:-}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_9d_infer.py" ]]; then
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
    --galaxies "${GALAXIES}"
    --stack-draws "${STACK_DRAWS}"
    --nwalkers "${N_WALKERS}"
    --burnin "${BURNIN}"
    --production "${PRODUCTION}"
    --batch-size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --seed "${SEED}"
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
python diagnostics/likelihood_stack_nre_9d_infer.py "${ARGS[@]}"
python diagnostics/likelihood_stack_nre_9d_corner_report.py \
    --parent-npe "${PARENT_NPE}" \
    --nre-name "${NRE_NAME}" \
    --model-root "${MODEL_ROOT}" \
    --cache-root "${CHAIN_ROOT}" \
    --run-name "${RUN_NAME}" \
    --report-dir "${REPORT_DIR}" \
    --galaxies "${GALAXIES}" \
    --stack-draws "${STACK_DRAWS}" \
    --overwrite
python diagnostics/likelihood_stack_write_index.py --report-root "${REPORT_ROOT}"
