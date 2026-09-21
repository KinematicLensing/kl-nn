#!/bin/bash
#SBATCH --job-name=kl_nre9d_infer
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_nre9d_infer_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=24:00:00
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
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/03_nre_9d}"
DEVICE="${DEVICE:-cuda}"
N_WALKERS="${N_WALKERS:-32}"
BURNIN="${BURNIN:-100}"
PRODUCTION="${PRODUCTION:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_GALAXIES="${MAX_GALAXIES:-}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"

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

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nre_9d_infer.py "${ARGS[@]}"
python diagnostics/likelihood_stack_write_index.py --report-root "${REPORT_ROOT}"
