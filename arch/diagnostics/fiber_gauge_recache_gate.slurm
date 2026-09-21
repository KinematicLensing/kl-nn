#!/bin/bash
#SBATCH --job-name=kl_fib_gate
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/fiber_gauge_gate_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/fiber_gauge_recache_gate.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch fiber-gauge-gate

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

: "${CASE_1:?Set CASE_1 to MODEL:CACHE_DATASET}"
: "${OUTPUT:?Set OUTPUT to the gate JSON path}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
CASE_2="${CASE_2:-}"

ARGS=(
    --cache-root "${CACHE_ROOT}"
    --output "${OUTPUT}"
    --case "${CASE_1}"
)
if [[ -n "${CASE_2}" ]]; then
    ARGS+=(--case "${CASE_2}")
fi

cd "${KLNN_REPO_ROOT}/arch/diagnostics"
python fiber_gauge_recache_gate.py "${ARGS[@]}"
