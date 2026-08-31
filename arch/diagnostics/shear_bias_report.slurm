#!/bin/bash
#SBATCH --job-name=klnn_test_report
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/shear_bias_report_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G

set -euo pipefail
ulimit -c 0

source /jet/home/xwang30/kl-nn/shared_job_scratch.sh
setup_shared_job_scratch shear-bias-report

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

: "${CASE_1:?Set CASE_1 to MODEL:CACHE_DATASET}"
: "${OUTPUT:?Set OUTPUT to the destination HTML report}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
CASE_2="${CASE_2:-}"
CASE_3="${CASE_3:-}"
MAX_GALAXIES="${MAX_GALAXIES:-}"
BINS="${BINS:-10}"
LOW_G="${LOW_G:-0.02}"
# Preserve the historical production behavior while allowing an equal-galaxy
# test-set report with WEIGHTED=0.
WEIGHTED="${WEIGHTED:-1}"

ARGS=(
    --test-set
    --cache-root "${CACHE_ROOT}"
    --output "${OUTPUT}"
    --bins "${BINS}"
    --low-g "${LOW_G}"
    --case "${CASE_1}"
)
case "${WEIGHTED}" in
    1) ARGS+=(--weighted) ;;
    0) ;;
    *) echo "WEIGHTED must be 0 or 1, got ${WEIGHTED}" >&2; exit 2 ;;
esac
if [[ -n "${CASE_2}" ]]; then
    ARGS+=(--case "${CASE_2}")
fi
if [[ -n "${CASE_3}" ]]; then
    ARGS+=(--case "${CASE_3}")
fi
if [[ -n "${MAX_GALAXIES}" ]]; then
    ARGS+=(--max-galaxies "${MAX_GALAXIES}")
fi

cd /jet/home/xwang30/kl-nn/arch/diagnostics
python shear_bias_report.py "${ARGS[@]}"
