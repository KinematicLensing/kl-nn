#!/bin/bash
#SBATCH --job-name=kl_like_cov
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_coverage_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_coverage.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
CASE="${CASE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/likelihood-stack/02_coverage}"
REPORT_ROOT="${REPORT_ROOT:-$(dirname "${REPORT_DIR}")}"
ABS_G_SPLIT="${ABS_G_SPLIT:-0.05}"
OVERWRITE="${OVERWRITE:-1}"
STATUS="${REPORT_ROOT}/STATUS.txt"
mkdir -p "${REPORT_DIR}" "$(dirname "${STATUS}")"
{
    echo
    echo "COVERAGE_JOB=${SLURM_JOB_ID:-unknown}  started $(date -R)"
    echo "02_coverage  16-84% coverage on |g|<0.05 vs |g|>0.05"
} >> "${STATUS}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "case=${CASE}"
echo "report_dir=${REPORT_DIR}"
echo "abs_g_split=${ABS_G_SPLIT}"

ARGS=(
    --cache-root "${CACHE_ROOT}"
    --case "${CASE}"
    --report-dir "${REPORT_DIR}"
    --abs-g-split "${ABS_G_SPLIT}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_coverage.py "${ARGS[@]}"
python diagnostics/likelihood_stack_write_index.py --report-root "$(dirname "${REPORT_DIR}")"
