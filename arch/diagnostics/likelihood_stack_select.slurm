#!/bin/bash
#SBATCH --job-name=kl_like_sel
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/likelihood_stack_select_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_select.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch likelihood-stack-select

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/phy250048p/shared/cache}"
CASE="${CASE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/likelihood-stack/00_select}"
N_SELECTED="${N_SELECTED:-128}"
ABS_G_MIN="${ABS_G_MIN:-0.02}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"
REUSE_GALAXIES="${REUSE_GALAXIES:-0}"

export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "case=${CASE}"
echo "report_dir=${REPORT_DIR}"
echo "n_selected=${N_SELECTED} abs_g_min=${ABS_G_MIN} seed=${SEED}"

ARGS=(
    --cache-root "${CACHE_ROOT}"
    --case "${CASE}"
    --report-dir "${REPORT_DIR}"
    --n-selected "${N_SELECTED}"
    --abs-g-min "${ABS_G_MIN}"
    --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${REUSE_GALAXIES}" == "1" ]]; then
    ARGS+=(--reuse-galaxies)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_select.py "${ARGS[@]}"
python diagnostics/likelihood_stack_write_index.py --report-root "$(dirname "${REPORT_DIR}")"
