#!/bin/bash
#SBATCH --job-name=kl_fib_repair
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/repair_fiber_gauge_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/data_generate/repair_fiber_gauge_campaign.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch fiber-gauge-repair

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate kl-nn
set -u

PHASE="${PHASE:?Set PHASE to dry-run, apply, or verify}"
OUTPUT="${OUTPUT:?Set OUTPUT to the audit JSON path}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

cd "${KLNN_REPO_ROOT}/data_generate"
python repair_fiber_gauge_campaign.py \
    --phase "${PHASE}" \
    --output "${OUTPUT}" \
    --jobs "${JOBS}"
