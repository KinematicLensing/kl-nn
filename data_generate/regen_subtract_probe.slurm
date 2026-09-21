#!/bin/bash
#SBATCH --job-name=kl_regen_sub
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/regen_subtract_probe_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-11
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/data_generate/compare_regen_subtract.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

echo "render index=${SLURM_ARRAY_TASK_ID} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown}"
cd "${KLNN_REPO_ROOT}/data_generate"
python compare_regen_subtract.py --render-index "${SLURM_ARRAY_TASK_ID}"
