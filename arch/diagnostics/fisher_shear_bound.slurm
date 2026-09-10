#!/bin/bash
#SBATCH --job-name=kl_fisher_bound
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/fisher_shear_bound_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/shared_job_scratch.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch fisher-shear-bound

module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u

FULL_FITS="${FULL_FITS:-/ocean/projects/phy250048p/shared/fits/fisher_full_simv3_cosi}"
G5_FITS="${G5_FITS:-/ocean/projects/phy250048p/shared/fits/fisher_g5_simv3_cosi}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/ocean/projects/phy250048p/shared/samples}"
OUTPUT="${OUTPUT:-/ocean/projects/phy250048p/shared/reports/fisher_shear_bound.json}"
HTML="${HTML:-/ocean/projects/phy250048p/shared/reports/fisher_shear_bound.html}"

cd "${KLNN_REPO_ROOT}"
python arch/diagnostics/fisher_shear_bound.py \
    --full-fits "${FULL_FITS}" \
    --full-samples "${SAMPLE_ROOT}/samples_fisher_full_simv3_cosi.csv" \
    --full-manifest "${SAMPLE_ROOT}/samples_fisher_full_simv3_cosi_manifest.csv" \
    --g5-fits "${G5_FITS}" \
    --g5-samples "${SAMPLE_ROOT}/samples_fisher_g5_simv3_cosi.csv" \
    --g5-manifest "${SAMPLE_ROOT}/samples_fisher_g5_simv3_cosi_manifest.csv" \
    --output "${OUTPUT}" \
    --html "${HTML}"

PAPER_HTML="${PAPER_HTML:-${KLNN_REPO_ROOT}/Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/fisher_shear_bound.html}"
cp -f "${HTML}" "${PAPER_HTML}"
echo "Copied HTML to ${PAPER_HTML}"
