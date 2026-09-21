#!/bin/bash
#SBATCH --job-name=kl_desi_test_samples
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_desi_test_sets_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

set -eo pipefail
ulimit -c 0

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
if [[ ! -f "${KLNN_REPO_ROOT}/shared_job_scratch.sh" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"
setup_shared_job_scratch desi-test-samples

module load anaconda3
source ~/.bashrc
conda activate kl-nn
set -u

CATALOG_ROOT="${CATALOG_ROOT:-/ocean/projects/phy250048p/shared/desi_cut_catalogs}"
SAMPLE_ROOT="${SAMPLE_ROOT:-/ocean/projects/phy250048p/shared/samples}"
PRODUCTION_TOTAL="100000"
REQUESTED_TOTAL="${TOTAL:-${PRODUCTION_TOTAL}}"
BASE_SEED="${BASE_SEED:-42000}"
CATALOG_BLOCK_SIZE="${CATALOG_BLOCK_SIZE:-500000}"
TF_SLOPE="${TF_SLOPE:--7.22}"
TF_INTERCEPT="${TF_INTERCEPT:-36.0}"
TF_SCATTER_DEX="${TF_SCATTER_DEX:-0.1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SAMPLE_CUTS=(1 3 5)

if [[ "${REQUESTED_TOTAL}" != "${PRODUCTION_TOTAL}" ]]; then
    echo "This production launcher requires TOTAL=${PRODUCTION_TOTAL}; got TOTAL=${REQUESTED_TOTAL}" >&2
    exit 2
fi
if [[ -n "${SAMPLE_NAME:-}" ]]; then
    echo "SAMPLE_NAME overrides are unsupported: each array task uses its canonical Xu-cut name" >&2
    exit 2
fi
TOTAL="${PRODUCTION_TOTAL}"
if (( CATALOG_BLOCK_SIZE <= 0 )); then
    echo "CATALOG_BLOCK_SIZE must be positive" >&2
    exit 2
fi
if [[ "${ALLOW_OVERWRITE}" != "0" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    echo "ALLOW_OVERWRITE must be 0 or 1" >&2
    exit 2
fi
TASK_INDEX="$((SLURM_ARRAY_TASK_ID - 1))"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#SAMPLE_CUTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID must be in 1-${#SAMPLE_CUTS[@]}" >&2
    exit 2
fi

CUT="${SAMPLE_CUTS[${TASK_INDEX}]}"
CATALOG="${CATALOG_ROOT}/xu_sample_${CUT}_fullfootprint.fits"
CANONICAL_SAMPLE_NAME="test_100k_simv3_cosi_xu${CUT}_tf"
OUTPUT="${SAMPLE_ROOT}/${CANONICAL_SAMPLE_NAME}.csv"
MANIFEST="${SAMPLE_ROOT}/${CANONICAL_SAMPLE_NAME}.manifest.json"
SEED="$((BASE_SEED + CUT))"

if [[ ! -f "${CATALOG}" ]]; then
    echo "Missing DESI cut catalog: ${CATALOG}" >&2
    exit 2
fi
if [[ "${ALLOW_OVERWRITE}" != "1" && ( -e "${OUTPUT}" || -e "${MANIFEST}" ) ]]; then
    echo "Refusing to overwrite existing sample output or manifest: ${OUTPUT}" >&2
    exit 2
fi

cd "${KLNN_REPO_ROOT}/data_generate"
python latin_hypercube.py \
    --test-set \
    --nsamples="${TOTAL}" \
    --seed="${SEED}" \
    --catalog="${CATALOG}" \
    --catalog-block-size="${CATALOG_BLOCK_SIZE}" \
    --tf-slope="${TF_SLOPE}" \
    --tf-intercept="${TF_INTERCEPT}" \
    --tf-scatter-dex="${TF_SCATTER_DEX}" \
    --output="${OUTPUT}"
