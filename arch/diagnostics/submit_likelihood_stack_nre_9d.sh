#!/bin/bash
# Submit the separate full-9D NRE training and emcee inference stage.
set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre9d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"
SEED="${SEED:-42}"
GPU_EXCLUDE="${GPU_EXCLUDE:-v005}"
N_WALKERS="${N_WALKERS:-20}"
BURNIN="${BURNIN:-32}"
PRODUCTION="${PRODUCTION:-64}"
MAX_GALAXIES="${MAX_GALAXIES:-128}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_9d_train.slurm" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
if [[ "${NRE_NAME}" == "${PARENT_NPE}" ]]; then
    echo "NRE_NAME must differ from PARENT_NPE" >&2
    exit 2
fi

mkdir -p "${REPORT_ROOT}/03_nre_9d"
TRAIN_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},PARENT_NPE=${PARENT_NPE},NRE_NAME=${NRE_NAME},TRAIN_DATA=${TRAIN_DATA},VALID_DATA=${VALID_DATA},SEED=${SEED},OVERWRITE=1"
INFER_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},PARENT_NPE=${PARENT_NPE},NRE_NAME=${NRE_NAME},DATA_DIR=${DATA_DIR},REPORT_ROOT=${REPORT_ROOT},REPORT_DIR=${REPORT_ROOT}/03_nre_9d,N_WALKERS=${N_WALKERS},BURNIN=${BURNIN},PRODUCTION=${PRODUCTION},MAX_GALAXIES=${MAX_GALAXIES},SEED=${SEED},OVERWRITE=1"

TRAIN_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" \
    --export="${TRAIN_EXPORT}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_9d_train.slurm")
INFER_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" \
    --dependency=afterok:${TRAIN_JOB} \
    --export="${INFER_EXPORT}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_9d_infer.slurm")

STATUS="${REPORT_ROOT}/STATUS.txt"
{
    echo
    echo "NRE9D submitted $(date -R)"
    echo "PARENT_NPE=${PARENT_NPE}"
    echo "NRE_NAME=${NRE_NAME}"
    echo "NRE9D_TRAIN_JOB=${TRAIN_JOB}"
    echo "NRE9D_INFER_JOB=${INFER_JOB} afterok:${TRAIN_JOB}"
    echo "03_nre_9d train full normalized 9D NRE head"
    echo "03_nre_9d infer bounded emcee with TF prior replacement (${MAX_GALAXIES} galaxies, ${N_WALKERS} walkers, burnin=${BURNIN}, production=${PRODUCTION})"
} >> "${STATUS}"
echo "Wrote ${STATUS}"
