#!/bin/bash
# Submit the fiber-gauge campaign:
# dry-run -> apply -> verify -> recache g02/g002 -> report -> gate
# -> ±0.1 CCL -> frozen NPE -> cache -> three-way report.
# The gate job fails if theta_int / g02 m did not recover, which blocks retrain.
set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/fiber-gauge}"
SEED="${SEED:-42}"
CACHE_TAG="${CACHE_TAG:-testset_tfweighted_v2_10k_s42_righthanded}"
DATASET="${DATASET:-test_100k_simv3_cosi_xu3_tf}"
G02_MODEL="${G02_MODEL:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_train100k_g02_s42_45802965}"
G002_MODEL="${G002_MODEL:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_train100k_g002_s42_45802967}"
CCL_NAME="${CCL_NAME:-CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k_s42_fibrepair}"
NPE_NAME="${NPE_NAME:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi/}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi/}"
GPU_EXCLUDE="${GPU_EXCLUDE:-v005}"

if [[ ! -f "${KLNN_REPO_ROOT}/data_generate/repair_fiber_gauge.slurm" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

mkdir -p \
    "${REPORT_ROOT}/01_repair_audit" \
    "${REPORT_ROOT}/02_g02_g002_recache" \
    "${REPORT_ROOT}/03_g01_retrain"

G02_CASE="${G02_MODEL}:${DATASET}_${CACHE_TAG}"
G002_CASE="${G002_MODEL}:${DATASET}_${CACHE_TAG}"
G01_CASE="${NPE_NAME}:${DATASET}_${CACHE_TAG}"
REPAIR_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT}"
TRAIN_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},SEED=${SEED},TRAIN_SIZE=100000,VALID_SIZE=10000,TRAIN_DATA=${TRAIN_DATA},VALID_DATA=${VALID_DATA},SHEAR_BOUND=0.1"
CACHE_COMMON="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},DATASET=${DATASET},NPARTS=100,NGALS=1000,NSAMPLES=10000,TEST_SET=1,CACHE_TAG=${CACHE_TAG},SEED=${SEED}"

DRY_JOB=$(sbatch --parsable \
    --export="${REPAIR_EXPORT},PHASE=dry-run,OUTPUT=${REPORT_ROOT}/01_repair_audit/dry_run.json" \
    "${KLNN_REPO_ROOT}/data_generate/repair_fiber_gauge.slurm")
echo "DRY_JOB=${DRY_JOB}"

APPLY_JOB=$(sbatch --parsable --dependency=afterok:${DRY_JOB} \
    --export="${REPAIR_EXPORT},PHASE=apply,OUTPUT=${REPORT_ROOT}/01_repair_audit/apply.json" \
    "${KLNN_REPO_ROOT}/data_generate/repair_fiber_gauge.slurm")
echo "APPLY_JOB=${APPLY_JOB}"

VERIFY_JOB=$(sbatch --parsable --dependency=afterok:${APPLY_JOB} \
    --export="${REPAIR_EXPORT},PHASE=verify,OUTPUT=${REPORT_ROOT}/01_repair_audit/verify.json" \
    "${KLNN_REPO_ROOT}/data_generate/repair_fiber_gauge.slurm")
echo "VERIFY_JOB=${VERIFY_JOB}"

CACHE_G02_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" --array=1-100 \
    --dependency=afterok:${VERIFY_JOB} \
    --export="${CACHE_COMMON},MODEL_NAME=${G02_MODEL}" \
    "${KLNN_REPO_ROOT}/arch/cache_posteriors.slurm")
echo "CACHE_G02_JOB=${CACHE_G02_JOB}"

CACHE_G002_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" --array=1-100 \
    --dependency=afterok:${VERIFY_JOB} \
    --export="${CACHE_COMMON},MODEL_NAME=${G002_MODEL}" \
    "${KLNN_REPO_ROOT}/arch/cache_posteriors.slurm")
echo "CACHE_G002_JOB=${CACHE_G002_JOB}"

REPORT_G02_JOB=$(sbatch --parsable --dependency=afterok:${CACHE_G02_JOB}:${CACHE_G002_JOB} \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},CASE_1=${G02_CASE},CASE_2=${G002_CASE},OUTPUT=${REPORT_ROOT}/02_g02_g002_recache/report.html,WEIGHTED=1" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/shear_bias_report.slurm")
echo "REPORT_G02_JOB=${REPORT_G02_JOB}"

GATE_JOB=$(sbatch --parsable --dependency=afterok:${REPORT_G02_JOB} \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},CASE_1=${G02_CASE},CASE_2=${G002_CASE},OUTPUT=${REPORT_ROOT}/02_g02_g002_recache/gate.json" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/fiber_gauge_recache_gate.slurm")
echo "GATE_JOB=${GATE_JOB}"

CCL_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" --dependency=afterok:${GATE_JOB} \
    --export="${TRAIN_EXPORT},MODEL_NAME=${CCL_NAME},MODEL_STEM=CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k" \
    "${KLNN_REPO_ROOT}/arch/pretrain_ccl.slurm")
echo "CCL_JOB=${CCL_JOB} MODEL_NAME=${CCL_NAME}"

NPE_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" --dependency=afterok:${CCL_JOB} \
    --export="${TRAIN_EXPORT},PRETRAINED_NAME=${CCL_NAME},MODEL_NAME=${NPE_NAME},FREEZE_FEATURE_EXTRACTOR=1,IMAGE_SPECTRUM_FUSION=0,MODEL_STEM=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen" \
    "${KLNN_REPO_ROOT}/arch/train_npe.slurm")
echo "NPE_JOB=${NPE_JOB} MODEL_NAME=${NPE_NAME}"

CACHE_G01_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" --array=1-100 \
    --dependency=afterok:${NPE_JOB} \
    --export="${CACHE_COMMON},MODEL_NAME=${NPE_NAME}" \
    "${KLNN_REPO_ROOT}/arch/cache_posteriors.slurm")
echo "CACHE_G01_JOB=${CACHE_G01_JOB}"

REPORT_G01_JOB=$(sbatch --parsable --dependency=afterok:${CACHE_G01_JOB} \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},CASE_1=${G02_CASE},CASE_2=${G01_CASE},CASE_3=${G002_CASE},OUTPUT=${REPORT_ROOT}/03_g01_retrain/report.html,WEIGHTED=1" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/shear_bias_report.slurm")
echo "REPORT_G01_JOB=${REPORT_G01_JOB}"

STATUS="${REPORT_ROOT}/STATUS.txt"
{
    echo "fiber-gauge campaign"
    echo "===================="
    echo
    echo "Submitted $(date -R)"
    echo "CACHE_TAG=${CACHE_TAG}"
    echo "G02_CASE=${G02_CASE}"
    echo "G002_CASE=${G002_CASE}"
    echo "G01_CASE=${G01_CASE}"
    echo
    echo "DRY_JOB=${DRY_JOB}"
    echo "APPLY_JOB=${APPLY_JOB}"
    echo "VERIFY_JOB=${VERIFY_JOB}"
    echo "CACHE_G02_JOB=${CACHE_G02_JOB}"
    echo "CACHE_G002_JOB=${CACHE_G002_JOB}"
    echo "REPORT_G02_JOB=${REPORT_G02_JOB}"
    echo "GATE_JOB=${GATE_JOB}  (blocks retrain unless theta_int / g02 m recover)"
    echo "CCL_JOB=${CCL_JOB} MODEL_NAME=${CCL_NAME}"
    echo "NPE_JOB=${NPE_JOB} MODEL_NAME=${NPE_NAME}"
    echo "CACHE_G01_JOB=${CACHE_G01_JOB}"
    echo "REPORT_G01_JOB=${REPORT_G01_JOB}"
    echo
    echo "00_regen_subtract  noiseless FITS subtract: ±0.1/xu3 are swap_minor; g02/g002 already match"
    echo "01_repair_audit    in-place minor-axis fiber swap on legacy catalogs"
    echo "02_g02_g002_recache  recache matched frozen g02/g002 NPEs on repaired xu3"
    echo "03_g01_retrain     CCL + frozen no-fusion NPE on repaired valid_100k, then three-way report"
    echo
    echo "Do not recache 1m / better-spec / production 45255702 on repaired xu3."
} > "${STATUS}"

echo "Wrote ${STATUS}"
