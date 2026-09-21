#!/bin/bash
# Submit the isolated fixed-shear data chain and benchmark inference.
# Every job ID is appended to the stage-04 STATUS file.

set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
STATUS="${REPORT_ROOT}/STATUS.txt"
mkdir -p "${REPORT_ROOT}"

PREP_JOB="$(sbatch --parsable \
    "${KLNN_REPO_ROOT}/data_generate/likelihood_stack_benchmark_prepare.slurm")"
FITS_JOB="$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
    "${KLNN_REPO_ROOT}/data_generate/likelihood_stack_benchmark_fits.slurm")"
DB_JOB="$(sbatch --parsable --dependency="afterok:${FITS_JOB}" \
    "${KLNN_REPO_ROOT}/data_generate/likelihood_stack_benchmark_db.slurm")"
MERGE_JOB="$(sbatch --parsable --dependency="afterok:${DB_JOB}" \
    "${KLNN_REPO_ROOT}/data_generate/likelihood_stack_benchmark_merge.slurm")"
NOISE_JOB="$(sbatch --parsable --dependency="afterok:${MERGE_JOB}" \
    "${KLNN_REPO_ROOT}/data_generate/likelihood_stack_benchmark_noise.slurm")"
BENCHMARK_JOB="$(sbatch --parsable --dependency="afterok:${NOISE_JOB}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_benchmark.slurm")"
INDEX_JOB="$(sbatch --parsable --dependency="afterok:${BENCHMARK_JOB}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_write_index.slurm")"

{
    echo
    echo "Fixed-shear benchmark submitted $(date -R)"
    echo "BENCHMARK_PREP_JOB=${PREP_JOB}"
    echo "BENCHMARK_FITS_JOB=${FITS_JOB}"
    echo "BENCHMARK_DB_JOB=${DB_JOB}"
    echo "BENCHMARK_MERGE_JOB=${MERGE_JOB}"
    echo "BENCHMARK_NOISE_JOB=${NOISE_JOB}"
    echo "BENCHMARK_EVAL_JOB=${BENCHMARK_JOB}"
    echo "BENCHMARK_INDEX_JOB=${INDEX_JOB}"
} >> "${STATUS}"

printf 'PREP=%s FITS=%s DB=%s MERGE=%s NOISE=%s EVAL=%s INDEX=%s\n' \
    "${PREP_JOB}" "${FITS_JOB}" "${DB_JOB}" "${MERGE_JOB}" \
    "${NOISE_JOB}" "${BENCHMARK_JOB}" "${INDEX_JOB}"
