#!/bin/bash
#SBATCH --job-name=kl_bench_fits
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_bench_fits_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-25

set -euo pipefail
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
module load anaconda3
set +u
source ~/.bashrc
conda activate kl-nn
set -u

SAMPLE_NAME="${SAMPLE_NAME:-likelihood_stack_04_benchmark}"
DATASET_NAME="${DATASET_NAME:-likelihood_stack_04_benchmark}"
SAMPLE_FILE="${SAMPLE_FILE:-${SAMPLE_NAME}.csv}"
CELL_SIZE="${CELL_SIZE:-32}"
PART="${SLURM_ARRAY_TASK_ID}"
START="$(( (PART - 1) * CELL_SIZE ))"
STOP="$(( PART * CELL_SIZE ))"

cd "${KLNN_REPO_ROOT}"
python data_generate/generate_fits_wrapper.py \
  -i="${START}" -j="${STOP}" -n="${PART}" \
  -s="${SAMPLE_FILE}" -d="${DATASET_NAME}" --skip-existing
