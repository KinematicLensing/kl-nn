#!/bin/bash
#SBATCH --job-name=kl_bench_eval
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_bench_eval_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-32:1

set -euo pipefail
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
module load anaconda3
module load cuda
set +u
source ~/.bashrc
conda activate kl-nn
set -u

DATASET_NAME="${DATASET_NAME:-likelihood_stack_04_benchmark}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/likelihood-stack/04_benchmark}"
NOISE_DATA="${NOISE_DATA:-/ocean/projects/phy250048p/shared/datasets/${DATASET_NAME}/benchmark_noisy.npz}"
SEED="${BENCHMARK_SEED:-420042}"

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_benchmark.py \
  --data-dir="/ocean/projects/phy250048p/shared/datasets/${DATASET_NAME}" \
  --noise-data="${NOISE_DATA}" \
  --report-dir="${REPORT_DIR}" \
  --device=cuda --seed="${SEED}" --overwrite
