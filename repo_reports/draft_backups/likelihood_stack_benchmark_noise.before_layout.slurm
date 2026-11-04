#!/bin/bash
#SBATCH --job-name=kl_bench_noise
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_bench_noise_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
module load anaconda3
set +u
source ~/.bashrc
conda activate kl-nn
set -u

DATASET_NAME="${DATASET_NAME:-likelihood_stack_04_benchmark}"
SEED="${BENCHMARK_SEED:-420042}"
cd "${KLNN_REPO_ROOT}"
python data_generate/materialize_likelihood_stack_benchmark.py \
  --data-dir="/ocean/projects/phy250048p/shared/datasets/${DATASET_NAME}" \
  --seed="${SEED}" --overwrite
