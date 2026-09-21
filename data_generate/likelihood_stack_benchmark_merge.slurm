#!/bin/bash
#SBATCH --job-name=kl_bench_merge
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_bench_merge_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
module load anaconda3
set +u
source ~/.bashrc
conda activate kl-nn
set -u

SAMPLE_NAME="${SAMPLE_NAME:-likelihood_stack_04_benchmark}"
DATASET_NAME="${DATASET_NAME:-likelihood_stack_04_benchmark}"
N_CELLS="${N_CELLS:-25}"
CELL_SIZE="${CELL_SIZE:-32}"

cd "${KLNN_REPO_ROOT}/data_generate"
python make_database.py \
  -s="${SAMPLE_NAME}" -d="${DATASET_NAME}" \
  -N="${N_CELLS}" -n="${CELL_SIZE}" --nspec=5 \
  --num_shards="${N_CELLS}" --merge
