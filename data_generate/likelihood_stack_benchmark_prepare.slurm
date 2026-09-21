#!/bin/bash
#SBATCH --job-name=kl_bench_prepare
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_bench_prepare_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
module load anaconda3
set +u
source ~/.bashrc
conda activate kl-nn
set -u

cd "${KLNN_REPO_ROOT}"
python data_generate/prepare_likelihood_stack_benchmark.py --overwrite
