#!/bin/bash
#SBATCH --job-name=h_cluster_val
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_validate_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=15600M
# Cell-stacking validation on synthetic noise realizations (Gaussian likelihoods with the real NPE widths).
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TAG="${TAG:-s3_sub1}"
/jet/home/xwang30/.conda/envs/kl-nn/bin/python /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/h_cluster_validate.py \
    --tag "${TAG}" --realizations 30
