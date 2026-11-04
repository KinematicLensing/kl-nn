#!/bin/bash
#SBATCH --job-name=h_cluster_tune
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_tune_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=7800M
#SBATCH --array=0-19
# kappa-map tuning on synthetic data: one noise realization per array task (aggregate afterwards with --aggregate).
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TAG="${TAG:-s3_sub1}"
/jet/home/xwang30/.conda/envs/kl-nn/bin/python /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/h_cluster_tune.py \
    --tag "${TAG}" --only "${SLURM_ARRAY_TASK_ID}"
