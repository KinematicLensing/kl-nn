#!/bin/bash
#SBATCH --job-name=h_ens_ana
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_ens_analysis_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=7800M
# NPE analysis of one ensemble realization (TAG) with the map settings tuned on s3_sub1; no halo fit.
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${TAG:?}"
/jet/home/xwang30/.conda/envs/kl-nn/bin/python /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/h_cluster_analysis.py \
    --tag "${TAG}" --source npe --no-fit --tuned --tuning-tag s3_sub1
