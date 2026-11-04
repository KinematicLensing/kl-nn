#!/bin/bash
#SBATCH --job-name=h_cluster_ana
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_analysis_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=15600M
# Aggregate the tuning parts, then run the cell + kappa-map analysis with the tuned settings
# (no halo fit: residual = reconstruction - true main-halo kappa) on the NPE and synthetic data.
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TAG="${TAG:-s3_sub1}"
PY=/jet/home/xwang30/.conda/envs/kl-nn/bin/python
D=/jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
$PY $D/h_cluster_tune.py --tag "${TAG}" --aggregate
for SRC in npe synthetic; do
    $PY $D/h_cluster_analysis.py --tag "${TAG}" --source "${SRC}" --no-fit --tuned
done
