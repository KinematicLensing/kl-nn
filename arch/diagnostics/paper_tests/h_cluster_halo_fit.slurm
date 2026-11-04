#!/bin/bash
#SBATCH --job-name=h_cluster_fit
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_halo_fit_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=4000M
# Parametric NFW fits (main / main+sub / main+sub_free) to the per-galaxy likelihood grids.
# TAG=s3_sub1 sbatch h_cluster_halo_fit.slurm, or as an array over the ensemble:
# sbatch --array=1-50 h_cluster_halo_fit.slurm  (task r -> s3_sub1_nRR)
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/jet/home/xwang30/.conda/envs/kl-nn/bin/python
D=/jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then TAG=$(printf "s3_sub1_n%02d" "${SLURM_ARRAY_TASK_ID}"); fi
TAG="${TAG:-s3_sub1}"
$PY $D/h_cluster_halo_fit.py --tag "${TAG}"
