#!/bin/bash
#SBATCH --job-name=a_shear_supplement2
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/a_shear_supplement2_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000M

set -euo pipefail
R=/ocean/projects/phy250048p/shared/reports/paper-tests
/jet/home/xwang30/.conda/envs/kl-nn/bin/python -u /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/a_shear_supplement2.py \
  --table $R/A_shear/per_galaxy.npz --outdir $R/A_shear --n-boot 300
