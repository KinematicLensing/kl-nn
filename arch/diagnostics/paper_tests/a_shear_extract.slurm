#!/bin/bash
#SBATCH --job-name=a_shear_extract
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/a_shear_extract_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000M

set -euo pipefail
OUT=/ocean/projects/phy250048p/shared/reports/paper-tests/A_shear/per_galaxy.npz
/jet/home/xwang30/.conda/envs/kl-nn/bin/python /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/a_shear_extract.py --out "$OUT"
