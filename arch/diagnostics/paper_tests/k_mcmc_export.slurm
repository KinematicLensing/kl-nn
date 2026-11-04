#!/bin/bash
#SBATCH --job-name=k_mcmc_export
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_k_mcmc_export_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --exclude=v005
# Noisy data vectors (exact 1M Xu3-cache noise) for the MCMC comparison galaxies.
set -euo pipefail
ulimit -c 0
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/arch
python diagnostics/paper_tests/k_mcmc_export.py --ids "${IDS:-6447,1682,96201,40690,62769,13804,26386}"
