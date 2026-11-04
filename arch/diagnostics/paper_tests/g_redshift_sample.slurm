#!/bin/bash
#SBATCH --job-name=g_redshift
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_g_redshift_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --exclude=v005
# Redshift scan: 1000 rows (100 bases x 10 z) x 10 paired noise seeds.
# MODEL defaults to the 1M paper model; set MODEL for a dry run.
set -euo pipefail
ulimit -c 0
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/arch
MODEL="${MODEL:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis}"
NOISE_SEEDS="${NOISE_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
python diagnostics/paper_tests/g_redshift_sample.py --model "${MODEL}" --noise-seeds "${NOISE_SEEDS}" --nsamples 10000
