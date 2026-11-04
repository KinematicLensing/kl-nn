#!/bin/bash
#SBATCH --job-name=b_r2b_1m
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_r2b_1m_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --exclude=v005
# R2b example posteriors for the 1M measured-axis paper model (Sec. 5 Fig. posterior_snr,
# appendix 9D corner). Same galaxy (6447) and rows as before, measured-axis data.
set -euo pipefail
ulimit -c 0
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/arch
S=/ocean/projects/phy250048p/shared
M=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis
COMMON="--model $M --dataset $S/datasets/test_100k_simv3_cosi_xu3_tf_measaxis --out $S/cache/paper_tests_B_inference/r2b_1m_measaxis --nsamples 10000"
python diagnostics/paper_tests/b_r2b_sample.py $COMMON --galaxy 6447 --snrs 50,200,catalog --noise-seeds 0,1,2,3,4
python diagnostics/paper_tests/b_r2b_sample.py $COMMON --galaxy 6447 --field central_halpha_snr --snrs 8,60 --noise-seeds 0,1,2,3,4
