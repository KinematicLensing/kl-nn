#!/bin/bash
#SBATCH --job-name=b_r2b
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_r2b_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --exclude=v005
set -euo pipefail
ulimit -c 0
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/arch
# R2b galaxy (6447): two image S/N levels plus catalog, 5 noise seeds
python diagnostics/paper_tests/b_r2b_sample.py --galaxy 6447 --snrs 50,200,catalog --noise-seeds 0,1,2,3,4 --nsamples 10000
# alternate R2b galaxy (503), same design
python diagnostics/paper_tests/b_r2b_sample.py --galaxy 503 --snrs 50,200,catalog --noise-seeds 0,1,2,3,4 --nsamples 10000
# full 9D samples at catalog S/N for corner plots
for G in 54792 5327; do
    python diagnostics/paper_tests/b_r2b_sample.py --galaxy ${G} --snrs catalog --noise-seeds 0 --nsamples 10000
done
