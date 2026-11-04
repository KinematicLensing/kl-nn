#!/bin/bash
#SBATCH --job-name=b_xu3like_db
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_xu3like_database_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=4000M
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py -s=../paper_tests_B_inference/samples/samples_paper_tests_B_shear_response_xu3like_5k \
    -d=paper_tests_B_shear_response_xu3like_5k -N=100 -n=50 --nspec=5 --num_shards=1
