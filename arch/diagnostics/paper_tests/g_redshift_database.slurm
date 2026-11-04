#!/bin/bash
#SBATCH --job-name=g_redshift_db
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_g_redshift_database_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=4000M
# LMDB for the redshift scan: 10 parts (one per redshift) x 100 galaxies.
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py -s=paper_tests_G_redshift_scan_xu3like_1k_measaxis \
    -d=paper_tests_G_redshift_scan_xu3like_1k_measaxis -N=10 -n=100 --nspec=5 --num_shards=1
