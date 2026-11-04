#!/bin/bash
#SBATCH --job-name=h_cluster_db
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_h_cluster_database_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=4000M
# LMDB for one cluster-test scenario: NAME (sample table and dataset name), N galaxies in one part.
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
: "${NAME:?}" "${N:?}"
cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py -s="${NAME}" -d="${NAME}" -N=1 -n="${N}" --nspec=5 --num_shards=1
