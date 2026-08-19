#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/merge_shear_response_db_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G

set -euo pipefail
module load anaconda3
source ~/.bashrc
conda activate kl-nn

SAMPLE=${SAMPLE:-shear_response_simv2_galaxyaxis_halpha_5k}
DATASET=${DATASET:-shear_response_simv2_galaxyaxis_halpha_5k}
NBASE=${NBASE:-1000}
NFOLDERS=100
NSHARDS=10
TOTAL=$((5 * NBASE))
CHUNK=$((TOTAL / NFOLDERS))

python make_database.py -s="${SAMPLE}" -d="${DATASET}" \
  -N="${NFOLDERS}" -n="${CHUNK}" --nspec=5 \
  --num_shards="${NSHARDS}" --merge
