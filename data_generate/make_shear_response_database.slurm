#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_shear_response_db_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-10

set -euo pipefail
ulimit -c 0
module load anaconda3
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn

SAMPLE="${SAMPLE:-shear_response_simv3_galaxyaxis_central_halpha_5k}"
DATASET="${DATASET:-shear_response_simv3_galaxyaxis_central_halpha_5k}"
NBASE="${NBASE:-1000}"
NFOLDERS=100
NSHARDS=10
TOTAL=$((5 * NBASE))
if (( TOTAL % NFOLDERS != 0 )); then
  echo "5*NBASE must be divisible by ${NFOLDERS}; got ${TOTAL}" >&2
  exit 2
fi
CHUNK=$((TOTAL / NFOLDERS))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

cd /jet/home/xwang30/kl-nn/data_generate
python make_database.py -s="${SAMPLE}" -d="${DATASET}" \
  -N="${NFOLDERS}" -n="${CHUNK}" --nspec=5 \
  --shard_idx="${SHARD_INDEX}" --num_shards="${NSHARDS}"
