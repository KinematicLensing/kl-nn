#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_shear_response_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-100

set -euo pipefail
ulimit -c 0
module load anaconda3
source ~/.bashrc
conda activate kl-nn

SAMPFILE=${SAMPFILE:-samples_shear_response_simv2_galaxyaxis_halpha_5k.csv}
DATASET=${DATASET:-shear_response_simv2_galaxyaxis_halpha_5k}
NBASE=${NBASE:-1000}
NTASKS=100
TOTAL=$((5 * NBASE))
if (( TOTAL % NTASKS != 0 )); then
  echo "5*NBASE must be divisible by ${NTASKS}; got ${TOTAL}" >&2
  exit 2
fi
CHUNK=$((TOTAL / NTASKS))
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
START=$((INDEX * CHUNK))
STOP=$((START + CHUNK))

mkdir -p "/ocean/projects/phy250048p/shared/fits/${DATASET}"
python generate_fits_wrapper.py -i="${START}" -j="${STOP}" \
  -n="${SLURM_ARRAY_TASK_ID}" -s="${SAMPFILE}" -d="${DATASET}" \
  --observation-model-version 2 --fiber-layout galaxy_axis
