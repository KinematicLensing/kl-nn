#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/shear_response_inference_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mem=8G

set -euo pipefail
ulimit -c 0
module load anaconda3
source ~/.bashrc
conda activate kl-nn

: "${MODEL:?Set MODEL}"
: "${EPOCH:?Set EPOCH}"
DATASET=${DATASET:-shear_response_simv2_galaxyaxis_halpha_5k}
SAMPFILE=${SAMPFILE:-samples_shear_response_simv2_galaxyaxis_halpha_5k.csv}
NBASE=${NBASE:-1000}
NSAMPS=${NSAMPS:-4096}
NPARTS=10
TOTAL=$((5 * NBASE))
if (( TOTAL % NPARTS != 0 )); then
  echo "5*NBASE must be divisible by ${NPARTS}" >&2
  exit 2
fi
NGALS=$((TOTAL / NPARTS))
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

python '[scr]_tf_analysis.py' -i="${INDEX}" --nparts="${NPARTS}" \
  --ngals="${NGALS}" --nsamples="${NSAMPS}" --stem="${MODEL}" \
  --epoch="${EPOCH}" --dataset="${DATASET}" --sample-set="${SAMPFILE}" \
  --mode="${MODE:-1}" --tf-inference="${TF_INFERENCE:-none}" \
  --cache-tag="${TAG:-response_simv2_halpha}" --matched-group-size=5
