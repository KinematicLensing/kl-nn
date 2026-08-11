#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_factorial_candidates_%A_%a.out
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

# Submit once for each controlled model/evaluation-population pair. Always use
# the same cached SNR directory across model pairs.
: "${MODEL:?Set MODEL}"
: "${EPOCH:?Set EPOCH}"
: "${DATASET:?Set DATASET}"
: "${SAMPFILE:?Set SAMPFILE}"
: "${SNR_DIR:?Set SNR_DIR to a directory containing partXof10.npy}"

NGALS=${NGALS:-10000}
NSAMPS=${NSAMPS:-4096}
NPARTS=${NPARTS:-10}
TAG=${TAG:-tf_factorial_raw}
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

python '[scr]_tf_analysis.py' \
  -i="${INDEX}" --nparts="${NPARTS}" --ngals="${NGALS}" \
  --nsamples="${NSAMPS}" --stem="${MODEL}" --epoch="${EPOCH}" \
  --dataset="${DATASET}" --sample-set="${SAMPFILE}" \
  --mode=1 --cache-tag="${TAG}" --cached-snrs-path="${SNR_DIR}"
