#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_factorial_offline_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-10

set -euo pipefail
module load anaconda3
source ~/.bashrc
conda activate kl-nn

: "${BANK_ROOT:?Set BANK_ROOT to model/dataset cache directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
NPARTS=${NPARTS:-10}
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

python diagnostics/tf_factorial_analyze.py \
  --bank-root="${BANK_ROOT}" --partition="${INDEX}" --nparts="${NPARTS}" \
  --output-root="${OUTPUT_ROOT}" --slope="${TF_SLOPE:--7.22}" \
  --intercept="${TF_INTERCEPT:-36.0}" --scatter-dex="${TF_SCATTER:-0.1}"
