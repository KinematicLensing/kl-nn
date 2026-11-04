#!/bin/bash
#SBATCH --job-name=b_t2b_xu3ma
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_t2b_xu3ma_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=15000M
#SBATCH --time=2:00:00
#SBATCH --array=1-100
# T2b Xu3-like stencil: measured-axis NPE on the measured-axis stencil, CPU, shared noise within each 5-shear group.
# five-shear stencil, shared noise within each 5-shear group, 100 x 50 rows.
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

S=/ocean/projects/phy250048p/shared
MODEL_NAME="${MODEL_NAME:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_measaxis}"
DATASET=paper_tests_B_shear_response_xu3like_5k_measaxis
PART_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
cd /jet/home/xwang30/kl-nn/arch
python cache_posteriors.py \
    --partition-index "${PART_INDEX}" --nparts 100 --ngals 50 \
    --nsamples 10000 --model-name "${MODEL_NAME}" --dataset "${DATASET}" \
    --model-root "${S}/models" --data-root "${S}/datasets" \
    --cache-root "${S}/cache/paper_tests_B_inference" \
    --cache-tag stencil_10k_s42_cpu --matched-group-size 5 --seed 42 --device cpu
