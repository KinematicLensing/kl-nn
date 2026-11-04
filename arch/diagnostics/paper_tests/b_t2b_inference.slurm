#!/bin/bash
#SBATCH --job-name=b_t2b_stencil
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_t2b_inference_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --array=1-5
#SBATCH --mem=16G
#SBATCH --exclude=v005
# T2b: fibrepair NPE on the matched five-shear stencil (shared noise per group).
# Reads the paper_tests copy of the legacy stencil LMDB (verified bitwise
# identical to a current-simulator re-render, fiber gauge "match").
set -euo pipefail
ulimit -c 0
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn

S=/ocean/projects/phy250048p/shared
MODEL_NAME=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair
DATASET=paper_tests_B_shear_response_simv3_cosi_5k_fibrepair
NPARTS=5
NGALS=1000
PART_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
cd /jet/home/xwang30/kl-nn/arch
python cache_posteriors.py \
    --partition-index "${PART_INDEX}" --nparts "${NPARTS}" --ngals "${NGALS}" \
    --nsamples 10000 --model-name "${MODEL_NAME}" --dataset "${DATASET}" \
    --model-root "${S}/models" --data-root "${S}/datasets" \
    --cache-root "${S}/cache/paper_tests_B_inference" \
    --cache-tag stencil_10k_s42 --matched-group-size 5 --seed 42 --device cuda
