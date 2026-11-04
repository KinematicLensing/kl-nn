#!/bin/bash
#SBATCH --job-name=b_xu3like_gen
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_b_xu3like_generate_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --array=1-100
# T2b second stencil: 1000 Xu3-like bases (from test_100k_simv3_xu3_tf, the
# non-paper Xu3 draw, H-alpha S/N <= 150) x 5 shears, current simulator.
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
SAMPFILE=../paper_tests_B_inference/samples/samples_paper_tests_B_shear_response_xu3like_5k.csv
DATASET=paper_tests_B_shear_response_xu3like_5k
CHUNK=50
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
START=$((INDEX * CHUNK)); STOP=$((START + CHUNK))
mkdir -p "/ocean/projects/phy250048p/shared/fits/${DATASET}"
cd /jet/home/xwang30/kl-nn/data_generate
python generate_fits_wrapper.py -i="${START}" -j="${STOP}" -n="${SLURM_ARRAY_TASK_ID}" -s="${SAMPFILE}" -d="${DATASET}"
