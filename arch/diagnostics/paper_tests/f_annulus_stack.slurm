#!/bin/bash
#SBATCH --job-name=f_annulus
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/f_annulus_stack_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

# Usage: sbatch f_annulus_stack.slurm --out <dir> [--cache <xu3 cache>] [--synthetic] ...
set -eo pipefail
ulimit -c 0
module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u
cd /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
nvidia-smi --query-gpu=name,memory.total --format=csv
/jet/home/xwang30/.conda/envs/kl-nn/bin/python f_annulus_stack.py "$@"
