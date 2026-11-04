#!/bin/bash
#SBATCH --job-name=pt_C_stack
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_C_stack_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

set -eo pipefail
ulimit -c 0
module load anaconda3
module load cuda
KLNN_CONDA_BASE="$(conda info --base)"
source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate kl-nn
set -u
cd /jet/home/xwang30/kl-nn/arch
nvidia-smi --query-gpu=name,memory.total --format=csv
/jet/home/xwang30/.conda/envs/kl-nn/bin/python diagnostics/paper_tests/c_stack_analysis.py --device cuda "$@"
