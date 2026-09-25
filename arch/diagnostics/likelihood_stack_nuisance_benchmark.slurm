#!/bin/bash
#SBATCH --job-name=kl_nuisance_bench
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/kl_nuisance_bench_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-32:1

set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/jet/home/xwang30/kl-nn}}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/likelihood_stack_04_benchmark}"
REPORT_DIR="${REPORT_DIR:-/ocean/projects/phy250048p/shared/reports/likelihood-stack/04_benchmark_nuisance}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_2D_NAME="${NRE_2D_NAME:-CNN-CNN-Meta-nre2d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_9D_NAME="${NRE_9D_NAME:-CNN-CNN-Meta-nre9d-factorized-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
SEED="${BENCHMARK_SEED:-420042}"
NPE_SAMPLES="${NPE_SAMPLES:-1024}"
GRID_N="${GRID_N:-21}"
OVERWRITE="${OVERWRITE:-0}"

set +u
module load anaconda3
module load cuda
source ~/.bashrc
conda activate kl-nn
set -u

ARGS=(
  --data-dir "${DATA_DIR}"
  --parent-npe "${PARENT_NPE}"
  --nre-2d-name "${NRE_2D_NAME}"
  --nre-9d-name "${NRE_9D_NAME}"
  --report-dir "${REPORT_DIR}"
  --device cuda
  --seed "${SEED}"
  --npe-samples "${NPE_SAMPLES}"
  --grid-n "${GRID_N}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "${KLNN_REPO_ROOT}/arch"
python diagnostics/likelihood_stack_nuisance_benchmark.py "${ARGS[@]}"
