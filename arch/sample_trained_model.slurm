#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/sample_trained_model.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
module load cuda
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT_DIR=/jet/home/xwang30/kl-nn/arch
SCRIPT=${SCRIPT_DIR}/sample_trained_model.py
cd ${SCRIPT_DIR}

MODEL_NAME=${MODEL_NAME:-CNN-CNN-flow_1m_tf}
DATASET_NAME=${DATASET_NAME:-test_1m_low_hlr}
SAMPLES_PER_GALAXY=${SAMPLES_PER_GALAXY:-1000}
N_GALAXIES=${N_GALAXIES:-10000}
CHECKPOINT_SUFFIX=${CHECKPOINT_SUFFIX:-latest}
MODE=${MODE:-2}

MODEL_ROOT=${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}
DATA_ROOT=${DATA_ROOT:-/ocean/projects/phy250048p/shared/datasets}
CACHE_DIR=${CACHE_DIR:-/ocean/projects/phy250048p/shared/cache}
OUTPUT_STEM=${OUTPUT_STEM:-${MODEL_NAME}_${DATASET_NAME}_slurm}

python ${SCRIPT} \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --samples-per-galaxy "${SAMPLES_PER_GALAXY}" \
  --n-galaxies "${N_GALAXIES}" \
  --checkpoint-suffix "${CHECKPOINT_SUFFIX}" \
  --mode "${MODE}" \
  --model-root "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --output-stem "${OUTPUT_STEM}" \
  --device cuda \
  --overwrite
