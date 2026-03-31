#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/sample_trained_model_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --array=0-9
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
SAMPLES_PER_GALAXY=${SAMPLES_PER_GALAXY:-10000}
N_GALAXIES=${N_GALAXIES:-100000}
CHECKPOINT_SUFFIX=${CHECKPOINT_SUFFIX:-latest}
MODE=${MODE:-2}

MODEL_ROOT=${MODEL_ROOT:-/ocean/projects/phy250048p/shared/models}
DATA_ROOT=${DATA_ROOT:-/ocean/projects/phy250048p/shared/datasets}
CACHE_DIR=${CACHE_DIR:-/ocean/projects/phy250048p/shared/cache}
OUTPUT_STEM=${OUTPUT_STEM:-${MODEL_NAME}_${DATASET_NAME}}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

if [ "${TASK_COUNT}" -gt 1 ] && [ "${N_GALAXIES}" -lt 0 ]; then
  echo "ERROR: N_GALAXIES must be >= 0 when using job arrays so split indices can be computed."
  exit 1
fi

if [ "${N_GALAXIES}" -lt 0 ]; then
  SPLIT_START=0
  SPLIT_END=-1
else
  CHUNK_SIZE=$(( (N_GALAXIES + TASK_COUNT - 1) / TASK_COUNT ))
  SPLIT_START=$(( TASK_ID * CHUNK_SIZE ))
  SPLIT_END=$(( SPLIT_START + CHUNK_SIZE ))
  if [ "${SPLIT_END}" -gt "${N_GALAXIES}" ]; then
    SPLIT_END=${N_GALAXIES}
  fi
fi

if [ "${SPLIT_END}" -ge 0 ] && [ "${SPLIT_START}" -ge "${SPLIT_END}" ]; then
  echo "Task ${TASK_ID}: empty shard [${SPLIT_START}, ${SPLIT_END}), skipping."
  exit 0
fi

OUTPUT_STEM=${OUTPUT_STEM}_part${TASK_ID}of${TASK_COUNT}

echo "Task ${TASK_ID}/${TASK_COUNT}: processing split [${SPLIT_START}, ${SPLIT_END})"

python ${SCRIPT} \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --samples-per-galaxy "${SAMPLES_PER_GALAXY}" \
  --n-galaxies "${N_GALAXIES}" \
  --split-start "${SPLIT_START}" \
  --split-end "${SPLIT_END}" \
  --checkpoint-suffix "${CHECKPOINT_SUFFIX}" \
  --mode "${MODE}" \
  --model-root "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --output-stem "${OUTPUT_STEM}" \
  --device cuda \
  --overwrite
