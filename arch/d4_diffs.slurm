#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/d4_diffs_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --array=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=d4_diffs.py
MODEL=CNN-CNN-flow
EPOCH=199
SAMPFILE=samples_valid_1m.csv
DATASET=valid_1m
NSAMPS=10000
NPARTS=10
MODE=1
USE_OPTIMIZATION=1

i=$((${SLURM_ARRAY_TASK_ID}-1))
if [[ "${USE_OPTIMIZATION}" == "1" ]]; then
  OPT_FLAG="--use-optimization"
else
  OPT_FLAG="--no-use-optimization"
fi

python ${SCRIPT} -i=${i} --nparts=${NPARTS} --nsamples=${NSAMPS} --stem=${MODEL} --epoch=${EPOCH} --dataset=${DATASET} --sample-set=${SAMPFILE} --mode=${MODE} ${OPT_FLAG}
