#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_analysis_%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=\[scr\]_tf_analysis.py
MODEL=ViT-CNN-flow_pretrained
EPOCH=248
SAMPFILE=samples_valid_1m.csv
DATASET=valid_1m
NGALS=10000
NSAMPS=10000
NPARTS=10
# CACHED_SNRS_PATH=/ocean/projects/phy250048p/shared/cache/CNN-CNN-flow/small_1m/snr

i=$((${SLURM_ARRAY_TASK_ID}-1))
python ${SCRIPT} -i=${i} --nparts=${NPARTS} --ngals=${NGALS} --nsamples=${NSAMPS} --stem=${MODEL} --epoch=${EPOCH} --dataset=${DATASET} --sample-set=${SAMPFILE} --conform-to-tf
