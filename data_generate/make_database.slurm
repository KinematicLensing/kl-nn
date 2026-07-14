#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_database_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=make_database.py
SAMPLE=pretrain_5m
DATASET=pretrain_5m
NSAMPS=5000
NPARTS=1000
NSPEC=5
NSHARDS=100

# Capture the individual array worker index
SHARD_IDX=$((${SLURM_ARRAY_TASK_ID}-1))

python ${SCRIPT} \
    -s=${SAMPLE} \
    -d=${DATASET} \
    -N=${NPARTS} \
    -n=${NSAMPS} \
    --nspec=${NSPEC} \
    --shard_idx=${SHARD_IDX} \
    --num_shards=${NSHARDS}