#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_database_small.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=make_database.py
SAMPLE=small_1m
DATASET=small_1m
NSAMPS=400
NPARTS=25
NSPEC=5
hit=0

python ${SCRIPT} -s=${SAMPLE} -d=${DATASET} -N=${NPARTS} -n=${NSAMPS} --nspec=${NSPEC}
