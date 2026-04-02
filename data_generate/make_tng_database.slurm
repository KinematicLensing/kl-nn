#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/make_tng_database.out
#SBATCH --partition=RM-shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=make_tng_database.py
SAMPLE=tng_10k
DATASET=test_tng_10k
NSPEC=5
NGAL=50
ROWSPERGAL=10000
IDSTRIDE=10000

python ${SCRIPT} -s=${SAMPLE} -d=${DATASET} --nspec=${NSPEC} --ngal=${NGAL} --rows-per-gal=${ROWSPERGAL} --id-stride=${IDSTRIDE}
