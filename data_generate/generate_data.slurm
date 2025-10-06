#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_data_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
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

SCRIPT=../scripts/generate_fits_wrapper.py
SAMPFILE=samples_small.csv
DATASET=small
NSAMPS=100
hit=0

mkdir /ocean/projects/phy250048p/shared/fits/${DATASET}

for (( c=0; c<101; c++ ))
do
  hit=$((${hit}+1))
  if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
  then
    i=$(((${SLURM_ARRAY_TASK_ID}-1)*${NSAMPS}))
    j=$((${i}+${NSAMPS}))
    echo ${i} ${j}
    python ${SCRIPT} -i=${i} -j=${j} -n=${SLURM_ARRAY_TASK_ID} -s=${SAMPFILE} -d=${DATASET}
  fi
done
