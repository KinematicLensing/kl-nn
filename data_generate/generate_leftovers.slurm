#!/bin/bash
#SBATCH --job-name=kl_training_data
#SBATCH --output=/xdisk/timeifler/wxs0703/outputs/full_set_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=191-195
#SBATCH --ntasks-per-node=1
### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wxs0703@arizona.edu

ulimit -c 0

module load anaconda
conda init bash
source ~/.bashrc

conda activate kl-nn

SCRIPT=generate_training_wrapper.py
NCPUS=${SLURM_NTASKS}
NSAMPS=4000
hit=190

for (( c=0; c<7; c++ ))
do
  hit=$((${hit}+1))
  if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
  then
    i=$(((${SLURM_ARRAY_TASK_ID}-1)*${NSAMPS}+3900))
    j=$((${i}+100))
    echo ${i} ${j}
    python ${SCRIPT} -i=${i} -j=${j} -n=${SLURM_ARRAY_TASK_ID}
  fi
done
