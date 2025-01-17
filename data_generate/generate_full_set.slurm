#!/bin/bash
#SBATCH --job-name=kl_training_data
#SBATCH --output=/xdisk/timeifler/wxs0703/outputs/gen_train_data_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-50
#SBATCH --cpus-per-task=1
### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wxs0703@arizona.edu

ulimit -c 0

module load gsl/2.6
module load anaconda
conda init bash
source ~/.bashrc

conda activate kltools

SAMPDIR=/xdisk/timeifler/wxs0703/desi_nn_mcmc/samples/samples.csv
SCRIPT=generate_training_wrapper.py
NCPUS=${SLURM_NTASKS}
NSAMPS=200
hit=0

for (( c=0; c<50; c++ ))
do
  hit=$((${hit}+1))
  if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
  then
    i=$(((${SLURM_ARRAY_TASK_ID}-1)*${NSAMPS}))
    j=$((${i}+${NSAMPS}))
    echo ${i} ${j}
    python ${SCRIPT} -i=${i} -j=${j}
  fi
done
