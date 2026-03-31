#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/tf_analysis.out
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
MODEL=CNN-CNN-flow_1m_tf
SAMPFILE=samples_test_1m_low_hlr.csv
DATASET=test_1m_low_hlr
NGALS=10000
NSAMPS=10000
hit=0

for (( c=0; c<11; c++ ))
do
  hit=$((${hit}+1))
  if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
  then
    i=$((${SLURM_ARRAY_TASK_ID}-1))
    python ${SCRIPT} -i=${i} --ngals=${NGALS} --nsamples=${NSAMPS} --stem=${MODEL} --dataset=${DATASET} --sample-set=${SAMPFILE}
  fi
done
