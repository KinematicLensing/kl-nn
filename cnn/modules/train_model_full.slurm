#!/bin/bash
#SBATCH --job-name=kl_nn
#SBATCH --output=/xdisk/timeifler/wxs0703/outputs/train_NN_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
### >>> High priority purchase-in time
###SBATCH --partition=high_priority
###SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:volta:2

#SBATCH --account=timeifler

#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wxs0703@arizona.edu

ulimit -c 0

module load anaconda
conda init bash
source ~/.bashrc

conda activate kl-nn

SCRIPT=train_model.py

python ${SCRIPT}
