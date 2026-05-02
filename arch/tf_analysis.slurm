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
MODEL=CNN-CNN-flow_all_params_unif_snr
EPOCH=180
SAMPFILE=samples_test_1m_low_hlr.csv
DATASET=test_1m_low_hlr
NGALS=10000
NSAMPS=10000
NPARTS=10

i=$((${SLURM_ARRAY_TASK_ID}-1))
python ${SCRIPT} -i=${i} --nparts=${NPARTS} --ngals=${NGALS} --nsamples=${NSAMPS} --stem=${MODEL} --epoch=${EPOCH} --dataset=${DATASET} --sample-set=${SAMPFILE}
