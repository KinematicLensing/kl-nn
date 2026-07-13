#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_model_ViT-CNN-flow_pretrained.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
module load cuda
conda init bash
source ~/.bashrc
conda activate kl-nn

export TORCH_NCCL_TRACE_BUFFER_SIZE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

SCRIPT=\[scr\]_train_model.py

python ${SCRIPT} \
    --model KLNPE \
    --trainer NPETrainer \
    --train_type train \