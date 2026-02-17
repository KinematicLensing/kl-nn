#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/train_model_CNN-CNN-flow_low_hlr.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
module load cuda
conda init bash
source ~/.bashrc
conda activate kl-nn

SCRIPT=\[scr\]_train_model.py

cp config.py /ocean/projects/phy250048p/shared/configs/cfg_train_model_CNN-CNN-flow_low_hlr.py

python ${SCRIPT}
