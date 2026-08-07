#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/merge_database_pretrain_5m.out
#SBATCH --partition=RM-shared
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

module load anaconda3
source ~/.bashrc
conda activate kl-nn

python make_database.py -s=pretrain_5m -d=pretrain_5m -N=1000 -n=5000 --nspec=5 --num_shards=100 --merge