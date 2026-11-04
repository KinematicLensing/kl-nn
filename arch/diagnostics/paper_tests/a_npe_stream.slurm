#!/bin/bash
#SBATCH --job-name=klnn_a_npe
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_a_npe_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G

set -eo pipefail
ulimit -c 0
cd /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
/jet/home/xwang30/.conda/envs/kl-nn/bin/python a_npe_stream.py
