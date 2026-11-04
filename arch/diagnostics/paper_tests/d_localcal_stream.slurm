#!/bin/bash
#SBATCH --job-name=d_localcal_stream
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/d_localcal_stream_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000M

set -euo pipefail
/jet/home/xwang30/.conda/envs/kl-nn/bin/python /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests/d_localcal_stream.py
