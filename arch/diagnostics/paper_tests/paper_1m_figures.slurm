#!/bin/bash
#SBATCH --job-name=paper_1m_figs
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_1m_figures_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=30000M
# Final step for the 1M measured-axis paper model: T8 stacking report, then the
# Sec. 5 and Sec. 6 paper figures/numbers, all under $S/reports/paper-tests-1m/.
set -euo pipefail
ulimit -c 0
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kl-nn
PY=/jet/home/xwang30/.conda/envs/kl-nn/bin/python
cd /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
S=/ocean/projects/phy250048p/shared
R=$S/reports/paper-tests-1m
M=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis
C=$S/cache/paper_tests_C_stacking/$M
mkdir -p $R/C_stacking $R/paper_figs $R/paper_figs_final
$PY -u c_stack_report.py --analysis $C/analysis_v1.npz \
    --samples $C/likelihood_stack_04_benchmark_measaxis_r90_8192 \
    --report-dir $R/C_stacking --figs-dir $R/paper_figs \
    --model $M --dataset-name likelihood_stack_04_benchmark_measaxis
export KLNN_PAPER_TESTS_ROOT=$R KLNN_PAPER_MODEL=$M KLNN_R2B_DIR=$S/cache/paper_tests_B_inference/r2b_1m_measaxis
$PY -u make_paper_figures_sec5.py
$PY -u make_paper_figures_sec6.py
