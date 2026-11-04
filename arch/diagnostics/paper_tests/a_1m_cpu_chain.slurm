#!/bin/bash
#SBATCH --job-name=a_1m_chain
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/paper_tests_a_1m_chain_%j.out
#SBATCH --partition=RM-shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000M
# Sec. 5/6 Xu3 analyses for the 1M measured-axis paper model. Writes only under
# $S/reports/paper-tests-1m/ (the old paper-tests/ outputs are left untouched).
set -euo pipefail
ulimit -c 0
PY=/jet/home/xwang30/.conda/envs/kl-nn/bin/python
cd /jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
S=/ocean/projects/phy250048p/shared
R=$S/reports/paper-tests-1m
M=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis
T=test_100k_simv3_cosi_xu3_tf_measaxis_testset_tfweighted_v2_10k_s42_righthanded
mkdir -p $R/A_shear $R/A_npe $R/paper_figs
# Sec. 5: NPE validation (loss, P-P, coverage, TF ESS)
$PY -u a_npe_stream.py --case "$M:$T" --output $R/A_npe/a_npe_stream.npz
$PY -u a_npe_report.py --case "$M:$T" --out-dir $R/A_npe --figs-dir $R/paper_figs
# Sec. 6: per-galaxy table, then raw bias / shrinkage / weights / cuts
$PY -u a_shear_extract.py --model $M --dataset $T --out $R/A_shear/per_galaxy.npz
$PY -u a_shear_analysis.py --table $R/A_shear/per_galaxy.npz --outdir $R/A_shear --paperdir $R/paper_figs --n-boot 1000
