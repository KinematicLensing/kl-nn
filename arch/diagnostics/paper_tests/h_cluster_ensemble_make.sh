#!/bin/bash
# Option-B ensemble for the cluster test: N_REAL independent realizations of the s3_sub1 setup
# (new Poisson galaxy count, positions and Xu3 source galaxies each time; subhalo at (1, 0) Mpc).
# Writes tags s3_sub1_nNN (NN = 01..N_REAL). A seed whose draw puts a shear outside the training box is
# skipped and the next seed is tried. The seed used for each tag is listed in ensemble_seeds.txt.
set -uo pipefail
N_REAL="${N_REAL:-50}"
PY=/jet/home/xwang30/.conda/envs/kl-nn/bin/python
D=/jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests
OUT=/ocean/projects/phy250048p/shared/reports/paper-tests-1m/H_cluster/ensemble
mkdir -p "$OUT"; : > "$OUT/ensemble_seeds.txt"
seed=20261101
for r in $(seq 1 "$N_REAL"); do
    tag=$(printf "s3_sub1_n%02d" "$r")
    until $PY "$D/h_cluster_make.py" --tag "$tag" --sub-x-mpc 1.0 --seed "$seed" 2>&1 | grep -v -i warn; [ "${PIPESTATUS[0]}" -eq 0 ]; do
        echo "  seed $seed rejected for $tag"; seed=$((seed + 1))
    done
    echo "$tag $seed" >> "$OUT/ensemble_seeds.txt"
    seed=$((seed + 1))
done
