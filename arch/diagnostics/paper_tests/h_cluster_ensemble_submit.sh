#!/bin/bash
# Submit generation -> LMDB -> NPE posteriors -> analysis for every tag in ensemble_seeds.txt.
# Main and sub of one realization share the inference noise seed (SEED = 42 + 1000 r).
set -euo pipefail
cd /jet/home/xwang30/kl-nn
M=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis
E=/ocean/projects/phy250048p/shared/reports/paper-tests-1m/H_cluster/ensemble
: > "$E/ensemble_jobs.txt"
while read -r tag seed; do
    r=$((10#${tag##*_n}))
    n=$(( $(wc -l < /ocean/projects/phy250048p/shared/samples/paper_tests_H_cluster_${tag}_main.csv) - 1 ))
    caches=""
    for sc in main sub; do
        NAME=paper_tests_H_cluster_${tag}_${sc}
        G=$(SAMPLE_NAME=$NAME DATASET_NAME=$NAME TOTAL=$n CHUNK_SIZE=$n sbatch --parsable --export=ALL --array=1 \
            --time=0:45:00 --job-name=he_fits --mail-type=NONE data_generate/generate_simulator_v3.slurm)
        D=$(NAME=$NAME N=$n sbatch --parsable --export=ALL --dependency=afterok:$G arch/diagnostics/paper_tests/h_cluster_database.slurm)
        C=$(MODEL_NAME=$M DATASET=$NAME NPARTS=1 NGALS=$n SEED=$((42 + 1000 * r)) CACHE_TAG=r0 sbatch --parsable --export=ALL \
            --array=1 --time=0:30:00 --job-name=he_cache --mail-type=NONE --dependency=afterok:$D arch/cache_posteriors.slurm)
        caches="$caches:$C"
        echo "$tag $sc n=$n fits=$G db=$D cache=$C" >> "$E/ensemble_jobs.txt"
    done
    A=$(TAG=$tag sbatch --parsable --export=ALL --dependency=afterok$caches arch/diagnostics/paper_tests/h_cluster_ensemble_analysis.slurm)
    echo "$tag analysis=$A" >> "$E/ensemble_jobs.txt"
done < "$E/ensemble_seeds.txt"
echo "submitted; see $E/ensemble_jobs.txt"
