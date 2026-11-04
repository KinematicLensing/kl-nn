#!/bin/bash
# Remove the partial LMDB outputs left by the 2026-10-03 bus-error crashes
# (26 failed shards plus 2 crashed merge targets). All are rebuilt from FITS.
set -u
cd /ocean/projects/phy250048p/shared/datasets || exit 1

targets=(
    small_10k_simv3_cosi_measaxis
    paper_tests_B_shear_response_xu3like_5k_measaxis
)
for i in 28 29 43 46 47; do
    targets+=("valid_100k_simv3_cosi_measaxis_shard_${i}_of_100")
done
for i in 0 7 20 22 39 42 43 44 46 48 49 53 69 70 73 83 85 86 89 92 94; do
    targets+=("test_100k_simv3_cosi_xu3_tf_measaxis_shard_${i}_of_100")
done

for d in "${targets[@]}"; do
    rm -rf -- "$d"
done

left=0
for d in "${targets[@]}"; do
    [[ -e "$d" ]] && { echo "still present: $d"; left=$((left + 1)); }
done
echo "removed $(( ${#targets[@]} - left )) of ${#targets[@]}"
