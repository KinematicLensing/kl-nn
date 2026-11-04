#!/bin/bash
# Second cleanup after the 2026-10-03 bus errors: 8 more crashed
# valid_100k_simv3_cosi_measaxis shards (tasks 4, 6, 9, 10, 23, 24, 27, 28)
# and the partial merge target (3000 of 100000 records). All rebuilt from FITS.
set -u
cd /ocean/projects/phy250048p/shared/datasets || exit 1

targets=(valid_100k_simv3_cosi_measaxis)
for i in 3 5 8 9 22 23 26 27; do
    targets+=("valid_100k_simv3_cosi_measaxis_shard_${i}_of_100")
done

for d in "${targets[@]}"; do
    rm -rf -- "$d"
done

left=0
for d in "${targets[@]}"; do
    [[ -e "$d" ]] && { echo "still present: $d"; left=$((left + 1)); }
done
echo "removed $(( ${#targets[@]} - left )) of ${#targets[@]}"
