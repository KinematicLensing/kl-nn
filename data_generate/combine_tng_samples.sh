#!/usr/bin/env bash

set -euo pipefail

input_dir="${1:-/ocean/projects/phy250048p/shared/samples}"
output_file="${2:-${input_dir}/samples_tng_10k_all.csv}"

shopt -s nullglob
mapfile -t files < <(
    for file in "${input_dir}"/samples_tng_10k_*.csv; do
        [[ "${file}" == "${output_file}" ]] && continue
        printf '%s\n' "${file}"
    done | sort -V
)

if ((${#files[@]} == 0)); then
    echo "No CSV files found in ${input_dir}" >&2
    exit 1
fi

first_file=1
offset=0

{
    for file in "${files[@]}"; do
        if (( first_file )); then
            head -n 1 "${file}"
            first_file=0
        fi

        awk -v offset="${offset}" '
            BEGIN { FS = OFS = "," }
            NR > 1 {
                $1 += offset
                print
            }
        ' "${file}"

        offset=$((offset + 10000))
    done
} > "${output_file}"

echo "Wrote ${output_file}"