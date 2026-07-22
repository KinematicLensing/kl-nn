#!/bin/bash

set -u

DATASET=${DATASET:-pretrain_5m}
FITS_ROOT=${FITS_ROOT:-/ocean/projects/phy250048p/shared/fits/${DATASET}}
NGAL=${NGAL:-1000}
ROWSPERGAL=${ROWSPERGAL:-5000}
IDSTRIDE=${IDSTRIDE:-5000}
OUTFILE=${1:-missing_fits_${DATASET}.txt}
PREVFILE=${2:-missing_fits_${DATASET}.txt}

missing=0
found=0

# Check if a previous results file exists and extract galaxies to scan
declare -a GALAXIES_TO_SCAN
if [ -f "${PREVFILE}" ]; then
  echo "[INFO] Found previous results file: ${PREVFILE}"
  echo "[INFO] Extracting galaxies from previous scan..."
  while IFS= read -r line; do
    if [[ $line =~ \[INFO\]\ Checking\ galaxy\ folder\ ([0-9]+) ]]; then
      GALAXIES_TO_SCAN+=("${BASH_REMATCH[1]}")
    fi
  done < "${PREVFILE}"
  
  if [ ${#GALAXIES_TO_SCAN[@]} -eq 0 ]; then
    echo "[WARN] No galaxies found in ${PREVFILE}; falling back to full scan"
    for ((gal=0; gal<NGAL; gal++)); do
      GALAXIES_TO_SCAN+=("$gal")
    done
  else
    echo "[INFO] Found ${#GALAXIES_TO_SCAN[@]} galaxies to re-scan from previous results"
  fi
else
  echo "[INFO] No previous results file found; performing full scan"
  for ((gal=0; gal<NGAL; gal++)); do
    GALAXIES_TO_SCAN+=("$gal")
  done
fi

: > "${OUTFILE}"

echo "[INFO] Checking FITS under ${FITS_ROOT}" | tee -a "${OUTFILE}"
echo "[INFO] Expected galaxies: ${NGAL}, rows per galaxy: ${ROWSPERGAL}, id stride: ${IDSTRIDE}" | tee -a "${OUTFILE}"

for gal in "${GALAXIES_TO_SCAN[@]}"; do
  gal=$((${gal}+640))
  pid=$((${gal}+1))
  gal_dir="${FITS_ROOT}/part_${pid}"
  start_id=$((gal * IDSTRIDE))
  end_id=$((start_id + ROWSPERGAL - 1))
  echo "[INFO] Checking galaxy folder ${pid}" | tee -a "${OUTFILE}"

  if [ ! -d "${gal_dir}" ]; then
    echo "[MISSING] ${start_id}-${end_id}" | tee -a "${OUTFILE}"
    missing=$((missing + ROWSPERGAL))
    continue
  fi

  first_missing=""
  last_missing=""
  missing_in_gal=0

  for ((row=0; row<ROWSPERGAL; row++)); do
    global_id=$((gal * IDSTRIDE + row))
    fit_file="${gal_dir}/gal_${global_id}.fits"

    if [ -f "${fit_file}" ]; then
      found=$((found + 1))
    else
      missing=$((missing + 1))
      missing_in_gal=$((missing_in_gal + 1))
      if [ -z "${first_missing}" ]; then
        first_missing=${global_id}
      fi
      last_missing=${global_id}
    fi
  done

  if [ "${missing_in_gal}" -gt 0 ]; then
    echo "[MISSING] ${first_missing}-${last_missing}" | tee -a "${OUTFILE}"
  fi
done

echo "[INFO] Found: ${found}" | tee -a "${OUTFILE}"
echo "[INFO] Missing: ${missing}" | tee -a "${OUTFILE}"
echo "[INFO] Missing list written to ${OUTFILE}"
