#!/bin/bash

# Completeness/integrity check for simulator-v3 FITS generation.
# Override SAMPLE_NAME, DATASET_NAME, SAMPLE_PATH, FITS_ROOT, CHUNK_SIZE,
# TOTAL, REPORT_PATH, or VERIFY_FITS in the environment when needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_NAME="${SAMPLE_NAME:-train_1m_simv3}"
DATASET_NAME="${DATASET_NAME:-${SAMPLE_NAME}}"
SAMPLE_PATH="${SAMPLE_PATH:-/ocean/projects/phy250048p/shared/samples/${SAMPLE_NAME}.csv}"
FITS_ROOT="${FITS_ROOT:-/ocean/projects/phy250048p/shared/fits}"
FITS_PATH="${FITS_PATH:-${FITS_ROOT}/${DATASET_NAME}}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
REPORT_PATH="${REPORT_PATH:-${PWD}/checksum_${DATASET_NAME}.tsv}"
VERIFY_FITS="${VERIFY_FITS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=(
    --sample "${SAMPLE_PATH}"
    --fits-dir "${FITS_PATH}"
    --chunk-size "${CHUNK_SIZE}"
    --report "${REPORT_PATH}"
)
if [[ -n "${TOTAL:-}" ]]; then
    ARGS+=(--total "${TOTAL}")
fi
case "${VERIFY_FITS}" in
    0) ;;
    1) ARGS+=(--verify-fits) ;;
    *) echo "VERIFY_FITS must be 0 or 1" >&2; exit 2 ;;
esac

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/check_fits_generation.py" "${ARGS[@]}" "$@"
