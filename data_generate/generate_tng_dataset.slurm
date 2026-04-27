#!/bin/bash
#SBATCH --output=/ocean/projects/phy250048p/shared/logs/generate_tng_dataset_%A_%a.out
#SBATCH --partition=RM-shared
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=401-480
#SBATCH --mail-type=ALL
#SBATCH --mail-user=easonw2@andrew.cmu.edu

ulimit -c 0

module load anaconda3
conda init bash
source ~/.bashrc
conda activate klpipe

set -u

SCRIPT=/jet/home/xwang30/kl-nn/data_generate/generate_tng_sample.py
SAMPROOT=/ocean/projects/phy250048p/shared/samples
OUTROOT=/ocean/projects/phy250048p/shared/fits/test_tng_10k
CHUNK_SIZE=500

SAMPLE_NAME=${SAMPLE_NAME:-test_tng_10k}
DATASET_NAME=${DATASET_NAME:-test_tng_10k}
NGAL=${NGAL:-60}
ROWS_PER_GAL=${ROWS_PER_GAL:-10000}
SECTIONS_PER_GAL=${SECTIONS_PER_GAL:-8}
ID_STRIDE=${ID_STRIDE:-10000}

# Test-mode controls (override at submit time, e.g. --export=ALL,TEST_MODE=1,TEST_MAX_ROWS=20)
TEST_MODE=${TEST_MODE:-0}
TEST_MAX_ROWS=${TEST_MAX_ROWS:-20}
TEST_CHUNK_SIZE=${TEST_CHUNK_SIZE:-50}

if [ ${SECTIONS_PER_GAL} -le 0 ]; then
  echo "[ERROR] SECTIONS_PER_GAL must be >= 1, got ${SECTIONS_PER_GAL}"
  exit 1
fi

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
GAL_IDX=$(( TASK_INDEX / SECTIONS_PER_GAL ))
SECTION_IDX=$(( TASK_INDEX % SECTIONS_PER_GAL ))

if [ ${GAL_IDX} -ge ${NGAL} ]; then
  echo "[INFO] Task ${SLURM_ARRAY_TASK_ID} maps to galaxy ${GAL_IDX}, beyond NGAL=${NGAL}; skipping"
  exit 0
fi

BASE_ROWS=$((ROWS_PER_GAL / SECTIONS_PER_GAL))
EXTRA_ROWS=$((ROWS_PER_GAL % SECTIONS_PER_GAL))
if [ ${SECTION_IDX} -lt ${EXTRA_ROWS} ]; then
  SECTION_ROWS=$((BASE_ROWS + 1))
  START_REALIZ=$((SECTION_IDX * SECTION_ROWS))
else
  SECTION_ROWS=${BASE_ROWS}
  START_REALIZ=$((EXTRA_ROWS * (BASE_ROWS + 1) + (SECTION_IDX - EXTRA_ROWS) * BASE_ROWS))
fi
END_REALIZ=$((START_REALIZ + SECTION_ROWS))

if [ ${SECTION_ROWS} -le 0 ]; then
  echo "[INFO] Galaxy ${GAL_IDX} section ${SECTION_IDX}: empty section; skipping"
  exit 0
fi

OUTROOT=/ocean/projects/phy250048p/shared/fits/${DATASET_NAME}
SAMPFILE=${SAMPROOT}/samples_${SAMPLE_NAME}_${GAL_IDX}.csv
GAL_DIR=${OUTROOT}/galaxy_${GAL_IDX}

mkdir -p ${OUTROOT}
mkdir -p ${GAL_DIR}

if [ ! -f "${SAMPFILE}" ]; then
  echo "[ERROR] Sample file not found: ${SAMPFILE}"
  exit 1
fi

# For this section, we process SECTION_ROWS rows (or fewer in TEST_MODE)
TOTAL_ROWS=${SECTION_ROWS}
if [ "${TEST_MODE}" -eq 1 ]; then
  if [ ${TEST_MAX_ROWS} -lt ${TOTAL_ROWS} ]; then
    TOTAL_ROWS=${TEST_MAX_ROWS}
  fi
  CHUNK_SIZE=${TEST_CHUNK_SIZE}
  echo "[INFO] TEST_MODE enabled: processing first ${TOTAL_ROWS} rows with chunk size ${CHUNK_SIZE}"
fi

SECTION_END=$((START_REALIZ + TOTAL_ROWS))

NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
SUCCESS=0
FAIL=0

echo "[INFO] Start galaxy ${GAL_IDX}, section ${SECTION_IDX}/${SECTIONS_PER_GAL} (realizations ${START_REALIZ}-$((END_REALIZ - 1)))"
echo "[INFO] Input sample file: ${SAMPFILE}"
echo "[INFO] Output directory: ${GAL_DIR}"
echo "[INFO] Total rows for this section: ${TOTAL_ROWS}; chunk size: ${CHUNK_SIZE}; chunks: ${NUM_CHUNKS}"

for (( chunk=0; chunk<NUM_CHUNKS; chunk++ ))
do
  START=$((START_REALIZ + chunk * CHUNK_SIZE))
  END=$((START + CHUNK_SIZE))
  if [ ${END} -gt ${SECTION_END} ]; then
    END=${SECTION_END}
  fi

  echo "[INFO] Galaxy ${GAL_IDX} section ${SECTION_IDX}: processing chunk $((chunk + 1))/${NUM_CHUNKS}, rows ${START}..$((END - 1))"

  while IFS=, read -r row_id g1 g2 theta_int incl v0 vcirc rscale
  do
    if [ -z "${row_id}" ]; then
      continue
    fi

    GLOBAL_ID=$((GAL_IDX * ID_STRIDE + row_id))
    echo "[INFO] Galaxy ${GAL_IDX}: sample row ${row_id}, global ID ${GLOBAL_ID}"

    python ${SCRIPT} \
      -s=${GAL_IDX} \
      -g1=${g1} \
      -g2=${g2} \
      -theta_int=${theta_int} \
      -i=${incl} \
      -v0=${v0} \
      -vcirc=${vcirc} \
      -rscale=${rscale} \
      -ID=${GLOBAL_ID} \
      -d=${DATASET_NAME}

    STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
      SUCCESS=$((SUCCESS + 1))
    else
      FAIL=$((FAIL + 1))
      echo "[WARN] Failed sample row ${row_id} for galaxy ${GAL_IDX} (exit ${STATUS})"
    fi
  done < <(awk -F, -v s="${START}" -v e="${END}" 'NR>1 {idx=NR-2; if (idx>=s && idx<e) print $1","$2","$3","$4","$5","$6","$7","$8}' "${SAMPFILE}")
done

echo "[INFO] Finished galaxy ${GAL_IDX} section ${SECTION_IDX}"
echo "[INFO] Success count: ${SUCCESS}"
echo "[INFO] Failure count: ${FAIL}"
