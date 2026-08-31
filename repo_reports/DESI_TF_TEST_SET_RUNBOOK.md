# DESI TF-conformed test-set runbook

This workflow creates one independent 100,000-galaxy simulator-v3 test set
for each Xu selection cut, caches a compact TF-weighted posterior product, and writes
one shear-bias report comparing the three cuts. These datasets are evaluation
sets only; do not use them for training, early stopping, or checkpoint choice.

## Statistical contract

For each cut, sample 100,000 jointly eligible catalog rows uniformly without
replacement. `rmag`, `hlr`, `img_snr`, and `halpha_snr` come from the same
row, preserving their empirical dependence. `xu_effective_weight` and the
catalog redshift are retained only as source provenance: neither affects row
selection or report weights. Every simulated galaxy uses `z=0.3`.

Catalog support is restricted to the trained model's domain: r magnitude
15--23.4, image S/N 5--1000, central H-alpha S/N 1--200, and finite HLR in
the inclusive range 0.1--5 arcsec. Rows outside any support bound are excluded
before selection. The resulting preflight counts for the uploaded catalogs are:

| Cut | Catalog rows | Jointly eligible rows |
| --- | ---: | ---: |
| Xu 1 | 36,536,538 | 2,428,231 |
| Xu 3 | 10,445,307 | 2,327,774 |
| Xu 5 | 6,837,780 | 1,737,834 |

Draw `g1`, `g2`, `theta_int`, `v0`, and `rscale` by Latin-hypercube uniform
coverage of the usual parameter ranges. Draw uniform `cos(i)` and transform
with `sin(i)=sqrt(1-cos(i)^2)`. Draw `vcirc` from the truncated TF conditional
used by `arch/tf_prior.py`: slope -7.22, intercept 36, intrinsic scatter 0.1
dex, and physical support 60--540 km/s. Draw true H-alpha flux log-uniformly
over the training range.

In `--test-set` cache/report mode, posterior candidates receive the usual
within-galaxy TF prior-replacement weights,
`p_TF(vcirc | rmag_true) / p0(vcirc)`, because finite TF scatter still changes
the posterior mass within each galaxy. The generated truth population is
already TF-conformed, so galaxies retain equal population mass before
posterior-precision weighting and no truth-level population TF ratio is
applied. TF-weighted posterior Mean summaries, physical `(g1, g2)` candidates,
normalized candidate log weights, and candidate-weight diagnostics are cached.
The report uses those same candidate weights for posterior intervals and
variance. `--test-set` alone keeps equal mass across truth galaxies;
`--test-set --weighted` composes that mass with the standard posterior-precision
weight regularized by the fixed ensemble shape-noise floor. Both modes report
the posterior Mean only and omit MAP diagnostics.

## Submit the complete dependency chain

Run this block once from a login node. The sample-table array maps task 1 to
Xu1, task 2 to Xu3, and task 3 to Xu5; it fixes the production size at 100,000
and uses canonical per-cut names and seeds 42001, 42003, and 42005. A global
`SAMPLE_NAME` override is deliberately rejected so array tasks cannot collide.
Each task also writes `<sample>.manifest.json` with the catalog selection audit,
sampling laws, TF configuration, fixed redshift, table hash, and ID policy.

The explicit `afterok` edges below make the whole block copyable: each database
waits for its own FITS array, each merge waits for its own database array, each
cache waits for its own merge, and the report waits for all three cache arrays.

```bash
set -euo pipefail
cd /jet/home/xwang30/kl-nn

# Avoid forwarding an unrelated interactive-shell override to the production
# sample-table array.
unset SAMPLE_NAME TOTAL

TOTAL=100000
CHUNK_SIZE=2000
NPARTS=100
NGALS=1000
NSAMPLES=10000
SEED=42
MODEL=CNN-CNN-Meta-bounded-hybrid-simv3-r90_1m_s42_44603007
CACHE_TAG=testset_tfweighted_v2_10k_s42
NAME_XU1=test_100k_simv3_xu1_tf
NAME_XU3=test_100k_simv3_xu3_tf
NAME_XU5=test_100k_simv3_xu5_tf
REPORT_OUTPUT=/ocean/projects/phy250048p/shared/reports/desi_tf_test_set_tfweighted_v2_s42.html

# Generate all three CSV sample tables and generation manifests.
SAMPLES_JOB=$(sbatch --parsable --array=1-3 \
  --export=ALL,TOTAL=${TOTAL},ALLOW_OVERWRITE=1 \
  data_generate/generate_desi_test_sets.slurm)

# Render 50 FITS chunks per cut.
FITS_XU1=$(sbatch --parsable --array=1-50%10 --time=12:00:00 --dependency=afterok:${SAMPLES_JOB} \
  --export=ALL,SAMPLE_NAME=${NAME_XU1},DATASET_NAME=${NAME_XU1},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE},ALLOW_PARTIAL_ARRAY=0 \
  data_generate/generate_simulator_v3.slurm)
FITS_XU3=$(sbatch --parsable --array=1-50%10 --time=12:00:00 --dependency=afterok:${SAMPLES_JOB} \
  --export=ALL,SAMPLE_NAME=${NAME_XU3},DATASET_NAME=${NAME_XU3},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE},ALLOW_PARTIAL_ARRAY=0 \
  data_generate/generate_simulator_v3.slurm)
FITS_XU5=$(sbatch --parsable --array=1-50%10 --time=12:00:00 --dependency=afterok:${SAMPLES_JOB} \
  --export=ALL,SAMPLE_NAME=${NAME_XU5},DATASET_NAME=${NAME_XU5},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE},ALLOW_PARTIAL_ARRAY=0 \
  data_generate/generate_simulator_v3.slurm)

# Build 50 LMDB shards per cut.
DB_XU1=$(sbatch --parsable --array=1-50 --dependency=afterok:${FITS_XU1} \
  --export=ALL,SAMPLE_NAME=${NAME_XU1},DATASET_NAME=${NAME_XU1},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/make_database_simulator_v3.slurm)
DB_XU3=$(sbatch --parsable --array=1-50 --dependency=afterok:${FITS_XU3} \
  --export=ALL,SAMPLE_NAME=${NAME_XU3},DATASET_NAME=${NAME_XU3},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/make_database_simulator_v3.slurm)
DB_XU5=$(sbatch --parsable --array=1-50 --dependency=afterok:${FITS_XU5} \
  --export=ALL,SAMPLE_NAME=${NAME_XU5},DATASET_NAME=${NAME_XU5},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/make_database_simulator_v3.slurm)

# Merge each shard set. The merge also installs the validated generation
# manifest at datasets/${NAME}/manifest.json for compact-cache validation.
MERGE_XU1=$(sbatch --parsable --dependency=afterok:${DB_XU1} \
  --export=ALL,SAMPLE_NAME=${NAME_XU1},DATASET_NAME=${NAME_XU1},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/merge_database_simulator_v3.slurm)
MERGE_XU3=$(sbatch --parsable --dependency=afterok:${DB_XU3} \
  --export=ALL,SAMPLE_NAME=${NAME_XU3},DATASET_NAME=${NAME_XU3},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/merge_database_simulator_v3.slurm)
MERGE_XU5=$(sbatch --parsable --dependency=afterok:${DB_XU5} \
  --export=ALL,SAMPLE_NAME=${NAME_XU5},DATASET_NAME=${NAME_XU5},TOTAL=${TOTAL},CHUNK_SIZE=${CHUNK_SIZE} \
  data_generate/merge_database_simulator_v3.slurm)

# Cache 10,000 identity/R90 draws for each galaxy. Compact --test-set mode
# saves TF-weighted Mean summaries, physical shear candidates, normalized
# candidate log weights, and ESS diagnostics.
CACHE_XU1=$(sbatch --parsable --array=1-${NPARTS} --time=2:00:00 \
  --dependency=afterok:${MERGE_XU1} \
  --export=ALL,MODEL_NAME=${MODEL},DATASET=${NAME_XU1},NPARTS=${NPARTS},NGALS=${NGALS},NSAMPLES=${NSAMPLES},SEED=${SEED},CACHE_TAG=${CACHE_TAG},TEST_SET=1 \
  arch/cache_posteriors.slurm)
CACHE_XU3=$(sbatch --parsable --array=1-${NPARTS} --time=2:00:00 \
  --dependency=afterok:${MERGE_XU3} \
  --export=ALL,MODEL_NAME=${MODEL},DATASET=${NAME_XU3},NPARTS=${NPARTS},NGALS=${NGALS},NSAMPLES=${NSAMPLES},SEED=${SEED},CACHE_TAG=${CACHE_TAG},TEST_SET=1 \
  arch/cache_posteriors.slurm)
CACHE_XU5=$(sbatch --parsable --array=1-${NPARTS} --time=2:00:00 \
  --dependency=afterok:${MERGE_XU5} \
  --export=ALL,MODEL_NAME=${MODEL},DATASET=${NAME_XU5},NPARTS=${NPARTS},NGALS=${NGALS},NSAMPLES=${NSAMPLES},SEED=${SEED},CACHE_TAG=${CACHE_TAG},TEST_SET=1 \
  arch/cache_posteriors.slurm)

# Analyze only after every compact cache partition succeeds.
REPORT_JOB=$(sbatch --parsable \
  --dependency=afterok:${CACHE_XU1}:${CACHE_XU3}:${CACHE_XU5} \
  --export=ALL,CASE_1=${MODEL}:${NAME_XU1}_${CACHE_TAG},CASE_2=${MODEL}:${NAME_XU3}_${CACHE_TAG},CASE_3=${MODEL}:${NAME_XU5}_${CACHE_TAG},OUTPUT=${REPORT_OUTPUT},WEIGHTED=1 \
  arch/diagnostics/shear_bias_report.slurm)

printf 'samples=%s\n' "${SAMPLES_JOB}"
printf 'xu1: fits=%s db=%s merge=%s cache=%s\n' "${FITS_XU1}" "${DB_XU1}" "${MERGE_XU1}" "${CACHE_XU1}"
printf 'xu3: fits=%s db=%s merge=%s cache=%s\n' "${FITS_XU3}" "${DB_XU3}" "${MERGE_XU3}" "${CACHE_XU3}"
printf 'xu5: fits=%s db=%s merge=%s cache=%s\n' "${FITS_XU5}" "${DB_XU5}" "${MERGE_XU5}" "${CACHE_XU5}"
printf 'report=%s output=%s\n' "${REPORT_JOB}" "${REPORT_OUTPUT}"
```

The sample launcher and database merge fail closed on existing outputs. The
block above deliberately sets `ALLOW_OVERWRITE=1` because the old
cap-after-selection CSV/manifest pairs are known to be obsolete; do not carry
that override into an unrelated run. Remove or archive database outputs
separately before a clean database rebuild. The versioned cache tag keeps new
TF-weighted cache products separate from the obsolete equal-candidate cache.
FITS rendering validates existing files before skipping them, so old
science-row fingerprints are regenerated and a partial FITS array can be
resumed safely.

## Recover an interrupted FITS array

Keep `CHUNK_SIZE=2000` when resuming: changing it would change the assignment
of proposal rows to `part_N` directories. First identify incomplete parts and
validate every published FITS against its proposal row:

```bash
(
  source /jet/home/xwang30/kl-nn/shared_job_scratch.sh
  setup_shared_job_scratch "fits-check-${DATASET_NAME}"
  FAILURE_REPORT=${SHARED_JOB_TMPDIR}/${DATASET_NAME}_fits_failures.tsv
  python data_generate/check_fits_generation.py \
    --sample=/ocean/projects/phy250048p/shared/samples/${SAMPLE_NAME}.csv \
    --fits-dir=/ocean/projects/phy250048p/shared/fits/${DATASET_NAME} \
    --chunk-size=2000 --total=100000 --verify-fits \
    --report=${FAILURE_REPORT}
  sed -n '1,200p' "${FAILURE_REPORT}"
)
```

The checker prints an `Incomplete array parts:` list. Review the displayed TSV
before the subshell exits; its unique child under
`/ocean/projects/phy250048p/shared/tmp` is then deleted automatically. Resume
only those parts, with at most ten concurrent tasks per cut:

```bash
FITS_RETRY=$(sbatch --parsable \
  --array=${INCOMPLETE_PARTS}%10 --time=12:00:00 \
  --export=ALL,SAMPLE_NAME=${SAMPLE_NAME},DATASET_NAME=${DATASET_NAME},TOTAL=100000,CHUNK_SIZE=2000,ALLOW_PARTIAL_ARRAY=1 \
  data_generate/generate_simulator_v3.slurm)
```

`--skip-existing` is built into the launcher. It skips a file only after its
structure, observation metadata, row ID, and full science-row fingerprint
match the current CSV. Atomic publication prevents a killed generator from
replacing a valid final path. Logs are combined stdout/stderr files named
`/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_<array-job>_<task>.out`;
there is no standalone `<array-job>.out` for an array.

The compact cache case names are `${MODEL}:${NAME_XU1}_${CACHE_TAG}` and the
corresponding Xu3/Xu5 forms. Compact candidates occupy roughly 12 GB per cut.
The final report validates complete partition coverage and shared provenance,
then shows one TF-conformed population per cut with TF-weighted, Mean-only
shear/nuisance calibration, candidate-weight health, coverage, conditional
diagnostics, the generation audit, and a cross-cut shear summary. The launcher
defaults to `WEIGHTED=1` to preserve the production report above. Submit it with
`WEIGHTED=0` and a distinct `OUTPUT` to report the equal-galaxy-mass ensemble
from the same compact caches.
