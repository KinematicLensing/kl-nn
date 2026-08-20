# Matched shear-response runbook

This workflow estimates finite-difference shear response on matched simulator
realizations. Each base galaxy produces five rows: zero shear, positive and
negative `g1`, and positive and negative `g2`. The nuisance parameters,
magnitude, H-alpha flux, and noise-control metadata are shared within the
five-row group.

Run every command below from the repository root.

The maintained workflow has one bounded posterior and one cache schema. The
cache contains both broad-proposal summaries and TF-weighted summaries under
explicit names.

## 1. Create the matched sample table

Choose a simulator-v2 source table that was not used for final scientific
testing:

```bash
python data_generate/make_shear_response_samples.py \
  --input /ocean/projects/phy250048p/shared/samples/samples_small_1m_simv2_galaxyaxis_halpha.csv \
  --output /ocean/projects/phy250048p/shared/samples/samples_shear_response_simv2_galaxyaxis_halpha_5k.csv \
  --manifest /ocean/projects/phy250048p/shared/samples/shear_response_simv2_galaxyaxis_halpha_5k_manifest.csv \
  --nbase 1000 \
  --delta-g 0.01 \
  --seed 1729
```

Audit the output before rendering: it must contain exactly 5,000 rows, every
`base_id` must have the five named states, and all non-shear fields must agree
inside each group.

## 2. Render the five matched observations

```bash
SAMPFILE=samples_shear_response_simv2_galaxyaxis_halpha_5k.csv \
DATASET=shear_response_simv2_galaxyaxis_halpha_5k \
NBASE=1000 \
sbatch data_generate/generate_shear_response.slurm
```

Wait for every array task and check the FITS manifest before packaging. A
missing FITS file should fail packaging; do not silently drop a response state.

## 3. Build and merge the database

```bash
SAMPLE=shear_response_simv2_galaxyaxis_halpha_5k \
DATASET=shear_response_simv2_galaxyaxis_halpha_5k \
NBASE=1000 \
sbatch data_generate/make_shear_response_database.slurm
```

After all ten database shards succeed:

```bash
SAMPLE=shear_response_simv2_galaxyaxis_halpha_5k \
DATASET=shear_response_simv2_galaxyaxis_halpha_5k \
NBASE=1000 \
sbatch data_generate/merge_shear_response_database.slurm
```

Validate that the merged database has 5,000 rows in manifest order and that
the nine-target schema matches the trained model.

## 4. Cache posterior candidates

`EPOCH` is an optional integer checkpoint suffix, for example `EPOCH=199`.
Omitting it requires and selects the saved best checkpoint. A numbered
checkpoint is used only when `EPOCH` is set explicitly.

```bash
export MODEL_NAME=your-completed-npe-run
# export EPOCH=199  # Optional numbered checkpoint.
DATASET=shear_response_simv2_galaxyaxis_halpha_5k \
NBASE=1000 \
CACHE_TAG=shear_response_candidates \
sbatch arch/shear_response_inference.slurm
```

The launcher sets `matched_group_size=5`. Each partition writes physical-unit
base candidates, base log densities, within-galaxy TF weights and ESS, the
truth-level population TF ratio, named proposal/TF-target summaries, and a
manifest. Candidate counts must be positive and even for the identity/R90
mixture.

Before reporting, require all partitions and verify their manifests agree on
checkpoint, target order, TF hyperparameters, group size, and R90 policy.

## 5. Measure response

Use `arch/diagnostics/shear_response_report.py` with a named posterior source,
never a numeric selector. For example, after setting `CACHE_DIR` to the cache
directory containing the partitioned arrays:

```bash
python arch/diagnostics/shear_response_report.py \
  --cache-root "${CACHE_DIR}" \
  --manifest /ocean/projects/phy250048p/shared/samples/shear_response_simv2_galaxyaxis_halpha_5k_manifest.csv \
  --posterior-source tf_target \
  --estimator mean \
  --calibration-fraction 0.5 \
  --seed 31415 \
  --output repo_reports/shear_response_tf_target.html
```

Run the same report with `--posterior-source proposal` to separate behavior of
the broad base posterior from the assumed TF target population.

For TF-target response, load `population_tf_log_ratio`, verify it is identical
across the five matched rows of each base galaxy, and reduce to one ratio per
base. Normalize ratios globally over the complete calibration ensemble and
again over the complete holdout ensemble. Use those weights in response,
additive bias, uncertainty, and any downstream selection statistics. Do not
normalize population weights partition by partition.

## 6. Interpretation guardrails

- Split by `base_id`, so no matched state of one galaxy crosses between
  calibration and holdout.
- Compute the central response matrix from the `+/- delta-g` pairs and use the
  zero-shear state for additive response.
- Apply the response correction learned from the calibration bases only.
- Quote the corrected response and additive term on holdout bases.
- Report effective sample size for both posterior candidate weights and
  population weights; low ESS is a failure warning, not extra precision.
- Keep response-calibrated results distinct from the ordinary shear-bias fit.
  The response pilot diagnoses estimator nonlinearity but does not replace
  model repair or held-out posterior calibration.
