# Matched shear-response runbook

This workflow estimates finite-difference shear response on matched simulator
realizations. Each base galaxy produces five rows: zero shear, positive and
negative `g1`, and positive and negative `g2`. The nuisance parameters,
magnitude, H-alpha flux, and noise-control metadata are shared within the
five-row group. Simulator sample tables still store `sini`; packaging derives
the inference target `cosi`.

Run every command below from the repository root. Submit GPU arrays with
`--exclude=v005`. Set `KLNN_REPO_ROOT=/jet/home/xwang30/kl-nn`.

The maintained workflow has one bounded posterior and one cache schema. The
cache contains both broad-proposal summaries and TF-weighted summaries under
explicit names. FITS are noiseless; `sample_density(..., matched_group_size=5)`
applies one shared noise draw per base galaxy at cache time.

Do not use `test_100k_simv3_cosi_xu3_tf` as the generator input. Inverse-R
catalog correction is not the production shear estimator.

## 1. Create the matched sample table

Choose a simulator-v3-cosi source table that was not used for final scientific
testing:

```bash
python data_generate/make_shear_response_samples.py \
  --input /ocean/projects/phy250048p/shared/samples/train_1m_simv3_cosi.csv \
  --output /ocean/projects/phy250048p/shared/samples/samples_shear_response_simv3_cosi_5k.csv \
  --manifest /ocean/projects/phy250048p/shared/samples/shear_response_simv3_cosi_5k_manifest.csv \
  --nbase 1000 \
  --delta-g 0.01 \
  --seed 1729
```

Audit the output before rendering: it must contain exactly 5,000 rows, every
`base_id` must have the five named states, and all non-shear fields must agree
inside each group.

The generate launcher reads `SAMPFILE` with the `.csv` suffix. Database
packaging reads `SAMPLE` without the suffix, because `make_database.py` opens
`{SAMPLE}.csv`. Keep those two names aligned:

- `SAMPFILE=samples_shear_response_simv3_cosi_5k.csv`
- `SAMPLE=samples_shear_response_simv3_cosi_5k`
- `DATASET=shear_response_simv3_cosi_5k`

## 2. Render the five matched observations

```bash
SAMPFILE=samples_shear_response_simv3_cosi_5k.csv \
DATASET=shear_response_simv3_cosi_5k \
NBASE=1000 \
sbatch data_generate/generate_shear_response.slurm
```

Wait for every array task and check the FITS manifest before packaging. A
missing FITS file should fail packaging; do not silently drop a response state.

## 3. Build and merge the database

```bash
SAMPLE=samples_shear_response_simv3_cosi_5k \
DATASET=shear_response_simv3_cosi_5k \
NBASE=1000 \
sbatch data_generate/make_shear_response_database.slurm
```

After all ten database shards succeed:

```bash
SAMPLE=samples_shear_response_simv3_cosi_5k \
DATASET=shear_response_simv3_cosi_5k \
NBASE=1000 \
sbatch data_generate/merge_shear_response_database.slurm
```

Validate that the merged database has 5,000 rows in manifest order and that
the nine-target schema matches the trained model.

## 4. Cache posterior candidates

`EPOCH` is an optional integer checkpoint suffix, for example `EPOCH=199`.
Omitting it requires and selects the saved best checkpoint. A numbered
checkpoint is used only when `EPOCH` is set explicitly. Archived concat
checkpoints that omit `use_image_spectrum_fusion` load as identity fusion.

```bash
export KLNN_REPO_ROOT=/jet/home/xwang30/kl-nn
export MODEL_NAME=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702
# export EPOCH=199  # Optional numbered checkpoint.
DATASET=shear_response_simv3_cosi_5k \
NBASE=1000 \
CACHE_TAG=shear_response_candidates \
sbatch --exclude=v005 arch/shear_response_inference.slurm
```

Resume a sparse array with `ALLOW_PARTIAL_ARRAY=1`. The launcher sets
`matched_group_size=5`. Each partition writes physical-unit base candidates,
base log densities, within-galaxy TF weights and ESS, the truth-level
population TF ratio, named proposal/TF-target summaries, and a manifest.
Candidate counts must be positive and even for the identity/R90 mixture.

Before reporting, require all partitions and verify their manifests agree on
checkpoint, target order, TF hyperparameters, group size, and R90 policy.

## 5. Measure raw response

Use `arch/diagnostics/shear_response_report.py` with a named posterior source,
never a numeric selector. Inverse-R correction stays off unless
`--apply-response-correction` is passed. Quote raw `R`, additive `c`, ESS, and
the SNR / `cosi` bins, especially `faint_low_halpha`.

```bash
python arch/diagnostics/shear_response_report.py \
  --cache-root "${CACHE_DIR}" \
  --manifest /ocean/projects/phy250048p/shared/samples/shear_response_simv3_cosi_5k_manifest.csv \
  --posterior-source tf_target \
  --estimator mean \
  --calibration-fraction 0.5 \
  --seed 31415 \
  --output /ocean/projects/phy250048p/shared/reports/shear-response/shear_response_tf_target.html
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
- Interpret diagonal `R` on the faint / low-Hα bin, not only the
  training-prior average.
- Do not apply the calibration-split inverse as the paper's shear estimator.
- Report effective sample size for both posterior candidate weights and
  population weights; low ESS is a failure warning, not extra precision.
- Keep this diagnostic distinct from the ordinary xu3 shear-bias fit.

## 7. Pair-NLL follow-up and xu3 comparison

Stage-4 faint-bin raw \(R\sim 0.5\) is the go criterion for mixing pair rows
into concat NPE with the CNN still frozen and `IMAGE_SPECTRUM_FUSION=0`.
Do not honor a leftover `MODEL_NAME` from a cache job: `train_npe.slurm`
now refuses to overwrite an existing model directory unless
`ALLOW_MODEL_OVERWRITE=1`. Job `45467143` continued the frozen concat
checkpoint in place, so the files currently named
`CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702`
are the pair-NPE weights. The original frozen xu3 TF-weighted Mean cache
written on 2026-09-05 remains

`.../45255702/test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42/`

and must not be overwritten. Job `45467143` also overwrote that model's
network snapshot with the current spec CNN, which does not match the saved
weights; instantiate `KLNPE` from the sibling concat snapshot
`CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_unfrozen_s42_45255704`.
Cache the pair-NPE xu3 comparison with a distinct `CACHE_TAG`, for example
`testset_tfweighted_v2_10k_s42_pairs_45467143`.

```bash
export KLNN_REPO_ROOT=/jet/home/xwang30/kl-nn
sbatch --exclude=v005 --array=1-100 --time=1:30:00 \
  --export=ALL,KLNN_REPO_ROOT=/jet/home/xwang30/kl-nn,MODEL_NAME=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702,NETWORKS_NAME=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_unfrozen_s42_45255704,DATASET=test_100k_simv3_cosi_xu3_tf,NPARTS=100,NGALS=1000,NSAMPLES=10000,TEST_SET=1,CACHE_TAG=testset_tfweighted_v2_10k_s42_pairs_45467143,SEED=42,CHECKPOINT=/ocean/projects/phy250048p/shared/models/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702best \
  arch/cache_posteriors.slurm
```

Then compare TF-weighted Mean against the frozen cache:

```bash
CASE_1=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42 \
CASE_2=CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_pairs_45467143 \
OUTPUT=/ocean/projects/phy250048p/shared/reports/arch-ablations/valid100k_frozen_vs_pairs_xu3_tfweighted_s42.html \
WEIGHTED=1 \
sbatch arch/diagnostics/shear_bias_report.slurm
```
