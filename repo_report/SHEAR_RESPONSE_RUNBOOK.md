# Matched finite-shear response pilot

This is an offline calibration experiment, not an inference-time simulator
requirement. The simulator is run once to measure a response matrix; applying
the fitted matrix to real galaxies is a small matrix operation and generates no
new images or spectra.

The default pilot uses 1,000 nuisance-parameter bases and five matched shear
states per base: zero, ±g1, and ±g2. At five seconds per simulation that is
about 6.9 serial core-hours, or roughly 4–10 minutes across 100 CPU tasks before
filesystem overhead.

From `data_generate/`, prepare the input and manifest:

```bash
python make_shear_response_samples.py \
  --input=/ocean/projects/phy250048p/shared/samples/samples_valid_1m_simv2_galaxyaxis_halpha.csv \
  --output=/ocean/projects/phy250048p/shared/samples/samples_shear_response_simv2_galaxyaxis_halpha_5k.csv \
  --manifest=/ocean/projects/phy250048p/shared/samples/shear_response_simv2_galaxyaxis_halpha_5k_manifest.csv \
  --nbase=1000 --delta-g=0.01
```

Then run, in dependency order:

```bash
GEN=$(sbatch --parsable generate_shear_response.slurm)
DB=$(sbatch --parsable --dependency=afterok:$GEN make_shear_response_database.slurm)
MERGE=$(sbatch --parsable --dependency=afterok:$DB merge_shear_response_database.slurm)
```

After the merge, submit `arch/shear_response_inference.slurm` with `MODEL` and
`EPOCH`. It defaults to the v2 mode-1 base posterior; set
`TF_INFERENCE=prior_replacement` only when the response should include the
external TF prior. Finally run `arch/diagnostics/shear_response_report.py` against the
resulting cache and the manifest. It splits by `base_id`, so no matched galaxy
can leak between calibration and holdout. The inference job also reuses the
same latent `rmag_true` and `halpha_flux_true`, catalog-flux draw, spectral
quality, and injected image/spectral noise realization across each group of five;
otherwise observation differences would dominate the finite difference.
