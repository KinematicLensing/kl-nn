# TF factorial runbook

The expensive stage is four raw model passes, not twelve: two training
treatments (unweighted and TF-weighted) by two evaluation populations
(ordinary and TF-conformed). Each raw candidate bank is then reused for all
three inference treatments (`none`, `multiply`, and `replace`) on CPU.

Use one fixed scalar-SNR realization for both models within an evaluation
population. The candidate job accepts `SNR_DIR`, containing `part0of10.npy`
through `part9of10.npy`. This controls noise realization and prevents a TF
scatter draw from being mistaken for a model effect.

Example submission from `arch/`:

```bash
MODEL=CNN-CNN-flow EPOCH=... DATASET=valid_1m \
SAMPFILE=samples_valid_1m.csv \
SNR_DIR=/ocean/projects/phy250048p/shared/cache/REFERENCE/valid_1m/snr \
sbatch --export=ALL tf_factorial_candidates.slurm
```

Repeat for the TF-trained model, then repeat both models with the common
TF-conformed SNR directory. Set a distinct `TAG` if needed. A bank is written
under `cache/$MODEL/${DATASET}_${TAG}`.

Then submit the CPU treatment for each bank:

```bash
BANK_ROOT=/ocean/projects/phy250048p/shared/cache/MODEL/DATASET_tf_factorial_raw \
OUTPUT_ROOT=/ocean/projects/phy250048p/shared/tf_factorial/MODEL/DATASET \
sbatch --export=ALL tf_factorial_offline.slurm
```

Start with 4096 candidates. Inspect the saved ESS arrays before increasing
that count: only banks with poor ESS need rerunning. The `replace` treatment
uses a documented histogram approximation to the historical one-dimensional
KDE, avoiding an O(N²) calculation for every galaxy.
