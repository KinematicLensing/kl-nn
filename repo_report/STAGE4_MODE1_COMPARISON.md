# Stage 4 mode-1 paired comparison

This run isolates inference-time Tully-Fisher replacement for the completed
Stage 4 D4 posterior. The checkpoint itself was trained with TF weighting;
mode 1 only disables the additional inference-time TF replacement.

With its defaults, the launcher processes all 10,000 galaxies in ten array
tasks of 1,000 galaxies each. Task `i + 1` matches `part{i}of10` from the
completed mode-2 run, including its dataset range, deterministically generated
TF-conformed SNR, and injected image/spectral noise realization. The base seed
is 42 and each task derives the same partition seed as mode 2. The D4 posterior
draws 5,000 samples per galaxy, which is divisible by eight.

Submit from the repository root:

```bash
cd /jet/home/xwang30/kl-nn
sbatch arch/tf_analysis_stage4_mode1_10k.slurm
```

Default checkpoint:

```text
CNN-SetAttn-D4-flow_tf_stage4_s42_43435707, epoch suffix 19
```

Expected mode-1 cache root:

```text
/ocean/projects/phy250048p/shared/cache/CNN-SetAttn-D4-flow_tf_stage4_s42_43435707/small_1m_tf_conformed_d4_exact_mode1_raw_10k_s42
```

Completed mode-2 reference:

```text
/ocean/projects/phy250048p/shared/cache/CNN-SetAttn-D4-flow_tf_stage4_s42_43435707/small_1m_tf_conformed_d4_exact
```

After all ten array tasks complete, build the paired 10,000-galaxy report from
the repository root:

```bash
python arch/diagnostics/shear_bias_report.py \
  --case CNN-SetAttn-D4-flow_tf_stage4_s42_43435707:small_1m_tf_conformed_d4_exact \
  --case CNN-SetAttn-D4-flow_tf_stage4_s42_43435707:small_1m_tf_conformed_d4_exact_mode1_raw_10k_s42 \
  --mode 0 \
  --output repo_report/SHEAR_BIAS_STAGE4_MODE1_VS_MODE2_10K.html
```

Here `--mode 0` selects the sole cached result axis in each case; it does not
change either case's model inference mode.

Each new manifest records the checkpoint epoch, TF-conformance flag, global
seed, partition seed, and the correct half-open galaxy range. The launcher
requires ten 1,000-galaxy partitions. Keep its other defaults for an exactly
paired comparison; overriding the seed, dataset, sample set, model, or epoch
changes the pairing.
