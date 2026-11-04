# KL-NN

Kinematic lensing (KL) measures weak-lensing shear from a galaxy's shape
together with its rotation. Fitting each galaxy with MCMC is slow. KL-NN trains
a neural network on simulated galaxies so that, once trained, it returns the
posterior for a galaxy in under a second.

For each galaxy, the network takes
- an r-band image, and
- five Hα fiber spectra (DESI-like),

together with the galaxy's r-band magnitude and the noise level of the image and
spectra, which are treated as known. It returns posterior samples of

| Parameter | Meaning |
|---|---|
| `g1`, `g2` | shear |
| `theta_int` | intrinsic position angle of the disk |
| `cosi` | cosine of the inclination |
| `v0` | systemic velocity |
| `vcirc` | rotation speed |
| `rscale` | rotation-curve scale radius |
| `hlr` | half-light radius |
| `halpha_flux_true` | Hα flux |

The network is trained without assuming the Tully–Fisher relation. The
relation can be added afterwards by reweighting the posterior samples.

This repository covers the pipeline: simulating training galaxies, training the
network, and running it on new galaxies. Everything done with a trained network
(tests, figures and reports for the paper) is in
[kl-nn-analysis](https://github.com/KinematicLensing/kl-nn-analysis).

## Folders

| Folder | Contents |
|---|---|
| `data_generate/` | Simulate galaxies (image + spectra) with [kl-tools](https://github.com/wxs0703/kl-tools) and pack them into a database for training |
| `arch/` | The network, training, and running a trained network on new galaxies |
| `tests/` | Automated tests |

## Setup

1. Put this repository, `kl-tools` and (optionally) `kl-nn-analysis` in the
   same parent folder.
2. Create the Python environment:
   ```bash
   conda env create -f environment.yml
   conda activate kl-nn
   ```
3. Open `paths.json` and check the paths. It is the only place where folders
   on disk are written down (simulated data, trained models, results, logs).
   On a new machine, you usually only need to change `shared_root`.

## Running the pipeline

Jobs are Slurm scripts (`.slurm`). Submit them from this folder with
`./submit.sh`, which works like `sbatch` and puts the log file in the `logs`
folder from `paths.json`:

```bash
./submit.sh arch/train_npe.slurm
```

Settings such as dataset names or the number of galaxies are listed at the
top of each script. They can be changed through environment variables, e.g.
`PRETRAINED_NAME=... ./submit.sh arch/train_npe.slurm`.

1. **Draw galaxy parameters.** `python data_generate/latin_hypercube.py`
   writes a table of galaxy parameters: `--nsamples` rows (default 100,000)
   spread evenly over the allowed ranges. Make separate tables, with
   different `--seed`s and `--output` names, for training, validation and
   testing. Never use the test set to choose between trained networks.
   For realistic test galaxies drawn from DESI catalogs, add `--test-set
   --catalog <catalog file>`.
2. **Simulate the galaxies.** `data_generate/generate_simulator_v3.slurm`
   renders the image and spectra of every galaxy in the table.
3. **Build the training database.**
   `data_generate/make_database_simulator_v3.slurm`, then
   `data_generate/merge_database_simulator_v3.slurm`.
4. **Pretrain the image and spectrum encoders.** `arch/pretrain_ccl.slurm`.
5. **Train the network.** `arch/train_npe.slurm`, with `PRETRAINED_NAME` set
   to the run from step 4.
6. **Run the trained network on test galaxies.** `arch/cache_posteriors.slurm`,
   with `DATASET` set to a test database. It saves posterior samples for each
   galaxy, with and without the Tully–Fisher reweighting.
7. **Analyse the results** with the scripts in kl-nn-analysis.

## Tests

```bash
conda activate kl-nn
pytest -q tests
```

## Notes

- Datasets and trained networks from earlier simulator versions (1 and 2)
  cannot be read by the current code (the scripts above are version 3).
  Regenerate them with the steps above.
- When a network is trained, its settings and the network code are saved next
  to it. Keep them together.
- `environment.full.yml` is the exact environment used on Bridges-2, kept for
  reference. Use `environment.yml` for new installs.
- The history of the analysis code, from before it moved to kl-nn-analysis
  in 2026-11, is on branch `nre`.
