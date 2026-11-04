# KL-NN

Neural posterior estimation for kinematic lensing. A network is trained on
simulated galaxies (an r-band image and five Hα fiber spectra each) and then
returns posterior samples of the shear and the galaxy parameters for new galaxies.

This repository covers simulating the training data, training the network, and
running it on test galaxies. Analyses that use a trained network (tests and
figures for the paper) are in [kl-nn-analysis](https://github.com/KinematicLensing/kl-nn-analysis).

## Quick Start

```bash
conda env create -f environment.yml   # creates the 'kl-nn' environment
conda activate kl-nn
pytest -q tests
```

Check the paths in `paths.json` before running anything (see [Paths](#paths)).

## Repository Structure

```
arch/                         # Network, training, and inference
├── networks.py               # Image/spectrum encoders and the posterior network
├── data.py                   # Data loading and on-the-fly noise
├── config.py                 # Training settings and parameter ranges
├── train.py                  # Training loop
├── train_model.py            # Training entry point
├── model_registry.py         # Save and load trained networks with their settings
├── cache_posteriors.py       # Run a trained network on a test set, save posterior samples
├── cache_contract.py         # Read saved posterior samples
├── tf_prior.py               # Tully-Fisher reweighting of posterior samples
├── inclination_prior.py      # Reweighting for a different inclination prior
├── map_laplace.py            # Shear point estimates and errors from posterior samples
├── circular_spline.py        # Spline flow for the (periodic) position angle
├── plot_losses.py            # Plot training and validation losses
├── pretrain_ccl.slurm        # Job: pretrain the image/spectrum encoders
├── train_npe.slurm           # Job: train the posterior network
├── cache_posteriors.slurm    # Job: posterior samples for a test set
└── shear_response_inference.slurm  # Job: posterior samples for the shear-response set

data_generate/                # Simulated data (uses kl-tools)
├── latin_hypercube.py        # Draw galaxy parameters (training, or test sets from DESI catalogs)
├── desi_test_set_sampling.py # Draw test galaxies from DESI catalogs
├── generate_fits.py          # Render one galaxy's image and spectra
├── generate_fits_wrapper.py  # Render a range of galaxies from a parameter table
├── observation_schema.py     # Observation settings (wavelengths, fibers, noise)
├── fiber_targeting.py        # Fiber placement from the observed image
├── make_database.py          # Pack rendered galaxies into a training database
├── generation_integrity.py   # Checks that rendering finished correctly
├── check_fits_generation.py  # Check a rendered dataset is complete
├── checksum.sh               # Same check, as a shell script
├── make_shear_response_samples.py   # Galaxies re-rendered at several shears (shear calibration)
├── make_fisher_stencil_samples.py   # Galaxies at small parameter offsets (Fisher forecast)
├── repair_minor_axis_fiber_sign.py  # Fix for a fiber-placement bug in older datasets
├── generate_simulator_v3.slurm      # Job: render galaxies
├── make_database_simulator_v3.slurm # Job: build database pieces
├── merge_database_simulator_v3.slurm  # Job: merge them into one database
├── generate_desi_test_sets.slurm    # Job: test sets from DESI catalogs
├── generate_shear_response.slurm, make/merge_shear_response_database.slurm
├── generate_fisher_stencil.slurm
└── *_simulator_v2.slurm      # Jobs for the previous simulator version

tests/                        # Unit tests (pytest)
paths.json                    # All folders on disk (data, models, results, logs)
klnn_paths.py, klnn_paths.sh  # Read paths.json from Python and shell
submit.sh                     # Submit a job with its log in the logs folder
shared_job_scratch.sh         # Temporary folder for jobs, removed when the job ends
environment.yml               # Conda environment
environment.full.yml          # Exact environment used on Bridges-2 (reference only)
```

## Installation

kl-tools is installed from the `data_generate` branch of
[wxs0703/kl-tools](https://github.com/wxs0703/kl-tools) by `environment.yml`.
The simulator also reads data files from a kl-tools checkout, which by default is
expected next to this repository (`../kl-tools`). To work on kl-tools itself,
install that checkout instead:

```bash
pip install -e ../kl-tools
```

## Paths

Every folder on disk is set in `paths.json`; nothing else in the code has a
path written in it. On a new machine, change `shared_root` (and anything else
that moved), or put your own values in `paths.local.json` (not tracked by git).

```bash
python klnn_paths.py --all    # print every path
```

In Python, use `from klnn_paths import path; path("models")`. In shell scripts,
`source klnn_paths.sh` sets `KLNN_MODELS`, `KLNN_DATASETS`, etc.

## Running Jobs

Submit Slurm jobs from the repository root with `./submit.sh`, which takes the
same arguments as `sbatch` and writes the log to the `logs` folder:

```bash
./submit.sh arch/train_npe.slurm
./submit.sh --array=1-500 data_generate/generate_simulator_v3.slurm
```

Settings are listed at the top of each job script and can be changed with
environment variables, e.g. `PRETRAINED_NAME=<run> ./submit.sh arch/train_npe.slurm`.

## Workflow

1. Draw galaxy parameters: `python data_generate/latin_hypercube.py --nsamples N --seed S --output NAME`.
   Make separate tables for training, validation and testing. For test galaxies
   drawn from DESI catalogs, add `--test-set --catalog <file>`.
2. Render the galaxies: `data_generate/generate_simulator_v3.slurm`.
3. Build the database: `make_database_simulator_v3.slurm`, then `merge_database_simulator_v3.slurm`.
4. Pretrain the encoders: `arch/pretrain_ccl.slurm`.
5. Train the network: `arch/train_npe.slurm` with `PRETRAINED_NAME` from step 4.
6. Get posterior samples for a test set: `arch/cache_posteriors.slurm` with `DATASET` set.

Each trained network is saved with its settings and a copy of the network code.
Datasets and networks from the previous simulator version cannot be read by the
current code.

## Testing

```bash
pytest -q tests
```

## Development

The history of the analysis code that moved to kl-nn-analysis is on branch `nre`.
