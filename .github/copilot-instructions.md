# Contributor instructions for KL-NN

## Supported surface

Work only against the current simulator-v2 pipeline described in `README.md`
and `repo_reports/KL-NN_SIMV2_ARCH.md`. Do not add compatibility branches for
removed architectures or inference modes. If an old artifact needs analysis,
use its checkpointed repository revision instead of expanding the live API.

The canonical entry points are:

- `arch/train_model.py --stage pretrain`
- `arch/train_model.py --stage npe`
- `arch/cache_posteriors.py`
- `arch/pretrain_ccl.slurm`
- `arch/train_npe.slurm`
- `arch/cache_posteriors.slurm`

## Contracts to preserve

- Target order is exactly `config.TARGET_NAMES` and includes
  `halpha_flux_true`.
- Scalar context is exactly `config.ORACLE_CONTEXT_FIELDS`:
  `rmag_true` and `spectral_reference_quality`.
- The feature extractor receives the identity/R90 pair. Parameters and fiber
  positions must be transformed with the image and returned posterior samples
  must be inverse-aligned before mixing.
- The posterior is bounded for every non-angular target and circular for
  `theta_int`.
- The simulation proposal is independent and uniform in `vcirc`; TF
  information is post-training importance weighting only.
- Posterior TF weights normalize within galaxy. Population TF weights normalize
  only after concatenating every cache partition used in an ensemble statistic.
- Image and spectral noise levels are independent. Keep the Gaussian noise
  family unless the scientific scope is explicitly changed.

## Tests and repository hygiene

Activate the `kl-nn` environment and run `pytest -q tests`. A focused test can
be run as `pytest -q tests/<file>.py -k <case>`. Several tests import modules
directly from `arch/` through `tests/conftest.py`.

Do not commit `__pycache__`, notebook checkpoints, generated plots, local model
checkpoints, cache arrays, or scheduler logs. Preserve user data under
`/ocean/projects/phy250048p/shared`; tests should use temporary directories.
