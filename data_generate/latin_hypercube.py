"""Generate the simulator-v3 coverage proposal without a Tully--Fisher relation."""

from argparse import ArgumentParser
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

try:
    from .observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_HALPHA_LOG10_FLUX_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )
except ImportError:
    from observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_HALPHA_LOG10_FLUX_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )


SAMPLE_ROOT = "/ocean/projects/phy250048p/shared/samples"
PARAMETER_LIMITS = {
    "g1": (-0.1, 0.1),
    "g2": (-0.1, 0.1),
    "theta_int": (-np.pi, np.pi),
    "sini": (0.0, 1.0),
    "v0": (-30.0, 30.0),
    "vcirc": (60.0, 540.0),
    "rscale": (0.1, 5.0),
    "hlr": (0.1, 5.0),
}
DEFAULT_RMAG_RANGE = (15.0, 23.4)


def _lhs_column(nsamples, seed_sequence, lower, upper):
    sampler = LatinHypercube(
        1,
        scramble=True,
        seed=np.random.default_rng(seed_sequence),
    )
    return lower + sampler.random(nsamples)[:, 0] * (upper - lower)


def generate_samples(
    nsamples: int,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw targets and observation-quality controls independently.

    H-alpha flux is Latin-hypercube-uniform in log10 physical flux. Image and
    central-line S/N are Latin-hypercube-uniform in linear S/N and are stored
    once per galaxy for reuse across noise realizations and matched shears.
    """

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    (
        parameter_seed,
        magnitude_seed,
        halpha_seed,
        image_snr_seed,
        central_halpha_snr_seed,
    ) = np.random.SeedSequence(seed).spawn(5)
    names = tuple(PARAMETER_LIMITS)
    unit = LatinHypercube(
        len(names),
        scramble=True,
        seed=np.random.default_rng(parameter_seed),
    ).random(nsamples)
    result = pd.DataFrame(unit, columns=names)
    for name, (lower, upper) in PARAMETER_LIMITS.items():
        result[name] = lower + result[name] * (upper - lower)
    result[RMAG_TRUE_COLUMN] = _lhs_column(
        nsamples, magnitude_seed, *DEFAULT_RMAG_RANGE
    )
    log10_halpha_flux = _lhs_column(
        nsamples, halpha_seed, *DEFAULT_HALPHA_LOG10_FLUX_RANGE
    )
    result[HALPHA_FLUX_TRUE_COLUMN] = np.power(10.0, log10_halpha_flux)
    result[IMAGE_SNR_COLUMN] = _lhs_column(
        nsamples, image_snr_seed, *DEFAULT_IMAGE_SNR_RANGE
    )
    result[CENTRAL_HALPHA_SNR_COLUMN] = _lhs_column(
        nsamples,
        central_halpha_snr_seed,
        *DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
    )
    result[FIBER_LAYOUT_COLUMN] = GALAXY_AXIS_FIBER_LAYOUT
    result[OBSERVATION_MODEL_VERSION_COLUMN] = np.int16(
        OBSERVATION_MODEL_VERSION
    )
    return result


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--nsamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output",
        default=join(
            SAMPLE_ROOT,
            "samples_train_1m_simv3_galaxyaxis_central_halpha.csv",
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    samples = generate_samples(
        args.nsamples,
        seed=args.seed,
    )
    samples.to_csv(args.output, index_label="ID")


if __name__ == "__main__":
    main()
