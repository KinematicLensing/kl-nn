"""Generate simulator-v3 training proposals or catalog-backed test sets."""

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
DEFAULT_OUTPUT = join(
    SAMPLE_ROOT,
    "train_1m_simv3_cosi.csv",
)
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
ALLOWED_G_LIMITS = (0.02, 0.1, 0.2)
DEFAULT_G_LIMIT = 0.1
DEFAULT_RMAG_RANGE = (15.0, 23.4)


def resolve_g_limit(g_limit=DEFAULT_G_LIMIT) -> float:
    candidate = float(g_limit)
    if not np.isfinite(candidate) or candidate <= 0.0:
        raise ValueError("g-limit must be a positive finite value")
    for allowed in ALLOWED_G_LIMITS:
        if np.isclose(candidate, allowed, rtol=0.0, atol=1e-12):
            return float(allowed)
    raise ValueError(
        f"g-limit must be one of {ALLOWED_G_LIMITS!r}; got {candidate!r}"
    )


def parameter_limits_with_g_limit(g_limit=DEFAULT_G_LIMIT) -> dict[str, tuple[float, float]]:
    half_width = resolve_g_limit(g_limit)
    limits = dict(PARAMETER_LIMITS)
    limits["g1"] = (-half_width, half_width)
    limits["g2"] = (-half_width, half_width)
    return limits


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
    g_limit: float = DEFAULT_G_LIMIT,
) -> pd.DataFrame:
    """Draw targets and observation-quality controls independently.

    Inclination is Latin-hypercube-uniform in ``cos(i)`` and converted to the
    simulator's ``sin(i)`` parameterization. H-alpha flux is uniform in log10
    physical flux. Image and central-line S/N are uniform in linear S/N and
    are stored once per galaxy for reuse across noise realizations and matched
    shears. ``g_limit`` is the allowlisted half-width for ``g1`` and ``g2``.
    """

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    limits = parameter_limits_with_g_limit(g_limit)
    (
        parameter_seed,
        magnitude_seed,
        halpha_seed,
        image_snr_seed,
        central_halpha_snr_seed,
    ) = np.random.SeedSequence(seed).spawn(5)
    names = tuple(limits)
    unit = LatinHypercube(
        len(names),
        scramble=True,
        seed=np.random.default_rng(parameter_seed),
    ).random(nsamples)
    result = pd.DataFrame(unit, columns=names)
    for name, (lower, upper) in limits.items():
        if name == "sini":
            cosi = lower + result[name].to_numpy() * (upper - lower)
            result[name] = np.sqrt(np.maximum(0.0, 1.0 - np.square(cosi)))
        else:
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


def generate_test_set_samples(
    nsamples: int,
    *,
    catalog: str,
    seed: int | None = None,
    catalog_extension: str = "SELECTION",
    catalog_block_size: int = 500_000,
    tf_slope: float = -7.22,
    tf_intercept: float = 36.0,
    tf_scatter_dex: float = 0.1,
):
    """Draw the DESI-backed TF-conformed proposal and its manifest payload."""

    try:
        from .desi_test_set_sampling import generate_catalog_test_set
    except ImportError:  # Support direct execution from data_generate/.
        from desi_test_set_sampling import generate_catalog_test_set

    try:
        from arch.tf_prior import TFPrior
    except ModuleNotFoundError:  # Direct execution is handled by the sampler module.
        from desi_test_set_sampling import TFPrior

    prior = TFPrior(
        slope=tf_slope,
        intercept=tf_intercept,
        scatter_dex=tf_scatter_dex,
        vcirc_min=PARAMETER_LIMITS["vcirc"][0],
        vcirc_max=PARAMETER_LIMITS["vcirc"][1],
    )
    return generate_catalog_test_set(
        nsamples,
        catalog_path=catalog,
        parameter_limits=PARAMETER_LIMITS,
        seed=seed,
        catalog_extension=catalog_extension,
        catalog_block_size=catalog_block_size,
        tf_prior=prior,
    )


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--nsamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="generate a DESI-backed TF-conformed test population",
    )
    parser.add_argument(
        "--catalog",
        help="DESI cut FITS catalog (required with --test-set)",
    )
    parser.add_argument("--catalog-extension", default="SELECTION")
    parser.add_argument("--catalog-block-size", type=int, default=500_000)
    parser.add_argument("--tf-slope", type=float, default=-7.22)
    parser.add_argument("--tf-intercept", type=float, default=36.0)
    parser.add_argument("--tf-scatter-dex", type=float, default=0.1)
    parser.add_argument(
        "--output",
        help=(
            "sample CSV; required with --test-set so a test population cannot "
            "overwrite the default training proposal"
        ),
    )
    parser.add_argument(
        "--g-limit",
        type=float,
        default=DEFAULT_G_LIMIT,
        help=(
            "Half-width a for training-proposal g1,g2 ~ U[-a, a]; "
            f"must be one of {ALLOWED_G_LIMITS}. Ignored for --test-set."
        ),
    )
    args = parser.parse_args(argv)
    if args.test_set:
        if args.catalog is None:
            parser.error("--catalog is required with --test-set")
        if args.output is None:
            parser.error("--output is required with --test-set")
        if not np.isclose(args.g_limit, DEFAULT_G_LIMIT, rtol=0.0, atol=1e-12):
            parser.error("--g-limit applies only to generate_samples, not --test-set")
    else:
        if args.catalog is not None:
            parser.error("--catalog requires --test-set")
        if args.output is None:
            args.output = DEFAULT_OUTPUT
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest = None
    if args.test_set:
        samples, manifest = generate_test_set_samples(
            args.nsamples,
            catalog=args.catalog,
            seed=args.seed,
            catalog_extension=args.catalog_extension,
            catalog_block_size=args.catalog_block_size,
            tf_slope=args.tf_slope,
            tf_intercept=args.tf_intercept,
            tf_scatter_dex=args.tf_scatter_dex,
        )
    else:
        samples = generate_samples(
            args.nsamples,
            seed=args.seed,
            g_limit=args.g_limit,
        )
    samples.to_csv(args.output, index_label="ID")
    if manifest is not None:
        try:
            from .desi_test_set_sampling import write_generation_manifest
        except ImportError:  # Support direct execution from data_generate/.
            from desi_test_set_sampling import write_generation_manifest
        write_generation_manifest(manifest, args.output)


if __name__ == "__main__":
    main()
