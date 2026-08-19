import numpy as np
import pytest

from data_generate.latin_hypercube import (
    LEGACY_SUBPIXEL_PARAMETER_LIMITS,
    PARAMETER_LIMITS,
    generate_samples,
    parse_args,
)
from data_generate.observation_schema import (
    CURRENT_OBSERVATION_MODEL_VERSION,
    DEFAULT_HALPHA_FLUX_RANGE,
    FIBER_LAYOUT_COLUMN,
    GALAXY_AXIS_FIBER_LAYOUT,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_AXIS_FIBER_LAYOUT,
    LEGACY_OBSERVATION_MODEL_VERSION,
    OBSERVATION_MODEL_VERSION_COLUMN,
    RMAG_TRUE_COLUMN,
)


ORIGINAL_SIMULATION_PARAMETERS = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)


def test_v2_samples_exactly_eight_physical_parameters_and_two_nuisances():
    samples = generate_samples(
        4096,
        seed=23017,
        observation_model_version=CURRENT_OBSERVATION_MODEL_VERSION,
        rmag_range=(16.0, 23.4),
        halpha_flux_range=(2.0e-16, 250.0e-16),
    )

    assert tuple(PARAMETER_LIMITS) == ORIGINAL_SIMULATION_PARAMETERS
    assert list(samples.columns) == [
        *ORIGINAL_SIMULATION_PARAMETERS,
        RMAG_TRUE_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        FIBER_LAYOUT_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
    ]
    assert samples[RMAG_TRUE_COLUMN].between(16.0, 23.4, inclusive="neither").all()
    assert samples[HALPHA_FLUX_TRUE_COLUMN].between(
        2.0e-16,
        250.0e-16,
        inclusive="neither",
    ).all()
    assert (samples[OBSERVATION_MODEL_VERSION_COLUMN] == 2).all()
    assert samples[OBSERVATION_MODEL_VERSION_COLUMN].dtype == np.int16
    assert (samples[FIBER_LAYOUT_COLUMN] == GALAXY_AXIS_FIBER_LAYOUT).all()

    # Fixed-seed statistical guards against deriving either nuisance from a
    # physical parameter or from one another.
    correlations = samples[
        ["vcirc", RMAG_TRUE_COLUMN, HALPHA_FLUX_TRUE_COLUMN]
    ].corr()
    assert abs(correlations.loc["vcirc", RMAG_TRUE_COLUMN]) < 0.05
    assert abs(correlations.loc["vcirc", HALPHA_FLUX_TRUE_COLUMN]) < 0.05
    assert abs(
        correlations.loc[RMAG_TRUE_COLUMN, HALPHA_FLUX_TRUE_COLUMN]
    ) < 0.05


def test_default_halpha_flux_range_matches_requested_desi_kl_range():
    assert DEFAULT_HALPHA_FLUX_RANGE == pytest.approx(
        (1.2e-16, 301.43e-16),
        rel=1.0e-12,
        abs=0.0,
    )


def test_no_argument_cli_preserves_legacy_observation_schema():
    args = parse_args([])

    assert args.observation_model_version == LEGACY_OBSERVATION_MODEL_VERSION
    assert args.fiber_layout is None
    assert args.output.endswith("samples_pretrain_5m.csv")


def test_rmag_range_only_changes_magnitude_samples():
    common = dict(
        nsamples=256,
        seed=9876,
        observation_model_version=CURRENT_OBSERVATION_MODEL_VERSION,
    )
    narrow = generate_samples(**common, rmag_range=(19.0, 20.0))
    broad = generate_samples(**common, rmag_range=(15.0, 23.4))

    for name in PARAMETER_LIMITS:
        np.testing.assert_array_equal(narrow[name], broad[name])
    np.testing.assert_array_equal(
        narrow[HALPHA_FLUX_TRUE_COLUMN],
        broad[HALPHA_FLUX_TRUE_COLUMN],
    )
    assert not np.array_equal(narrow[RMAG_TRUE_COLUMN], broad[RMAG_TRUE_COLUMN])


def test_halpha_range_only_changes_halpha_samples():
    common = dict(
        nsamples=256,
        seed=9876,
        observation_model_version=CURRENT_OBSERVATION_MODEL_VERSION,
    )
    narrow = generate_samples(
        **common,
        halpha_flux_range=(10.0e-16, 20.0e-16),
    )
    broad = generate_samples(
        **common,
        halpha_flux_range=(1.2e-16, 301.43e-16),
    )

    for name in PARAMETER_LIMITS:
        np.testing.assert_array_equal(narrow[name], broad[name])
    np.testing.assert_array_equal(narrow[RMAG_TRUE_COLUMN], broad[RMAG_TRUE_COLUMN])
    assert not np.array_equal(
        narrow[HALPHA_FLUX_TRUE_COLUMN],
        broad[HALPHA_FLUX_TRUE_COLUMN],
    )


def test_sample_generation_is_reproducible_with_explicit_seed():
    first = generate_samples(128, seed=91)
    second = generate_samples(128, seed=91)

    for column in first:
        np.testing.assert_array_equal(first[column], second[column])


def test_legacy_samples_keep_old_physical_schema_and_version_marker():
    samples = generate_samples(
        32,
        seed=5,
        observation_model_version=LEGACY_OBSERVATION_MODEL_VERSION,
    )

    assert list(samples.columns) == [
        *PARAMETER_LIMITS,
        *LEGACY_SUBPIXEL_PARAMETER_LIMITS,
        FIBER_LAYOUT_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
    ]
    assert RMAG_TRUE_COLUMN not in samples
    assert HALPHA_FLUX_TRUE_COLUMN not in samples
    assert (samples[OBSERVATION_MODEL_VERSION_COLUMN] == 1).all()
    assert (samples[FIBER_LAYOUT_COLUMN] == IMAGE_AXIS_FIBER_LAYOUT).all()


def test_explicit_fiber_layout_overrides_version_default():
    samples = generate_samples(
        16,
        seed=8,
        observation_model_version=CURRENT_OBSERVATION_MODEL_VERSION,
        fiber_layout=IMAGE_AXIS_FIBER_LAYOUT,
    )

    assert (samples[FIBER_LAYOUT_COLUMN] == IMAGE_AXIS_FIBER_LAYOUT).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nsamples": 0}, "nsamples must be positive"),
        ({"nsamples": 8, "observation_model_version": 3}, "must be 1 or 2"),
        ({"nsamples": 8, "rmag_range": (20.0, 20.0)}, "finite increasing"),
        ({"nsamples": 8, "rmag_range": (np.nan, 23.0)}, "finite increasing"),
        (
            {"nsamples": 8, "halpha_flux_range": (2.0e-16, 2.0e-16)},
            "finite increasing",
        ),
        (
            {"nsamples": 8, "halpha_flux_range": (np.nan, 2.0e-16)},
            "finite increasing",
        ),
        ({"nsamples": 8, "fiber_layout": "diagonal"}, "fiber_layout"),
    ],
)
def test_invalid_sample_requests_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_samples(**kwargs)
