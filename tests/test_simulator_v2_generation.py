import ast
from pathlib import Path

import numpy as np
import pytest

from data_generate.latin_hypercube import (
    DEFAULT_RMAG_RANGE,
    PARAMETER_LIMITS,
    generate_samples,
    parse_args,
)
from data_generate.observation_schema import (
    CENTRAL_HALPHA_SNR_COLUMN,
    DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
    DEFAULT_HALPHA_FLUX_RANGE,
    DEFAULT_HALPHA_LOG10_FLUX_RANGE,
    DEFAULT_IMAGE_SNR_RANGE,
    FIBER_LAYOUT_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_SNR_COLUMN,
    OBSERVATION_MODEL_VERSION_COLUMN,
    RMAG_TRUE_COLUMN,
)
ROOT = Path(__file__).resolve().parents[1]
FIXED_CENTERING = ("dx_disk", "dy_disk", "dx_spec", "dy_spec")


TARGETS_8 = ("g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr")


def test_proposal_has_eight_targets_plus_independent_observation_controls():
    samples = generate_samples(4096, seed=23017)
    assert tuple(PARAMETER_LIMITS) == TARGETS_8
    assert tuple(samples) == (
        *TARGETS_8,
        RMAG_TRUE_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
    )
    assert not any(name.startswith(("dx_", "dy_")) for name in samples)
    for name, (low, high) in PARAMETER_LIMITS.items():
        assert samples[name].between(low, high, inclusive="neither").all()
    assert samples[RMAG_TRUE_COLUMN].between(*DEFAULT_RMAG_RANGE, inclusive="neither").all()
    assert samples[HALPHA_FLUX_TRUE_COLUMN].between(
        *DEFAULT_HALPHA_FLUX_RANGE, inclusive="neither"
    ).all()
    assert samples[IMAGE_SNR_COLUMN].between(
        *DEFAULT_IMAGE_SNR_RANGE, inclusive="neither"
    ).all()
    assert samples[CENTRAL_HALPHA_SNR_COLUMN].between(
        *DEFAULT_CENTRAL_HALPHA_SNR_RANGE, inclusive="neither"
    ).all()
    assert (samples[FIBER_LAYOUT_COLUMN] == "galaxy_axis").all()
    assert (samples[OBSERVATION_MODEL_VERSION_COLUMN] == 3).all()
    independent = samples[
        [
            "vcirc",
            RMAG_TRUE_COLUMN,
            IMAGE_SNR_COLUMN,
            CENTRAL_HALPHA_SNR_COLUMN,
        ]
    ]
    correlations = independent.corr()
    assert np.max(np.abs(correlations.to_numpy()[np.triu_indices(4, 1)])) < 0.06


def test_halpha_is_lhs_uniform_in_log_flux_and_snrs_are_linear_uniform():
    count = 257
    samples = generate_samples(count, seed=381)
    quantities = (
        (
            np.log10(samples[HALPHA_FLUX_TRUE_COLUMN]),
            DEFAULT_HALPHA_LOG10_FLUX_RANGE,
        ),
        (samples[IMAGE_SNR_COLUMN], DEFAULT_IMAGE_SNR_RANGE),
        (samples[CENTRAL_HALPHA_SNR_COLUMN], DEFAULT_CENTRAL_HALPHA_SNR_RANGE),
    )
    for values, (lower, upper) in quantities:
        strata = np.floor(count * (values - lower) / (upper - lower)).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(count))


@pytest.mark.parametrize("name", ("rmag_range", "halpha_flux_range"))
def test_generation_api_rejects_range_overrides(name):
    with pytest.raises(TypeError):
        generate_samples(2, seed=91, **{name: (1.0, 2.0)})


def test_generation_is_seed_reproducible_and_cli_has_no_schema_switches():
    first = generate_samples(128, seed=12)
    second = generate_samples(128, seed=12)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    args = parse_args([])
    assert not hasattr(args, "observation_model_version")
    assert not hasattr(args, "fiber_layout")
    assert not hasattr(args, "rmag_min")
    assert not hasattr(args, "halpha_flux_min")
    assert args.nsamples == 100_000
    assert args.output.endswith(
        "samples_train_1m_simv3_galaxyaxis_central_halpha.csv"
    )
    for removed in (
        "--rmag-min", "--rmag-max", "--halpha-flux-min", "--halpha-flux-max"
    ):
        with pytest.raises(SystemExit):
            parse_args([removed, "19"])


def test_generator_supplies_fixed_zero_centering_to_renderer_theta():
    tree = ast.parse(
        (ROOT / "data_generate" / "generate_fits.py").read_text()
    )
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"sampled_pars", "sampled_pars_value_dict"}
    }

    sampled_pars = tuple(ast.literal_eval(assignments["sampled_pars"]))
    assert sampled_pars == (*TARGETS_8, *FIXED_CENTERING)

    value_node = assignments["sampled_pars_value_dict"]
    fixed_values = {
        ast.literal_eval(key): ast.literal_eval(value)
        for key, value in zip(value_node.keys, value_node.values)
        if ast.literal_eval(key) in FIXED_CENTERING
    }
    assert fixed_values == {name: 0.0 for name in FIXED_CENTERING}


def test_nonpositive_sample_count_is_rejected():
    with pytest.raises(ValueError):
        generate_samples(0)
