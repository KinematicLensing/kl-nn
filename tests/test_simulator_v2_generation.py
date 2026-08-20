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
    DEFAULT_HALPHA_FLUX_RANGE,
    FIBER_LAYOUT_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    OBSERVATION_MODEL_VERSION_COLUMN,
    RMAG_TRUE_COLUMN,
)
ROOT = Path(__file__).resolve().parents[1]
FIXED_CENTERING = ("dx_disk", "dy_disk", "dx_spec", "dy_spec")


TARGETS_8 = ("g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr")


def test_proposal_is_exactly_eight_parameters_plus_two_independent_latents():
    samples = generate_samples(4096, seed=23017)
    assert tuple(PARAMETER_LIMITS) == TARGETS_8
    assert tuple(samples) == (
        *TARGETS_8,
        RMAG_TRUE_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
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
    assert (samples[FIBER_LAYOUT_COLUMN] == "galaxy_axis").all()
    assert (samples[OBSERVATION_MODEL_VERSION_COLUMN] == 2).all()
    correlations = samples[["vcirc", RMAG_TRUE_COLUMN, HALPHA_FLUX_TRUE_COLUMN]].corr()
    assert np.max(np.abs(correlations.to_numpy()[np.triu_indices(3, 1)])) < 0.06


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
        "samples_valid_1m_simv2_galaxyaxis_halpha.csv"
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
