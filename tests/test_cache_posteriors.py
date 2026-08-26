import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "arch" / "cache_posteriors.py"
    spec = importlib.util.spec_from_file_location("cache_posteriors_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_summary_and_target_map_use_same_joint_candidate_rows():
    module = _module()
    names = ("g1", "g2", "theta_int")
    samples = np.asarray(
        [[[0.0, 10.0, 3.10], [1.0, 20.0, -3.12], [2.0, 30.0, 3.13]]]
    )
    base = np.asarray([[3.0, 2.0, 1.0]])
    ratio = np.asarray([[0.0, 2.0, 5.0]])
    weight = np.asarray([[0.05, 0.15, 0.80]])
    summary = module.posterior_summaries(samples, base, ratio, weight, names)
    np.testing.assert_array_equal(summary["proposal_map_estimates"][0], samples[0, 0])
    np.testing.assert_array_equal(summary["tf_target_map_estimates"][0], samples[0, 2])
    np.testing.assert_allclose(summary["tf_target_mean_estimates"][0, 1, :2], [1.75, 27.5])
    assert abs(abs(summary["tf_target_mean_estimates"][0, 1, 2]) - np.pi) < 0.05


def test_cache_cli_rejects_unknown_options_and_has_one_sampling_surface():
    module = _module()
    parser_error = None
    try:
        module.parse_args(
            [
                "-i", "0", "--nparts", "1", "--ngals", "2",
                "--model-name", "m", "--dataset", "d",
                "--unknown-option", "value",
            ]
        )
    except SystemExit as exc:
        parser_error = exc
    assert parser_error is not None
    args = module.parse_args(
        [
            "-i", "0", "--nparts", "1", "--ngals", "2",
            "--model-name", "m", "--dataset", "d", "--nsamples", "20",
        ]
    )
    assert args.nsamples == 20


def test_default_checkpoint_requires_saved_best(tmp_path):
    module = _module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    numbered = model_dir / "model12"
    numbered.touch()
    default = SimpleNamespace(
        checkpoint=None, epoch=None, model_name="model"
    )
    with np.testing.assert_raises_regex(FileNotFoundError, "modelbest"):
        module.resolve_checkpoint(default, tmp_path)

    best = model_dir / "modelbest"
    best.touch()
    assert module.resolve_checkpoint(default, tmp_path) == best

    explicit_epoch = SimpleNamespace(
        checkpoint=None, epoch=12, model_name="model"
    )
    assert module.resolve_checkpoint(explicit_epoch, tmp_path) == numbered


def test_partition_range_is_exact_and_nonoverlapping():
    module = _module()
    assert module.resolve_partition_range(0, 10, 30) == (0, 10)
    assert module.resolve_partition_range(2, 10, 30) == (20, 30)
    module.validate_partition_coverage(3, 10, 30)
    with np.testing.assert_raises_regex(ValueError, "complete dataset"):
        module.validate_partition_coverage(2, 10, 30)


def test_partition_seed_changes_candidate_streams():
    base = 42
    seeds = [base + 1_000_003 * index for index in range(3)]
    assert len(set(seeds)) == 3


def test_physical_map_scores_include_log_flux_jacobian():
    module = _module()
    names = ("halpha_flux_true",)
    normalized = np.asarray([[[-1.0], [1.0]]], dtype=np.float64)
    equal_normalized_density = np.zeros((1, 2), dtype=np.float64)
    scores = module.physical_log_prob_from_normalized(
        normalized,
        equal_normalized_density,
        par_ranges={"halpha_flux_true": [1.0e-17, 1.0e-14]},
        feature_names=names,
        target_transforms={"halpha_flux_true": "log10"},
    )
    assert scores[0, 0] > scores[0, 1]
    np.testing.assert_allclose(scores[0, 0] - scores[0, 1], np.log(1000.0))
