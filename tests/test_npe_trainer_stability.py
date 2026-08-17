import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    SequentialLR,
)

import config
import train


class _TinyLikelihood(nn.Module):
    mode = 0

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, img, spec, fid, fp=None):
        prediction = self.weight * img.reshape(img.shape[0], -1).mean(dim=1)
        return (prediction - fid[:, 0]).square().mean()


class _ModuleWrapper(nn.Module):
    """Small CPU stand-in for DistributedDataParallel."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _bare_npe_trainer(*, use_amp=False):
    trainer = object.__new__(train.NPETrainer)
    trainer.device = torch.device("cpu")
    trainer.model = _ModuleWrapper(_TinyLikelihood())
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.use_amp = use_amp
    trainer.amp_dtype = torch.bfloat16
    trainer.gradient_clip_norm = 0.25
    trainer.invalid_loss_count = 0
    trainer.invalid_gradient_count = 0
    trainer._preclip_grad_norm_sum = 0.0
    trainer._preclip_grad_norm_count = 0
    trainer._preclip_grad_norm_max = 0.0
    return trainer


def _batch():
    return (
        torch.tensor([[[[1.0]]], [[[2.0]]]]),
        torch.zeros((2, 1, 1, 1)),
        torch.tensor([[0.0], [1.0]]),
    )


def test_global_metric_reduces_float64_sum_and_count(monkeypatch):
    trainer = object.__new__(train.Trainer)
    trainer.device = torch.device("cpu")
    observed = {}

    monkeypatch.setattr(trainer, "_distributed_is_initialized", lambda: True)

    def fake_all_reduce(tensor, op):
        observed["dtype"] = tensor.dtype
        observed["op"] = op
        # The remote rank contributes a sum of 8 over three valid batches.
        tensor.add_(torch.tensor([8.0, 3.0], dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    result = trainer._global_mean_from_sum_count(2.0, 1)

    assert result == pytest.approx(2.5)
    assert observed["dtype"] == torch.float64
    assert observed["op"] == torch.distributed.ReduceOp.SUM


def test_fixed_validation_streams_repeat_while_training_streams_change(
    monkeypatch,
):
    trainer = object.__new__(train.Trainer)
    trainer.device = torch.device("cpu")
    trainer.base_seed = 1234
    trainer.gpu_id = 0
    trainer.deterministic = False
    trainer.fixed_validation_streams = True
    monkeypatch.setattr(train, "seed_everything", lambda *args, **kwargs: None)

    trainer._reset_epoch_rngs(3)
    train_first = torch.rand(
        8, generator=trainer.epoch_generators["train_img_noise"]
    )
    valid_first = torch.rand(
        8, generator=trainer.epoch_generators["valid_img_noise"]
    )
    valid_numpy_first = trainer.epoch_numpy_generators["valid"].normal(size=8)

    trainer._reset_epoch_rngs(4)
    train_second = torch.rand(
        8, generator=trainer.epoch_generators["train_img_noise"]
    )
    valid_second = torch.rand(
        8, generator=trainer.epoch_generators["valid_img_noise"]
    )
    valid_numpy_second = trainer.epoch_numpy_generators["valid"].normal(size=8)

    assert not torch.equal(train_first, train_second)
    assert torch.equal(valid_first, valid_second)
    np.testing.assert_array_equal(valid_numpy_first, valid_numpy_second)


def test_amp_unscales_before_configurable_clipping(monkeypatch):
    trainer = _bare_npe_trainer(use_amp=True)
    events = []

    class ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class FakeScaler:
        def scale(self, loss):
            events.append("scale")
            return ScaledLoss(loss)

        def unscale_(self, optimizer):
            events.append("unscale")

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self, new_scale=None):
            events.append("update")

        def get_scale(self):
            return 128.0

        def get_backoff_factor(self):
            return 0.5

    trainer.scaler = FakeScaler()
    original_clip = nn.utils.clip_grad_norm_

    def checked_clip(parameters, max_norm, error_if_nonfinite):
        events.append(("clip", max_norm))
        return original_clip(
            parameters,
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
        )

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", checked_clip)
    img, spec, fid = _batch()

    trainer._run_batch(img, spec, fid, mode="train")

    assert events == [
        "scale",
        "backward",
        "unscale",
        ("clip", 0.25),
        "step",
        "update",
    ]
    assert np.isfinite(trainer.last_preclip_grad_norm)


def test_globally_invalid_gradient_skips_optimizer_step(monkeypatch):
    trainer = _bare_npe_trainer(use_amp=False)
    decisions = iter((True, False))  # finite loss, invalid gradient on one rank
    monkeypatch.setattr(trainer, "_all_ranks_true", lambda value: next(decisions))
    before = trainer.model.module.weight.detach().clone()
    img, spec, fid = _batch()

    trainer._run_batch(img, spec, fid, mode="train")

    torch.testing.assert_close(trainer.model.module.weight, before)
    assert trainer.last_batch_step_valid is False
    assert trainer.invalid_gradient_count == 1


def test_warmup_cosine_and_legacy_plateau_scheduler(monkeypatch):
    parameter = nn.Parameter(torch.tensor(1.0))

    warmup_trainer = object.__new__(train.NPETrainer)
    warmup_trainer.optimizer = torch.optim.SGD([parameter], lr=3e-4)
    warmup_trainer.scheduler_type = "warmup_cosine"
    monkeypatch.setitem(config.train, "warmup_epochs", 2)
    monkeypatch.setitem(config.train, "min_learning_rate", 1e-5)
    warmup_trainer._configure_scheduler(total_epochs=10)
    assert isinstance(warmup_trainer.scheduler, SequentialLR)

    cosine_trainer = object.__new__(train.NPETrainer)
    cosine_trainer.optimizer = torch.optim.SGD(
        [nn.Parameter(torch.tensor(2.0))], lr=3e-4
    )
    cosine_trainer.scheduler_type = "cosine"
    cosine_trainer._configure_scheduler(total_epochs=10)
    assert isinstance(cosine_trainer.scheduler, CosineAnnealingLR)

    invalid_trainer = object.__new__(train.NPETrainer)
    invalid_trainer.optimizer = torch.optim.SGD(
        [nn.Parameter(torch.tensor(3.0))], lr=3e-4
    )
    invalid_trainer.scheduler_type = "warmup_cosine"
    monkeypatch.setitem(config.train, "warmup_epochs", 10)
    with pytest.raises(ValueError, match="smaller than epoch_number"):
        invalid_trainer._configure_scheduler(total_epochs=10)

    plateau_trainer = object.__new__(train.NPETrainer)
    plateau_trainer.optimizer = torch.optim.SGD([parameter], lr=3e-4)
    plateau_trainer.scheduler_type = "plateau"
    plateau_trainer._configure_scheduler(total_epochs=10)
    assert isinstance(plateau_trainer.scheduler, ReduceLROnPlateau)
    assert plateau_trainer.scheduler.patience == 10


def test_early_stopping_respects_min_delta_and_patience(monkeypatch):
    trainer = object.__new__(train.Trainer)
    trainer.device = torch.device("cpu")
    trainer.gpu_id = 0
    trainer.log_rank = 0
    trainer.best_validation_loss = float("inf")
    trainer.epochs_without_improvement = 0
    trainer.early_stopping_min_delta = 0.1
    trainer.early_stopping_patience = 2
    monkeypatch.setattr(trainer, "_distributed_is_initialized", lambda: False)

    assert trainer._update_training_control(1.0) == (True, False)
    assert trainer._update_training_control(0.95) == (False, False)
    assert trainer._update_training_control(0.94) == (False, True)
    assert trainer.best_validation_loss == pytest.approx(1.0)


def test_best_checkpoint_has_json_metadata(tmp_path, monkeypatch):
    trainer = _bare_npe_trainer(use_amp=False)
    trainer.scheduler_type = "warmup_cosine"
    trainer.early_stopping_min_delta = 1e-3
    trainer.early_stopping_patience = 5
    trainer.logger = train.logging.getLogger("best-checkpoint-test")
    monkeypatch.setitem(config.train, "model_path", str(tmp_path))
    monkeypatch.setitem(config.train, "model_name", "hybrid-test")

    trainer._save_best_checkpoint(epoch=3, train_loss=-10.0, valid_loss=-9.5)

    checkpoint_path = tmp_path / "hybrid-test" / "hybrid-testbest"
    metadata_path = tmp_path / "hybrid-test" / "best.json"
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text())
    torch.testing.assert_close(restored["weight"], trainer.model.module.weight)
    assert metadata["epoch"] == 4
    assert metadata["epoch_index"] == 3
    assert metadata["validation_loss"] == pytest.approx(-9.5)
    assert metadata["checkpoint"] == "hybrid-test3"
    assert metadata["checkpoint_suffix"] == "3"
    assert metadata["named_best_checkpoint"] == "hybrid-testbest"
    assert metadata["named_best_checkpoint_suffix"] == "best"
    assert metadata["next_epoch_learning_rates"] == pytest.approx([0.1])


def test_hybrid_optimizer_groups_are_complete_unique_and_use_branch_lrs():
    class HybridFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.affine_flow = nn.Linear(2, 2)
            self.theta_transform = nn.Linear(2, 1)

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(1, 2)
            self.flow = HybridFlow()
            self.frozen = nn.Parameter(torch.ones(()), requires_grad=False)

    model = HybridModel()
    groups = train._npe_optimizer_parameters(
        model,
        {
            "initial_learning_rate": 3e-4,
            "affine_learning_rate": 3e-4,
            "theta_learning_rate": 1e-4,
        },
    )
    grouped_parameters = [
        parameter for group in groups for parameter in group["params"]
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert len(grouped_parameters) == len({id(p) for p in grouped_parameters})
    assert {id(p) for p in grouped_parameters} == {id(p) for p in expected}
    learning_rates = {group["group_name"]: group["lr"] for group in groups}
    assert learning_rates == {
        "shared": pytest.approx(3e-4),
        "affine_flow": pytest.approx(3e-4),
        "theta_transform": pytest.approx(1e-4),
    }


def test_training_health_diagnostics_use_expected_epoch_reductions():
    trainer = _bare_npe_trainer(use_amp=False)
    trainer._reset_training_diagnostics()
    trainer.model.module.last_training_diagnostics = {
        "raw_feature_rms": torch.tensor(2.0),
        "effective_sample_size": torch.tensor(10.0),
        "effective_sample_fraction": torch.tensor(0.2),
        "max_normalized_weight": torch.tensor(5.0),
        "theta_raw_logit_abs_max": torch.tensor(7.0),
        "theta_bounded_logit_abs_max": torch.tensor(4.0),
        "theta_logdet_min": torch.tensor(-3.0),
        "theta_logdet_max": torch.tensor(2.0),
        "affine_log_prob_mean": torch.tensor(-4.0),
        "theta_log_prob_mean": torch.tensor(1.0),
        "theta_derivative_min": torch.tensor(0.2),
        "theta_derivative_max": torch.tensor(3.0),
        "theta_wrap_count": torch.tensor(1.0),
        "theta_max_wrap_excursion": torch.tensor(1e-7),
    }
    trainer._capture_training_diagnostics()
    trainer.model.module.last_training_diagnostics = {
        "raw_feature_rms": torch.tensor(4.0),
        "effective_sample_size": torch.tensor(6.0),
        "effective_sample_fraction": torch.tensor(0.1),
        "max_normalized_weight": torch.tensor(8.0),
        "theta_raw_logit_abs_max": torch.tensor(9.0),
        "theta_bounded_logit_abs_max": torch.tensor(5.0),
        "theta_logdet_min": torch.tensor(-6.0),
        "theta_logdet_max": torch.tensor(3.0),
        "affine_log_prob_mean": torch.tensor(-2.0),
        "theta_log_prob_mean": torch.tensor(0.0),
        "theta_derivative_min": torch.tensor(0.1),
        "theta_derivative_max": torch.tensor(4.0),
        "theta_wrap_count": torch.tensor(2.0),
        "theta_max_wrap_excursion": torch.tensor(2e-7),
    }
    trainer._capture_training_diagnostics()

    diagnostics = trainer._finalize_training_diagnostics()

    assert diagnostics["raw_feature_rms"] == pytest.approx(3.0)
    assert diagnostics["effective_sample_size"] == pytest.approx(8.0)
    assert diagnostics["effective_sample_fraction"] == pytest.approx(0.15)
    assert diagnostics["max_normalized_weight"] == pytest.approx(8.0)
    assert diagnostics["theta_raw_logit_abs_max"] == pytest.approx(9.0)
    assert diagnostics["theta_bounded_logit_abs_max"] == pytest.approx(5.0)
    assert diagnostics["theta_logdet_min"] == pytest.approx(-6.0)
    assert diagnostics["theta_logdet_max"] == pytest.approx(3.0)
    assert diagnostics["affine_log_prob_mean"] == pytest.approx(-3.0)
    assert diagnostics["theta_log_prob_mean"] == pytest.approx(0.5)
    assert diagnostics["theta_derivative_min"] == pytest.approx(0.1)
    assert diagnostics["theta_derivative_max"] == pytest.approx(4.0)
    assert diagnostics["theta_wrap_count"] == pytest.approx(2.0)
    assert diagnostics["theta_max_wrap_excursion"] == pytest.approx(2e-7)


def test_bounded_flow_diagnostics_use_expected_epoch_reductions():
    """Compact-flow means and extrema retain their physical meaning."""
    trainer = _bare_npe_trainer(use_amp=False)
    trainer._reset_training_diagnostics()
    trainer.model.module.last_training_diagnostics = {
        "bounded_log_prob_mean": torch.tensor(-5.0),
        "bounded_support_violation_count": torch.tensor(0.0),
        "bounded_raw_logit_abs_max": torch.tensor(7.0),
        "bounded_logit_abs_max": torch.tensor(4.0),
        "bounded_derivative_min": torch.tensor(0.2),
        "bounded_derivative_max": torch.tensor(3.0),
        "bounded_logdet_min": torch.tensor(-3.0),
        "bounded_logdet_max": torch.tensor(2.0),
    }
    trainer._capture_training_diagnostics()
    trainer.model.module.last_training_diagnostics = {
        "bounded_log_prob_mean": torch.tensor(-3.0),
        "bounded_support_violation_count": torch.tensor(2.0),
        "bounded_raw_logit_abs_max": torch.tensor(9.0),
        "bounded_logit_abs_max": torch.tensor(5.0),
        "bounded_derivative_min": torch.tensor(0.1),
        "bounded_derivative_max": torch.tensor(4.0),
        "bounded_logdet_min": torch.tensor(-6.0),
        "bounded_logdet_max": torch.tensor(3.0),
    }
    trainer._capture_training_diagnostics()

    diagnostics = trainer._finalize_training_diagnostics()

    assert diagnostics["bounded_log_prob_mean"] == pytest.approx(-4.0)
    assert diagnostics["bounded_support_violation_count"] == pytest.approx(2.0)
    assert diagnostics["bounded_raw_logit_abs_max"] == pytest.approx(9.0)
    assert diagnostics["bounded_logit_abs_max"] == pytest.approx(5.0)
    assert diagnostics["bounded_derivative_min"] == pytest.approx(0.1)
    assert diagnostics["bounded_derivative_max"] == pytest.approx(4.0)
    assert diagnostics["bounded_logdet_min"] == pytest.approx(-6.0)
    assert diagnostics["bounded_logdet_max"] == pytest.approx(3.0)
