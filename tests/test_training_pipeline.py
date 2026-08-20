import json
import logging
from contextlib import nullcontext

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
from diagnostics.plot_losses import load_losses


class _ModuleWrapper(nn.Module):
    """Small CPU stand-in for DistributedDataParallel."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _TinyCCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.seen_batch_sizes = []

    def forward(
        self,
        image,
        spectra,
        positions,
        *,
        labels,
        observation_context,
        return_diagnostics=False,
    ):
        del spectra, positions
        assert set(observation_context) == set(config.ORACLE_CONTEXT_FIELDS)
        self.seen_batch_sizes.append(image.shape[0])
        loss = (self.weight * image.mean() - labels[:, 0].mean()).square()
        return (loss, {}) if return_diagnostics else loss


class _TinyNPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.feature_extractor = nn.Linear(1, 1)
        self.last_training_diagnostics = {"bounded_logdet_min": torch.tensor(-2.0)}
        self.seen_batch_sizes = []

    def forward(
        self,
        image,
        spectra,
        targets,
        *,
        fiber_positions,
        observation_context,
    ):
        del spectra, fiber_positions
        assert set(observation_context) == set(config.ORACLE_CONTEXT_FIELDS)
        self.seen_batch_sizes.append(image.shape[0])
        return (self.weight * image.mean() - targets[:, 0].mean()).square()


def _bare_batch_trainer(trainer_type, model):
    trainer = object.__new__(trainer_type)
    trainer.device = torch.device("cpu")
    trainer.model = _ModuleWrapper(model)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.use_amp = False
    trainer.amp_dtype = torch.bfloat16
    trainer.channels_last = False
    trainer.gradient_clip_norm = 0.25
    trainer.preclip_grad_norm_history = []
    trainer.training_diagnostic_history = []
    trainer.fid_train = torch.zeros(2, len(config.TARGET_NAMES))
    trainer.fid_valid = trainer.fid_train.clone()
    trainer.fibpos_train = torch.zeros(2, 5, 2)
    trainer.fibpos_valid = trainer.fibpos_train.clone()
    trainer.rmag_train = torch.tensor([18.0, 20.0])
    trainer.rmag_valid = trainer.rmag_train.clone()

    image = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    spectra = torch.zeros(2, 1, 5, 8)
    quality = torch.tensor([5.0, 20.0])
    trainer._noisy_batch = lambda indices, split: (
        image[indices], spectra[indices], quality[indices]
    )
    trainer._all_ranks_true = lambda value: bool(value)
    return trainer


def test_fixed_validation_streams_repeat_while_training_streams_change():
    trainer = object.__new__(train.Trainer)
    trainer.device = torch.device("cpu")
    trainer.base_seed = 1234
    trainer.gpu_id = 0
    trainer.fixed_validation_streams = True

    train_first = torch.rand(
        8, generator=trainer._generator(3, "train_img_noise")
    )
    train_second = torch.rand(
        8, generator=trainer._generator(4, "train_img_noise")
    )
    valid_first = torch.rand(
        8, generator=trainer._generator(3, "valid_img_noise", validation=True)
    )
    valid_second = torch.rand(
        8, generator=trainer._generator(4, "valid_img_noise", validation=True)
    )

    assert not torch.equal(train_first, train_second)
    assert torch.equal(valid_first, valid_second)


def test_configured_dataset_size_selects_a_prefix_and_rejects_oversize():
    dataset = list(range(7))
    limited = train.limit_dataset(dataset, 4, split="training")
    assert [limited[index] for index in range(len(limited))] == [0, 1, 2, 3]
    assert train.limit_dataset(dataset, 7, split="training") is dataset
    with pytest.raises(ValueError, match="contains only 7"):
        train.limit_dataset(dataset, 8, split="training")


def test_multigpu_ccl_converts_batch_norm_and_checkpoint_remains_loadable():
    class _BatchNormBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.image = nn.BatchNorm2d(3)
            self.projector = nn.BatchNorm1d(3)

    ordinary = _BatchNormBackbone()
    optimizer_parameter_ids = {id(parameter) for parameter in ordinary.parameters()}
    synchronized = train._synchronize_pretrain_batch_norm(
        ordinary, train_mode="pretrain", world_size=4
    )
    assert isinstance(synchronized.image, nn.SyncBatchNorm)
    assert isinstance(synchronized.projector, nn.SyncBatchNorm)
    assert {id(parameter) for parameter in synchronized.parameters()} == (
        optimizer_parameter_ids
    )

    restored = _BatchNormBackbone()
    restored.load_state_dict(synchronized.state_dict(), strict=True)
    assert isinstance(restored.image, nn.BatchNorm2d)
    assert not isinstance(restored.image, nn.SyncBatchNorm)

    single = _BatchNormBackbone()
    assert train._synchronize_pretrain_batch_norm(
        single, train_mode="pretrain", world_size=1
    ) is single
    npe = _BatchNormBackbone()
    assert train._synchronize_pretrain_batch_norm(
        npe, train_mode="train", world_size=4
    ) is npe


def _matched_analysis_records(*, corrupt_nuisance=False, corrupt_stencil=False):
    shear = ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1))
    records = []
    for index, (g1, g2) in enumerate(shear):
        truth = torch.zeros(len(config.TARGET_NAMES))
        truth[0], truth[1] = g1, g2
        truth[2:] = torch.arange(2, len(config.TARGET_NAMES), dtype=torch.float32)
        if corrupt_nuisance and index == 3:
            truth[5] += 0.25
        if corrupt_stencil and index == 4:
            truth[1] = -0.2
        records.append(
            {"fid_pars": truth, "rmag_true": 20.0, "halpha_flux_true": 1e-15}
        )
    return records


class _AnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("image_noise_sigma", torch.tensor(1.0))
        self.register_buffer("spectral_reference_line_norm", torch.tensor(1.0))


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (_matched_analysis_records(corrupt_nuisance=True), "non-shear truth"),
        (_matched_analysis_records(corrupt_stencil=True), "shear order"),
    ],
)
def test_matched_analysis_rejects_corrupt_truth_groups(monkeypatch, records, message):
    monkeypatch.setattr(
        train,
        "validate_observation_record",
        lambda record, **_: (record["rmag_true"], record["halpha_flux_true"]),
    )
    with pytest.raises(ValueError, match=message):
        train.sample_density(
            _AnalysisModel(), records, 2, matched_group_size=5
        )


def test_global_mean_reduces_float64_sum_and_count(monkeypatch):
    trainer = object.__new__(train.Trainer)
    trainer.device = torch.device("cpu")
    observed = {}
    monkeypatch.setattr(trainer, "_distributed", lambda: True)

    def fake_all_reduce(tensor, op):
        observed["dtype"] = tensor.dtype
        observed["op"] = op
        tensor.add_(torch.tensor([8.0, 3.0], dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    assert trainer._global_mean(2.0, 1) == pytest.approx(2.5)
    assert observed == {
        "dtype": torch.float64,
        "op": torch.distributed.ReduceOp.SUM,
    }


def test_ccl_identity_r90_pair_is_one_optimizer_step(monkeypatch):
    model = _TinyCCL()
    trainer = _bare_batch_trainer(train.FETrainer, model)
    steps = []
    original_step = trainer.optimizer.step
    monkeypatch.setattr(
        trainer.optimizer,
        "step",
        lambda *args, **kwargs: (steps.append(1), original_step(*args, **kwargs))[1],
    )

    loss, _ = trainer._batch_loss(torch.tensor([0, 1]), "train", train=True)

    assert torch.isfinite(loss)
    assert model.seen_batch_sizes == [4]
    assert len(steps) == 1


def test_ccl_invalid_global_gradient_skips_optimizer_step(monkeypatch):
    model = _TinyCCL()
    trainer = _bare_batch_trainer(train.FETrainer, model)
    steps = []
    monkeypatch.setattr(
        trainer.optimizer,
        "step",
        lambda *args, **kwargs: steps.append(1),
    )
    decisions = iter((True, False))
    trainer._all_ranks_true = lambda value: next(decisions)

    before = model.weight.detach().clone()
    loss, _ = trainer._batch_loss(
        torch.tensor([0, 1]), "train", train=True
    )

    assert torch.isfinite(loss)
    assert not trainer._last_batch_valid
    torch.testing.assert_close(model.weight, before)
    assert not steps


def test_npe_pair_is_one_step_and_invalid_global_gradient_skips(monkeypatch):
    model = _TinyNPE()
    trainer = _bare_batch_trainer(train.NPETrainer, model)
    steps = []
    original_step = trainer.optimizer.step
    monkeypatch.setattr(
        trainer.optimizer,
        "step",
        lambda *args, **kwargs: (steps.append(1), original_step(*args, **kwargs))[1],
    )

    loss, valid = trainer._batch_loss(torch.tensor([0, 1]), "train", train=True)
    assert torch.isfinite(loss) and valid
    assert model.seen_batch_sizes == [4]
    assert len(steps) == 1
    assert trainer.preclip_grad_norm_history
    assert trainer.training_diagnostic_history[-1]["bounded_logdet_min"] == -2.0

    before = model.weight.detach().clone()
    decisions = iter((True, False))
    trainer._all_ranks_true = lambda value: next(decisions)
    _, valid = trainer._batch_loss(torch.tensor([0, 1]), "train", train=True)
    assert not valid
    torch.testing.assert_close(model.weight, before)
    assert len(steps) == 1


def test_amp_unscales_before_gradient_clipping(monkeypatch):
    model = _TinyNPE()
    trainer = _bare_batch_trainer(train.NPETrainer, model)
    trainer.use_amp = True
    events = []

    class _ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class _Scaler:
        def scale(self, loss):
            events.append("scale")
            return _ScaledLoss(loss)

        def unscale_(self, optimizer):
            events.append("unscale")

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")

    trainer.scaler = _Scaler()
    original_clip = nn.utils.clip_grad_norm_

    def checked_clip(parameters, max_norm, error_if_nonfinite):
        events.append(("clip", max_norm))
        return original_clip(
            parameters, max_norm=max_norm, error_if_nonfinite=error_if_nonfinite
        )

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", checked_clip)
    trainer._batch_loss(torch.tensor([0, 1]), "train", train=True)
    assert events == [
        "scale", "backward", "unscale", ("clip", 0.25), "step", "update"
    ]


def test_optimizer_groups_are_unique_and_use_bounded_branch_rates():
    class _Flow(nn.Module):
        def __init__(self):
            super().__init__()
            self.non_theta_flow = nn.Linear(2, 2)
            self.theta_transform = nn.Linear(2, 1)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(1, 2)
            self.flow = _Flow()
            self.frozen = nn.Parameter(torch.ones(()), requires_grad=False)

    model = _Model()
    groups = train._npe_optimizer_parameters(
        model,
        {
            "initial_learning_rate": 3e-4,
            "non_theta_learning_rate": 2e-4,
            "theta_learning_rate": 1e-4,
        },
    )
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in expected
    }
    assert {group["group_name"]: group["lr"] for group in groups} == {
        "shared": pytest.approx(3e-4),
        "non_theta_flow": pytest.approx(2e-4),
        "theta_transform": pytest.approx(1e-4),
    }


@pytest.mark.parametrize(
    ("scheduler_type", "expected_type"),
    [
        ("plateau", ReduceLROnPlateau),
        ("cosine", CosineAnnealingLR),
        ("warmup_cosine", SequentialLR),
    ],
)
def test_current_npe_schedulers(scheduler_type, expected_type, monkeypatch):
    monkeypatch.setitem(config.train, "scheduler_type", scheduler_type)
    monkeypatch.setitem(config.train, "epoch_number", 10)
    monkeypatch.setitem(config.train, "warmup_epochs", 2)
    parameter = nn.Parameter(torch.tensor(1.0))
    trainer = train.NPETrainer(
        1,
        nn.Linear(1, 1),
        [],
        [],
        torch.optim.SGD([parameter], lr=1e-3),
        0,
        1,
        1,
    )
    assert isinstance(trainer.scheduler, expected_type)


def test_best_checkpoint_persists_current_metadata(tmp_path):
    trainer = object.__new__(train.Trainer)
    trainer.stage_config = {"model_path": str(tmp_path)}
    trainer.model_name = "bounded-current"
    trainer.model = _ModuleWrapper(nn.Linear(1, 1))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    trainer._save_checkpoint(
        3, best=True, train_loss=-10.0, valid_loss=-9.5
    )
    directory = tmp_path / "bounded-current"
    state = torch.load(directory / "bounded-currentbest", weights_only=True)
    metadata = json.loads((directory / "best.json").read_text())
    assert "weight" in state
    assert metadata["epoch"] == 4
    assert metadata["epoch_index"] == 3
    assert metadata["validation_loss"] == pytest.approx(-9.5)
    assert metadata["next_epoch_learning_rates"] == pytest.approx([0.01])


def test_epoch_diagnostics_are_logged_and_saved_with_losses(tmp_path):
    trainer = object.__new__(train.Trainer)
    trainer.stage_config = {
        "model_path": str(tmp_path),
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
    }
    trainer.model_name = "diagnostic-run"
    trainer.model = _ModuleWrapper(nn.Linear(1, 1))
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.gpu_id = trainer.log_rank = 0
    trainer.save_every = 1
    trainer.logger = logging.getLogger("diagnostic-test")
    trainer.best_validation_loss = float("inf")
    trainer.epochs_without_improvement = 0
    trainer.epoch_diagnostics = {"train": {}, "valid": {}}
    trainer._set_tensors = lambda: None
    trainer._prepare_epoch = lambda epoch: None
    trainer._step_scheduler = lambda valid_loss: None
    trainer._save_checkpoint = lambda *args, **kwargs: None

    def train_epoch(epoch):
        trainer.epoch_diagnostics["train"] = {
            "target_entropy": 1.25,
            "excess_loss": 0.75,
        }
        return 2.0

    def valid_epoch(epoch):
        trainer.epoch_diagnostics["valid"] = {
            "target_entropy": 1.5,
            "excess_loss": 0.5,
        }
        return 1.8

    trainer._train_epoch = train_epoch
    trainer._valid_epoch = valid_epoch
    trainer.train(1)

    loss_path = tmp_path / "losses" / "losses_diagnostic-run.csv"
    rows = loss_path.read_text()
    assert "train_target_entropy" in rows
    assert "valid_target_entropy" in rows
    assert "train_excess_loss" in rows
    assert "valid_excess_loss" in rows
    train_losses, valid_losses = load_losses(str(loss_path))
    assert train_losses.tolist() == [2.0]
    assert valid_losses.tolist() == [1.8]

    invalid_path = tmp_path / "nonfinite_losses.csv"
    invalid_path.write_text("train,valid,diagnostic\n1.0,2.0,nan\n")
    with pytest.raises(ValueError, match="finite"):
        load_losses(str(invalid_path))
