import os
import random

import numpy as np
import pytest
import torch

from train import RNG_STREAM_IDS, derive_stream_seed, seed_everything


@pytest.fixture(autouse=True)
def _restore_determinism_state():
    algorithms = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    if python_hash_seed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = python_hash_seed
    if cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas


def test_seed_everything_repeats_python_numpy_and_torch_draws():
    seed_everything(123456)
    first = (random.random(), np.random.random(), torch.rand(5))

    seed_everything(123456)
    second = (random.random(), np.random.random(), torch.rand(5))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert os.environ["PYTHONHASHSEED"] == "123456"


def test_current_rng_stream_ids_are_stable_distinct_and_tf_free():
    assert RNG_STREAM_IDS == {
        "ambient": 0,
        "train_order": 1,
        "valid_order": 2,
        "train_img_noise": 3,
        "train_spec_noise": 4,
        "valid_img_noise": 5,
        "valid_spec_noise": 6,
        "train_spec_quality": 7,
        "valid_spec_quality": 8,
        "train_npe_view": 9,
        "valid_npe_view": 10,
        "train_image_snr": 11,
        "train_central_halpha_snr": 12,
        "valid_image_snr": 13,
        "valid_central_halpha_snr": 14,
    }
    first = {
        stream: derive_stream_seed(20260810, rank=2, epoch=7, stream=stream)
        for stream in RNG_STREAM_IDS
    }
    repeated = {
        stream: derive_stream_seed(20260810, rank=2, epoch=7, stream=stream)
        for stream in RNG_STREAM_IDS
    }

    assert first == repeated
    assert len(set(first.values())) == len(first)
    assert all("tf" not in stream.lower() for stream in RNG_STREAM_IDS)
    assert first["train_spec_quality"] != derive_stream_seed(
        20260810, rank=3, epoch=7, stream="train_spec_quality"
    )
    assert first["train_spec_quality"] != derive_stream_seed(
        20260810, rank=2, epoch=8, stream="train_spec_quality"
    )
    with pytest.raises(ValueError, match="unknown RNG stream"):
        derive_stream_seed(1, stream="unknown_stream")


def test_explicit_torch_streams_repeat_independently():
    image_seed = derive_stream_seed(9, rank=0, epoch=4, stream="train_img_noise")
    spectrum_seed = derive_stream_seed(
        9, rank=0, epoch=4, stream="train_spec_noise"
    )

    image_1 = torch.randn(16, generator=torch.Generator().manual_seed(image_seed))
    image_2 = torch.randn(16, generator=torch.Generator().manual_seed(image_seed))
    spectrum = torch.randn(
        16, generator=torch.Generator().manual_seed(spectrum_seed)
    )

    assert torch.equal(image_1, image_2)
    assert not torch.equal(image_1, spectrum)


def test_deterministic_flag_controls_torch_backends():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed_everything(77, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    seed_everything(77, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
