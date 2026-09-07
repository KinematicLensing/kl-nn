"""Contracts for the additive Xu-map geometric NPE."""

import copy

import pytest
import torch

import config
import train
from data import rotate_90_datavector, rotate_90_parameters
from geom_arch import (
    EPS_V_KMS,
    GeometricKLNPE,
    GeometricStatEncoder,
    _physical_vcirc,
    odd_axis_velocities,
    photometric_quadrupole,
    thin_disk_e_int,
    wavelength_grid_nm,
    wavelength_to_velocity_kms,
    xu_reduced_shear,
)
from networks import BoundedHybridCircularFlow, ImgCNN, JointSpecCNN
from utils import denormalize


@pytest.fixture(autouse=True)
def _restore_model_config():
    original = copy.deepcopy(config.MODEL_CONFIG)
    yield
    config.set_model_config(original)


def _oracle(batch_size=2):
    return {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size),
        "image_snr": torch.linspace(50.0, 500.0, batch_size),
        "central_halpha_snr": torch.linspace(20.0, 120.0, batch_size),
    }


def _gaussian_ellipse(batch_size, q, theta, sigma=6.0, size=48):
    yy = torch.arange(size, dtype=torch.float32)
    xx = torch.arange(size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    x = grid_x - 0.5 * (size - 1)
    y = grid_y - 0.5 * (size - 1)
    cos_t = torch.cos(torch.as_tensor(theta))
    sin_t = torch.sin(torch.as_tensor(theta))
    xp = cos_t * x + sin_t * y
    yp = -sin_t * x + cos_t * y
    image = torch.exp(
        -0.5 * ((xp / sigma) ** 2 + (yp / (q * sigma)) ** 2)
    )
    return image.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, size, size).contiguous()


def _line_spectra(velocities_kms):
    batch, n_fibers = velocities_kms.shape
    grid = wavelength_grid_nm(dtype=velocities_kms.dtype)
    lam_c = 656.4589 * 1.3 * (1.0 + velocities_kms / 299792.458)
    delta = grid[None, None, :] - lam_c[..., None]
    flux = torch.exp(-0.5 * (delta / 0.08) ** 2)
    return flux.unsqueeze(1)


def _axis_fibers(theta, offset=1.5):
    major = torch.tensor([offset * torch.cos(theta), offset * torch.sin(theta)])
    minor = torch.tensor(
        [offset * torch.cos(theta + 0.5 * torch.pi), offset * torch.sin(theta + 0.5 * torch.pi)]
    )
    return torch.stack(
        (major, -major, torch.zeros(2), minor, -minor), dim=0
    ).unsqueeze(0)


def test_physical_vcirc_matches_nine_target_denormalize():
    normalized = torch.linspace(-0.9, 0.9, 9).unsqueeze(0)
    got = _physical_vcirc(normalized)
    full = denormalize(
        normalized,
        config.par_ranges,
        feature_names=tuple(config.TARGET_NAMES),
        target_transforms=config.TARGET_TRANSFORMS,
    )
    index = list(config.TARGET_NAMES).index("vcirc")
    torch.testing.assert_close(got, full[..., index])
    assert config.TARGET_TRANSFORMS["vcirc"] == "identity"


def test_quadrupole_is_finite_on_fp16_noisy_bright_images():
    image = _gaussian_ellipse(4, q=0.55, theta=0.3, sigma=8.0) * 400.0
    image = image + torch.randn_like(image) * 60.0
    _e1, _e2, e_obs = photometric_quadrupole(image.half())
    assert torch.isfinite(e_obs).all()
    assert float(e_obs.max()) < 1.0


def test_thin_disk_e_int_matches_axis_ratio():
    cosi = torch.tensor([0.5, 1.0, 0.0])
    e_int = thin_disk_e_int(cosi)
    q = cosi
    expected = (1.0 - q) / (1.0 + q)
    torch.testing.assert_close(e_int, expected)


def test_xu_map_recovers_signed_cross_shear():
    sini = torch.tensor([0.8])
    cosi = torch.sqrt(1.0 - sini * sini)
    vcirc = torch.tensor([220.0])
    e_int = thin_disk_e_int(cosi)
    e_obs = e_int + 0.02
    gcross = torch.tensor([0.04])
    v_major = -vcirc * sini
    v_minor = (
        vcirc
        * sini
        * cosi
        * (1.0 + (1.0 + e_obs * e_obs) / (2.0 * e_int))
        * gcross
    )
    theta = torch.tensor([0.3])
    g1, g2 = xu_reduced_shear(e_obs, theta, v_major, v_minor, vcirc)
    two = 2.0 * theta
    recovered_cross = g1 * torch.sin(two) + g2 * torch.cos(two)
    torch.testing.assert_close(recovered_cross, gcross, atol=2e-4, rtol=0)


def test_xu_map_plus_has_correct_sign_and_order():
    cosi = torch.tensor([0.6])
    e_int = thin_disk_e_int(cosi)
    vcirc = torch.tensor([200.0])
    sini = torch.sqrt(1.0 - cosi * cosi)
    v_major = -vcirc * sini
    v_minor = torch.zeros_like(v_major)
    theta = torch.zeros(1)
    e_obs = e_int + 0.05
    g1, g2 = xu_reduced_shear(e_obs, theta, v_major, v_minor, vcirc)
    assert float(g1) > 0.0
    assert abs(float(g2)) < 1.0e-5
    assert 0.005 < float(g1) < 0.2


def test_xu_map_clamps_face_on_and_tiny_v_major():
    e_obs = torch.tensor([0.2, 0.2])
    theta = torch.zeros(2)
    v_major = torch.tensor([0.0, 0.1])
    v_minor = torch.tensor([5.0, 5.0])
    vcirc = torch.tensor([200.0, 200.0])
    g1, g2 = xu_reduced_shear(e_obs, theta, v_major, v_minor, vcirc)
    assert torch.allclose(g1, torch.zeros(2))
    assert torch.allclose(g2, torch.zeros(2))
    assert float(v_major[1]) < EPS_V_KMS


def test_quadrupole_recovers_gaussian_ellipticity():
    q = 0.5
    theta = torch.tensor(0.4)
    image = _gaussian_ellipse(1, q=q, theta=theta, sigma=8.0)
    e1, e2, e_obs = photometric_quadrupole(
        image, pixel_scale_arcsec=0.2637, psf_fwhm_arcsec=1.0
    )
    expected_e = (1.0 - q) / (1.0 + q)
    assert abs(float(e_obs) - expected_e) < 0.05
    recovered_theta = 0.5 * torch.atan2(e2, e1)
    delta = 0.5 * torch.atan2(
        torch.sin(2.0 * (recovered_theta - theta)),
        torch.cos(2.0 * (recovered_theta - theta)),
    )
    assert abs(float(delta)) < 0.08


def test_stats_and_xu_recover_injected_cross_shear():
    theta = torch.tensor(0.25)
    q = 0.65
    vcirc = torch.tensor([210.0])
    cosi = torch.as_tensor(q)
    sini = torch.sqrt(1.0 - cosi * cosi)
    e_int = thin_disk_e_int(cosi)
    image = _gaussian_ellipse(1, q=q, theta=theta, sigma=8.0)
    fibers = _axis_fibers(theta)
    _e1, _e2, e_obs = photometric_quadrupole(image)
    gcross = torch.tensor([0.035])
    v0 = torch.tensor([3.0])
    v_maj = -vcirc * sini
    v_min = (
        vcirc
        * sini
        * cosi
        * (1.0 + (1.0 + e_obs * e_obs) / (2.0 * e_int))
        * gcross
    )
    velocities = torch.stack(
        (v0 + v_maj, v0 - v_maj, v0, v0 + v_min, v0 - v_min), dim=-1
    )
    spectra = _line_spectra(velocities)
    encoder = GeometricStatEncoder().eval()
    stats = encoder.sufficient_stats(image, spectra, fibers)
    g1, g2 = xu_reduced_shear(
        stats["e_obs"],
        stats["theta_phot"],
        stats["v_major"],
        stats["v_minor"],
        vcirc,
    )
    two = 2.0 * stats["theta_phot"]
    recovered_cross = g1 * torch.sin(two) + g2 * torch.cos(two)
    recovered_plus = g1 * torch.cos(two) - g2 * torch.sin(two)
    torch.testing.assert_close(recovered_cross, gcross, atol=3e-3, rtol=0)
    assert abs(float(recovered_plus.detach())) < 0.05


def test_centroids_recover_injected_velocities():
    v0 = torch.tensor([8.0])
    v_maj = torch.tensor([-90.0])
    v_min = torch.tensor([6.0])
    velocities = torch.stack(
        (
            v0 + v_maj,
            v0 - v_maj,
            v0,
            v0 + v_min,
            v0 - v_min,
        ),
        dim=-1,
    )
    spectra = _line_spectra(velocities)
    grid = wavelength_grid_nm()
    flux = spectra[:, 0]
    total = flux.sum(dim=-1, keepdim=True)
    centroid = (flux * grid).sum(dim=-1, keepdim=True) / total
    recovered = wavelength_to_velocity_kms(centroid.squeeze(-1))
    got_v0, got_maj, got_min = odd_axis_velocities(recovered)
    torch.testing.assert_close(got_v0, v0, atol=0.5, rtol=0)
    torch.testing.assert_close(got_maj, v_maj, atol=0.5, rtol=0)
    torch.testing.assert_close(got_min, v_min, atol=0.5, rtol=0)


def test_xu_map_commutes_with_r90_datavector():
    theta = torch.tensor(0.35)
    q = 0.55
    vcirc = torch.tensor([240.0])
    sini = torch.sqrt(1.0 - torch.as_tensor(q) ** 2)
    v_maj = -vcirc * sini
    v_min = torch.tensor([12.0])
    v0 = torch.tensor([4.0])
    image = _gaussian_ellipse(1, q=q, theta=theta)
    fibers = _axis_fibers(theta)
    velocities = torch.stack(
        (v0 + v_maj, v0 - v_maj, v0, v0 + v_min, v0 - v_min), dim=-1
    )
    spectra = _line_spectra(velocities)
    _e1, _e2, e_obs = photometric_quadrupole(image)
    g1, g2 = xu_reduced_shear(e_obs, theta.unsqueeze(0), v_maj, v_min, vcirc)
    image_r, spectra_r, _, fibers_r = rotate_90_datavector(
        image, spectra, fiber_positions=fibers
    )
    _er1, _er2, e_obs_r = photometric_quadrupole(image_r)
    theta_r = torch.atan2(fibers_r[0, 0, 1], fibers_r[0, 0, 0])
    flux = spectra_r[:, 0]
    grid = wavelength_grid_nm()
    centroid = (flux * grid).sum(dim=-1) / flux.sum(dim=-1)
    vel_r = wavelength_to_velocity_kms(centroid)
    _, v_maj_r, v_min_r = odd_axis_velocities(vel_r)
    g1_r, g2_r = xu_reduced_shear(e_obs_r, theta_r.unsqueeze(0), v_maj_r, v_min_r, vcirc)
    torch.testing.assert_close(g1_r, -g1, atol=5e-3, rtol=0)
    torch.testing.assert_close(g2_r, -g2, atol=5e-3, rtol=0)
    parameters = torch.zeros(1, 9)
    parameters[0, 0] = g1 / 0.1
    parameters[0, 1] = g2 / 0.1
    rotated = rotate_90_parameters(parameters)
    torch.testing.assert_close(rotated[0, 0], -parameters[0, 0])
    torch.testing.assert_close(rotated[0, 1], -parameters[0, 1])


def test_kl_geom_does_not_instantiate_concat_towers():
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.architecture = "kl_geom"
    config.set_model_config(configured)
    model = GeometricKLNPE()
    assert not any(
        isinstance(module, (ImgCNN, JointSpecCNN)) for module in model.modules()
    )
    assert isinstance(model.flow, BoundedHybridCircularFlow)
    image = _gaussian_ellipse(2, q=0.6, theta=0.2)
    theta = torch.tensor(0.2)
    fibers = _axis_fibers(theta).expand(2, -1, -1).contiguous()
    velocities = torch.tensor(
        [[-80.0, 80.0, 3.0, 5.0, -5.0], [-70.0, 70.0, -2.0, 4.0, -4.0]]
    )
    spectra = _line_spectra(velocities)
    targets = torch.rand(2, 9) * 1.6 - 0.8
    targets[:, 5] = 0.2
    loss = model(
        image,
        spectra,
        targets,
        fiber_positions=fibers,
        observation_context=_oracle(2),
    )
    assert torch.isfinite(loss)
    assert "ghat_rms" in model.last_training_diagnostics
    samples = model.sample(
        image,
        spectra,
        4,
        fiber_positions=fibers,
        observation_context=_oracle(2),
    )
    assert samples.shape == (2, 4, 9)
    assert bool((samples.abs() < 1.0).all())


def test_kl_geom_compose_roundtrip_and_optimizer_groups():
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.architecture = "kl_geom"
    config.set_model_config(configured)
    model = GeometricKLNPE().eval()
    image = _gaussian_ellipse(1, q=0.5, theta=0.1)
    fibers = _axis_fibers(torch.tensor(0.1))
    velocities = torch.tensor([[0.2, -0.2, 0.0, 0.1, -0.1]])
    spectra = _line_spectra(velocities)
    context = _oracle(1)
    samples, sample_lp = model.sample(
        image,
        spectra,
        5,
        fiber_positions=fibers,
        observation_context=context,
        return_log_prob=True,
    )
    scored = model.posterior_log_prob(
        image, spectra, samples[0], fibers, context
    )
    torch.testing.assert_close(scored, sample_lp[0], atol=2e-4, rtol=2e-4)
    groups = train._npe_optimizer_parameters(model, config.train)
    assert [group["group_name"] for group in groups] == [
        "shared",
        "non_theta_flow",
        "theta_transform",
    ]
    assert all(len(group["params"]) > 0 for group in groups)


def test_kl_geom_channels_last_is_disabled():
    stage = {"channels_last": True, "architecture": "kl_geom"}
    assert train._use_channels_last(stage) is False
    concat = {"channels_last": True, "architecture": "concat"}
    assert train._use_channels_last(concat) is True


def test_kl_geom_compile_is_skipped_even_when_requested():
    class _Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compile_calls = []

        def compile(self, **kwargs):
            self.compile_calls.append(kwargs)

    model = _Recorder()
    compiled = train._maybe_compile_model(
        model,
        {"use_compile": True, "architecture": "kl_geom"},
    )
    assert compiled is model
    assert model.compile_calls == []


def test_load_model_resolves_kl_geom_class_from_snapshot(tmp_path):
    import model_registry

    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.model_name = "kl-geom-load"
    configured.train.architecture = "kl_geom"
    model_registry.save_model_artifacts(
        configured,
        configs_root=str(tmp_path / "configs"),
        networks_root=str(tmp_path / "networks"),
    )
    config.set_model_config(configured)
    model = GeometricKLNPE()
    checkpoint = tmp_path / "models" / "kl-geom-load" / "kl-geom-loadbest"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint)
    restored = train.load_model(
        GeometricKLNPE,
        path=str(checkpoint),
        model_name="kl-geom-load",
        networks_root=str(tmp_path / "networks"),
    )
    assert type(restored).__name__ == "GeometricKLNPE"
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
