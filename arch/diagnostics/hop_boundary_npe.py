#!/usr/bin/env python3
"""Matched-noise NPE comparison of hopped vs unhopped round-looking galaxies."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from data import (
    central_halpha_line_norm,
    image_matched_filter_norm,
    noise_sigma_from_target_snr,
)
from model_registry import load_model_config
from networks import KLNPE
from train import (
    _seeded_generator,
    build_observation_levels,
    load_model,
    seed_everything,
    validate_observation_record,
)
from utils import denormalize, resolve_feature_index

from data_generate.observation_schema import (
    classify_fiber_hop,
    major_axis_hop_angle_deg,
    observed_ellipticity,
)

DATA_ROOT = Path("/ocean/projects/phy250048p/shared/datasets")
MODEL_ROOT = Path("/ocean/projects/phy250048p/shared/models")
REPORT_DIR = Path(
    "/ocean/projects/phy250048p/shared/reports/fiber-gauge/07_hop_boundary"
)
G02_VAL = DATA_ROOT / "small_10k_simv3_cosi_g02"
G02_NPE = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_train100k_g02_s42_45802965"
ROUND_CUTS = (0.08, 0.09, 0.10)
MIN_GROUP_COUNT = 40
N_REALIZATIONS = 4
N_SAMPLES = 2048
SNR_FACTOR = 1.2
ABS_G_BIN_WIDTH = 0.01
FIGURE_DPI = 140
GROUP_ORDER = ("unhopped", "hopped")
GROUP_LABELS = {
    "unhopped": "fibers stayed put",
    "hopped": "fibers hopped",
}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=G02_VAL)
    parser.add_argument("--model-name", default=G02_NPE)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--checkpoint-suffix", default="best")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-galaxies",
        action="store_true",
        help="Rebuild figures and HTML from galaxies.npz without sampling the NPE",
    )
    return parser.parse_args(argv)


def checkpoint_file(model_root: Path, model_name: str, suffix: str = "best") -> Path:
    return Path(model_root) / model_name / f"{model_name}{suffix}"


def sini_from_cosi(cosi: np.ndarray) -> np.ndarray:
    cosi = np.asarray(cosi, dtype=np.float64)
    return np.sqrt(np.clip(1.0 - cosi * cosi, 0.0, 1.0))


def catalog_geometry(physical: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    g1 = physical[:, resolve_feature_index(names, "g1")]
    g2 = physical[:, resolve_feature_index(names, "g2")]
    theta = physical[:, resolve_feature_index(names, "theta_int")]
    cosi = physical[:, resolve_feature_index(names, "cosi")]
    sini = sini_from_cosi(cosi)
    ellipticity = np.empty(len(physical), dtype=np.float64)
    hop_angle = np.empty(len(physical), dtype=np.float64)
    hop_class = np.empty(len(physical), dtype=object)
    for index in range(len(physical)):
        ellipticity[index] = observed_ellipticity(
            g1=float(g1[index]),
            g2=float(g2[index]),
            theta_int=float(theta[index]),
            sini=float(sini[index]),
        )
        hop_class[index] = classify_fiber_hop(
            g1=float(g1[index]),
            g2=float(g2[index]),
            theta_int=float(theta[index]),
            sini=float(sini[index]),
        )
        hop_angle[index] = major_axis_hop_angle_deg(
            g1=float(g1[index]),
            g2=float(g2[index]),
            theta_int=float(theta[index]),
            sini=float(sini[index]),
        )
    return {
        "g1": g1,
        "g2": g2,
        "theta_int": theta,
        "cosi": cosi,
        "sini": sini,
        "abs_g": np.hypot(g1, g2),
        "ellipticity": ellipticity,
        "hop_angle_deg": hop_angle,
        "hop_class": hop_class,
    }


def choose_ellipticity_cut(
    ellipticity: np.ndarray,
    hop_class: np.ndarray,
    *,
    cuts=ROUND_CUTS,
    min_count: int = MIN_GROUP_COUNT,
) -> float:
    last = float(cuts[-1])
    for cut in cuts:
        hopped = int(
            np.sum((ellipticity < cut) & (hop_class == "hopped"))
        )
        unhopped = int(
            np.sum((ellipticity < cut) & (hop_class == "unhopped"))
        )
        if hopped >= min_count and unhopped >= min_count:
            return float(cut)
    return last


def selected_mask(ellipticity: np.ndarray, hop_class: np.ndarray, cut: float) -> np.ndarray:
    return (ellipticity < cut) & np.isin(hop_class, list(GROUP_ORDER))


def abs_g_overlap(abs_g: np.ndarray, hop_class: np.ndarray) -> tuple[float, float] | None:
    hopped = abs_g[hop_class == "hopped"]
    unhopped = abs_g[hop_class == "unhopped"]
    if hopped.size == 0 or unhopped.size == 0:
        return None
    low = float(max(hopped.min(), unhopped.min()))
    high = float(min(hopped.max(), unhopped.max()))
    if high <= low:
        return None
    return low, high


def abs_g_bin_index(
    abs_g: np.ndarray, *, width: float = ABS_G_BIN_WIDTH
) -> np.ndarray:
    values = np.asarray(abs_g, dtype=np.float64)
    if width <= 0.0:
        raise ValueError("abs_g bin width must be positive")
    return np.floor(values / width).astype(np.int64)


def match_abs_g_counts(
    abs_g: np.ndarray,
    hop_class: np.ndarray,
    *,
    width: float = ABS_G_BIN_WIDTH,
    seed: int = 42,
) -> np.ndarray:
    """Downsample so hopped and unhopped have the same count in each |g| bin."""

    abs_g = np.asarray(abs_g, dtype=np.float64)
    hop_class = np.asarray(hop_class)
    if abs_g.shape != hop_class.shape:
        raise ValueError("abs_g and hop_class must have the same length")
    bins = abs_g_bin_index(abs_g, width=width)
    keep = np.zeros(abs_g.shape[0], dtype=bool)
    rng = np.random.default_rng(seed)
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        hopped = np.flatnonzero(in_bin & (hop_class == "hopped"))
        unhopped = np.flatnonzero(in_bin & (hop_class == "unhopped"))
        n_keep = min(hopped.size, unhopped.size)
        if n_keep == 0:
            continue
        if hopped.size > n_keep:
            hopped = rng.choice(hopped, size=n_keep, replace=False)
        if unhopped.size > n_keep:
            unhopped = rng.choice(unhopped, size=n_keep, replace=False)
        keep[hopped] = True
        keep[unhopped] = True
    return keep


def linear_slope(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.size < 2 or np.allclose(truth, truth[0]):
        return float("nan"), float("nan")
    design = np.column_stack((truth, np.ones_like(truth)))
    slope, intercept = np.linalg.lstsq(design, prediction, rcond=None)[0]
    return float(slope), float(intercept)


def circular_residual(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return (prediction - truth + np.pi) % (2.0 * np.pi) - np.pi


def group_metrics(truth: dict[str, np.ndarray], estimate: dict[str, np.ndarray]) -> dict:
    g1_res = estimate["g1"] - truth["g1"]
    g2_res = estimate["g2"] - truth["g2"]
    theta_res = circular_residual(truth["theta_int"], estimate["theta_int"])
    g1_slope, g1_intercept = linear_slope(truth["g1"], estimate["g1"])
    g2_slope, g2_intercept = linear_slope(truth["g2"], estimate["g2"])
    return {
        "n": int(len(truth["g1"])),
        "g1_bias": float(np.mean(g1_res)),
        "g2_bias": float(np.mean(g2_res)),
        "g1_rms": float(np.sqrt(np.mean(g1_res ** 2))),
        "g2_rms": float(np.sqrt(np.mean(g2_res ** 2))),
        "g1_slope": g1_slope,
        "g2_slope": g2_slope,
        "g1_intercept": g1_intercept,
        "g2_intercept": g2_intercept,
        "g1_m": float(g1_slope - 1.0) if np.isfinite(g1_slope) else float("nan"),
        "g2_m": float(g2_slope - 1.0) if np.isfinite(g2_slope) else float("nan"),
        "sigma_g1": float(np.mean(estimate["sigma_g1"])),
        "sigma_g2": float(np.mean(estimate["sigma_g2"])),
        "mean_cos_theta": float(np.mean(np.cos(theta_res))),
        "theta_rmse": float(np.sqrt(np.mean(theta_res ** 2))),
        "mean_abs_g": float(np.mean(truth["abs_g"])),
        "mean_sini": float(np.mean(truth["sini"])),
        "mean_hlr": float(np.mean(truth["hlr"])),
        "mean_ellipticity": float(np.mean(truth["ellipticity"])),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def label_distributions(truth: dict) -> dict:
    payload = {}
    for name in GROUP_ORDER:
        mask = truth["hop_class"] == name
        payload[name] = {
            "abs_g": distribution_summary(truth["abs_g"][mask]),
            "sini": distribution_summary(truth["sini"][mask]),
            "hlr": distribution_summary(truth["hlr"][mask]),
            "ellipticity": distribution_summary(truth["ellipticity"][mask]),
        }
    return payload


def mean_or_nan(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def clip_snr(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
        }
    )
    return plt


def scan_catalog(dataset, par_ranges, names: tuple[str, ...]) -> dict:
    n = len(dataset)
    physical = np.empty((n, len(names)), dtype=np.float64)
    image_snr = np.empty(n, dtype=np.float64)
    spec_snr = np.empty(n, dtype=np.float64)
    hlr = np.empty(n, dtype=np.float64)
    for index in range(n):
        record = dataset[index]
        rmag, halpha, image_snr_i, spec_snr_i = validate_observation_record(
            record, location=f"catalog record {index}"
        )
        del rmag, halpha
        fid = np.asarray(record["fid_pars"], dtype=np.float64)
        physical[index] = denormalize(fid, par_ranges, feature_names=names)
        image_snr[index] = image_snr_i
        spec_snr[index] = spec_snr_i
        hlr[index] = physical[index, resolve_feature_index(names, "hlr")]
        if index == 0 or index + 1 == n or (index + 1) % 1000 == 0:
            print(f"geometry: scanned {index + 1}/{n}", flush=True)
    geometry = catalog_geometry(physical, names)
    geometry["hlr"] = hlr
    geometry["record_image_snr"] = image_snr
    geometry["record_spec_snr"] = spec_snr
    geometry["physical"] = physical
    return geometry


def load_selected_tensors(dataset, indices: np.ndarray, device: torch.device) -> dict:
    images = []
    spectra = []
    positions = []
    rmag = []
    for index in indices:
        record = dataset[int(index)]
        images.append(torch.as_tensor(record["img"]).float())
        spectra.append(torch.as_tensor(record["spec"]).float())
        positions.append(torch.as_tensor(record["fib_pos"]).float())
        rmag.append(
            float(validate_observation_record(record, location=f"load {index}")[0])
        )
    return {
        "img": torch.stack(images).to(device),
        "spec": torch.stack(spectra).to(device),
        "fib_pos": torch.stack(positions).to(device),
        "rmag": torch.tensor(rmag, device=device, dtype=torch.float32),
    }


def apply_shared_noise(
    tensors: dict,
    *,
    image_snr: float,
    spec_snr: float,
    image_seed: int,
    spec_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    img = tensors["img"]
    spec = tensors["spec"]
    image_gen = _seeded_generator(device, image_seed)
    spec_gen = _seeded_generator(device, spec_seed)
    unit_image = torch.randn(
        (1, *img.shape[1:]),
        device=device,
        dtype=img.dtype,
        generator=image_gen,
    )
    image_norm = image_matched_filter_norm(img)
    image_sigma = noise_sigma_from_target_snr(
        image_norm,
        torch.full_like(image_norm, float(image_snr)),
        name="image_snr",
    ).reshape((-1,) + (1,) * (img.ndim - 1))
    noisy_image = img + unit_image * image_sigma

    line_norm = central_halpha_line_norm(
        spec, center_fiber_index=config.observation["center_fiber_index"]
    )
    center_sigma = noise_sigma_from_target_snr(
        line_norm,
        torch.full_like(line_norm, float(spec_snr)),
        name="central_halpha_snr",
    )
    offset_ratio = math.sqrt(
        config.observation["offset_exposure_s"]
        / config.observation["center_exposure_s"]
    )
    fiber_sigma = center_sigma[:, None].expand(-1, spec.shape[-2]).clone()
    offset_mask = torch.ones(spec.shape[-2], dtype=torch.bool, device=device)
    offset_mask[int(config.observation["center_fiber_index"])] = False
    fiber_sigma[:, offset_mask] *= offset_ratio
    unit_spec = torch.randn(
        (1, *spec.shape[1:]),
        device=device,
        dtype=spec.dtype,
        generator=spec_gen,
    )
    valid = spec != 0
    noisy_spec = spec + unit_spec * fiber_sigma[:, None, :, None] * valid
    return noisy_image, noisy_spec


def sample_means(
    model,
    tensors: dict,
    noisy_image: torch.Tensor,
    noisy_spec: torch.Tensor,
    *,
    image_snr: float,
    spec_snr: float,
    n_samples: int,
    batch_size: int,
    channels_last: bool,
    names: tuple[str, ...],
    par_ranges,
) -> dict[str, np.ndarray]:
    n = noisy_image.shape[0]
    g1_idx = resolve_feature_index(names, "g1")
    g2_idx = resolve_feature_index(names, "g2")
    theta_idx = resolve_feature_index(names, "theta_int")
    means = np.empty((n, len(names)), dtype=np.float64)
    sigma_g1 = np.empty(n, dtype=np.float64)
    sigma_g2 = np.empty(n, dtype=np.float64)
    snr_image, snr_spec = build_observation_levels(
        torch.full((n,), image_snr, device=noisy_image.device),
        torch.full((n,), spec_snr, device=noisy_image.device),
    )
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        image = noisy_image[start:stop]
        spec = noisy_spec[start:stop]
        positions = tensors["fib_pos"][start:stop]
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spec = spec.contiguous(memory_format=torch.channels_last)
        context = {
            "rmag_true": tensors["rmag"][start:stop],
            "image_snr": snr_image[start:stop],
            "central_halpha_snr": snr_spec[start:stop],
        }
        with torch.inference_mode():
            samples = model.sample(
                image,
                spec,
                n_samples,
                fiber_positions=positions,
                observation_context=context,
            )
        if samples.ndim != 3:
            raise RuntimeError(f"unexpected sample shape {tuple(samples.shape)}")
        physical = denormalize(
            samples.float().cpu(), par_ranges, feature_names=names
        ).numpy()
        mean = physical.mean(axis=1)
        theta = physical[..., theta_idx]
        mean[:, theta_idx] = np.arctan2(
            np.sin(theta).mean(axis=1), np.cos(theta).mean(axis=1)
        )
        means[start:stop] = mean
        sigma_g1[start:stop] = physical[..., g1_idx].std(axis=1, ddof=0)
        sigma_g2[start:stop] = physical[..., g2_idx].std(axis=1, ddof=0)
        print(f"sampled galaxies {stop}/{n}", flush=True)
    return {
        "g1": means[:, g1_idx],
        "g2": means[:, g2_idx],
        "theta_int": means[:, theta_idx],
        "sigma_g1": sigma_g1,
        "sigma_g2": sigma_g2,
    }


def average_realizations(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    stacked = {key: np.stack([row[key] for row in rows], axis=0) for key in keys}
    out = {key: stacked[key].mean(axis=0) for key in keys if key != "theta_int"}
    theta = stacked["theta_int"]
    out["theta_int"] = np.arctan2(np.sin(theta).mean(axis=0), np.cos(theta).mean(axis=0))
    return out


def subset_truth(geometry: dict, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "g1": geometry["g1"][mask],
        "g2": geometry["g2"][mask],
        "theta_int": geometry["theta_int"][mask],
        "sini": geometry["sini"][mask],
        "hlr": geometry["hlr"][mask],
        "abs_g": geometry["abs_g"][mask],
        "ellipticity": geometry["ellipticity"][mask],
        "hop_class": geometry["hop_class"][mask],
        "hop_angle_deg": geometry["hop_angle_deg"][mask],
    }


def subset_estimate(estimate: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in estimate.items()}


def compare_groups(truth: dict, estimate: dict) -> dict:
    payload = {}
    for name in GROUP_ORDER:
        group_mask = truth["hop_class"] == name
        payload[name] = group_metrics(
            {key: value[group_mask] for key, value in truth.items() if key != "hop_class"},
            subset_estimate(estimate, group_mask),
        )
        payload[name]["hop_class"] = name
    return payload


def save_figures(
    truth: dict,
    estimate: dict,
    report_dir: Path,
    *,
    matched_truth: dict | None = None,
) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures = {}
    for key, xlabel, filename in (
        ("abs_g", "true shear amplitude", "abs_g.png"),
        ("sini", "sine of inclination", "sini.png"),
        ("hlr", "half-light radius", "hlr.png"),
    ):
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        for name, color in (("unhopped", "#1f4e79"), ("hopped", "#a31f34")):
            values = truth[key][truth["hop_class"] == name]
            axis.hist(
                values,
                bins=18,
                histtype="step",
                color=color,
                lw=1.6,
                label=GROUP_LABELS[name],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("number of galaxies")
        axis.legend(frameon=False)
        fig.tight_layout()
        path = report_dir / filename
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        figures[key] = path.name

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for axis, component in zip(axes, ("g1", "g2")):
        for name, color, marker in (
            ("unhopped", "#1f4e79", "o"),
            ("hopped", "#a31f34", "s"),
        ):
            mask = truth["hop_class"] == name
            axis.scatter(
                truth[component][mask],
                estimate[component][mask],
                s=22,
                color=color,
                marker=marker,
                label=GROUP_LABELS[name],
                alpha=0.85,
            )
        lo = float(np.min(truth[component]))
        hi = float(np.max(truth[component]))
        pad = 0.05 * max(hi - lo, 0.02)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#bbbbbb", lw=1)
        axis.set_xlabel(f"true {component}")
        axis.set_ylabel(f"mean recovered {component}")
        axis.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    path = report_dir / "g_recovery.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["g_recovery"] = path.name

    if matched_truth is not None and matched_truth["abs_g"].size:
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        max_g = float(np.max(matched_truth["abs_g"]))
        edges = np.arange(0.0, max_g + 2.0 * ABS_G_BIN_WIDTH, ABS_G_BIN_WIDTH)
        for name, color in (("unhopped", "#1f4e79"), ("hopped", "#a31f34")):
            values = matched_truth["abs_g"][matched_truth["hop_class"] == name]
            axis.hist(
                values,
                bins=edges,
                histtype="step",
                color=color,
                lw=1.6,
                label=GROUP_LABELS[name],
            )
        axis.set_xlabel("true shear amplitude")
        axis.set_ylabel("number of galaxies")
        axis.legend(frameon=False)
        fig.tight_layout()
        path = report_dir / "abs_g_matched.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        figures["abs_g_matched"] = path.name
    return figures


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    full = payload["full"]
    matched = payload.get("matched")
    rows = []
    for name in GROUP_ORDER:
        metrics = full[name]
        rows.append(
            "<tr>"
            f"<td>{html.escape(GROUP_LABELS[name])}</td>"
            f"<td>{metrics['n']}</td>"
            f"<td>{fmt(metrics['g1_m'])}</td>"
            f"<td>{fmt(metrics['g2_m'])}</td>"
            f"<td>{fmt(metrics['g1_bias'])}</td>"
            f"<td>{fmt(0.5 * (metrics['sigma_g1'] + metrics['sigma_g2']))}</td>"
            f"<td>{fmt(metrics['mean_cos_theta'])}</td>"
            f"<td>{fmt(metrics['theta_rmse'])}</td>"
            f"<td>{fmt(metrics['mean_abs_g'])}</td>"
            "</tr>"
        )
    makeup_rows = []
    for name in GROUP_ORDER:
        metrics = full[name]
        makeup_rows.append(
            "<tr>"
            f"<td>{html.escape(GROUP_LABELS[name])}</td>"
            f"<td>{metrics['n']}</td>"
            f"<td>{fmt(metrics['mean_abs_g'])}</td>"
            f"<td>{fmt(metrics['mean_sini'])}</td>"
            f"<td>{fmt(metrics['mean_hlr'])}</td>"
            f"<td>{fmt(metrics['mean_ellipticity'])}</td>"
            "</tr>"
        )
    cut_note = payload.get("ellipticity_cut_note", "")
    matched_rows = []
    matched_note = ""
    if matched is not None:
        for name in GROUP_ORDER:
            metrics = matched["metrics"][name]
            matched_rows.append(
                "<tr>"
                f"<td>{html.escape(GROUP_LABELS[name])}</td>"
                f"<td>{metrics['n']}</td>"
                f"<td>{fmt(metrics['g1_m'])}</td>"
                f"<td>{fmt(metrics['g2_m'])}</td>"
                f"<td>{fmt(metrics['g1_bias'])}</td>"
                f"<td>{fmt(0.5 * (metrics['sigma_g1'] + metrics['sigma_g2']))}</td>"
                f"<td>{fmt(metrics['mean_cos_theta'])}</td>"
                f"<td>{fmt(metrics['theta_rmse'])}</td>"
                f"<td>{fmt(metrics['mean_abs_g'])}</td>"
                "</tr>"
            )
        matched_note = (
            "The control keeps the same number of galaxies in each 0.01-wide "
            "shear-amplitude bin, by randomly dropping the extra galaxies in "
            f"the larger group. That leaves {matched['metrics']['unhopped']['n']} "
            "galaxies on each side."
        )
    else:
        matched_note = (
            "The two groups do not share any 0.01-wide shear-amplitude bin, "
            "so the matched-count control is omitted."
        )

    def takeaway(block: dict) -> str:
        hopped = block["hopped"]
        unhopped = block["unhopped"]
        m_gap = abs(hopped["g1_m"] - unhopped["g1_m"])
        cos_gap = abs(hopped["mean_cos_theta"] - unhopped["mean_cos_theta"])
        if m_gap < 0.05 and cos_gap < 0.05:
            return (
                "The two groups are nearly the same in shear slope and in "
                "position-angle recovery. A discrete fiber hop is a weak "
                "explanation of the worse wide-shear slope."
            )
        return (
            "The hopped and unhopped groups do not agree. The shear slope, "
            "the width, or the position-angle recovery jumps across the "
            "axis swap. That supports two spectrograph meters, and a single "
            "network that splits the difference."
        )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Round galaxies: hopped versus unhopped</title>
<style>
body {{ font: 17px/1.55 Palatino, "Palatino Linotype", serif; margin: 2rem auto; max-width: 980px; color: #1b1b1b; }}
h1, h2, h3 {{ font-weight: 600; }}
h1 {{ font-size: 1.85rem; }}
h2 {{ margin-top: 2.4rem; }}
p.lead {{ font-size: 1.08rem; }}
figure {{ margin: 1.4rem 0 2rem; }}
figcaption {{ font-size: 0.92rem; color: #444; margin-top: 0.45rem; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.55rem; }}
th {{ background: #f4f4f4; text-align: left; }}
.note {{ color: #444; }}
</style></head><body>
<h1>Do hopped round galaxies come back with a different shear and position angle?</h1>
<p class="lead">At large shear the spectrograph fibers can swap onto a new pair
of axes. This page asks whether a network trained out to shear 0.2 treats those
hopped galaxies differently from round-looking galaxies whose fibers stayed
put. The stamps are the same noiseless validation images. The noise is not:
every galaxy is observed at one shared S/N, a little above the catalog median,
and the four noise draws are shared across the sample.</p>

<h2>Who was selected</h2>
<p>From the ten-thousand-galaxy validation list we kept galaxies whose
observed ellipse is rounder than {fmt(payload['ellipticity_cut'])}.
{html.escape(cut_note)}
A galaxy is hopped if its major-axis fibers have rotated by more than
45 degrees relative to the unsheared placement. Galaxies within four
degrees of that cut are dropped, because the axis label is unstable there.
That leaves {full['unhopped']['n']} unhopped and {full['hopped']['n']} hopped
galaxies.</p>
<table>
<thead><tr>
<th>Group</th><th>N</th><th>mean |g|</th><th>mean sin i</th>
<th>mean half-light radius</th><th>mean observed ellipticity</th>
</tr></thead>
<tbody>
{"".join(makeup_rows)}
</tbody>
</table>
<figure>
<img src="{html.escape(figures['abs_g'])}" alt="True shear amplitude of the two groups">
<figcaption>True shear amplitude. Hopped round galaxies tend to live at
larger shear, because that is where the axes can swap.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['sini'])}" alt="Inclination of the two groups">
<figcaption>Sine of inclination. Edge-on disks rarely hop; round-looking
hopped galaxies are usually more face-on, with shear cancelling the remaining
flattening.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['hlr'])}" alt="Half-light radius of the two groups">
<figcaption>Half-light radius of the same round-looking sample.</figcaption>
</figure>

<h2>Recovered shear</h2>
<p>Each point is one galaxy. The recovered value is the mean of the posterior,
averaged over the four shared noise draws. A slope of 1 would mean the mean
tracks the truth. The number we quote as a multiplicative bias is that slope
minus 1.</p>
<figure>
<img src="{html.escape(figures['g_recovery'])}" alt="Recovered versus true shear">
<figcaption>Left: first shear component. Right: second. Navy circles stayed
on the original axes. Red squares hopped.</figcaption>
</figure>

<h2>Headline comparison</h2>
<table>
<thead><tr>
<th>Group</th><th>N</th><th>m of g1</th><th>m of g2</th>
<th>mean g1 residual</th><th>mean posterior width</th>
<th>mean cos of PA error</th><th>PA RMSE (rad)</th>
<th>mean |g|</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<p>{html.escape(takeaway(full))}</p>

<h2>Control: the same |g| histogram</h2>
<p>{html.escape(matched_note)}</p>
"""
    if matched is not None:
        matched_fig = ""
        if figures.get("abs_g_matched"):
            matched_fig = f"""<figure>
<img src="{html.escape(figures['abs_g_matched'])}" alt="Matched shear-amplitude histograms">
<figcaption>After matching, the two groups have the same count in every
0.01-wide shear-amplitude bin.</figcaption>
</figure>
"""
        body += matched_fig + f"""<table>
<thead><tr>
<th>Group</th><th>N</th><th>m of g1</th><th>m of g2</th>
<th>mean g1 residual</th><th>mean posterior width</th>
<th>mean cos of PA error</th><th>PA RMSE (rad)</th>
<th>mean |g|</th>
</tr></thead>
<tbody>
{"".join(matched_rows)}
</tbody>
</table>
<p>{html.escape(takeaway(matched['metrics']))}</p>
"""
    body += f"""
<p class="note">S/N is {fmt(payload['image_snr'])} in the image and
{fmt(payload['spec_snr'])} in the central line, 1.2 times the catalog medians
except where that would exceed the allowed range. Four noise realizations.
Identity samples only; a quarter-turn of the image would swap the photometric
axes and mix the hop into the ensemble.</p>

<h2>What this can and cannot say</h2>
<p>This is an in-distribution check of one frozen wide-shear network on
round-looking validation galaxies. It does not retrain, and it does not score
the Xu test catalog. A jump here means the hop is a second spectrograph meter
the network can see. A smooth match, especially after the |g| histograms
are forced to agree, means the hop is a weak story for the wide-shear
multiplicative bias.</p>
</body></html>
"""
    path = report_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def galaxies_file(report_dir: Path) -> Path:
    return Path(report_dir) / "galaxies.npz"


def save_galaxies(path: Path, truth: dict, estimate: dict, meta: dict) -> None:
    arrays = {f"est_{key}": np.asarray(value) for key, value in estimate.items()}
    for key, value in truth.items():
        array = np.asarray(value)
        if key == "hop_class":
            array = array.astype("U16")
        arrays[f"truth_{key}"] = array
    arrays["meta_json"] = np.asarray(json.dumps(json_safe(meta)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_galaxies(path: Path) -> tuple[dict, dict, dict]:
    data = np.load(path, allow_pickle=True)
    truth = {}
    estimate = {}
    for key in data.files:
        if key == "meta_json":
            continue
        if key.startswith("truth_"):
            truth[key[len("truth_") :]] = data[key]
        elif key.startswith("est_"):
            estimate[key[len("est_") :]] = data[key]
    truth["hop_class"] = np.asarray(truth["hop_class"]).astype(str)
    raw_meta = data["meta_json"]
    text = raw_meta.item() if getattr(raw_meta, "shape", None) == () else raw_meta.tolist()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    meta = json.loads(str(text))
    return truth, estimate, meta


def matched_control(
    truth: dict, estimate: dict, *, seed: int
) -> tuple[dict | None, dict | None]:
    mask = match_abs_g_counts(
        truth["abs_g"], truth["hop_class"], width=ABS_G_BIN_WIDTH, seed=seed
    )
    if not np.any(mask):
        return None, None
    matched_truth = {key: value[mask] for key, value in truth.items()}
    n_unhopped = int((matched_truth["hop_class"] == "unhopped").sum())
    n_hopped = int((matched_truth["hop_class"] == "hopped").sum())
    payload = {
        "bin_width": ABS_G_BIN_WIDTH,
        "n_unhopped": n_unhopped,
        "n_hopped": n_hopped,
        "metrics": compare_groups(matched_truth, subset_estimate(estimate, mask)),
        "distributions": label_distributions(matched_truth),
    }
    return payload, matched_truth


def write_outputs(args, truth: dict, estimate: dict, meta: dict) -> None:
    matched_payload, matched_truth = matched_control(
        truth, estimate, seed=args.seed
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figures = save_figures(
        truth, estimate, args.report_dir, matched_truth=matched_truth
    )
    payload = {
        **meta,
        "full": compare_groups(truth, estimate),
        "full_distributions": label_distributions(truth),
        "matched": matched_payload,
        "figures": figures,
    }
    write_html(payload, figures, args.report_dir)
    json_path = args.report_dir / "report.json"
    json_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)


def write_report(args) -> None:
    html_path = args.report_dir / "report.html"
    galaxies_path = galaxies_file(args.report_dir)
    if html_path.exists() and not args.overwrite and not args.reuse_galaxies:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    if args.reuse_galaxies:
        if not galaxies_path.is_file():
            raise FileNotFoundError(galaxies_path)
        truth, estimate, meta = load_galaxies(galaxies_path)
        write_outputs(args, truth, estimate, meta)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    configs_root = args.model_root.parent / "configs"
    model_config = load_model_config(
        args.model_name, configs_root=str(configs_root)
    )
    config.set_model_config(model_config)
    names = tuple(config.TARGET_NAMES)
    par_ranges = config.par_ranges
    config.require_matching_dataset_par_ranges(args.data_dir, par_ranges)

    dataset = pxt.TorchDataset(str(args.data_dir))
    geometry = scan_catalog(dataset, par_ranges, names)
    cut = choose_ellipticity_cut(geometry["ellipticity"], geometry["hop_class"])
    keep = selected_mask(geometry["ellipticity"], geometry["hop_class"], cut)
    selected_indices = np.flatnonzero(keep)
    selected = {key: value[keep] if isinstance(value, np.ndarray) else value for key, value in geometry.items() if key != "physical"}
    selected["hop_class"] = geometry["hop_class"][keep]
    n_unhopped = int((selected["hop_class"] == "unhopped").sum())
    n_hopped = int((selected["hop_class"] == "hopped").sum())
    n_knife = int(
        np.sum(
            (geometry["ellipticity"] < cut)
            & (geometry["hop_class"] == "knife_edge")
        )
    )
    cut_note = (
        "The first roundness cut already has at least forty galaxies on each side."
        if cut == ROUND_CUTS[0]
        else (
            f"The tighter roundness cuts did not yield forty galaxies on each "
            f"side, so the cut was relaxed to {cut:.2f}."
        )
    )
    print(
        f"ellipticity_cut={cut} selected={keep.sum()} "
        f"unhopped={n_unhopped} hopped={n_hopped} knife_edge={n_knife}",
        flush=True,
    )

    image_median = float(np.median(geometry["record_image_snr"]))
    spec_median = float(np.median(geometry["record_spec_snr"]))
    image_snr = clip_snr(
        SNR_FACTOR * image_median,
        float(config.observation["image_snr_min"]),
        float(config.observation["image_snr_max"]),
    )
    spec_snr = clip_snr(
        SNR_FACTOR * spec_median,
        float(config.observation["central_halpha_snr_min"]),
        float(config.observation["central_halpha_snr_max"]),
    )
    print(
        f"snr image median={image_median:.3f} used={image_snr:.3f}; "
        f"spec median={spec_median:.3f} used={spec_snr:.3f}",
        flush=True,
    )

    checkpoint = checkpoint_file(args.model_root, args.model_name, args.checkpoint_suffix)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    networks_root = args.model_root.parent / "networks"
    model = load_model(
        KLNPE,
        path=str(checkpoint),
        model_name=args.model_name,
        device=str(device),
        strict=True,
        networks_root=str(networks_root),
    )
    model.eval()
    channels_last = bool(model_config.train.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    tensors = load_selected_tensors(dataset, selected_indices, device)
    realization_rows = []
    for realization in range(args.n_realizations):
        print(f"realization {realization + 1}/{args.n_realizations}", flush=True)
        noisy_image, noisy_spec = apply_shared_noise(
            tensors,
            image_snr=image_snr,
            spec_snr=spec_snr,
            image_seed=args.seed + 1000 * realization,
            spec_seed=args.seed + 1000 * realization + 17,
            device=device,
        )
        if channels_last:
            noisy_image = noisy_image.contiguous(memory_format=torch.channels_last)
            noisy_spec = noisy_spec.contiguous(memory_format=torch.channels_last)
        realization_rows.append(
            sample_means(
                model,
                tensors,
                noisy_image,
                noisy_spec,
                image_snr=image_snr,
                spec_snr=spec_snr,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                channels_last=channels_last,
                names=names,
                par_ranges=par_ranges,
            )
        )
    estimate = average_realizations(realization_rows)
    truth = {
        "g1": selected["g1"],
        "g2": selected["g2"],
        "theta_int": selected["theta_int"],
        "sini": selected["sini"],
        "hlr": selected["hlr"],
        "abs_g": selected["abs_g"],
        "ellipticity": selected["ellipticity"],
        "hop_class": selected["hop_class"],
        "hop_angle_deg": selected["hop_angle_deg"],
    }
    meta = {
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "data_dir": str(args.data_dir),
        "ellipticity_cut": cut,
        "ellipticity_cut_note": cut_note,
        "n_knife_edge": n_knife,
        "image_snr": image_snr,
        "spec_snr": spec_snr,
        "image_snr_median": image_median,
        "spec_snr_median": spec_median,
        "n_realizations": args.n_realizations,
        "n_samples": args.n_samples,
    }
    save_galaxies(galaxies_path, truth, estimate, meta)
    print(f"Wrote {galaxies_path}", flush=True)
    write_outputs(args, truth, estimate, meta)


def main(argv=None) -> int:
    args = parse_args(argv)
    write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
