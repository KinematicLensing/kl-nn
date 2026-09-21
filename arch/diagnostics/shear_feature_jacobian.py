#!/usr/bin/env python3
"""Feature-space shear Jacobian: encoder cosine of small vs large shear steps."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from model_registry import load_model_config
from networks import (
    FEATURE_DIM,
    IMAGE_FEATURE_DIM,
    METADATA_FEATURE_DIM,
    SPECTRAL_FEATURE_DIM,
    CCLPretrain,
)
from train import load_model

from data_generate.shear_jacobian_probe import (
    DATASET,
    DIRECTIONS,
    GALAXY_CSV,
    LARGE_PAIR,
    RENDER_CSV,
    SINI_BINS,
    SMALL_PAIR,
    amplitude_ratio,
    cosine,
    inclination_label,
    json_safe,
    load_render,
    mean_or_nan,
)

SAMPLE_ROOT = Path("/ocean/projects/phy250048p/shared/samples")
FITS_ROOT = Path("/ocean/projects/phy250048p/shared/fits")
MODEL_ROOT = Path("/ocean/projects/phy250048p/shared/models")
REPORT_DIR = Path(
    "/ocean/projects/phy250048p/shared/reports/fiber-gauge/06_feature_jacobian"
)
DATA_SPACE_JSON = Path(
    "/ocean/projects/phy250048p/shared/reports/fiber-gauge/05_shear_jacobian/report.json"
)
G01_PREFERRED = "CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k_s42_fibrepair"
G01_FALLBACK = "CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k_s42_45252895"
G02_NAME = "CNN-CNN-Meta-CCL-simv3-cosi-r90_train100k_g02_s42_45802963"
ENCODER_ORDER = ("g01", "g02")
ENCODER_LABELS = {
    "g01": "trained with shear out to 0.1",
    "g02": "trained with shear out to 0.2",
}
IMAGE_SLICE = slice(0, IMAGE_FEATURE_DIM)
SPECTRAL_SLICE = slice(
    IMAGE_FEATURE_DIM, IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM
)
METADATA_SLICE = slice(IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM, FEATURE_DIM)
BRANCH_SLICES = {
    "image": IMAGE_SLICE,
    "spectral": SPECTRAL_SLICE,
    "metadata": METADATA_SLICE,
    "concat": slice(0, FEATURE_DIM),
}
CONTEXT_FIELDS = ("rmag_true", "image_snr", "central_halpha_snr")
DEFAULT_WAVELENGTH_COUNT = 64
FIGURE_DPI = 140


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--fits-root", type=Path, default=FITS_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--data-space-json", type=Path, default=DATA_SPACE_JSON)
    parser.add_argument("--checkpoint-suffix", default="best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def checkpoint_file(model_root: Path, model_name: str, suffix: str = "best") -> Path:
    return Path(model_root) / model_name / f"{model_name}{suffix}"


def resolve_g01_name(model_root: Path, suffix: str = "best") -> str:
    preferred = checkpoint_file(model_root, G01_PREFERRED, suffix)
    if preferred.is_file():
        return G01_PREFERRED
    fallback = checkpoint_file(model_root, G01_FALLBACK, suffix)
    if fallback.is_file():
        return G01_FALLBACK
    raise FileNotFoundError(
        "neither preferred nor fallback ±0.1 encoder checkpoint exists: "
        f"{preferred} / {fallback}"
    )


def resolve_encoders(model_root: Path, suffix: str = "best") -> dict[str, str]:
    g02 = checkpoint_file(model_root, G02_NAME, suffix)
    if not g02.is_file():
        raise FileNotFoundError(f"±0.2 encoder checkpoint not found: {g02}")
    return {"g01": resolve_g01_name(model_root, suffix), "g02": G02_NAME}


def pack_product(
    product: dict,
    *,
    wavelength_count: int = DEFAULT_WAVELENGTH_COUNT,
) -> dict[str, np.ndarray]:
    image = np.asarray(product["image"], dtype=np.float32)
    spectra = np.asarray(product["spectra"], dtype=np.float32)
    positions = np.asarray(product["positions"], dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D; got shape {image.shape}")
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be (fibers, wave); got shape {spectra.shape}")
    if positions.shape != (spectra.shape[0], 2):
        raise ValueError(
            "fiber positions must have shape "
            f"({spectra.shape[0]}, 2); got {positions.shape}"
        )
    nspec, nwave = spectra.shape
    if nwave > wavelength_count:
        raise ValueError(
            f"spectrum length {nwave} exceeds wavelength_count {wavelength_count}"
        )
    packed_spec = np.zeros((nspec, wavelength_count), dtype=np.float32)
    packed_spec[:, :nwave] = spectra
    return {
        "img": image[None, ...],
        "spec": packed_spec[None, ...],
        "fib_pos": positions.astype(np.float32, copy=False),
    }


def split_features(features: np.ndarray) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"features must end with width {FEATURE_DIM}; got {features.shape}"
        )
    return {name: features[..., sl] for name, sl in BRANCH_SLICES.items()}


def branch_delta_metrics(
    small_lo: np.ndarray,
    small_hi: np.ndarray,
    large_lo: np.ndarray,
    large_hi: np.ndarray,
) -> dict[str, dict[str, float]]:
    small_lo = np.asarray(small_lo, dtype=np.float64).reshape(-1)
    small_hi = np.asarray(small_hi, dtype=np.float64).reshape(-1)
    large_lo = np.asarray(large_lo, dtype=np.float64).reshape(-1)
    large_hi = np.asarray(large_hi, dtype=np.float64).reshape(-1)
    if not (
        small_lo.shape
        == small_hi.shape
        == large_lo.shape
        == large_hi.shape
        == (FEATURE_DIM,)
    ):
        raise ValueError(
            "all feature vectors must have shape "
            f"({FEATURE_DIM},); got {small_lo.shape}, {small_hi.shape}, "
            f"{large_lo.shape}, {large_hi.shape}"
        )
    metrics = {}
    for name, sl in BRANCH_SLICES.items():
        d_small = small_hi[sl] - small_lo[sl]
        d_large = large_hi[sl] - large_lo[sl]
        metrics[name] = {
            "cosine": cosine(d_small, d_large),
            "amplitude_ratio": amplitude_ratio(d_small, d_large),
        }
    return metrics


def image_frozen_match(moving: np.ndarray, frozen: np.ndarray) -> float:
    moving = np.asarray(moving, dtype=np.float64).reshape(-1)
    frozen = np.asarray(frozen, dtype=np.float64).reshape(-1)
    return cosine(moving[IMAGE_SLICE], frozen[IMAGE_SLICE])


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


def hopped(row: dict) -> bool:
    value = row.get("axis_flip_g")
    return value is not None and np.isfinite(value)


def load_data_space_rows(path: Path) -> dict[tuple[int, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = {}
    for row in payload["rows"]:
        key = (int(row["galaxy_index"]), str(row["direction"]))
        index[key] = {
            "image_cosine": float(row["moving"]["image_cosine"]),
            "spectra_moving": float(row["moving"]["spectra_mean_cosine"]),
            "spectra_frozen": float(row["frozen"]["spectra_mean_cosine"]),
            "axis_flip_g": row["analytic"].get("axis_flip_g"),
        }
    return index


def load_encoder(
    model_name: str,
    *,
    model_root: Path,
    suffix: str,
    device: torch.device,
):
    configs_root = model_root.parent / "configs"
    model_config = load_model_config(
        model_name, configs_root=str(configs_root)
    )
    config.set_model_config(model_config)
    checkpoint = checkpoint_file(model_root, model_name, suffix)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    networks_root = model_root.parent / "networks"
    model = load_model(
        CCLPretrain,
        path=str(checkpoint),
        model_name=model_name,
        device=str(device),
        strict=True,
        networks_root=str(networks_root),
    )
    model.eval()
    channels_last = bool(model_config.pretrain.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model, channels_last, checkpoint


def extract_features_batch(
    model,
    packed_rows: list[dict],
    *,
    device: torch.device,
    channels_last: bool,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(packed_rows), batch_size):
            chunk = packed_rows[start : start + batch_size]
            img = torch.stack(
                [torch.as_tensor(row["img"]) for row in chunk]
            ).float().to(device)
            spec = torch.stack(
                [torch.as_tensor(row["spec"]) for row in chunk]
            ).float().to(device)
            fp = torch.stack(
                [torch.as_tensor(row["fib_pos"]) for row in chunk]
            ).float().to(device)
            context = {
                name: torch.tensor(
                    [float(row["context"][name]) for row in chunk],
                    device=device,
                    dtype=torch.float32,
                )
                for name in CONTEXT_FIELDS
            }
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            features = model.extract_features(
                img,
                spec,
                fp,
                observation_context=context,
            )
            outputs.append(features.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def collect_packed_renders(
    galaxies: pd.DataFrame,
    renders: pd.DataFrame,
    *,
    fits_root: Path,
    wavelength_count: int,
) -> tuple[list[dict], dict[tuple, int]]:
    packed_rows = []
    index = {}
    galaxy_by_id = {
        int(row["galaxy_index"]): row for _, row in galaxies.iterrows()
    }
    for _, render in renders.iterrows():
        galaxy_index = int(render["galaxy_index"])
        galaxy = galaxy_by_id[galaxy_index]
        product = load_render(render, fits_root=fits_root)
        packed = pack_product(product, wavelength_count=wavelength_count)
        packed["context"] = {
            name: float(galaxy[name]) for name in CONTEXT_FIELDS
        }
        key = (
            galaxy_index,
            str(render["direction"]),
            float(render["g_amp"]),
            str(render["fiber_mode"]),
        )
        index[key] = len(packed_rows)
        packed_rows.append(packed)
    return packed_rows, index


def feature_for(
    features: np.ndarray,
    index: dict[tuple, int],
    *,
    galaxy_index: int,
    direction: str,
    g_amp: float,
    fiber_mode: str,
) -> np.ndarray:
    matches = [
        features[slot]
        for key, slot in index.items()
        if key[0] == galaxy_index
        and key[1] == direction
        and np.isclose(key[2], g_amp)
        and key[3] == fiber_mode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one feature vector for galaxy={galaxy_index} "
            f"{direction} g={g_amp} mode={fiber_mode}; got {len(matches)}"
        )
    return matches[0]


def compare_encoder_pairs(
    galaxies: pd.DataFrame,
    features: np.ndarray,
    index: dict[tuple, int],
    data_space: dict[tuple[int, str], dict],
) -> list[dict]:
    rows = []
    for _, galaxy in galaxies.iterrows():
        galaxy_index = int(galaxy["galaxy_index"])
        for direction in DIRECTIONS:
            moving_small_lo = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=SMALL_PAIR[0],
                fiber_mode="moving",
            )
            moving_small_hi = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=SMALL_PAIR[1],
                fiber_mode="moving",
            )
            moving_large_lo = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=LARGE_PAIR[0],
                fiber_mode="moving",
            )
            moving_large_hi = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=LARGE_PAIR[1],
                fiber_mode="moving",
            )
            frozen_large_lo = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=LARGE_PAIR[0],
                fiber_mode="frozen",
            )
            frozen_large_hi = feature_for(
                features,
                index,
                galaxy_index=galaxy_index,
                direction=direction,
                g_amp=LARGE_PAIR[1],
                fiber_mode="frozen",
            )
            moving = branch_delta_metrics(
                moving_small_lo,
                moving_small_hi,
                moving_large_lo,
                moving_large_hi,
            )
            frozen = branch_delta_metrics(
                moving_small_lo,
                moving_small_hi,
                frozen_large_lo,
                frozen_large_hi,
            )
            space = data_space[(galaxy_index, direction)]
            rows.append(
                {
                    "galaxy_index": galaxy_index,
                    "source_id": int(galaxy["source_id"]),
                    "inclination_bin": str(galaxy["inclination_bin"]),
                    "size_bin": str(galaxy["size_bin"]),
                    "pa_bin": str(galaxy["pa_bin"]),
                    "sini": float(galaxy["sini"]),
                    "hlr": float(galaxy["hlr"]),
                    "theta_int": float(galaxy["theta_int"]),
                    "direction": direction,
                    "axis_flip_g": space["axis_flip_g"],
                    "data_space": {
                        "image_cosine": space["image_cosine"],
                        "spectra_moving": space["spectra_moving"],
                        "spectra_frozen": space["spectra_frozen"],
                    },
                    "moving": moving,
                    "frozen": frozen,
                    "image_frozen_match": {
                        "g15": image_frozen_match(
                            moving_large_lo, frozen_large_lo
                        ),
                        "g17": image_frozen_match(
                            moving_large_hi, frozen_large_hi
                        ),
                    },
                }
            )
    return rows


def save_image_cosine(payload: dict, report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    bins = [name for name, _, _ in SINI_BINS]
    rng = np.random.default_rng(0)
    colors = {"data": "#888888", "g01": "#1f4e79", "g02": "#a31f34"}
    for axis, direction, title in (
        (axes[0], "image_g1", "Shear along the first camera axis"),
        (axes[1], "galaxy_plus", "Shear along the disk major axis"),
    ):
        for offset, sini_name in enumerate(bins):
            data_values = []
            for encoder_key in ENCODER_ORDER:
                subset = [
                    row
                    for row in payload["encoders"][encoder_key]["rows"]
                    if row["direction"] == direction
                    and row["inclination_bin"] == sini_name
                ]
                if encoder_key == "g01":
                    data_values = [row["data_space"]["image_cosine"] for row in subset]
                jitter = rng.normal(0.0, 0.035, size=len(subset))
                axis.scatter(
                    np.full(len(subset), offset) + jitter,
                    [row["moving"]["image"]["cosine"] for row in subset],
                    s=36,
                    color=colors[encoder_key],
                    zorder=3,
                    label=ENCODER_LABELS[encoder_key] if offset == 0 else None,
                )
            jitter = rng.normal(0.0, 0.035, size=len(data_values))
            axis.scatter(
                np.full(len(data_values), offset) + jitter,
                data_values,
                s=42,
                facecolors="none",
                edgecolors=colors["data"],
                linewidths=1.1,
                zorder=2,
                label="simulator image" if offset == 0 else None,
            )
        axis.set_xticks(range(len(bins)))
        axis.set_xticklabels([inclination_label(name) for name in bins], rotation=15)
        axis.set_title(title)
        axis.set_ylim(-0.05, 1.05)
        axis.axhline(1.0, color="#bbbbbb", lw=0.8)
        axis.axhline(0.0, color="#bbbbbb", lw=0.8)
    axes[0].set_ylabel("cosine of the two feature changes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Do the image-feature changes point in the same direction?")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    path = report_dir / "image_cosine.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_spectrum_cosine(payload: dict, report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    x = np.arange(len(SINI_BINS))
    width = 0.36
    for axis, encoder_key in zip(axes, ENCODER_ORDER):
        rows = payload["encoders"][encoder_key]["rows"]
        moving = []
        frozen = []
        for sini_name, _, _ in SINI_BINS:
            subset = [
                row
                for row in rows
                if row["direction"] == "image_g1"
                and row["inclination_bin"] == sini_name
            ]
            moving.append(
                mean_or_nan([row["moving"]["spectral"]["cosine"] for row in subset])
            )
            frozen.append(
                mean_or_nan([row["frozen"]["spectral"]["cosine"] for row in subset])
            )
        axis.bar(
            x - width / 2,
            moving,
            width,
            label="apertures move with shear",
            color="#1f4e79",
        )
        axis.bar(
            x + width / 2,
            frozen,
            width,
            label="apertures held at the small-shear positions",
            color="#c47b15",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [inclination_label(name) for name, _, _ in SINI_BINS], rotation=15
        )
        axis.set_ylim(-1.05, 1.05)
        axis.axhline(0.0, color="#bbbbbb", lw=0.8)
        axis.set_title(ENCODER_LABELS[encoder_key])
    axes[0].set_ylabel("mean cosine of the spectral-feature change")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Spectrum features: moving apertures versus frozen apertures",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    path = report_dir / "spectrum_cosine.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_frozen_vs_moving(payload: dict, report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharex=True, sharey=True)
    for axis, encoder_key in zip(axes, ENCODER_ORDER):
        rows = [
            row
            for row in payload["encoders"][encoder_key]["rows"]
            if row["direction"] == "image_g1"
        ]
        stayed = [row for row in rows if not hopped(row)]
        flipped = [row for row in rows if hopped(row)]
        if stayed:
            axis.scatter(
                [row["moving"]["spectral"]["cosine"] for row in stayed],
                [row["frozen"]["spectral"]["cosine"] for row in stayed],
                s=42,
                color="#1f4e79",
                label="fibers stayed put",
                zorder=3,
            )
        if flipped:
            axis.scatter(
                [row["moving"]["spectral"]["cosine"] for row in flipped],
                [row["frozen"]["spectral"]["cosine"] for row in flipped],
                s=52,
                marker="s",
                color="#a31f34",
                label="fibers hopped",
                zorder=4,
            )
        axis.plot([-1, 1], [-1, 1], color="#bbbbbb", lw=1)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(ENCODER_LABELS[encoder_key])
        axis.set_aspect("equal")
        axis.set_xlabel("cosine with moving apertures")
    axes[0].set_ylabel("cosine with frozen apertures")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Each galaxy, shear along the first camera axis")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    path = report_dir / "frozen_vs_moving.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def encoder_takeaway(rows: list[dict]) -> str:
    camera = [row for row in rows if row["direction"] == "image_g1"]
    hopped_rows = [row for row in camera if hopped(row)]
    if not hopped_rows:
        return (
            "None of these twelve galaxies hopped. Spectrum-feature cosines "
            "stay high, matching the simulator."
        )
    moving = mean_or_nan(
        [row["moving"]["spectral"]["cosine"] for row in hopped_rows]
    )
    frozen = mean_or_nan(
        [row["frozen"]["spectral"]["cosine"] for row in hopped_rows]
    )
    data_moving = mean_or_nan(
        [row["data_space"]["spectra_moving"] for row in hopped_rows]
    )
    if moving < 0.5 and frozen > 0.8:
        return (
            f"On the galaxies whose fibers hopped, the spectrum-feature change "
            f"at large shear points a different way (mean cosine {moving:.2f}), "
            f"close to the simulator ({data_moving:.2f}). Locking the apertures "
            f"restores the match (mean cosine {frozen:.2f}). This network "
            "inherited the hop; it did not undo it."
        )
    if moving > 0.8:
        return (
            f"On the galaxies whose fibers hopped, the spectrum-feature change "
            f"still points the same way (mean cosine {moving:.2f}), even though "
            f"the raw spectra do not (mean cosine {data_moving:.2f}). This "
            "network already factored the hop out."
        )
    return (
        f"On the galaxies whose fibers hopped, the spectrum-feature cosine is "
        f"{moving:.2f} with moving apertures and {frozen:.2f} with frozen "
        f"apertures. The simulator moving-aperture cosine is {data_moving:.2f}."
    )


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    summary_rows = []
    for encoder_key in ENCODER_ORDER:
        rows = payload["encoders"][encoder_key]["rows"]
        for sini_name, _, _ in SINI_BINS:
            subset = [
                row
                for row in rows
                if row["inclination_bin"] == sini_name
                and row["direction"] == "image_g1"
            ]
            phot = [
                row
                for row in rows
                if row["inclination_bin"] == sini_name
                and row["direction"] == "galaxy_plus"
            ]
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(ENCODER_LABELS[encoder_key])}</td>"
                f"<td>{html.escape(inclination_label(sini_name))}</td>"
                f"<td>{fmt(mean_or_nan([row['data_space']['spectra_moving'] for row in subset]))}</td>"
                f"<td>{fmt(mean_or_nan([row['moving']['spectral']['cosine'] for row in subset]))}</td>"
                f"<td>{fmt(mean_or_nan([row['frozen']['spectral']['cosine'] for row in subset]))}</td>"
                f"<td>{fmt(mean_or_nan([row['moving']['image']['cosine'] for row in phot]))}</td>"
                "</tr>"
            )
    takeaways = []
    for encoder_key in ENCODER_ORDER:
        takeaways.append(
            f"<li><strong>{html.escape(ENCODER_LABELS[encoder_key])}.</strong> "
            f"{html.escape(encoder_takeaway(payload['encoders'][encoder_key]['rows']))}</li>"
        )
    image_match = mean_or_nan(
        [
            0.5
            * (
                row["image_frozen_match"]["g15"]
                + row["image_frozen_match"]["g17"]
            )
            for encoder_key in ENCODER_ORDER
            for row in payload["encoders"][encoder_key]["rows"]
            if row["direction"] == "image_g1"
        ]
    )
    if np.isfinite(image_match) and image_match < 0.999:
        image_match_note = (
            "A check: image features at the same large shear, with and without "
            f"moving the apertures, agree only to cosine {image_match:.3f}. "
            "The image branch should not see the spectrograph pointing."
        )
    else:
        image_match_note = (
            "A check: at the same large shear, image features are the same "
            "whether the apertures move or stay put. The image branch does not "
            "see the spectrograph pointing."
        )

    g01_image = mean_or_nan(
        [
            row["moving"]["image"]["cosine"]
            for row in payload["encoders"]["g01"]["rows"]
            if row["direction"] == "galaxy_plus"
        ]
    )
    g02_image = mean_or_nan(
        [
            row["moving"]["image"]["cosine"]
            for row in payload["encoders"]["g02"]["rows"]
            if row["direction"] == "galaxy_plus"
        ]
    )
    data_image = mean_or_nan(
        [
            row["data_space"]["image_cosine"]
            for row in payload["encoders"]["g01"]["rows"]
            if row["direction"] == "galaxy_plus"
        ]
    )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Does the encoder still see the fiber hop?</title>
<style>
body {{ font: 17px/1.55 Palatino, "Palatino Linotype", serif; margin: 2rem auto; max-width: 980px; color: #1b1b1b; }}
h1, h2, h3 {{ font-weight: 600; }}
h1 {{ font-size: 1.85rem; }}
h2 {{ margin-top: 2.4rem; }}
p.lead {{ font-size: 1.08rem; }}
figure {{ margin: 1.4rem 0 2rem; }}
figcaption {{ font-size: 0.92rem; color: #444; margin-top: 0.45rem; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.55rem; }}
th {{ background: #f4f4f4; text-align: left; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.note {{ color: #444; }}
</style></head><body>
<h1>Does a trained encoder still see the spectral flip after a fiber hop?</h1>
<p class="lead">The simulator said a 0.02 shear bump rearranges the image the
same way at small and large shear, but the spectra anti-align once the
apertures hop onto a new pair of axes. This page asks the same cosine
question of two frozen encoders: one trained with shear out to 0.1, and one
trained with shear out to 0.2. No network is retrained here.</p>

<h2>What was held fixed, and what was changed</h2>
<p>The same twelve galaxies as the simulator test. Size, brightness,
inclination, position angle, and rotation speed stay the same. Only shear
changes. We bump shear by the same amount, 0.02, in two places:</p>
<ul>
<li>a small-shear neighborhood, from 0.01 to 0.03</li>
<li>a large-shear neighborhood, from 0.15 to 0.17</li>
</ul>
<p>The stamps are noiseless. For each stamp we read the image half of the
encoder, the spectrum half, and the small metadata half that knows where the
fibers sat. The number we quote is the cosine of the angle between those two
feature changes: 1 means the encoder still sees the same pattern, 0 means
unrelated patterns, and a negative value means the change pointed the opposite
way. We repeat the large-shear pair with the fibers locked at their 0.01
placement.</p>

<h2>Image features</h2>
<p>Open circles are the simulator image cosine from the earlier test. Filled
points are the image half of each encoder.</p>
<figure>
<img src="{html.escape(figures['image_cosine'])}" alt="Cosine similarity of image-feature differences">
<figcaption>Each point is one galaxy. Values near 1 mean the small-shear and
large-shear image-feature changes point the same way. Simulator mean along the
disk major axis: {fmt(data_image)}. Encoder trained to 0.1:
{fmt(g01_image)}. Encoder trained to 0.2: {fmt(g02_image)}.</figcaption>
</figure>
<p>{html.escape(image_match_note)}</p>

<h2>Spectrum features, moving versus frozen</h2>
<p>This comparison uses shear along the first camera axis, the direction in
which the fibers can hop. Squares in the scatter are galaxies whose apertures
swapped onto a new pair of axes somewhere between shear 0 and 0.2.</p>
<figure>
<img src="{html.escape(figures['spectrum_cosine'])}" alt="Spectral-feature cosine with moving versus frozen fibers">
<figcaption>Mean over the four galaxies in each inclination bin. If freezing
the apertures restores the cosine, the encoder still sees the hop. If the
cosine stays high even when the raw spectra anti-align, the encoder already
factored the hop out.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['frozen_vs_moving'])}" alt="Per-galaxy frozen versus moving spectral-feature cosine">
<figcaption>Each point is one galaxy. Points on the diagonal are unaffected
by locking the fibers. Points above the diagonal mean frozen apertures make
the large-shear spectral-feature change look more like the small-shear
change.</figcaption>
</figure>
<ul>
{"".join(takeaways)}
</ul>

<h2>Summary</h2>
<table>
<thead><tr>
<th>Network</th>
<th>Inclination</th>
<th>Simulator spectra, moving</th>
<th>Encoder spectra, moving</th>
<th>Encoder spectra, frozen</th>
<th>Encoder image</th>
</tr></thead>
<tbody>
{"".join(summary_rows)}
</tbody>
</table>
<p class="note">Numbers are means over the four galaxies in that inclination
bin. The spectra columns use shear along the first camera axis. The image
column uses shear along the disk major axis. Frozen means the spectrograph
pointing is held at the small-shear placement while the galaxy is still
sheared.</p>

<h2>What this can and cannot say</h2>
<p>This is a check of two frozen encoders on noiseless stamps. It does not
measure the recovered shear slope, and it does not prove that a later
posterior used these features on weakly sheared test galaxies. If the
spectrum-feature cosine drops on hopped galaxies and freezing the apertures
restores it, the encoder inherited the simulator hop. Connecting that to
multiplicative bias would still take a later readout that trains on one shear
range and scores another. If the spectrum-feature cosine stays high where the
raw spectra flip, the hop is a weaker story for that bias.</p>
</body></html>
"""
    html_path = report_dir / "report.html"
    html_path.write_text(body, encoding="utf-8")
    print(f"Wrote {html_path}", flush=True)


def write_report(
    *,
    sample_root: Path,
    fits_root: Path,
    model_root: Path,
    report_dir: Path,
    data_space_json: Path,
    checkpoint_suffix: str,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    html_path = report_dir / "report.html"
    if html_path.exists() and not overwrite:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    if not data_space_json.is_file():
        raise FileNotFoundError(
            f"simulator Jacobian JSON not found: {data_space_json}"
        )
    galaxies = pd.read_csv(sample_root / GALAXY_CSV, float_precision="round_trip")
    renders = pd.read_csv(sample_root / RENDER_CSV, float_precision="round_trip")
    data_space = load_data_space_rows(data_space_json)
    encoders = resolve_encoders(model_root, checkpoint_suffix)
    packed_rows, render_index = collect_packed_renders(
        galaxies,
        renders,
        fits_root=fits_root,
        wavelength_count=DEFAULT_WAVELENGTH_COUNT,
    )
    payload = {
        "dataset": DATASET,
        "n_renders": len(packed_rows),
        "feature_dimension": FEATURE_DIM,
        "slices": {
            "image": [IMAGE_SLICE.start, IMAGE_SLICE.stop],
            "spectral": [SPECTRAL_SLICE.start, SPECTRAL_SLICE.stop],
            "metadata": [METADATA_SLICE.start, METADATA_SLICE.stop],
        },
        "encoders": {},
    }
    for encoder_key in ENCODER_ORDER:
        model_name = encoders[encoder_key]
        print(f"encoder={encoder_key} model={model_name}", flush=True)
        model, channels_last, checkpoint = load_encoder(
            model_name,
            model_root=model_root,
            suffix=checkpoint_suffix,
            device=device,
        )
        features = extract_features_batch(
            model,
            packed_rows,
            device=device,
            channels_last=channels_last,
            batch_size=batch_size,
        )
        if features.shape != (len(packed_rows), FEATURE_DIM):
            raise ValueError(
                "encoder features must have shape "
                f"({len(packed_rows)}, {FEATURE_DIM}); got {features.shape}"
            )
        rows = compare_encoder_pairs(galaxies, features, render_index, data_space)
        payload["encoders"][encoder_key] = {
            "label": ENCODER_LABELS[encoder_key],
            "model_name": model_name,
            "checkpoint": str(checkpoint),
            "rows": rows,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "image_cosine": save_image_cosine(payload, report_dir),
        "spectrum_cosine": save_spectrum_cosine(payload, report_dir),
        "frozen_vs_moving": save_frozen_vs_moving(payload, report_dir),
    }
    write_html(payload, figures, report_dir)
    sidecar = {
        "dataset": payload["dataset"],
        "n_renders": payload["n_renders"],
        "feature_dimension": payload["feature_dimension"],
        "slices": payload["slices"],
        "encoders": json_safe(payload["encoders"]),
        "figures": figures,
    }
    json_path = report_dir / "report.json"
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    write_report(
        sample_root=args.sample_root,
        fits_root=args.fits_root,
        model_root=args.model_root,
        report_dir=args.report_dir,
        data_space_json=args.data_space_json,
        checkpoint_suffix=args.checkpoint_suffix,
        device=device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
