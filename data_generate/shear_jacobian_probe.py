#!/usr/bin/env python3
"""Small-vs-large shear Jacobian probe: tables, analytic fibers, renders, HTML."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

try:
    from .generation_integrity import simulator_v3_output_path
    from .observation_schema import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        compute_fiber_offsets,
    )
except ImportError:
    from generation_integrity import simulator_v3_output_path
    from observation_schema import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        compute_fiber_offsets,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_ROOT = Path("/ocean/projects/phy250048p/shared/samples")
FITS_ROOT = Path("/ocean/projects/phy250048p/shared/fits")
REPORT_DIR = Path(
    "/ocean/projects/phy250048p/shared/reports/fiber-gauge/05_shear_jacobian"
)
SOURCE_CSV = "valid_100k_simv3_cosi.csv"
GALAXY_CSV = "shear_jacobian_galaxies.csv"
RENDER_CSV = "shear_jacobian_renders.csv"
DATASET = "shear_jacobian_probe"
PART_SIZE = 2000
SPECTRUM_HDU_COUNT = 5
IMAGE_HDU = 6
SHEAR_VALUES = (0.01, 0.03, 0.15, 0.17)
SMALL_PAIR = (0.01, 0.03)
LARGE_PAIR = (0.15, 0.17)
FIBER_BASE = 0.01
DIRECTIONS = ("image_g1", "galaxy_plus")
FIBER_NAMES = (
    "major-axis fiber A",
    "major-axis fiber B",
    "center fiber",
    "minor-axis fiber A",
    "minor-axis fiber B",
)
SINI_BINS = (
    ("edge-on", 0.85, 1.01),
    ("inclined", 0.60, 0.85),
    ("moderate", 0.40, 0.60),
)
FIGURE_DPI = 140


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-tables",
        action="store_true",
        help="select 12 galaxies and write probe CSVs",
    )
    parser.add_argument(
        "--render-index",
        type=int,
        help="1-based galaxy index (1-12) to render",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="write analytic-fiber figures, cosine metrics, and HTML",
    )
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--fits-root", type=Path, default=FITS_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def wrap_angle(value: np.ndarray | float, target: float) -> np.ndarray:
    delta = (np.asarray(value, dtype=float) - target + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(delta)


def galaxy_frame_plus(g_plus: float, theta_int: float) -> tuple[float, float]:
    two_theta = 2.0 * float(theta_int)
    return (
        float(g_plus) * float(np.cos(two_theta)),
        float(g_plus) * float(np.sin(two_theta)),
    )


def shear_components(direction: str, g_amp: float, theta_int: float) -> tuple[float, float]:
    if direction == "image_g1":
        return float(g_amp), 0.0
    if direction == "galaxy_plus":
        return galaxy_frame_plus(g_amp, theta_int)
    raise ValueError(f"unknown direction {direction!r}")


def direction_label(direction: str) -> str:
    if direction == "image_g1":
        return "shear along the first camera axis"
    if direction == "galaxy_plus":
        return "shear stretching the disk along its major axis"
    raise ValueError(f"unknown direction {direction!r}")


def inclination_label(name: str) -> str:
    return {
        "edge-on": "nearly edge-on",
        "inclined": "inclined",
        "moderate": "moderately inclined",
    }[name]


def size_label(name: str) -> str:
    return "compact" if name == "compact" else "extended"


def select_galaxies(table: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    used = set()
    for sini_name, lo, hi in SINI_BINS:
        subset = table[(table["sini"] >= lo) & (table["sini"] < hi)].copy()
        if subset.empty:
            raise ValueError(f"no galaxies with sin i in [{lo}, {hi})")
        median_hlr = float(subset["hlr"].median())
        size_masks = (
            ("compact", subset["hlr"] <= median_hlr),
            ("extended", subset["hlr"] > median_hlr),
        )
        for size_name, size_mask in size_masks:
            pool = subset.loc[size_mask]
            for pa_name, target in (("aligned", 0.0), ("perpendicular", 0.5 * np.pi)):
                available = pool.loc[~pool["ID"].isin(used)]
                if available.empty:
                    raise ValueError(
                        f"empty pool for {sini_name} {size_name} {pa_name}"
                    )
                distance = wrap_angle(available["theta_int"].to_numpy(), target)
                winner = available.iloc[int(np.argmin(distance))]
                used.add(int(winner["ID"]))
                row = winner.to_dict()
                row["galaxy_index"] = len(chosen)
                row["source_id"] = int(winner["ID"])
                row["inclination_bin"] = sini_name
                row["size_bin"] = size_name
                row["pa_bin"] = pa_name
                chosen.append(row)
    out = pd.DataFrame(chosen)
    out["ID"] = np.arange(len(out), dtype=int)
    return out


def build_render_table(galaxies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    render_id = 0
    for _, galaxy in galaxies.iterrows():
        theta_int = float(galaxy["theta_int"])
        for direction in DIRECTIONS:
            for g_amp in SHEAR_VALUES:
                g1, g2 = shear_components(direction, g_amp, theta_int)
                fiber_g1, fiber_g2 = g1, g2
                rows.append(
                    _render_row(
                        render_id,
                        galaxy,
                        direction,
                        g_amp,
                        g1,
                        g2,
                        fiber_g1,
                        fiber_g2,
                        "moving",
                    )
                )
                render_id += 1
            base_g1, base_g2 = shear_components(direction, FIBER_BASE, theta_int)
            for g_amp in LARGE_PAIR:
                g1, g2 = shear_components(direction, g_amp, theta_int)
                rows.append(
                    _render_row(
                        render_id,
                        galaxy,
                        direction,
                        g_amp,
                        g1,
                        g2,
                        base_g1,
                        base_g2,
                        "frozen",
                    )
                )
                render_id += 1
    return pd.DataFrame(rows)


def _render_row(
    render_id: int,
    galaxy: pd.Series,
    direction: str,
    g_amp: float,
    g1: float,
    g2: float,
    fiber_g1: float,
    fiber_g2: float,
    fiber_mode: str,
) -> dict:
    row = {name: galaxy[name] for name in galaxy.index}
    row["ID"] = int(render_id)
    row["galaxy_index"] = int(galaxy["galaxy_index"])
    row["source_id"] = int(galaxy["source_id"])
    row["direction"] = direction
    row["g_amp"] = float(g_amp)
    row["g1"] = float(g1)
    row["g2"] = float(g2)
    row["fiber_g1"] = float(fiber_g1)
    row["fiber_g2"] = float(fiber_g2)
    row["fiber_mode"] = fiber_mode
    return row


def write_tables(*, sample_root: Path) -> tuple[Path, Path]:
    source = pd.read_csv(sample_root / SOURCE_CSV, float_precision="round_trip")
    galaxies = select_galaxies(source)
    renders = build_render_table(galaxies)
    galaxy_path = sample_root / GALAXY_CSV
    render_path = sample_root / RENDER_CSV
    galaxies.to_csv(galaxy_path, index=False)
    renders.to_csv(render_path, index=False)
    print(f"Wrote {galaxy_path} ({len(galaxies)} galaxies)")
    print(f"Wrote {render_path} ({len(renders)} renders)")
    return galaxy_path, render_path


def part_number(sample_id: int) -> int:
    return int(sample_id) // PART_SIZE + 1


def render_galaxy(galaxy_index: int, *, sample_root: Path, fits_root: Path) -> None:
    renders = pd.read_csv(sample_root / RENDER_CSV, float_precision="round_trip")
    subset = renders.loc[renders["galaxy_index"] == galaxy_index]
    if subset.empty:
        raise ValueError(f"no renders for galaxy_index={galaxy_index}")
    for _, row in subset.iterrows():
        sample_id = int(row["ID"])
        command = [
            sys.executable,
            str(SCRIPT_DIR / "generate_fits_wrapper.py"),
            f"-i={sample_id}",
            f"-j={sample_id + 1}",
            f"-n={part_number(sample_id)}",
            f"-s={RENDER_CSV}",
            f"-d={DATASET}",
            "--skip-existing",
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=str(SCRIPT_DIR))
        output = simulator_v3_output_path(
            fits_root, DATASET, part_number(sample_id), sample_id
        )
        if not output.is_file():
            raise FileNotFoundError(f"Generator did not write {output}")


def load_product(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdus:
        if len(hdus) != SPECTRUM_HDU_COUNT + 2:
            raise ValueError(f"{path} has {len(hdus)} HDUs, expected 7")
        spectra = np.stack(
            [
                np.asarray(hdus[index].data, dtype=np.float64)
                for index in range(1, SPECTRUM_HDU_COUNT + 1)
            ]
        )
        image = np.asarray(hdus[IMAGE_HDU].data, dtype=np.float64)
        positions = np.asarray(
            [
                (
                    float(hdus[index].header["FIBERDX"]),
                    float(hdus[index].header["FIBERDY"]),
                )
                for index in range(1, SPECTRUM_HDU_COUNT + 1)
            ],
            dtype=np.float64,
        )
    return {"spectra": spectra, "image": image, "positions": positions}


def cosine(a: np.ndarray, b: np.ndarray, *, zero: float = 1.0e-12) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= zero and right_norm <= zero:
        return 1.0
    if left_norm <= zero or right_norm <= zero:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def amplitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    left = float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))
    right = float(np.linalg.norm(np.asarray(b, dtype=np.float64).ravel()))
    if left == 0.0:
        return float("nan")
    return right / left


def high_frequency_fraction(image: np.ndarray) -> float:
    field = np.asarray(image, dtype=np.float64)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(field)))
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    rows, cols = np.ogrid[:height, :width]
    radius = np.hypot(rows - center_y, cols - center_x) / max(center_y, center_x, 1)
    total = float(spectrum.sum())
    if total == 0.0:
        return float("nan")
    return float(spectrum[radius > 0.4].sum() / total)


def shift_register(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    corr = np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(mov)))
    corr = np.fft.fftshift(np.real(corr))
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift_y = int(peak[0] - corr.shape[0] // 2)
    shift_x = int(peak[1] - corr.shape[1] // 2)
    registered = np.roll(np.roll(mov, shift_y, axis=0), shift_x, axis=1)
    return registered, (shift_y, shift_x)


def lookup_render(
    renders: pd.DataFrame,
    *,
    galaxy_index: int,
    direction: str,
    g_amp: float,
    fiber_mode: str,
) -> pd.Series:
    match = renders.loc[
        (renders["galaxy_index"] == galaxy_index)
        & (renders["direction"] == direction)
        & np.isclose(renders["g_amp"], g_amp)
        & (renders["fiber_mode"] == fiber_mode)
    ]
    if len(match) != 1:
        raise ValueError(
            f"expected one render for galaxy={galaxy_index} {direction} "
            f"g={g_amp} mode={fiber_mode}; got {len(match)}"
        )
    return match.iloc[0]


def load_render(row: pd.Series, *, fits_root: Path) -> dict:
    sample_id = int(row["ID"])
    path = simulator_v3_output_path(
        fits_root, DATASET, part_number(sample_id), sample_id
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    product = load_product(path)
    product["path"] = str(path)
    return product


def fiber_paths(galaxy: pd.Series, direction: str, g_grid: np.ndarray) -> np.ndarray:
    theta_int = float(galaxy["theta_int"])
    sini = float(galaxy["sini"])
    tracks = []
    for g_amp in g_grid:
        g1, g2 = shear_components(direction, float(g_amp), theta_int)
        tracks.append(
            compute_fiber_offsets(
                fiber_offset=DEFAULT_FIBER_OFFSET_ARCSEC,
                g1=g1,
                g2=g2,
                theta_int=theta_int,
                sini=sini,
            )
        )
    return np.stack(tracks)


def analytic_fiber_metrics(galaxy: pd.Series, direction: str) -> dict:
    g_grid = np.linspace(0.0, 0.2, 41)
    tracks = fiber_paths(galaxy, direction, g_grid)
    small = fiber_paths(galaxy, direction, np.asarray(SMALL_PAIR))
    large = fiber_paths(galaxy, direction, np.asarray(LARGE_PAIR))
    d_small = small[1] - small[0]
    d_large = large[1] - large[0]
    travel = np.linalg.norm(tracks - tracks[0], axis=-1)
    step = np.linalg.norm(np.diff(tracks, axis=0), axis=-1).max(axis=1)
    flip_indices = np.flatnonzero(step > 0.2)
    per_fiber = [
        {
            "fiber": FIBER_NAMES[index],
            "cosine": cosine(d_small[index], d_large[index]),
            "amplitude_ratio": amplitude_ratio(d_small[index], d_large[index]),
            "small_step_arcsec": float(np.linalg.norm(d_small[index])),
            "large_step_arcsec": float(np.linalg.norm(d_large[index])),
        }
        for index in range(5)
    ]
    return {
        "g_grid": g_grid.tolist(),
        "tracks": tracks.tolist(),
        "d_small": d_small.tolist(),
        "d_large": d_large.tolist(),
        "per_fiber": per_fiber,
        "all_fibers_cosine": cosine(d_small, d_large),
        "max_travel_arcsec": float(travel.max()),
        "axis_flip_g": None if flip_indices.size == 0 else float(g_grid[int(flip_indices[0]) + 1]),
    }


def pair_products(
    renders: pd.DataFrame,
    *,
    galaxy_index: int,
    direction: str,
    fiber_mode: str,
    fits_root: Path,
    pair: tuple[float, float],
) -> tuple[dict, dict]:
    first = load_render(
        lookup_render(
            renders,
            galaxy_index=galaxy_index,
            direction=direction,
            g_amp=pair[0],
            fiber_mode=fiber_mode,
        ),
        fits_root=fits_root,
    )
    second = load_render(
        lookup_render(
            renders,
            galaxy_index=galaxy_index,
            direction=direction,
            g_amp=pair[1],
            fiber_mode=fiber_mode,
        ),
        fits_root=fits_root,
    )
    return first, second


def compare_pairs(small_lo: dict, small_hi: dict, large_lo: dict, large_hi: dict) -> dict:
    d_image_small = small_hi["image"] - small_lo["image"]
    d_image_large = large_hi["image"] - large_lo["image"]
    registered, shift = shift_register(d_image_small, d_image_large)
    spec_small = small_hi["spectra"] - small_lo["spectra"]
    spec_large = large_hi["spectra"] - large_lo["spectra"]
    spectra = []
    for index, name in enumerate(FIBER_NAMES):
        spectra.append(
            {
                "fiber": name,
                "cosine": cosine(spec_small[index], spec_large[index]),
                "amplitude_ratio": amplitude_ratio(spec_small[index], spec_large[index]),
            }
        )
    return {
        "image_cosine": cosine(d_image_small, d_image_large),
        "image_cosine_shift_registered": cosine(d_image_small, registered),
        "image_amplitude_ratio": amplitude_ratio(d_image_small, d_image_large),
        "image_shift_pixels": [int(shift[0]), int(shift[1])],
        "image_d_small": d_image_small,
        "image_d_large": d_image_large,
        "spectra": spectra,
        "spec_d_small": spec_small,
        "spec_d_large": spec_large,
        "small_hi_high_frequency": high_frequency_fraction(small_hi["image"]),
        "large_hi_high_frequency": high_frequency_fraction(large_hi["image"]),
    }


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


def save_fiber_path_figure(galaxies: pd.DataFrame, report_dir: Path) -> str:
    plt = _setup_matplotlib()
    pa_rows = (("aligned", "disk long axis along the camera stretch"), ("perpendicular", "disk long axis across the camera stretch"))
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharex=True, sharey=True)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    g_grid = np.linspace(0.0, 0.2, 41)
    for row_index, (pa_name, pa_title) in enumerate(pa_rows):
        for col_index, (sini_name, _, _) in enumerate(SINI_BINS):
            match = galaxies.loc[
                (galaxies["inclination_bin"] == sini_name)
                & (galaxies["size_bin"] == "extended")
                & (galaxies["pa_bin"] == pa_name)
            ]
            galaxy = match.iloc[0]
            axis = axes[row_index, col_index]
            tracks = fiber_paths(galaxy, "image_g1", g_grid)
            for index, name in enumerate(FIBER_NAMES):
                axis.plot(
                    tracks[:, index, 0],
                    tracks[:, index, 1],
                    color=colors[index],
                    lw=1.6,
                    label=name.replace(" fiber", "") if row_index == 0 and col_index == 0 else None,
                )
                axis.scatter(
                    tracks[0, index, 0],
                    tracks[0, index, 1],
                    color=colors[index],
                    s=18,
                    zorder=3,
                )
            small_pos = fiber_paths(galaxy, "image_g1", np.asarray([0.02]))[0]
            large_pos = fiber_paths(galaxy, "image_g1", np.asarray([0.16]))[0]
            d_small = fiber_paths(galaxy, "image_g1", np.asarray(SMALL_PAIR))
            d_large = fiber_paths(galaxy, "image_g1", np.asarray(LARGE_PAIR))
            arrow_scale = 8.0
            for index in range(5):
                dx_s, dy_s = (d_small[1, index] - d_small[0, index]) * arrow_scale
                dx_l, dy_l = (d_large[1, index] - d_large[0, index]) * arrow_scale
                axis.annotate(
                    "",
                    xy=(small_pos[index, 0] + dx_s, small_pos[index, 1] + dy_s),
                    xytext=(small_pos[index, 0], small_pos[index, 1]),
                    arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.2},
                )
                axis.annotate(
                    "",
                    xy=(large_pos[index, 0] + dx_l, large_pos[index, 1] + dy_l),
                    xytext=(large_pos[index, 0], large_pos[index, 1]),
                    arrowprops={"arrowstyle": "->", "color": "#a31f34", "lw": 1.2},
                )
            axis.set_aspect("equal")
            if row_index == 0:
                axis.set_title(inclination_label(sini_name))
            if row_index == 1:
                axis.set_xlabel("east offset (arcsec)")
            if col_index == 0:
                axis.set_ylabel(f"{pa_title}\nnorth offset (arcsec)")
            axis.axhline(0.0, color="#bbbbbb", lw=0.6)
            axis.axvline(0.0, color="#bbbbbb", lw=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Fiber sky positions as camera-axis shear grows from 0 to 0.2",
        y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path = report_dir / "fiber_paths.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_image_diff_figure(rows: list[dict], report_dir: Path) -> str:
    plt = _setup_matplotlib()
    picks = []
    for sini_name, _, _ in SINI_BINS:
        for row in rows:
            if (
                row["inclination_bin"] == sini_name
                and row["direction"] == "galaxy_plus"
                and row["size_bin"] == "extended"
                and row["pa_bin"] == "aligned"
            ):
                picks.append(row)
                break
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 10.0))
    for row_index, row in enumerate(picks):
        small = np.asarray(row["moving"]["image_d_small"])
        large = np.asarray(row["moving"]["image_d_large"])
        vmax = np.percentile(np.abs(np.concatenate([small.ravel(), large.ravel()])), 99.5)
        vmax = max(float(vmax), 1.0e-12)
        for col, stamp, title in (
            (0, small, "change from 0.01 to 0.03"),
            (1, large, "change from 0.15 to 0.17"),
        ):
            axis = axes[row_index, col]
            axis.imshow(stamp, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title)
        axes[row_index, 0].set_ylabel(inclination_label(row["inclination_bin"]))
    fig.suptitle(
        "Image change for a 0.02 shear step, same color scale in each row",
        y=0.995,
    )
    fig.tight_layout()
    path = report_dir / "image_differences.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_cosine_strip(rows: list[dict], report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    bins = [name for name, _, _ in SINI_BINS]
    rng = np.random.default_rng(0)
    for axis, direction, title in (
        (axes[0], "image_g1", "Shear along the first camera axis"),
        (axes[1], "galaxy_plus", "Shear along the disk major axis"),
    ):
        for offset, sini_name in enumerate(bins):
            values = [
                row["moving"]["image_cosine"]
                for row in rows
                if row["direction"] == direction and row["inclination_bin"] == sini_name
            ]
            jitter = rng.normal(0.0, 0.04, size=len(values))
            axis.scatter(
                np.full(len(values), offset) + jitter,
                values,
                s=36,
                color="#1f4e79",
                zorder=3,
            )
        axis.set_xticks(range(len(bins)))
        axis.set_xticklabels([inclination_label(name) for name in bins], rotation=15)
        axis.set_title(title)
        axis.set_ylim(-0.05, 1.05)
        axis.axhline(1.0, color="#bbbbbb", lw=0.8)
        axis.axhline(0.0, color="#bbbbbb", lw=0.8)
    axes[0].set_ylabel("cosine of the two difference maps")
    fig.suptitle("Do the image changes point in the same direction?")
    fig.tight_layout()
    path = report_dir / "image_cosine.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_spectrum_cosine(rows: list[dict], report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axis = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(FIBER_NAMES))
    moving = []
    frozen = []
    for index in range(len(FIBER_NAMES)):
        moving.append(
            np.nanmean(
                [
                    row["moving"]["spectra"][index]["cosine"]
                    for row in rows
                    if row["direction"] == "image_g1"
                ]
            )
        )
        frozen.append(
            np.nanmean(
                [
                    row["frozen"]["spectra"][index]["cosine"]
                    for row in rows
                    if row["direction"] == "image_g1"
                ]
            )
        )
    width = 0.36
    axis.bar(x - width / 2, moving, width, label="apertures move with shear", color="#1f4e79")
    axis.bar(x + width / 2, frozen, width, label="apertures held at the small-shear positions", color="#c47b15")
    axis.set_xticks(x)
    axis.set_xticklabels([name.replace(" fiber", "\nfiber") for name in FIBER_NAMES])
    axis.set_ylim(-1.05, 1.05)
    axis.axhline(0.0, color="#bbbbbb", lw=0.8)
    axis.set_ylabel("mean cosine of the spectral difference")
    axis.set_title("Spectra: moving apertures versus frozen apertures")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path = report_dir / "spectrum_cosine.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def save_frozen_vs_moving(rows: list[dict], report_dir: Path) -> str:
    plt = _setup_matplotlib()
    fig, axis = plt.subplots(figsize=(5.6, 5.2))
    moving = [
        row["moving"]["spectra_mean_cosine"]
        for row in rows
        if row["direction"] == "image_g1"
    ]
    frozen = [
        row["frozen"]["spectra_mean_cosine"]
        for row in rows
        if row["direction"] == "image_g1"
    ]
    axis.scatter(moving, frozen, s=42, color="#1f4e79")
    axis.plot([-1, 1], [-1, 1], color="#bbbbbb", lw=1)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("cosine with moving apertures")
    axis.set_ylabel("cosine with frozen apertures")
    axis.set_title("Each galaxy, shear along the first camera axis")
    axis.set_aspect("equal")
    fig.tight_layout()
    path = report_dir / "frozen_vs_moving.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def fiber_takeaway(subset: list[dict]) -> str:
    travels = [row["analytic"]["max_travel_arcsec"] for row in subset]
    flips = [row["analytic"]["axis_flip_g"] for row in subset]
    finite_flips = [value for value in flips if value is not None]
    max_travel = max(travels) if travels else float("nan")
    if not np.isfinite(max_travel) or max_travel < 0.05:
        return (
            "The apertures stay on the same parts of the disk. Camera-axis shear "
            "does not rotate the observed axes enough to walk the fibers."
        )
    if finite_flips:
        lowest = min(finite_flips)
        return (
            f"When the disk is already across the camera stretch, the photometric "
            f"long axis can swap onto the stretch direction near shear {lowest:.2f}. "
            "The fibers then hop by about two arcseconds onto a new pair of axes. "
            "Disks already aligned with the camera do not show this hop."
        )
    return (
        "The apertures slide, but they do not jump to a new pair of axes in this bin."
    )


def build_payload(
    galaxies: pd.DataFrame,
    renders: pd.DataFrame,
    *,
    fits_root: Path,
) -> dict:
    rows = []
    for _, galaxy in galaxies.iterrows():
        galaxy_index = int(galaxy["galaxy_index"])
        for direction in DIRECTIONS:
            moving_lo, moving_hi = pair_products(
                renders,
                galaxy_index=galaxy_index,
                direction=direction,
                fiber_mode="moving",
                fits_root=fits_root,
                pair=SMALL_PAIR,
            )
            moving_large_lo, moving_large_hi = pair_products(
                renders,
                galaxy_index=galaxy_index,
                direction=direction,
                fiber_mode="moving",
                fits_root=fits_root,
                pair=LARGE_PAIR,
            )
            frozen_large_lo, frozen_large_hi = pair_products(
                renders,
                galaxy_index=galaxy_index,
                direction=direction,
                fiber_mode="frozen",
                fits_root=fits_root,
                pair=LARGE_PAIR,
            )
            moving = compare_pairs(
                moving_lo, moving_hi, moving_large_lo, moving_large_hi
            )
            frozen = compare_pairs(
                moving_lo, moving_hi, frozen_large_lo, frozen_large_hi
            )
            moving["spectra_mean_cosine"] = mean_or_nan(
                [item["cosine"] for item in moving["spectra"]]
            )
            frozen["spectra_mean_cosine"] = mean_or_nan(
                [item["cosine"] for item in frozen["spectra"]]
            )
            analytic = analytic_fiber_metrics(galaxy, direction)
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
                    "analytic": analytic,
                    "moving": moving,
                    "frozen": frozen,
                }
            )
    return {"galaxies": galaxies.to_dict(orient="records"), "rows": rows}


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    rows = payload["rows"]
    image_g1 = [row for row in rows if row["direction"] == "image_g1"]
    galaxy_plus = [row for row in rows if row["direction"] == "galaxy_plus"]

    def fmt(value: float) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{value:.3f}"

    def mean_for(subset, key):
        return mean_or_nan([float(np.asarray(row["moving"][key])) for row in subset])

    summary_rows = []
    takeaways = []
    for sini_name, _, _ in SINI_BINS:
        fiber_subset = [row for row in image_g1 if row["inclination_bin"] == sini_name]
        phot_subset = [row for row in galaxy_plus if row["inclination_bin"] == sini_name]
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(inclination_label(sini_name))}</td>"
            f"<td>{fmt(mean_or_nan([row['analytic']['all_fibers_cosine'] for row in fiber_subset]))}</td>"
            f"<td>{fmt(mean_or_nan([row['moving']['image_cosine'] for row in phot_subset]))}</td>"
            f"<td>{fmt(mean_or_nan([row['moving']['spectra_mean_cosine'] for row in fiber_subset]))}</td>"
            f"<td>{fmt(mean_or_nan([row['frozen']['spectra_mean_cosine'] for row in fiber_subset]))}</td>"
            "</tr>"
        )
        takeaways.append(
            f"<li><strong>{html.escape(inclination_label(sini_name))}.</strong> "
            f"{html.escape(fiber_takeaway(fiber_subset))}</li>"
        )
    ringing_ratio = mean_or_nan(
        [
            row["moving"]["large_hi_high_frequency"]
            / row["moving"]["small_hi_high_frequency"]
            for row in galaxy_plus
            if row["moving"]["small_hi_high_frequency"] not in (0.0, None)
            and np.isfinite(row["moving"]["small_hi_high_frequency"])
            and np.isfinite(row["moving"]["large_hi_high_frequency"])
        ]
    )
    if np.isfinite(ringing_ratio) and ringing_ratio > 2.5:
        ringing_note = (
            "The large-shear stamps carry more small-scale power than the small-shear "
            "stamps. Treat bright rings or checkerboards there as a renderer artifact, "
            "not as a new physical regime."
        )
    else:
        ringing_note = (
            "The large-shear stamps do not show a jump in small-scale power that would "
            "point to renderer ringing rather than a real change in the galaxy image."
        )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Does shear look different above 0.1?</title>
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
<h1>Does a shear of 0.16 change the observation in a different way than a shear of 0.02?</h1>
<p class="lead">Training a model with a wider shear range made the recovered shear
<em>slope</em> worse. Shape noise did not collapse. One possible reason is that,
once a galaxy is strongly sheared, both the image and the spectrograph
apertures become a different kind of observable. This page checks that idea
in the simulator, not in a trained network.</p>

<h2>What was held fixed, and what was changed</h2>
<p>Twelve galaxies. For each one, size, brightness, inclination, position
angle, and rotation speed stay the same. Only shear changes. We bump shear
by the same amount, 0.02, in two places:</p>
<ul>
<li>a small-shear neighborhood, from 0.01 to 0.03</li>
<li>a large-shear neighborhood, from 0.15 to 0.17</li>
</ul>
<p>If those two bumps rearrange the same pixels and the same spectral
channels, the observation is still the same shear meter, just evaluated
somewhere else. If they rearrange different pixels or different fibers, the
observation has changed character. The number we quote is the cosine of the
angle between the two difference maps: 1 means the same pattern, 0 means
unrelated patterns. The overall strength of the bump is allowed to differ.</p>
<p>We repeat the bump in two directions: along the first camera axis, and
along the galaxy’s own major axis. Nearly face-on disks are left out,
because they have no stable photometric axis.</p>

<h2>Test A — Where the fibers go</h2>
<p>The five spectrograph fibers sit on the observed major and minor axes of
the galaxy, not on its intrinsic axes. As shear grows, those observed axes
rotate, so the apertures can slide across the disk. This first test uses
only the geometry of that placement. No image is rendered.</p>
<figure>
<img src="{html.escape(figures['fiber_paths'])}" alt="Fiber sky tracks as shear increases">
<figcaption>Top row: the galaxy’s long axis already lies along the camera
stretch. Bottom row: it lies across that stretch. Dots mark the unsheared
placement. Navy arrows are a 0.02 bump near shear 0.02; red arrows are the
same bump near 0.16 (arrows magnified). A hop from one axis pair to the
other means the fibers have landed on a new part of the disk.</figcaption>
</figure>
<ul>
{''.join(takeaways)}
</ul>
<p>Shear along the disk’s own long axis does not rotate those axes, so the
fibers stay put. The interesting case is camera-axis shear. The cosine of
the 0.01–0.03 fiber step versus the 0.15–0.17 step is
{fmt(mean_or_nan([row['analytic']['all_fibers_cosine'] for row in image_g1]))}
(1 means those two small bumps move the fibers in the same way). That number
can stay high even after a hop that already happened in between; the tracks
above are the check for a jump.</p>

<h2>Test B — Does the image change the same way?</h2>
<p>Now the full simulator: noiseless images, apertures allowed to follow
the observed axes. Each row below is one galaxy. The left stamp is the
image change from 0.01 to 0.03. The right stamp is the change from 0.15 to
0.17. Both use the same color scale, so a fainter right-hand stamp means a
weaker response, not a different pattern.</p>
<figure>
<img src="{html.escape(figures['image_differences'])}" alt="Image difference stamps at small and large shear">
<figcaption>Shear along the disk major axis. If the two columns look like
scaled copies of each other, the image change has the same pattern. If
bright and dark lobes appear in new places, the pattern has changed.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['image_cosine'])}" alt="Cosine similarity of image differences">
<figcaption>Each point is one galaxy. Values near 1 mean the small-shear and
large-shear image changes point the same way in pixel space. Mean cosine
along the camera axis: {fmt(mean_for(image_g1, 'image_cosine'))}. Along the
disk major axis: {fmt(mean_for(galaxy_plus, 'image_cosine'))}.</figcaption>
</figure>
<p>That cosine ignores overall strength. The large-shear image change is on
average {fmt(mean_for(galaxy_plus, 'image_amplitude_ratio'))} times as strong
as the small-shear change when shear is along the disk major axis, and
{fmt(mean_for(image_g1, 'image_amplitude_ratio'))} times as strong along the
camera axis.</p>
<p>If a low cosine were only the galaxy sliding on the postage stamp, lining
the two difference images up with a pixel shift would restore it. After that
shift, the mean cosine along the disk major axis is
{fmt(mean_or_nan([row['moving']['image_cosine_shift_registered'] for row in galaxy_plus]))}.
A remaining drop is not a translation.</p>
<p>{html.escape(ringing_note)}</p>

<h2>Test C — Freeze the fiber positions</h2>
<p>The image does not know about the fibers. The spectra do. To separate
“the galaxy looks different” from “the apertures have moved,” we render the
large-shear pair a second time with the fibers locked at their 0.01
placement. The galaxy is still sheared; only the spectrograph pointing is
frozen. This comparison uses shear along the first camera axis, the
direction in which the fibers can hop.</p>
<figure>
<img src="{html.escape(figures['spectrum_cosine'])}" alt="Spectral cosine with moving versus frozen fibers">
<figcaption>Average over the twelve galaxies, shear along the first camera
axis (the direction in which apertures can hop). Center, major-axis pair,
and minor-axis pair are labeled in ordinary language. If freezing the
apertures restores the cosine, the mismatch was the fibers walking. If it
does not, the spectra themselves are a different function of shear.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['frozen_vs_moving'])}" alt="Per-galaxy frozen versus moving spectral cosine">
<figcaption>Each point is one galaxy. Points on the diagonal are unaffected
by locking the fibers. Points above the diagonal mean frozen apertures make
the large-shear spectral change look more like the small-shear change.
Galaxies whose fibers hopped land on the left: moving apertures anti-align
the two spectral differences, and freezing them brings the cosine back
near 1.</figcaption>
</figure>

<h2>Summary</h2>
<table>
<thead><tr>
<th>Inclination</th>
<th>Fiber-path cosine</th>
<th>Image cosine</th>
<th>Spectra, moving apertures</th>
<th>Spectra, frozen apertures</th>
</tr></thead>
<tbody>
{''.join(summary_rows)}
</tbody>
</table>
<p class="note">Numbers are means over the four galaxies in that inclination
bin. The fiber-path and spectra columns use shear along the first camera
axis, which is the case where apertures can hop. The image column uses shear
along the disk major axis. Frozen means the spectrograph pointing is held at
the small-shear placement while the galaxy is still sheared.</p>

<h2>What we saw</h2>
<p>The image change at large shear is almost the same pattern as at small
shear, and lining the stamps up does not change that. The fibers are a
different story. When the disk already lies across the camera stretch, a
shear of order 0.1 can swap which axis looks longer, and the five apertures
hop onto a new pair of axes. Then the spectral change at large shear points
the opposite way from the small-shear change, and locking the apertures in
place restores the match. That is an aperture effect, not a new look to the
galaxy image itself.</p>

<h2>What this can and cannot say</h2>
<p>This is a check of the forward model. It does not train a network, and it
does not prove that a network trained on large shears used these features on
weakly sheared test galaxies. If the cosines stay near 1, it is harder to
blame a qualitatively different observable, and the remaining puzzle is how
the training mixture weights large-shear labels. If the cosines drop, and
especially if freezing the fibers restores the spectra while the image stays
disagreed, then large-shear training really is seeing a different
instrument-plus-galaxy response. Connecting that to multiplicative bias
would still take a later training run that ignores labels with shear above
0.1.</p>
</body></html>
"""
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "report.html"
    html_path.write_text(body, encoding="utf-8")
    print(f"Wrote {html_path}")


def strip_arrays(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        moving = {
            key: value
            for key, value in row["moving"].items()
            if key
            not in {
                "image_d_small",
                "image_d_large",
                "spec_d_small",
                "spec_d_large",
            }
        }
        frozen = {
            key: value
            for key, value in row["frozen"].items()
            if key
            not in {
                "image_d_small",
                "image_d_large",
                "spec_d_small",
                "spec_d_large",
            }
        }
        analytic = dict(row["analytic"])
        analytic.pop("tracks", None)
        analytic.pop("d_small", None)
        analytic.pop("d_large", None)
        cleaned.append({**row, "moving": moving, "frozen": frozen, "analytic": analytic})
    return cleaned


def write_report(*, sample_root: Path, fits_root: Path, report_dir: Path) -> None:
    galaxies = pd.read_csv(sample_root / GALAXY_CSV, float_precision="round_trip")
    renders = pd.read_csv(sample_root / RENDER_CSV, float_precision="round_trip")
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(galaxies, renders, fits_root=fits_root)
    figures = {
        "fiber_paths": save_fiber_path_figure(galaxies, report_dir),
        "image_differences": save_image_diff_figure(payload["rows"], report_dir),
        "image_cosine": save_cosine_strip(payload["rows"], report_dir),
        "spectrum_cosine": save_spectrum_cosine(payload["rows"], report_dir),
        "frozen_vs_moving": save_frozen_vs_moving(payload["rows"], report_dir),
    }
    write_html(payload, figures, report_dir)
    sidecar = {
        "galaxies": payload["galaxies"],
        "rows": json_safe(strip_arrays(payload["rows"])),
        "figures": figures,
    }
    json_path = report_dir / "report.json"
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.write_tables or args.render_index is not None or args.report):
        raise SystemExit("specify --write-tables, --render-index, or --report")
    if args.write_tables:
        write_tables(sample_root=args.sample_root)
    if args.render_index is not None:
        if not (1 <= args.render_index <= 12):
            raise ValueError("render-index must be in 1..12")
        render_galaxy(
            args.render_index - 1,
            sample_root=args.sample_root,
            fits_root=args.fits_root,
        )
    if args.report:
        write_report(
            sample_root=args.sample_root,
            fits_root=args.fits_root,
            report_dir=args.report_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
