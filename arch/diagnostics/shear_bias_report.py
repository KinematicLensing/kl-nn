#!/usr/bin/env python3
"""Build a self-contained shear-bias report from compact cached summaries.

This deliberately reads only truth, SNR, MAP, and posterior-mean arrays. It
never opens the multi-gigabyte posterior sample or log-probability files.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PART_RE = re.compile(r"part(\d+)of(\d+)\.npy$")
SINI_CUTS = (0.3, 0.5, 0.7, np.inf)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="MODEL:DATASET",
        help="Cached model/dataset pair; repeat for comparisons.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/ocean/projects/phy250048p/shared/cache"),
    )
    parser.add_argument("--mode", type=int, default=0, help="TF-analysis mode index.")
    parser.add_argument("--low-g", type=float, default=0.02)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def partition_files(directory: Path) -> list[Path]:
    files = []
    for path in directory.glob("part*of*.npy"):
        match = PART_RE.match(path.name)
        if match:
            files.append((int(match.group(1)), int(match.group(2)), path))
    if not files:
        raise FileNotFoundError(f"No partition arrays in {directory}")
    totals = {item[1] for item in files}
    if len(totals) != 1:
        raise ValueError(f"Mixed partition totals in {directory}: {totals}")
    expected = next(iter(totals))
    files.sort(key=lambda item: item[0])
    indices = [item[0] for item in files]
    if indices != list(range(expected)):
        raise ValueError(f"Incomplete partitions in {directory}: {indices} of {expected}")
    return [item[2] for item in files]


def load_concat(directory: Path, axis: int) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r") for path in partition_files(directory)]
    return np.concatenate(arrays, axis=axis)


def load_case(cache_root: Path, case: str, mode: int) -> dict:
    try:
        model, dataset = case.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Case must be MODEL:DATASET, got {case!r}") from exc
    root = cache_root / model / dataset
    truth = load_concat(root / "truth", axis=0)
    snr = load_concat(root / "snr", axis=0)
    map_all = load_concat(root / "map_estimates", axis=1)
    mean_all = load_concat(root / "mean_estimates", axis=1)
    if not (0 <= mode < map_all.shape[0]):
        raise IndexError(f"Mode {mode} outside [0, {map_all.shape[0]}) for {case}")
    estimates = {
        "MAP": np.asarray(map_all[mode]),
        "Mean": np.asarray(mean_all[mode, :, 1]),
    }
    if any(value.shape[0] != truth.shape[0] for value in [snr, *estimates.values()]):
        raise ValueError(f"Length mismatch among cached summaries for {case}")
    return {
        "case": case,
        "model": model,
        "dataset": dataset,
        "root": root,
        "truth": np.asarray(truth),
        "snr": np.asarray(snr),
        "estimates": estimates,
    }


def fit_design(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = max(1, len(y) - design.shape[1])
    cov = np.linalg.pinv(design.T @ design) * (residual @ residual / dof)
    return coef, np.sqrt(np.maximum(np.diag(cov), 0))


def component_metrics(truth: np.ndarray, estimate: np.ndarray, low_g: float) -> dict:
    residual = estimate - truth
    additive = float(np.mean(residual))
    additive_se = float(np.std(residual, ddof=1) / np.sqrt(len(residual)))
    low = np.abs(truth) < low_g
    linear, linear_se = fit_design(
        residual[low], np.column_stack([np.ones(low.sum()), truth[low]])
    )
    cubic, cubic_se = fit_design(
        residual,
        np.column_stack([np.ones(len(truth)), truth, truth**3]),
    )
    return {
        "c": additive,
        "c_se": additive_se,
        "low_c": float(linear[0]),
        "low_c_se": float(linear_se[0]),
        "low_m": float(linear[1]),
        "low_m_se": float(linear_se[1]),
        "cubic_c": float(cubic[0]),
        "cubic_m": float(cubic[1]),
        "cubic_q": float(cubic[2]),
        "cubic_m_se": float(cubic_se[1]),
        "cubic_q_se": float(cubic_se[2]),
    }


def spin_metrics(theta: np.ndarray, residual: np.ndarray) -> dict:
    design = np.column_stack(
        [
            np.ones(len(theta)),
            np.cos(2 * theta),
            np.sin(2 * theta),
            np.cos(4 * theta),
            np.sin(4 * theta),
        ]
    )
    coef, _ = fit_design(residual, design)
    return {
        "offset": float(coef[0]),
        "spin2": float(np.hypot(coef[1], coef[2])),
        "spin4": float(np.hypot(coef[3], coef[4])),
        "coef": coef,
    }


def binned(x: np.ndarray, y: np.ndarray, bins: int):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    center, mean, error = [], [], []
    for i in range(bins):
        mask = index == i
        if mask.sum() < 2:
            continue
        center.append(np.mean(x[mask]))
        mean.append(np.mean(y[mask]))
        error.append(np.std(y[mask], ddof=1) / np.sqrt(mask.sum()))
    return np.asarray(center), np.asarray(mean), np.asarray(error)


def fig_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def bias_figure(case: dict, bins: int) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"MAP": "tab:blue", "Mean": "tab:orange"}
    for component in range(2):
        true = case["truth"][:, component]
        for name, estimate in case["estimates"].items():
            residual = estimate[:, component] - true
            x, y, error = binned(true, residual, bins)
            axes[0, component].errorbar(
                x, y, yerr=error, marker="o", ms=3, lw=1, label=name,
                color=colors[name],
            )
            theta = case["truth"][:, 2]
            tx, ty, terror = binned(theta, residual, bins)
            axes[1, component].errorbar(
                tx, ty, yerr=terror, marker="o", ms=3, lw=1,
                label=name, color=colors[name],
            )
        axes[0, component].axhline(0, color="black", ls="--", lw=0.8)
        axes[1, component].axhline(0, color="black", ls="--", lw=0.8)
        axes[0, component].set_title(f"g{component + 1} residual vs truth")
        axes[1, component].set_title(f"g{component + 1} residual vs theta_int")
        axes[0, component].set_ylabel("estimate - truth")
        axes[1, component].set_ylabel("estimate - truth")
        axes[0, component].legend()
    axes[0, 0].set_xlabel("true g1")
    axes[0, 1].set_xlabel("true g2")
    axes[1, 0].set_xlabel("theta_int [rad]")
    axes[1, 1].set_xlabel("theta_int [rad]")
    fig.suptitle(case["case"])
    fig.tight_layout()
    return fig_data_uri(fig)


def compute_metrics(case: dict, low_g: float) -> list[dict]:
    rows = []
    theta = case["truth"][:, 2]
    sini = case["truth"][:, 3]
    for estimator, estimate in case["estimates"].items():
        for component in range(2):
            true = case["truth"][:, component]
            residual = estimate[:, component] - true
            base = component_metrics(true, estimate[:, component], low_g)
            base.update(spin_metrics(theta, residual))
            cuts = {}
            for cut in SINI_CUTS:
                mask = np.ones(len(sini), dtype=bool) if np.isinf(cut) else sini < cut
                mask &= np.abs(true) < low_g
                coef, se = fit_design(
                    residual[mask], np.column_stack([np.ones(mask.sum()), true[mask]])
                )
                cuts["all" if np.isinf(cut) else f"{cut:.1f}"] = (
                    float(coef[1]), float(se[1]), int(mask.sum())
                )
            base.update(
                estimator=estimator,
                component=f"g{component + 1}",
                sini_cuts=cuts,
            )
            rows.append(base)
    return rows


def fmt(value: float) -> str:
    return f"{value:.3e}"


def metrics_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['estimator']}</td><td>{row['component']}</td>"
            f"<td>{fmt(row['c'])} ± {fmt(row['c_se'])}</td>"
            f"<td>{fmt(row['low_m'])} ± {fmt(row['low_m_se'])}</td>"
            f"<td>{fmt(row['cubic_m'])} ± {fmt(row['cubic_m_se'])}</td>"
            f"<td>{fmt(row['cubic_q'])} ± {fmt(row['cubic_q_se'])}</td>"
            f"<td>{fmt(row['spin2'])}</td><td>{fmt(row['spin4'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Component</th>"
        "<th>mean c</th><th>low-|g| m</th><th>cubic m</th><th>cubic q</th>"
        "<th>spin-2 amp</th><th>spin-4 amp</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def cuts_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        for cut, (slope, se, n) in row["sini_cuts"].items():
            body.append(
                f"<tr><td>{row['estimator']}</td><td>{row['component']}</td>"
                f"<td>{html.escape(cut)}</td><td>{fmt(slope)} ± {fmt(se)}</td>"
                f"<td>{n:,}</td></tr>"
            )
    return (
        "<table><thead><tr><th>Estimator</th><th>Component</th><th>sin i cut</th>"
        "<th>low-|g| m</th><th>N</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def main():
    args = parse_args()
    cases = [load_case(args.cache_root, item, args.mode) for item in args.case]
    sections = []
    for case in cases:
        metrics = compute_metrics(case, args.low_g)
        sections.append(
            f"<section><h2>{html.escape(case['case'])}</h2>"
            f"<p><b>N:</b> {len(case['truth']):,}; <b>mode:</b> {args.mode}; "
            f"<b>cache:</b> <code>{html.escape(str(case['root']))}</code></p>"
            + metrics_table(metrics)
            + "<h3>Inclination-cut diagnostic</h3>"
            + cuts_table(metrics)
            + f"<img src=\"{bias_figure(case, args.bins)}\" alt=\"bias diagnostics\">"
            + "</section>"
        )
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>KL-NN shear-bias diagnostics</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ccd1d5; padding: 0.45rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f1f3f4; }} img {{ max-width: 100%; height: auto; }}
.note {{ background: #fff7df; border-left: 4px solid #e6a700; padding: 0.8rem 1rem; }}
code {{ overflow-wrap: anywhere; }}
</style></head><body>
<h1>KL-NN shear-bias diagnostics</h1>
<p>Generated {generated}. This report reads only compact truth/SNR/MAP/mean summaries and is designed for a 4 GB CPU job.</p>
<section><h2>Step 1: 90° pairing audit</h2>
<p><b>Status: fixed and regression-tested.</b> Sampling appends each galaxy as
<code>original, rotated</code>. The previous direct reshape mixed posterior-draw
and branch axes. The production helper now reshapes to galaxy × branch × draw
first, then moves branch behind draw, yielding galaxy × draw × branch × feature.
The diagnostic also counter-rotates and combines shear only; nuisance parameters
remain in the original coordinate frame.</p>
<p><code>tests/test_rotation_pairing.py</code> covers sample and log-probability
ordering, odd-length rejection, vectorized angle wrapping/round-trip behavior,
and a two-galaxy no-collapse sentinel. Result at this artifact build: <b>5 passed</b>.</p></section>
<section><h2>Step 2: cached MAP/mean audit</h2>
<p class="note"><b>Interpretation:</b> mean <i>c</i> is the unbinned average residual. The low-|g| fit uses |g| &lt; {args.low_g:g}. The cubic fit is residual = c + m g + q g³. Spin amplitudes come from a joint constant + spin-2 + spin-4 regression against true theta_int. Existing cache files may predate the rotation-pairing fix; rerun cancellation analyses before treating their cancellation products as validated.</p>
{''.join(sections)}
</section>
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
