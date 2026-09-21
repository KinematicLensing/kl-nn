#!/usr/bin/env python3
"""Compile xu3 low-|g| m and ESS across weighting and catalog cuts.

Uses the existing 100k frozen-concat TF-conformed cache. Streams Tully–Fisher
posterior shear variances once (or reuses a sidecar), then evaluates the
selection table in memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys

import numpy as np

DIAG_DIR = Path(__file__).resolve().parent
ARCH_DIR = DIAG_DIR.parent
for path in (str(DIAG_DIR), str(ARCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from shear_bias_report import (
    component_metrics,
    compose_shape_noise_regularized_weights,
    load_case,
    posterior_component_variance,
    weighted_mean_and_se,
)
from tf_prior import effective_sample_size


LOGGER = logging.getLogger("compile_xu3_selection")
DEFAULT_CASE = (
    "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702:"
    "test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42"
)
DEFAULT_CACHE_ROOT = Path("/ocean/projects/phy250048p/shared/cache")
DEFAULT_OUTPUT = Path(
    "/ocean/projects/phy250048p/shared/reports/xu3-estimators/"
    "xu3_baseline100k_selection_m_ess_s42.html"
)
DEFAULT_SIDECAR = Path(
    "/ocean/projects/phy250048p/shared/reports/xu3-estimators/"
    "xu3_baseline100k_tf_shear_variances.npz"
)
DEFAULT_JSON = Path(
    "/ocean/projects/phy250048p/shared/reports/xu3-estimators/"
    "xu3_baseline100k_selection_m_ess_s42.json"
)
LOW_G = 0.02
CATALOG_N = 100_000
LOW_G_N = 20_000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--low-g", type=float, default=LOW_G)
    parser.add_argument(
        "--reuse-sidecar",
        action="store_true",
        help="Load variances from --sidecar if it exists instead of streaming.",
    )
    return parser.parse_args(argv)


def batch_weighted_variance(
    values: np.ndarray, log_weight: np.ndarray
) -> np.ndarray:
    """TF-weighted posterior variance of a shear component, one row per galaxy."""

    values = np.asarray(values, dtype=np.float64)
    log_weight = np.asarray(log_weight, dtype=np.float64)
    if values.shape != log_weight.shape:
        raise ValueError("values and log_weight must have the same shape")
    finite = np.isfinite(values) & np.isfinite(log_weight)
    filled_log = np.where(finite, log_weight, -np.inf)
    maximum = np.max(filled_log, axis=1, keepdims=True)
    scaled = np.exp(filled_log - maximum)
    scaled = np.where(finite, scaled, 0.0)
    total = scaled.sum(axis=1, keepdims=True)
    weight = np.divide(scaled, total, out=np.zeros_like(scaled), where=total > 0)
    filled_values = np.where(finite, values, 0.0)
    mean = np.sum(weight * filled_values, axis=1, keepdims=True)
    variance = np.sum(weight * np.square(filled_values - mean), axis=1)
    return np.where(total[:, 0] > 0.0, variance, np.nan)


def stream_tf_variances(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load TF-weighted g1/g2 posterior variances from the compact candidate bank."""

    truth = np.asarray(case["truth"])
    cache_partitions = case["cache_partitions"]
    sample_files = cache_partitions.files["shear_sample"]
    weight_files = cache_partitions.files["posterior_tf_log_weight"]
    truth_files = cache_partitions.files["truth"]
    g1 = np.full(len(truth), np.nan, dtype=np.float64)
    g2 = np.full(len(truth), np.nan, dtype=np.float64)
    offset = 0
    for part_index, (sample_path, weight_path, truth_path) in enumerate(
        zip(sample_files, weight_files, truth_files)
    ):
        samples = np.load(sample_path, mmap_mode="r")
        log_weight = np.load(weight_path, mmap_mode="r")
        stored_truth = np.load(truth_path, mmap_mode="r")
        take = min(samples.shape[0], len(truth) - offset)
        if not np.array_equal(
            np.asarray(stored_truth[:take]), truth[offset : offset + take],
            equal_nan=True,
        ):
            raise ValueError(f"Truth rows in {truth_path} do not align")
        draws = np.asarray(samples[:take, :, :2], dtype=np.float64)
        weights = np.asarray(log_weight[:take], dtype=np.float64)
        sl = slice(offset, offset + take)
        g1[sl] = batch_weighted_variance(draws[:, :, 0], weights)
        g2[sl] = batch_weighted_variance(draws[:, :, 1], weights)
        if part_index == 0:
            probe = posterior_component_variance(draws[0, :, 0], weights[0])
            if not np.isclose(g1[offset], probe, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    "vectorized TF variance does not match "
                    f"posterior_component_variance: {g1[offset]} vs {probe}"
                )
        offset += take
        LOGGER.info(
            "streamed partition %d/%d (%d galaxies)",
            part_index + 1,
            len(sample_files),
            offset,
        )
        del draws, weights, samples, log_weight, stored_truth
    if offset != len(truth):
        raise ValueError(f"Sample cache has {offset} galaxies, expected {len(truth)}")
    return g1, g2


def compose_unregularized_precision_weights(
    base_weight: np.ndarray,
    g1_variance: np.ndarray,
    g2_variance: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Inverse-variance galaxy weights with no ensemble shape-noise floor."""

    base_weight = np.asarray(base_weight, dtype=np.float64)
    g1_variance = np.asarray(g1_variance, dtype=np.float64)
    g2_variance = np.asarray(g2_variance, dtype=np.float64)
    galaxy_variance = 0.5 * (g1_variance + g2_variance)
    valid_base = np.isfinite(base_weight) & (base_weight > 0.0)
    valid_variance = (
        np.isfinite(g1_variance)
        & np.isfinite(g2_variance)
        & (g1_variance >= 0.0)
        & (g2_variance >= 0.0)
        & np.isfinite(galaxy_variance)
    )
    usable = valid_base & valid_variance
    if not np.any(usable) or np.sum(base_weight[usable]) <= 0.0:
        raise ValueError("no positive population mass has valid shear variances")
    galaxy_shape_noise = np.sqrt(np.clip(galaxy_variance, 0.0, None))
    combined = np.zeros(base_weight.shape, dtype=np.float64)
    zero_var = usable & (galaxy_variance <= 0.0)
    if np.any(zero_var):
        combined[zero_var] = base_weight[zero_var]
    else:
        combined[usable] = base_weight[usable] / galaxy_variance[usable]
    total = float(np.sum(combined))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("unregularized precision-weighted mass is not positive")
    combined /= total
    weighted_shape_noise, weighted_shape_noise_se, _ = weighted_mean_and_se(
        galaxy_shape_noise, combined
    )
    diagnostics = {
        "weighted_shape_noise": weighted_shape_noise,
        "weighted_shape_noise_se": weighted_shape_noise_se,
        "ess": effective_sample_size(combined),
        "usable_count": int(np.count_nonzero(combined > 0.0)),
    }
    return combined, diagnostics


def equal_weights(keep: np.ndarray) -> np.ndarray:
    weight = np.asarray(keep, dtype=np.float64)
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("selection keeps no galaxies")
    return weight / total


def empty_row(family: str, selection: str, weighting: str) -> dict:
    return {
        "family": family,
        "selection": selection,
        "weighting_key": weighting,
        "n_kept": 0,
        "catalog_n": CATALOG_N,
        "catalog_ess": float("nan"),
        "low_ess": float("nan"),
        "low_g_n": LOW_G_N,
        "low_kept_pct": float("nan"),
        "g1_m": float("nan"),
        "g1_m_se": float("nan"),
        "g2_m": float("nan"),
        "g2_m_se": float("nan"),
        "shape_noise": float("nan"),
        "shape_noise_se": float("nan"),
        "mean_abs_m_pct": float("nan"),
    }


def evaluate_row(
    *,
    truth: np.ndarray,
    estimate: np.ndarray,
    g1_variance: np.ndarray,
    g2_variance: np.ndarray,
    keep: np.ndarray,
    weighting: str,
    low_g: float,
    family: str,
    selection: str,
) -> dict:
    if not np.any(keep):
        return empty_row(family, selection, weighting)
    base = equal_weights(keep)
    sigma_gal = np.sqrt(np.clip(0.5 * (g1_variance + g2_variance), 0.0, None))
    if weighting == "equal":
        weight = base
        reported_shape_noise, reported_shape_noise_se, _ = weighted_mean_and_se(
            sigma_gal, weight
        )
        catalog_ess = effective_sample_size(weight)
    elif weighting == "unregularized":
        weight, diagnostics = compose_unregularized_precision_weights(
            base, g1_variance, g2_variance
        )
        reported_shape_noise = diagnostics["weighted_shape_noise"]
        reported_shape_noise_se = diagnostics["weighted_shape_noise_se"]
        catalog_ess = diagnostics["ess"]
    elif weighting == "regularized":
        weight, diagnostics = compose_shape_noise_regularized_weights(
            base, g1_variance, g2_variance
        )
        reported_shape_noise = diagnostics["weighted_shape_noise"]
        reported_shape_noise_se = diagnostics["weighted_shape_noise_se"]
        catalog_ess = diagnostics["ess"]
    else:
        raise ValueError(f"unknown weighting {weighting!r}")

    g1 = component_metrics(truth[:, 0], estimate[:, 0], low_g, weight)
    g2 = component_metrics(truth[:, 1], estimate[:, 1], low_g, weight)
    low_ess = 0.5 * (g1["ess_low"] + g2["ess_low"])
    return {
        "family": family,
        "selection": selection,
        "weighting_key": weighting,
        "n_kept": int(np.count_nonzero(keep)),
        "catalog_n": CATALOG_N,
        "catalog_ess": catalog_ess,
        "low_ess": low_ess,
        "low_g_n": LOW_G_N,
        "low_kept_pct": 100.0 * low_ess / LOW_G_N,
        "g1_m": g1["low_m"],
        "g1_m_se": g1["low_m_se"],
        "g2_m": g2["low_m"],
        "g2_m_se": g2["low_m_se"],
        "shape_noise": reported_shape_noise,
        "shape_noise_se": reported_shape_noise_se,
        "mean_abs_m_pct": 50.0 * (abs(g1["low_m"]) + abs(g2["low_m"])),
    }


def cut_mask(
    *,
    cosi: np.ndarray,
    rmag: np.ndarray,
    snr: np.ndarray,
    sigma_perp: np.ndarray,
    cosi_min: float | None = None,
    rmag_max: float | None = None,
    snr_min: float | None = None,
    sigma_perp_max: float | None = None,
) -> np.ndarray:
    keep = np.ones(len(cosi), dtype=bool)
    if cosi_min is not None:
        keep &= np.isfinite(cosi) & (cosi > cosi_min)
    if rmag_max is not None:
        keep &= np.isfinite(rmag) & (rmag < rmag_max)
    if snr_min is not None:
        keep &= np.isfinite(snr) & (snr > snr_min)
    if sigma_perp_max is not None:
        keep &= np.isfinite(sigma_perp) & (sigma_perp < sigma_perp_max)
    return keep


def inclination_phrase(cosi_min: float) -> str:
    degrees = math.degrees(math.acos(min(max(cosi_min, 0.0), 1.0)))
    return f"more face-on than about {degrees:.0f}° (cos i > {cosi_min:g})"


def format_pm(value: float, error: float, scale: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{scale * value:.3f} ± {scale * error:.3f}"


def format_ess(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value >= 100.0:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}%"


def render_html(rows: list[dict], catalog_n: int, low_g_n: int) -> str:
    body = []
    current_family = None
    for row in rows:
        if row["family"] != current_family:
            current_family = row["family"]
            body.append(
                f'<tr class="family"><th colspan="8">{html_escape(current_family)}</th></tr>'
            )
        body.append(
            "<tr>"
            f"<td>{html_escape(row['selection'])}</td>"
            f"<td>{row['n_kept']:,} / {catalog_n:,}</td>"
            f"<td>{format_ess(row['catalog_ess'])} / {catalog_n:,}</td>"
            f"<td>{format_ess(row['low_ess'])} / {low_g_n:,}</td>"
            f"<td>{format_pct(row['low_kept_pct'])}</td>"
            f"<td>{format_pm(row['g1_m'], row['g1_m_se'], 1e2)}</td>"
            f"<td>{format_pm(row['g2_m'], row['g2_m_se'], 1e2)}</td>"
            f"<td>{row['shape_noise']:.5g}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xu sample 3 shear m and effective sample size</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#1b1b1b;line-height:1.45}}
p,li{{max-width:46rem}}
h2{{margin-top:1.6rem;font-size:1.05rem}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.4rem .55rem;text-align:right;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f3f3f3}}
tr.family th{{text-align:left;background:#ececec;font-weight:600}}
</style>
</head>
<body>
<h1>Xu sample 3: multiplicative shear bias and effective sample size</h1>
<p>
One already-trained network (frozen image and spectrum encoder, 100,000 training
galaxies) evaluated on 100,000 Xu sample 3–like test galaxies (DESI BGS Any,
half-light radius at least 1 arcsec). No response calibration is applied.
Multiplicative bias <i>m</i> is a linear fit of (estimated − true) versus true
shear on galaxies with |<i>g</i>| &lt; 0.02.
</p>

<h2>What shear estimate is used</h2>
<p>
The network is trained with a uniform circular-velocity prior. After it draws
posterior samples for a galaxy, those draws are reweighted by the Tully–Fisher
prior <i>p</i>(<i>v</i><sub>circ</sub> | <i>r</i>) / <i>p</i><sub>train</sub>(<i>v</i><sub>circ</sub>).
The number quoted here is the mean of that reweighted posterior, not the mean
of a network that was trained on Tully–Fisher.
</p>

<h2>Two different kinds of weight</h2>
<p>
Within one galaxy, Tully–Fisher reweighting changes which posterior draws
count toward that galaxy’s shear mean. Across the catalog, a second weight
decides how much each galaxy counts in the ensemble <i>m</i> fit. Inverse-variance
weights across galaxies are used because a galaxy with a tighter shear
posterior is more informative about the mean shear.
</p>

<h2>Shape-noise floor</h2>
<p>
Raw inverse-variance weights <i>w</i> ∝ 1/σ<sub>gal</sub><sup>2</sup> let a few
galaxies with tiny posterior width dominate the fit. The regularized weight is
<i>w</i> ∝ 1 / (σ<sub>gal</sub><sup>2</sup> + σ<sub>shape</sub><sup>2</sup>).
Here σ<sub>gal</sub> = sqrt((Var <i>g</i><sub>1</sub> + Var <i>g</i><sub>2</sub>) / 2)
is the usual per-component RMS, and σ<sub>shape</sub> is the equal-weight mean of
σ<sub>gal</sub> among galaxies that survive that row’s catalog cut, frozen
<i>before</i> the precision weights are applied. After a cut, that floor is
recomputed on the kept subset only.
</p>

<h2>Shear-width cuts</h2>
<p>
<i>g</i><sub>1</sub> and <i>g</i><sub>2</sub> are treated as independent 1D
posteriors, so a width cut uses the quadrature combination
σ<sub>⊥</sub> = sqrt(σ<sub>g1</sub><sup>2</sup> + σ<sub>g2</sub><sup>2</sup>)
= sqrt(Var <i>g</i><sub>1</sub> + Var <i>g</i><sub>2</sub>).
When the two components have similar width, that is √2 times the
per-component RMS σ<sub>gal</sub> used in the weights above. Galaxies are
kept if σ<sub>⊥</sub> is below 0.10, 0.075, or 0.05. This quadrature
combination is a selection statistic only; the weights and the reported
σ<sub>shape</sub> still use σ<sub>gal</sub>. A cut at 0.10 keeps every galaxy
in this catalog.
</p>

<h2>How to read ESS</h2>
<p>
Catalog ESS is the effective number of galaxies after across-catalog weighting,
out of 100,000. The <i>m</i> fit uses only |<i>g</i>| &lt; 0.02, which is 20,000
galaxies in this catalog, so low-|g| ESS is out of 20,000. The percent column is
that low-|g| ESS fraction after both cutting and weighting: 100% means the fit
still has the information of all 20,000 low-shear galaxies.
</p>

<table>
<thead>
<tr>
<th>Selection</th>
<th>Galaxies kept</th>
<th>Catalog ESS</th>
<th>Low-|g| ESS</th>
<th>Low-|g| kept</th>
<th>10<sup>2</sup> m<sub>g1</sub> ± SE</th>
<th>10<sup>2</sup> m<sub>g2</sub> ± SE</th>
<th>σ<sub>shape</sub></th>
</tr>
</thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</body>
</html>
"""


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def json_ready(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)):
                converted[key] = None if not np.isfinite(value) else float(value)
            elif isinstance(value, (np.integer, int)):
                converted[key] = int(value)
            else:
                converted[key] = value
        out.append(converted)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    case = load_case(args.cache_root, args.case, test_set=True)
    names = list(case["feature_names"])
    truth = np.asarray(case["truth"], dtype=np.float64)
    estimate = np.asarray(
        next(iter(case["populations"].values()))["mean"], dtype=np.float64
    )
    rmag = np.asarray(case["rmag_true"], dtype=np.float64)
    snr = np.asarray(case["spectral_condition"], dtype=np.float64)
    cosi = truth[:, names.index("cosi")]
    n = len(truth)
    LOGGER.info("loaded %s galaxies from %s", f"{n:,}", case["case"])

    if args.reuse_sidecar and args.sidecar.is_file():
        payload = np.load(args.sidecar)
        g1_variance = np.asarray(payload["g1_variance"], dtype=np.float64)
        g2_variance = np.asarray(payload["g2_variance"], dtype=np.float64)
        LOGGER.info("loaded TF variances from %s", args.sidecar)
    else:
        g1_variance, g2_variance = stream_tf_variances(case)
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.sidecar,
            g1_variance=g1_variance,
            g2_variance=g2_variance,
        )
        LOGGER.info("wrote TF variances to %s", args.sidecar)
    if g1_variance.shape != (n,) or g2_variance.shape != (n,):
        raise ValueError("variance sidecar length does not match the catalog")
    sigma_perp = np.sqrt(np.clip(g1_variance + g2_variance, 0.0, None))

    def row(**kwargs) -> dict:
        return evaluate_row(
            truth=truth,
            estimate=estimate,
            g1_variance=g1_variance,
            g2_variance=g2_variance,
            low_g=args.low_g,
            **kwargs,
        )

    full = np.ones(n, dtype=bool)
    rows = [
        row(
            keep=full,
            weighting="equal",
            family="Full catalog",
            selection=(
                "Posterior mean after Tully–Fisher reweighting of the draws; "
                "each galaxy weighted equally."
            ),
        ),
        row(
            keep=full,
            weighting="unregularized",
            family="Full catalog",
            selection=(
                "Same estimator; galaxies weighted by inverse shear posterior "
                "variance, with no shape-noise floor."
            ),
        ),
        row(
            keep=full,
            weighting="regularized",
            family="Full catalog",
            selection=(
                "Same estimator; galaxies weighted by inverse shear posterior "
                "variance, with a shape-noise floor taken from this full catalog."
            ),
        ),
    ]

    for cosi_min in (0.3, 0.5, 0.7):
        keep = cut_mask(
            cosi=cosi, rmag=rmag, snr=snr, sigma_perp=sigma_perp, cosi_min=cosi_min
        )
        rows.append(
            row(
                keep=keep,
                weighting="regularized",
                family="Inclination cuts",
                selection=(
                    "Shape-noise-regularized inverse-variance weights, keeping "
                    f"only galaxies {inclination_phrase(cosi_min)}."
                ),
            )
        )

    for sigma_max in (0.1, 0.075, 0.05):
        keep = cut_mask(
            cosi=cosi,
            rmag=rmag,
            snr=snr,
            sigma_perp=sigma_perp,
            sigma_perp_max=sigma_max,
        )
        rows.append(
            row(
                keep=keep,
                weighting="regularized",
                family="Shear-width cuts",
                selection=(
                    "Shape-noise-regularized inverse-variance weights, keeping "
                    "only galaxies whose combined shear posterior width "
                    f"σ⊥ = sqrt(σ_g1² + σ_g2²) is below {sigma_max:g}."
                ),
            )
        )

    for rmag_max, snr_min in ((19.5, 10.0), (19.5, 25.0), (19.0, 10.0), (19.0, 25.0)):
        keep = cut_mask(
            cosi=cosi,
            rmag=rmag,
            snr=snr,
            sigma_perp=sigma_perp,
            rmag_max=rmag_max,
            snr_min=snr_min,
        )
        rows.append(
            row(
                keep=keep,
                weighting="regularized",
                family="Brightness and emission-line S/N",
                selection=(
                    "Shape-noise-regularized inverse-variance weights, keeping "
                    f"galaxies brighter than r = {rmag_max:g} with emission-line "
                    f"S/N above {snr_min:g}."
                ),
            )
        )

    payload = {
        "case": case["case"],
        "n": n,
        "low_g": args.low_g,
        "catalog_n": CATALOG_N,
        "low_g_n": LOW_G_N,
        "width_statistic": "quadrature",
        "rows": json_ready(rows),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(rows, CATALOG_N, LOW_G_N), encoding="utf-8")
    LOGGER.info("wrote %s", args.output)
    LOGGER.info("wrote %s", args.json_output)


if __name__ == "__main__":
    main()
