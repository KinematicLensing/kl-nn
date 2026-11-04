#!/usr/bin/env python3
"""Paper test group A (NPE validation): one streaming pass over the Xu3 cache.

Reuses ``shear_bias_report.load_case`` and
``shear_bias_report.load_shear_posterior_diagnostics`` (TF-weighted g1/g2
PIT ranks and posterior variances, exactly as in the campaign reports), then
does one vectorized pass over the stored TF log weights to recompute the
per-galaxy Kish ESS and max weight, and saves the candidate (g1, g2) draws of
the corner-plot galaxies.  Read-only on the cache; writes one NPZ.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
ARCH = DIAG.parent
for path in (ARCH, DIAG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shear_bias_report import load_case, load_shear_posterior_diagnostics  # noqa: E402
from likelihood_stack_core import CACHE_ROOT, G01_CASE  # noqa: E402

OUT_DIR = Path("/ocean/projects/phy250048p/shared/reports/paper-tests/A_npe")
TARGET_IMAGE_SNR = 170.0
TARGET_HALPHA_SNR = 25.0
COSI_RANGE = (0.4, 0.7)


def pick_corner_galaxies(case: dict, n_pick: int = 2) -> list[int]:
    """Moderately inclined, typical-SNR galaxies with healthy TF weights."""

    names = list(case["feature_names"])
    truth = case["truth"]
    cosi = truth[:, names.index("cosi")]
    g = np.hypot(truth[:, 0], truth[:, 1])
    hlr = truth[:, names.index("hlr")]
    ess_frac = case["posterior_tf_ess_fraction"]
    img = case["image_snr"]
    ha = case["spectral_condition"]
    ok = (
        (cosi >= COSI_RANGE[0]) & (cosi <= COSI_RANGE[1])
        & (g > 0.01) & (g < 0.06)
        & (hlr > np.quantile(hlr, 0.25)) & (hlr < np.quantile(hlr, 0.75))
        & (ess_frac > np.median(ess_frac))
    )
    score = np.abs(np.log(img / TARGET_IMAGE_SNR)) + np.abs(
        np.log(ha / TARGET_HALPHA_SNR)
    )
    score = np.where(ok, score, np.inf)
    order = np.argsort(score)
    picks = [int(order[0])]
    # second galaxy: different inclination half of the window
    first_cosi = cosi[picks[0]]
    for index in order[1:]:
        if not np.isfinite(score[index]):
            break
        if abs(cosi[index] - first_cosi) > 0.12:
            picks.append(int(index))
            break
    return picks[:n_pick]


def recompute_weight_health(case: dict) -> dict[str, np.ndarray]:
    files = case["cache_partitions"].files["posterior_tf_log_weight"]
    ess, maxw, nfinite = [], [], []
    for path in files:
        logw = np.load(path, mmap_mode="r")
        for start in range(0, logw.shape[0], 250):
            block = np.asarray(logw[start : start + 250], dtype=np.float64)
            finite = np.isfinite(block)
            block = np.where(finite, block, -np.inf)
            peak = np.max(block, axis=1, keepdims=True)
            w = np.exp(block - peak)
            w /= np.sum(w, axis=1, keepdims=True)
            ess.append(1.0 / np.sum(w * w, axis=1))
            maxw.append(np.max(w, axis=1))
            nfinite.append(np.sum(finite, axis=1))
        del logw
    return {
        "ess_recomputed": np.concatenate(ess),
        "max_weight_recomputed": np.concatenate(maxw),
        "n_finite_weight": np.concatenate(nfinite),
    }


def load_galaxy_draws(case: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    ranges = case["cache_partitions"].row_ranges
    for part, (start, end) in enumerate(ranges):
        if start <= index < end:
            s = np.load(case["cache_partitions"].files["shear_sample"][part], mmap_mode="r")
            w = np.load(
                case["cache_partitions"].files["posterior_tf_log_weight"][part],
                mmap_mode="r",
            )
            return (
                np.asarray(s[index - start], dtype=np.float64),
                np.asarray(w[index - start], dtype=np.float64),
            )
    raise IndexError(index)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--case", default=G01_CASE, help="MODEL:CACHE_DATASET")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "a_npe_stream.npz")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    t0 = time.time()
    case = load_case(CACHE_ROOT, args.case, args.max_galaxies, test_set=True)
    print(f"load_case {time.time() - t0:.1f}s n={case['analyzed_size']}", flush=True)

    picks = pick_corner_galaxies(case)
    print("corner picks", picks, flush=True)
    draws = {}
    for rank, index in enumerate(picks):
        values, logw = load_galaxy_draws(case, index)
        draws[f"corner{rank}_draws"] = values
        draws[f"corner{rank}_logw"] = logw

    t1 = time.time()
    health = recompute_weight_health(case)
    n = case["analyzed_size"]
    for key in health:
        health[key] = health[key][:n]
    print(f"weight pass {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    diag = load_shear_posterior_diagnostics(case)["test_set"]
    print(f"shear diagnostics {time.time() - t2:.1f}s", flush=True)

    population = case["populations"]["TF-conformed test set / TF posterior"]
    np.savez(
        args.output,
        feature_names=np.asarray(case["feature_names"]),
        truth=case["truth"],
        tf_summary=population["summary"],
        proposal_summary=case["base_summary"],
        rmag_true=case["rmag_true"],
        image_snr=case["image_snr"],
        central_halpha_snr=case["spectral_condition"],
        ess_stored=case["posterior_tf_ess"],
        ess_fraction_stored=case["posterior_tf_ess_fraction"],
        max_weight_stored=case["posterior_tf_max_weight"],
        pit_g1=diag["g1"],
        pit_g2=diag["g2"],
        var_g1=diag["g1_variance"],
        var_g2=diag["g2_variance"],
        conditional_ess_g1=diag["g1_conditional_ess"],
        retained_g1=diag["g1_retained"],
        retained_g2=diag["g2_retained"],
        corner_index=np.asarray(picks),
        meta_json=json.dumps(
            {
                "case": args.case,
                "cache_root": str(CACHE_ROOT),
                "dataset_size": case["dataset_size"],
                "analyzed_size": n,
                "seconds": time.time() - t0,
            }
        ),
        **health,
        **draws,
    )
    print(f"wrote {args.output} total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
