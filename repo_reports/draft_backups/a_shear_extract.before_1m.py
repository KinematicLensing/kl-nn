#!/usr/bin/env python3
"""Paper tests, Group A (shear): stream the Xu3 fibrepair posterior cache once.

Writes one compact per-galaxy table (``per_galaxy.npz``) with TF-weighted
posterior summaries of (g1, g2): mean, variance, covariance and weighted
quantiles, plus truths and SNR columns. All downstream Group-A shear tests
(R-raw, T1, T3, T4, T5, R6, T2a, T7) read only this table.

Posterior summaries use the cached TF log weights (normalized within each
galaxy). Variances are cross-checked against
``shear_bias_report.posterior_component_variance`` and quantiles against
``shear_bias_report.weighted_quantile`` on a subset of galaxies.

Read-only on the cache. Never overwrites an existing output file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DIAG_DIR = Path(__file__).resolve().parents[1]
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import shear_bias_report as sbr  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
DATASET = "test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded"
QUANTILES = np.asarray([0.025, 0.16, 0.5, 0.84, 0.975], dtype=np.float64)


def weighted_quantiles_batch(x: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized version of ``sbr.weighted_quantile`` (same midpoint convention).

    x, w: (n_gal, n_draw); w normalized along axis 1. Returns (n_gal, len(q)).
    """
    order = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    coord = np.cumsum(ws, axis=1) - 0.5 * ws
    out = np.empty((x.shape[0], len(q)))
    for i in range(x.shape[0]):
        out[i] = np.interp(q, coord[i], xs[i], left=xs[i, 0], right=xs[i, -1])
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--block", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=97,
                        help="cross-check every k-th galaxy against sbr helpers")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    t0 = time.time()
    case = sbr.load_case(S / "cache", f"{MODEL}:{DATASET}", test_set=True)
    parts = case["cache_partitions"]
    sample_files = parts.files["shear_sample"]
    weight_files = parts.files["posterior_tf_log_weight"]
    truth_files = parts.files["truth"]
    if args.max_parts is not None:
        sample_files = sample_files[: args.max_parts]
    truth_all = np.asarray(case["truth"], dtype=np.float64)

    rows = []
    checks = {"var_maxabs": 0.0, "q_maxabs": 0.0, "mean_vs_cache_maxabs": 0.0, "n": 0}
    target_mean = np.asarray(case["populations"][case["test_set_population_label"]]["mean"],
                             dtype=np.float64)
    offset = 0
    for pi, (sp, wp, tp) in enumerate(zip(sample_files, weight_files, truth_files)):
        samples = np.load(sp, mmap_mode="r")
        logw_all = np.load(wp, mmap_mode="r")
        stored_truth = np.load(tp, mmap_mode="r")
        n_part = samples.shape[0]
        if not np.array_equal(np.asarray(stored_truth, dtype=np.float64),
                              truth_all[offset: offset + n_part]):
            raise ValueError(f"truth misaligned in {tp}")
        for a in range(0, n_part, args.block):
            b = min(n_part, a + args.block)
            x = np.asarray(samples[a:b], dtype=np.float64)  # (n, d, 2)
            lw = np.asarray(logw_all[a:b], dtype=np.float64)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(lw))):
                raise ValueError(f"non-finite samples/weights in part {pi}")
            lw = lw - lw.max(axis=1, keepdims=True)
            w = np.exp(lw)
            w /= w.sum(axis=1, keepdims=True)
            ess = 1.0 / np.sum(w * w, axis=1)
            mean = np.einsum("nd,ndk->nk", w, x)
            dx = x - mean[:, None, :]
            var = np.einsum("nd,ndk->nk", w, dx * dx)
            cov12 = np.einsum("nd,nd->n", w, dx[:, :, 0] * dx[:, :, 1])
            q1 = weighted_quantiles_batch(x[:, :, 0], w, QUANTILES)
            q2 = weighted_quantiles_batch(x[:, :, 1], w, QUANTILES)
            umean = x.mean(axis=1)
            uvar = x.var(axis=1)
            for j in range(b - a):
                gi = offset + a + j
                if gi % args.check_every == 0:
                    for k in range(2):
                        v_ref = sbr.posterior_component_variance(x[j, :, k], lw[j])
                        q_ref = sbr.weighted_quantile(x[j, :, k], QUANTILES, w[j])
                        checks["var_maxabs"] = max(checks["var_maxabs"], abs(v_ref - var[j, k]))
                        qq = q1[j] if k == 0 else q2[j]
                        checks["q_maxabs"] = max(checks["q_maxabs"], float(np.max(np.abs(q_ref - qq))))
                    checks["n"] += 1
            checks["mean_vs_cache_maxabs"] = max(
                checks["mean_vs_cache_maxabs"],
                float(np.max(np.abs(mean - target_mean[offset + a: offset + b, :2]))),
            )
            rows.append(dict(mean=mean, var=var, cov12=cov12, q1=q1, q2=q2,
                             umean=umean, uvar=uvar, ess_recomputed=ess))
        offset += n_part
        del samples, logw_all
        print(f"part {pi + 1}/{len(sample_files)} done, {time.time() - t0:.1f}s", flush=True)

    n = offset
    cat = {k: np.concatenate([r[k] for r in rows], axis=0) for k in rows[0]}
    fn = case["feature_names"]
    out = dict(
        truth=truth_all[:n],
        feature_names=np.asarray(fn),
        tf_mean=cat["mean"],
        tf_var=cat["var"],
        tf_cov12=cat["cov12"],
        tf_q_g1=cat["q1"],
        tf_q_g2=cat["q2"],
        quantile_levels=QUANTILES,
        unweighted_mean=cat["umean"],
        unweighted_var=cat["uvar"],
        ess_recomputed=cat["ess_recomputed"],
        cache_tf_mean=target_mean[:n, :2],
        cache_proposal_mean=np.asarray(case["base_summary"], dtype=np.float64)[:n, 1, :2],
        posterior_tf_ess=np.asarray(case["posterior_candidate_ess"], dtype=np.float64)[:n],
        posterior_tf_ess_fraction=np.asarray(case["posterior_candidate_ess_fraction"], dtype=np.float64)[:n],
        image_snr=np.asarray(case["image_snr"], dtype=np.float64)[:n],
        central_halpha_snr=np.asarray(case["spectral_condition"], dtype=np.float64)[:n],
        rmag_true=np.asarray(case["rmag_true"], dtype=np.float64)[:n],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    meta = dict(model=MODEL, dataset=DATASET, n_galaxies=int(n), n_draws=int(x.shape[1]),
                checks=checks, seconds=time.time() - t0,
                ess_recomputed_vs_cache_maxabs=float(np.max(np.abs(out["ess_recomputed"] - out["posterior_tf_ess"]))))
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
