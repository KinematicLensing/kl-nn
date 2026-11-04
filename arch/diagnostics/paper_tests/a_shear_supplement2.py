#!/usr/bin/env python3
"""Paper tests, Group A (shear), follow-up S5: is the shape-noise-regularized
*weighting* (no cut) also subject to noise-driven selection?

Compare the existing regularized weights built from the observed posterior
variances with the same weights built from a shear-free, truth-level
prediction of sigma_eps,i (cross-fitted GBM, see a_shear_analysis), and the
difference expected from shrinkage alone (uniform-prior toy evaluated with
each estimator's galaxy weights). Paired bootstrap over galaxies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_shear_analysis as A  # noqa: E402
from a_shear_analysis import sbr  # noqa: E402


def evaluate(D, counts, var_pair, toy_m):
    base = np.ones(D["n"]) if counts is None else counts
    w, diag = sbr.compose_shape_noise_regularized_weights(base / base.sum(), *var_pair)
    m1, _ = A.m_c_fast(D["true"]["g1"], D["est"]["g1"], w)
    m2, _ = A.m_c_fast(D["true"]["g2"], D["est"]["g2"], w)
    ww = w / w.sum()
    return 0.5 * (m1 + m2), float(np.sum(ww * toy_m)), m1, m2, diag["ess"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=300)
    a = p.parse_args(argv)
    if (a.outdir / "supplement2.json").exists():
        raise SystemExit("refusing to overwrite supplement2.json")
    D = A.load(a.table)
    toy = A.uniform_prior_toy()
    logsig_pred, r2, _ = A.gbm_predicted_log_sigma(D)
    toy_m = 0.5 * (A.toy_m_from_post_std(np.sqrt(D["v1"]), toy) + A.toy_m_from_post_std(np.sqrt(D["v2"]), toy))
    vp = 0.5 * np.exp(2 * logsig_pred)
    variants = {"observed variances": (D["v1"], D["v2"]), "GBM-predicted (shear-free)": (vp, vp)}
    ones = np.ones(D["n"])
    raw = (0.5 * sum(A.m_c_fast(D["true"][c], D["est"][c], ones / D["n"])[0] for c in ("g1", "g2")),
           float(np.mean(toy_m)))
    pts = {k: evaluate(D, None, v, toy_m) for k, v in variants.items()}
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(a.n_boot):
        counts = rng.multinomial(D["n"], np.full(D["n"], 1.0 / D["n"])).astype(float)
        r = {k: evaluate(D, counts, v, toy_m) for k, v in variants.items()}
        mraw = 0.5 * sum(A.m_c_fast(D["true"][c], D["est"][c], counts / counts.sum())[0] for c in ("g1", "g2"))
        traw = float(np.sum(counts * toy_m) / counts.sum())
        o, g = r["observed variances"], r["GBM-predicted (shear-free)"]
        boots.append([o[0] - g[0], (o[0] - g[0]) - (o[1] - g[1]), o[0] - mraw, (o[0] - mraw) - (o[1] - traw)])
    b = np.array(boots)
    o, g = pts["observed variances"], pts["GBM-predicted (shear-free)"]
    out = {
        "gbm_r2": r2, "n_boot": a.n_boot,
        "raw": {"m_avg": raw[0], "toy": raw[1]},
        "weighted_observed": {"m_avg": o[0], "toy": o[1], "m1": o[2], "m2": o[3], "ess": o[4]},
        "weighted_gbm": {"m_avg": g[0], "toy": g[1], "m1": g[2], "m2": g[3], "ess": g[4]},
        "delta_obs_minus_gbm": o[0] - g[0], "delta_obs_minus_gbm_se": float(b[:, 0].std(ddof=1)),
        "excess_obs_minus_gbm": (o[0] - g[0]) - (o[1] - g[1]), "excess_obs_minus_gbm_se": float(b[:, 1].std(ddof=1)),
        "delta_obs_minus_raw": o[0] - raw[0], "delta_obs_minus_raw_se": float(b[:, 2].std(ddof=1)),
        "excess_obs_minus_raw": (o[0] - raw[0]) - (o[1] - raw[1]), "excess_obs_minus_raw_se": float(b[:, 3].std(ddof=1)),
    }
    (a.outdir / "supplement2.json").write_text(json.dumps(A.jsonable(out), indent=1))
    print(json.dumps(A.jsonable(out), indent=1))


if __name__ == "__main__":
    main()
