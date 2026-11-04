#!/usr/bin/env python3
"""D: stream the Xu3 cache once and store per-galaxy posterior masses in truth-shear bins.

This feeds the model-free "self-consistency" prediction of the conditional
coverage. For a correct posterior p(g|D) and truths drawn from the prior pi,
Bayes' theorem gives, for any truth bin B and any data-defined interval I(D),

  P(g in I(D) | g in B) = E_D[ int_B p(g|D) 1[g in I(D)] dg ] / E_D[ int_B p(g|D) dg ],

where E_D averages over the test set. Both integrals are TF-weighted sums over
each galaxy's posterior draws. So the NPE's own posteriors predict what the
truth-conditioned coverage should be if they are correct; no toy likelihood
or prior shape is assumed. (The denominator is also a check: averaged over
galaxies it should equal the prior mass of B.)

Per galaxy and component k, stores
  beta[k, b]      = posterior mass of g_k in bin b,
  alpha68[k, b]   = posterior mass of g_k in bin b AND in the galaxy's 16-84% interval of g_k,
  alpha95[k, b]   = same for the 2.5-97.5% interval,
for three bin sets: 20 bins of g_k on [-0.1, 0.1] ("gk"), 14 bins of
|g| = hypot(g1, g2) on [0, 0.1*sqrt(2)] ("absg"), and |g| < 0.05 / > 0.05
("split"). The intervals are the per_galaxy.npz quantiles (same TF weights and
quantile convention as a_shear_extract.py, so the coverage indicator is the
one used for the measured coverage).

Read-only on the cache. Refuses to overwrite its output.
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
PER_GALAXY = S / "reports/paper-tests/A_shear/per_galaxy.npz"
OUT = S / "reports/paper-tests/D_local_calibration/posterior_bin_masses.npz"

H = 0.1
N_GK, N_ABS = 20, 14
GK_EDGES = np.linspace(-H, H, N_GK + 1)
ABS_EDGES = np.linspace(0.0, np.sqrt(2) * H, N_ABS + 1)
SPLIT_EDGES = np.array([0.0, 0.05, np.inf])


def bin_mass(values, w, edges, extra=None):
    """(n_gal, n_bins) sum of w (times extra mask) over draws falling in each bin."""
    nb = len(edges) - 1
    idx = np.searchsorted(edges, values, side="right") - 1
    ok = (idx >= 0) & (idx < nb)
    ww = w * ok if extra is None else w * ok * extra
    idx = np.clip(idx, 0, nb - 1)
    flat = idx + nb * np.arange(values.shape[0])[:, None]
    return np.bincount(flat.ravel(), weights=ww.ravel(), minlength=nb * values.shape[0]).reshape(-1, nb)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--block", type=int, default=200)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    t0 = time.time()
    pg = np.load(PER_GALAXY)
    q = np.stack([pg["tf_q_g1"], pg["tf_q_g2"]], axis=1)  # (N, 2, 5)
    case = sbr.load_case(S / "cache", f"{MODEL}:{DATASET}", test_set=True)
    parts = case["cache_partitions"]
    sample_files = parts.files["shear_sample"]
    weight_files = parts.files["posterior_tf_log_weight"]
    if args.max_parts is not None:
        sample_files = sample_files[: args.max_parts]
    truth_all = np.asarray(case["truth"], dtype=np.float64)
    if not np.array_equal(truth_all, pg["truth"]):
        raise ValueError("per_galaxy.npz truth order differs from the cache")

    out = {}
    names = ("gk", "absg", "split")
    for nm in names:
        for a in ("beta", "alpha68", "alpha95"):
            out[f"{nm}_{a}"] = []
    mean_check = 0.0
    offset = 0
    for pi, (sp, wp) in enumerate(zip(sample_files, weight_files)):
        samples = np.load(sp, mmap_mode="r")
        logw_all = np.load(wp, mmap_mode="r")
        n_part = samples.shape[0]
        for a in range(0, n_part, args.block):
            b = min(n_part, a + args.block)
            x = np.asarray(samples[a:b], dtype=np.float64)  # (n, d, 2)
            lw = np.asarray(logw_all[a:b], dtype=np.float64)
            lw = lw - lw.max(axis=1, keepdims=True)
            w = np.exp(lw)
            w /= w.sum(axis=1, keepdims=True)
            sl = slice(offset + a, offset + b)
            mean = np.einsum("nd,ndk->nk", w, x)
            mean_check = max(mean_check, float(np.max(np.abs(mean - pg["tf_mean"][sl]))))
            qq = q[sl]
            absg = np.hypot(x[:, :, 0], x[:, :, 1])
            per = {nm: {"beta": [], "alpha68": [], "alpha95": []} for nm in names}
            for k in range(2):
                xk = x[:, :, k]
                in68 = (xk >= qq[:, k, 1, None]) & (xk <= qq[:, k, 3, None])
                in95 = (xk >= qq[:, k, 0, None]) & (xk <= qq[:, k, 4, None])
                for nm, vals, edges in (("gk", xk, GK_EDGES), ("absg", absg, ABS_EDGES),
                                        ("split", absg, SPLIT_EDGES)):
                    per[nm]["beta"].append(bin_mass(vals, w, edges))
                    per[nm]["alpha68"].append(bin_mass(vals, w, edges, in68))
                    per[nm]["alpha95"].append(bin_mass(vals, w, edges, in95))
            for nm in names:
                for key in ("beta", "alpha68", "alpha95"):
                    out[f"{nm}_{key}"].append(np.stack(per[nm][key], axis=1).astype(np.float32))
        offset += n_part
        del samples, logw_all
        print(f"part {pi + 1}/{len(sample_files)} done, {time.time() - t0:.1f}s", flush=True)

    arrays = {k: np.concatenate(v, axis=0) for k, v in out.items()}  # each (N, 2, nbins)
    arrays.update(gk_edges=GK_EDGES, absg_edges=ABS_EDGES, split_edges=SPLIT_EDGES,
                  truth=truth_all[:offset, :2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    meta = dict(model=MODEL, dataset=DATASET, n_galaxies=int(offset), n_draws=int(x.shape[1]),
                mean_vs_per_galaxy_maxabs=mean_check, seconds=time.time() - t0,
                shapes={k: list(v.shape) for k, v in arrays.items()})
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
