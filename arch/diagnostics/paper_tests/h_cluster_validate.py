#!/usr/bin/env python3
"""H: validate the cell-stacking estimator on many synthetic noise realizations.

Same galaxies and cells as h_cluster_analysis.py; each realization redraws the
Gaussian-likelihood noise (real per-galaxy NPE shear covariances). For an unbiased
estimator with correct errors, the per-cell pulls average to 0 with unit spread,
and the per-cell mean residual (over realizations) is consistent with 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h_cluster_analysis as H  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_r0")
    p.add_argument("--realizations", type=int, default=30)
    a = p.parse_args()
    cfg = json.loads((H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_config.json").read_text())
    man = pd.read_csv(H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_manifest.csv")
    out = H.S / "reports/paper-tests-1m/H_cluster" / a.tag / "synthetic_validation"
    out.mkdir(parents=True, exist_ok=True)
    truths = {sc: man[[f"g1_{sc}", f"g2_{sc}"]].to_numpy() for sc in H.SCENARIOS}
    covs = H.xu3_shear_covariance(man["source_id"].to_numpy())
    phi = man["phi"].to_numpy()
    pcells, pmem = H.polar_cells(man, cfg)
    ccells, cmem, _ = H.cartesian_cells(man, cfg)
    acc = {(sc, kind): [] for sc in H.SCENARIOS for kind in ("polar", "cartesian")}
    for r in range(a.realizations):
        draws = H.synthetic(truths, covs, seed=cfg["seed"] + 1000 + r, nsamp=4000)
        for sc in H.SCENARIOS:
            logl = H.kde_loglike_grid(*draws[sc])
            _, cg = H.grid_moments(logl)
            for kind, mem, nc, fr in (("polar", pmem, len(pcells), lambda i: H.tangential_matrix(phi[i])),
                                      ("cartesian", cmem, len(ccells), lambda i: np.eye(2))):
                rows = H.analyse_cells(logl, cg, truths[sc], mem, nc, fr)
                acc[(sc, kind)].append(rows)
        print(f"realization {r + 1}/{a.realizations} done", flush=True)
    summary = {}
    for (sc, kind), reals in acc.items():
        est = np.array([[row["est"] for row in rows] for rows in reals])  # (R, cells, 2)
        sd = np.array([[row["sd"] for row in rows] for rows in reals])
        ref = np.array([row["ref"] for row in reals[0]])
        n = np.array([row["n"] for row in reals[0]])
        pull = (est - ref[None]) / sd
        resid_mean = (est - ref[None]).mean(axis=0)
        resid_err = (est - ref[None]).std(axis=0) / np.sqrt(len(reals))
        z = resid_mean / resid_err
        summary[f"{sc}/{kind}"] = dict(
            n_cells=len(ref), pull_mean=pull.mean(axis=(0, 1)).tolist(),
            pull_std=pull.std(axis=(0, 1)).tolist(),
            cell_resid_z_mean=z.mean(axis=0).tolist(), cell_resid_z_rms=np.sqrt((z ** 2).mean(axis=0)).tolist(),
            frac_cells_abs_z_gt_3=(np.abs(z) > 3).mean(axis=0).tolist(),
            resid_vs_n=[dict(n=int(k), mean_resid=resid_mean[n == k].mean(axis=0).tolist(),
                             mean_sd=sd.mean(axis=0)[n == k].mean(axis=0).tolist()) for k in np.unique(n)])
        print(sc, kind, json.dumps({k: v for k, v in summary[f"{sc}/{kind}"].items() if k != "resid_vs_n"}), flush=True)
    (out / "validation.json").write_text(json.dumps(summary, indent=1))
    arrays = {}
    for (sc, kind), reals in acc.items():
        arrays[f"{sc}_{kind}_est"] = np.array([[row["est"] for row in rows] for rows in reals])
        arrays[f"{sc}_{kind}_sd"] = np.array([[row["sd"] for row in rows] for rows in reals])
        arrays[f"{sc}_{kind}_ref"] = np.array([row["ref"] for row in reals[0]])
        arrays[f"{sc}_{kind}_n"] = np.array([row["n"] for row in reals[0]])
        arrays[f"{sc}_{kind}_cell"] = np.array([row["cell"] for row in reals[0]])
    np.savez_compressed(out / "validation_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
