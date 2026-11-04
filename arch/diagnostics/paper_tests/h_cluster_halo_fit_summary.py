#!/usr/bin/env python3
"""Summarise the NFW halo fits (h_cluster_halo_fit.py) over the 50-realization ensemble.

For each realization (s3_sub1_nRR) and data scenario (main-only data, main+sub data):
  - posterior median and 16-84% interval of log10 M_main and log10 M_sub,
  - ln Bayes factors of main+sub (fixed position) and main+sub_free against main only,
  - for main+sub_free, the posterior-median subhalo position and its distance from the truth.
Pulls use the half 16-84% width as sigma. Writes ensemble/halo_fit_summary.json and prints a table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

H = Path("/ocean/projects/phy250048p/shared/reports/paper-tests-1m/H_cluster")
TAGS = [f"s3_sub1_n{r:02d}" for r in range(1, 51)]


def summ(p, truth):
    med, lo, hi = p["median"], p["q16"], p["q84"]
    return dict(median=med, q16=lo, q84=hi, pull=(med - truth) / (0.5 * (hi - lo)), covered=bool(lo <= truth <= hi))


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), err=float(x.std(ddof=1) / math.sqrt(len(x))), sd=float(x.std(ddof=1)),
                median=float(np.median(x)), q16=float(np.percentile(x, 16)), q84=float(np.percentile(x, 84)))


def main():
    rows = []
    for tag in TAGS:
        r = json.loads((H / tag / "npe/halo_fit.json").read_text())
        t = r["truth"]
        kpc = json.loads((Path("/ocean/projects/phy250048p/shared/samples") /
                          f"paper_tests_H_cluster_{tag}_config.json").read_text())["kpc_per_arcsec"]
        row = dict(tag=tag)
        for sc in ("main", "sub"):
            d = r[sc]
            row[f"{sc}_lnB_fixed"] = d["ln_B_main+sub_vs_main"]
            row[f"{sc}_lnB_free"] = d["ln_B_main+sub_free_vs_main"]
            row[f"{sc}_main_only_mmain"] = summ(d["main"]["params"]["log10_m_main"], t["log10_m_main"])
            free = d["main+sub_free"]["params"]
            dx = free["x_sub_arcsec"]["median"] - t["x_sub_arcsec"]
            dy = free["y_sub_arcsec"]["median"] - t["y_sub_arcsec"]
            row[f"{sc}_free_offset_mpc"] = math.hypot(dx, dy) * kpc / 1e3
        fx = r["sub"]["main+sub"]["params"]
        row["sub_fixed_mmain"] = summ(fx["log10_m_main"], t["log10_m_main"])
        row["sub_fixed_msub"] = summ(fx["log10_m_sub"], t["log10_m_sub"])
        rows.append(row)

    out = dict(n=len(rows), truth=dict(log10_m_main=15.0, log10_m_sub=math.log10(3e14)))
    for key in ("sub_fixed_msub", "sub_fixed_mmain", "main_main_only_mmain"):
        out[key] = dict(median=stats([r[key]["median"] for r in rows]), pull=stats([r[key]["pull"] for r in rows]),
                        coverage_68=float(np.mean([r[key]["covered"] for r in rows])))
    for sc in ("main", "sub"):
        for kind in ("fixed", "free"):
            b = np.array([r[f"{sc}_lnB_{kind}"] for r in rows])
            out[f"{sc}_lnB_{kind}"] = dict(**stats(b), frac_gt_1=float(np.mean(b > 1)), frac_gt_3=float(np.mean(b > 3)),
                                          values=b.tolist())
        off = np.array([r[f"{sc}_free_offset_mpc"] for r in rows])
        out[f"{sc}_free_offset_mpc"] = dict(**stats(off), frac_lt_05=float(np.mean(off < 0.5)), values=off.tolist())
    out["rows"] = rows
    (H / "ensemble/halo_fit_summary.json").write_text(json.dumps(out, indent=1))

    print(f"{len(rows)} realizations; truth log10 M_main 15.000, log10 M_sub {math.log10(3e14):.3f}")
    for key, lab in (("sub_fixed_msub", "M_sub   (main+sub data, fixed pos)"),
                     ("sub_fixed_mmain", "M_main  (main+sub data, fixed pos)"),
                     ("main_main_only_mmain", "M_main  (main-only data, main model)")):
        s = out[key]
        print(f"{lab}: median of medians {s['median']['median']:.3f}, mean {s['median']['mean']:.3f} "
              f"+- {s['median']['err']:.3f}; pull mean {s['pull']['mean']:+.2f} +- {s['pull']['err']:.2f}, "
              f"sd {s['pull']['sd']:.2f}; 68% coverage {s['coverage_68']:.2f}")
    for sc in ("sub", "main"):
        for kind in ("fixed", "free"):
            s = out[f"{sc}_lnB_{kind}"]
            print(f"ln B ({sc} data, {kind} position): median {s['median']:.2f} [{s['q16']:.2f}, {s['q84']:.2f}], "
                  f"P(>1) {s['frac_gt_1']:.2f}, P(>3) {s['frac_gt_3']:.2f}")
        s = out[f"{sc}_free_offset_mpc"]
        print(f"free-position offset ({sc} data): median {s['median']:.2f} Mpc, P(<0.5 Mpc) {s['frac_lt_05']:.2f}")


if __name__ == "__main__":
    main()
