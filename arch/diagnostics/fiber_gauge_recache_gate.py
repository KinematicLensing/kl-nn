#!/usr/bin/env python3
"""Fail unless g02/g002 recache on repaired xu3 recovered theta_int / g02 m."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

from shear_bias_report import (
    apply_precision_weighting,
    compute_metrics,
    coverage_metrics,
    load_case,
    load_shear_posterior_diagnostics,
    wrap_angle,
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-g", type=float, default=0.02)
    parser.add_argument("--min-mean-cos", type=float, default=0.25)
    parser.add_argument("--max-collapsed-m", type=float, default=0.85)
    return parser.parse_args(argv)


def _mean_cos_theta(case: dict) -> float:
    theta_index = case["feature_names"].index("theta_int")
    population = next(iter(case["populations"].values()))
    residual = wrap_angle(
        np.asarray(population["mean"][:, theta_index], dtype=np.float64)
        - np.asarray(case["truth"][:, theta_index], dtype=np.float64)
    )
    weight = np.asarray(population["galaxy_weight"], dtype=np.float64)
    finite = np.isfinite(residual) & np.isfinite(weight) & (weight >= 0.0)
    if not np.any(finite) or float(np.sum(weight[finite])) <= 0.0:
        return float("nan")
    return float(np.average(np.cos(residual[finite]), weights=weight[finite]))


def _mean_rows(metric_rows: list[dict]) -> list[dict]:
    return [
        row
        for row in metric_rows
        if row["estimator"] == "Mean" and row["frame"] == "image"
    ]


def evaluate_case(args, case_name: str) -> dict:
    case = load_case(args.cache_root, case_name, test_set=True)
    case["report_map"] = False
    pits = load_shear_posterior_diagnostics(case)
    apply_precision_weighting(case, pits)
    metric_rows = _mean_rows(compute_metrics(case, args.low_g))
    population = next(iter(case["populations"].values()))
    coverage = coverage_metrics(
        case["truth"],
        population["summary"],
        population["galaxy_weight"],
        case["feature_names"],
        next(iter(case["populations"])),
    )
    mean_cos = _mean_cos_theta(case)
    low_m = {
        row["component"]: {
            "low_m": float(row["low_m"]),
            "low_m_se": float(row["low_m_se"]),
        }
        for row in metric_rows
        if row["component"] in {"g1", "g2"}
    }
    theta_coverage = next(
        (
            float(row["coverage"])
            for row in coverage
            if row["parameter"] == "theta_int"
        ),
        float("nan"),
    )
    arm = "g02" if "_g02_" in case_name else "g002" if "_g002_" in case_name else "other"
    checks = {
        "theta_mean_cos": mean_cos >= args.min_mean_cos,
    }
    if arm == "g02":
        checks["g1_not_collapsed"] = abs(low_m["g1"]["low_m"]) < args.max_collapsed_m
        checks["g2_not_collapsed"] = abs(low_m["g2"]["low_m"]) < args.max_collapsed_m
    passed = all(checks.values())
    return {
        "case": case_name,
        "arm": arm,
        "mean_cos_theta": mean_cos,
        "theta_int_coverage": theta_coverage,
        "low_m": low_m,
        "checks": checks,
        "passed": passed,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    results = [evaluate_case(args, case_name) for case_name in args.case]
    payload = {
        "min_mean_cos": args.min_mean_cos,
        "max_collapsed_m": args.max_collapsed_m,
        "cases": results,
        "passed": all(item["passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
