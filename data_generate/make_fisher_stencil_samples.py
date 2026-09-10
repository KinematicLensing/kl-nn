#!/usr/bin/env python3
"""Build hybrid Fisher finite-difference sample tables and manifests.

Two catalogs, disjoint bases from one source table:

* full: native-(g1, g2) center plus bound-safe ± steps in all nine μ parameters
* g5: g=0 five-point shear stencil (same states as the existing response catalog)

Optional 2Δ audit rows for g1 and vcirc on the first N full bases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )
except ImportError:
    from observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )


SIMULATION_PARAMETERS = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
AUXILIARY_COLUMNS = (
    RMAG_TRUE_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_SNR_COLUMN,
    CENTRAL_HALPHA_SNR_COLUMN,
    FIBER_LAYOUT_COLUMN,
    OBSERVATION_MODEL_VERSION_COLUMN,
)
REQUIRED_COLUMNS = (*SIMULATION_PARAMETERS, *AUXILIARY_COLUMNS)
FISHER_PARAMETERS = (*SIMULATION_PARAMETERS, HALPHA_FLUX_TRUE_COLUMN)

PARAMETER_BOUNDS = {
    "g1": (-0.1, 0.1),
    "g2": (-0.1, 0.1),
    "theta_int": (-np.pi, np.pi),
    "sini": (0.0, 1.0),
    "v0": (-30.0, 30.0),
    "vcirc": (60.0, 540.0),
    "rscale": (0.1, 5.0),
    "hlr": (0.1, 5.0),
    HALPHA_FLUX_TRUE_COLUMN: (1.0e-17, 1.0e-14),
}
DEFAULT_STEPS = {
    "g1": ("additive", 0.01),
    "g2": ("additive", 0.01),
    "theta_int": ("additive", 0.05),
    "sini": ("additive", 0.02),
    "v0": ("additive", 1.0),
    "vcirc": ("additive", 5.0),
    "rscale": ("additive", 0.05),
    "hlr": ("additive", 0.05),
    HALPHA_FLUX_TRUE_COLUMN: ("log10", 0.02),
}
G5_STATES = (
    ("zero", 0.0, 0.0),
    ("g1_plus", 1.0, 0.0),
    ("g1_minus", -1.0, 0.0),
    ("g2_plus", 0.0, 1.0),
    ("g2_minus", 0.0, -1.0),
)
DOUBLE_DELTA_PARAMETERS = ("g1", "vcirc")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--g5-output", type=Path, required=True)
    parser.add_argument("--g5-manifest", type=Path, required=True)
    parser.add_argument("--n-full", type=int, default=250)
    parser.add_argument("--n-g5", type=int, default=500)
    parser.add_argument("--n-double-delta", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--delta-g", type=float, default=0.01)
    parser.add_argument("--full-bins", type=int, default=4)
    parser.add_argument("--g5-bins", type=int, default=10)
    return parser.parse_args(argv)


def wrap_theta(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def apply_step(name: str, value: float, sign: int, step: tuple[str, float]) -> float:
    kind, amount = step
    if kind == "additive":
        candidate = float(value) + sign * float(amount)
    elif kind == "log10":
        candidate = float(value) * (10.0 ** (sign * float(amount)))
    else:
        raise ValueError(f"unknown step kind {kind!r} for {name}")
    if name == "theta_int":
        return wrap_theta(candidate)
    return candidate


def in_bounds(name: str, value: float, *, atol: float = 1e-12) -> bool:
    lo, hi = PARAMETER_BOUNDS[name]
    return (lo - atol) <= float(value) <= (hi + atol)


def parameter_sides(name: str, value: float, step: tuple[str, float]) -> tuple[str, float, float]:
    plus = apply_step(name, value, 1, step)
    minus = apply_step(name, value, -1, step)
    plus_ok = in_bounds(name, plus)
    minus_ok = in_bounds(name, minus)
    if plus_ok and minus_ok:
        return "two_sided", plus, minus
    if plus_ok:
        return "plus_only", plus, float("nan")
    if minus_ok:
        return "minus_only", float("nan"), minus
    raise ValueError(f"no in-bounds finite-difference step for {name}={value}")


def cell_codes(frame: pd.DataFrame, keys: tuple[tuple[str, str], ...], n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    cells = np.zeros(len(frame), dtype=np.int64)
    for name, transform in keys:
        values = frame[name].to_numpy(dtype=np.float64)
        if transform == "log":
            values = np.log10(values)
        elif transform == "abs":
            values = np.abs(values)
        elif transform != "linear":
            raise ValueError(f"unknown transform {transform!r}")
        order = np.argsort(values, kind="mergesort")
        rank = np.empty(len(values), dtype=np.float64)
        rank[order] = np.linspace(0.0, 1.0, len(values), endpoint=False)
        bins = np.minimum((rank * n_bins).astype(np.int64), n_bins - 1)
        cells = cells * n_bins + bins
    return cells


def stratified_choice(
    n: int,
    cells: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    eligible = np.arange(len(cells)) if mask is None else np.flatnonzero(mask)
    if n > len(eligible):
        raise ValueError(f"requested {n} rows but only {len(eligible)} remain")
    grouped: dict[int, list[int]] = {}
    for index in eligible:
        grouped.setdefault(int(cells[index]), []).append(int(index))
    buckets = []
    for members in grouped.values():
        array = np.asarray(members, dtype=np.int64)
        rng.shuffle(array)
        buckets.append(array)
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        progressed = False
        for bucket in buckets:
            if depth < len(bucket):
                chosen.append(int(bucket[depth]))
                progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            raise RuntimeError("stratified sampler exhausted eligible rows")
        depth += 1
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _observation_metadata(row: pd.Series) -> dict:
    return {name: row[name] for name in AUXILIARY_COLUMNS}


def _append_row(
    sample_rows: list[dict],
    manifest_rows: list[dict],
    *,
    output_id: int,
    base_id: int,
    source_row: int,
    state: str,
    catalog: str,
    difference: str,
    stepped: str,
    values: dict,
    observation: dict,
    delta: float,
) -> int:
    sample_rows.append({"ID": output_id, **values, **observation})
    manifest_rows.append(
        {
            "ID": output_id,
            "base_id": base_id,
            "source_row": source_row,
            "catalog": catalog,
            "state": state,
            "difference": difference,
            "stepped": stepped,
            "delta": delta,
        }
    )
    return output_id + 1


def build_full_catalog(
    source: pd.DataFrame,
    chosen: np.ndarray,
    *,
    steps: dict[str, tuple[str, float]],
    n_double_delta: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict] = []
    manifest_rows: list[dict] = []
    output_id = 0
    parameter_columns = list(SIMULATION_PARAMETERS)
    for base_id, source_row in enumerate(chosen):
        row = source.iloc[int(source_row)]
        center = row[parameter_columns].astype(float).to_dict()
        center[HALPHA_FLUX_TRUE_COLUMN] = float(row[HALPHA_FLUX_TRUE_COLUMN])
        observation = _observation_metadata(row)
        output_id = _append_row(
            sample_rows,
            manifest_rows,
            output_id=output_id,
            base_id=base_id,
            source_row=int(source_row),
            state="center",
            catalog="full",
            difference="center",
            stepped="",
            values=center,
            observation=observation,
            delta=0.0,
        )
        for name in FISHER_PARAMETERS:
            kind, amount = steps[name]
            side, plus, minus = parameter_sides(name, center[name], (kind, amount))
            for label, candidate in (("plus", plus), ("minus", minus)):
                if not np.isfinite(candidate):
                    continue
                values = dict(center)
                values[name] = float(candidate)
                observation_step = dict(observation)
                if name == HALPHA_FLUX_TRUE_COLUMN:
                    observation_step[HALPHA_FLUX_TRUE_COLUMN] = float(candidate)
                output_id = _append_row(
                    sample_rows,
                    manifest_rows,
                    output_id=output_id,
                    base_id=base_id,
                    source_row=int(source_row),
                    state=f"{name}_{label}",
                    catalog="full",
                    difference=side,
                    stepped=name,
                    values=values,
                    observation=observation_step,
                    delta=float(amount),
                )
        if base_id < n_double_delta:
            for name in DOUBLE_DELTA_PARAMETERS:
                kind, amount = steps[name]
                double = (kind, 2.0 * float(amount))
                side, plus, minus = parameter_sides(name, center[name], double)
                if side in {"two_sided", "plus_only"}:
                    values = dict(center)
                    values[name] = plus
                    output_id = _append_row(
                        sample_rows,
                        manifest_rows,
                        output_id=output_id,
                        base_id=base_id,
                        source_row=int(source_row),
                        state=f"{name}_plus_2d",
                        catalog="full",
                        difference=side,
                        stepped=name,
                        values=values,
                        observation=observation,
                        delta=float(double[1]),
                    )
                if side in {"two_sided", "minus_only"}:
                    values = dict(center)
                    values[name] = minus
                    output_id = _append_row(
                        sample_rows,
                        manifest_rows,
                        output_id=output_id,
                        base_id=base_id,
                        source_row=int(source_row),
                        state=f"{name}_minus_2d",
                        catalog="full",
                        difference=side,
                        stepped=name,
                        values=values,
                        observation=observation,
                        delta=float(double[1]),
                    )

    samples = pd.DataFrame(
        sample_rows,
        columns=["ID", *SIMULATION_PARAMETERS, *AUXILIARY_COLUMNS],
    )
    manifest = pd.DataFrame(manifest_rows)
    return samples, manifest


def build_g5_catalog(
    source: pd.DataFrame,
    chosen: np.ndarray,
    *,
    delta_g: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict] = []
    manifest_rows: list[dict] = []
    output_id = 0
    parameter_columns = list(SIMULATION_PARAMETERS)
    for base_id, source_row in enumerate(chosen):
        row = source.iloc[int(source_row)]
        nuisance = row[parameter_columns].astype(float).to_dict()
        observation = _observation_metadata(row)
        for state, g1_sign, g2_sign in G5_STATES:
            values = dict(nuisance)
            values["g1"] = g1_sign * delta_g
            values["g2"] = g2_sign * delta_g
            output_id = _append_row(
                sample_rows,
                manifest_rows,
                output_id=output_id,
                base_id=base_id,
                source_row=int(source_row),
                state=state,
                catalog="g5",
                difference="two_sided" if state != "zero" else "center",
                stepped="" if state == "zero" else state.rsplit("_", 1)[0],
                values=values,
                observation=observation,
                delta=0.0 if state == "zero" else float(delta_g),
            )
    samples = pd.DataFrame(
        sample_rows,
        columns=["ID", *SIMULATION_PARAMETERS, *AUXILIARY_COLUMNS],
    )
    manifest = pd.DataFrame(manifest_rows)
    return samples, manifest


def load_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, float_precision="round_trip")
    unnamed = [column for column in source if str(column).startswith("Unnamed:")]
    if unnamed:
        source = source.drop(columns=unnamed)
    missing = [name for name in REQUIRED_COLUMNS if name not in source]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return source.reset_index(drop=True)


def write_catalog(samples: pd.DataFrame, manifest: pd.DataFrame, output: Path, manifest_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output, index=False)
    manifest.to_csv(manifest_path, index=False)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.n_full < 1 or args.n_g5 < 1:
        raise ValueError("n-full and n-g5 must be positive")
    if args.n_double_delta < 0 or args.n_double_delta > args.n_full:
        raise ValueError("n-double-delta must lie in [0, n-full]")
    if not 0.0 < args.delta_g <= 0.1:
        raise ValueError("delta-g must be in (0, 0.1]")
    source = load_source(args.input)
    if args.n_full + args.n_g5 > len(source):
        raise ValueError("requested bases exceed the source table")
    rng = np.random.default_rng(args.seed)
    steps = dict(DEFAULT_STEPS)
    steps["g1"] = ("additive", float(args.delta_g))
    steps["g2"] = ("additive", float(args.delta_g))

    ranked = source.copy()
    ranked["g_abs"] = np.hypot(
        source["g1"].to_numpy(dtype=np.float64),
        source["g2"].to_numpy(dtype=np.float64),
    )
    full_cells = cell_codes(
        ranked,
        (
            ("sini", "linear"),
            ("hlr", "log"),
            ("vcirc", "log"),
            ("g_abs", "linear"),
        ),
        args.full_bins,
    )
    full_chosen = stratified_choice(args.n_full, full_cells, rng)
    remaining = np.ones(len(source), dtype=bool)
    remaining[full_chosen] = False
    g5_cells = cell_codes(
        source,
        (("sini", "linear"), ("hlr", "log")),
        args.g5_bins,
    )
    g5_chosen = stratified_choice(args.n_g5, g5_cells, rng, mask=remaining)

    full_samples, full_manifest = build_full_catalog(
        source,
        full_chosen,
        steps=steps,
        n_double_delta=args.n_double_delta,
    )
    g5_samples, g5_manifest = build_g5_catalog(
        source, g5_chosen, delta_g=float(args.delta_g)
    )
    write_catalog(full_samples, full_manifest, args.full_output, args.full_manifest)
    write_catalog(g5_samples, g5_manifest, args.g5_output, args.g5_manifest)
    print(
        f"Wrote {len(full_samples)} full-stencil rows "
        f"({args.n_full} bases, {args.n_double_delta} with 2Δ audit)"
    )
    print(f"Full samples: {args.full_output}")
    print(f"Full manifest: {args.full_manifest}")
    print(f"Wrote {len(g5_samples)} g5-stencil rows ({args.n_g5} bases)")
    print(f"g5 samples: {args.g5_output}")
    print(f"g5 manifest: {args.g5_manifest}")


if __name__ == "__main__":
    main()
