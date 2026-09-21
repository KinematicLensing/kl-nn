#!/usr/bin/env python3
"""Re-render a few simulator-v3 rows and subtract the stored noiseless FITS."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

try:
    from .generation_integrity import simulator_v3_output_path
    from .observation_schema import (
        classify_fiber_offset_sign,
        swap_minor_axis_fibers,
    )
except ImportError:
    from generation_integrity import simulator_v3_output_path
    from observation_schema import (
        classify_fiber_offset_sign,
        swap_minor_axis_fibers,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_ROOT = Path("/ocean/projects/phy250048p/shared/samples")
FITS_ROOT = Path("/ocean/projects/phy250048p/shared/fits")
REPORT_HTML = Path(
    "/ocean/projects/phy250048p/shared/reports/fiber-gauge/"
    "00_regen_subtract/simv3_regen_subtract_g01_xu3.html"
)
REPORT_JSON = REPORT_HTML.with_suffix(".json")
PART_SIZE = 2000
SPECTRUM_HDU_COUNT = 5
IMAGE_HDU = 6
HEADER_KEYS = ("OBSMODV", "FIBLAY", "ADDNOISE", "ROWFP", "ROWID")

GALAXIES = (
    {
        "label": "valid_theta_-2.62",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 39244,
        "arm": "valid_g01",
    },
    {
        "label": "valid_theta_-1.57",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 66398,
        "arm": "valid_g01",
    },
    {
        "label": "valid_theta_-0.53",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 89167,
        "arm": "valid_g01",
    },
    {
        "label": "valid_theta_+0.52",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 23283,
        "arm": "valid_g01",
    },
    {
        "label": "valid_theta_+1.57",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 91803,
        "arm": "valid_g01",
    },
    {
        "label": "valid_theta_+2.62",
        "sample_csv": "valid_100k_simv3_cosi.csv",
        "original_dataset": "valid_100k_simv3_cosi",
        "regen_dataset": "valid_100k_simv3_cosi_regen_probe",
        "sample_id": 44440,
        "arm": "valid_g01",
    },
    {
        "label": "xu3_theta_-1.57",
        "sample_csv": "test_100k_simv3_cosi_xu3_tf.csv",
        "original_dataset": "test_100k_simv3_cosi_xu3_tf",
        "regen_dataset": "test_100k_simv3_cosi_xu3_tf_regen_probe",
        "sample_id": 81573,
        "arm": "xu3",
    },
    {
        "label": "xu3_theta_+0.00",
        "sample_csv": "test_100k_simv3_cosi_xu3_tf.csv",
        "original_dataset": "test_100k_simv3_cosi_xu3_tf",
        "regen_dataset": "test_100k_simv3_cosi_xu3_tf_regen_probe",
        "sample_id": 7021,
        "arm": "xu3",
    },
    {
        "label": "xu3_theta_+1.57",
        "sample_csv": "test_100k_simv3_cosi_xu3_tf.csv",
        "original_dataset": "test_100k_simv3_cosi_xu3_tf",
        "regen_dataset": "test_100k_simv3_cosi_xu3_tf_regen_probe",
        "sample_id": 84319,
        "arm": "xu3",
    },
    {
        "label": "g02_control_theta_+0.00",
        "sample_csv": "train_100k_simv3_cosi_g02.csv",
        "original_dataset": "train_100k_simv3_cosi_g02",
        "regen_dataset": "train_100k_simv3_cosi_g02_regen_probe",
        "sample_id": 12683,
        "arm": "g02_control",
    },
    {
        "label": "g002_control_theta_+0.00",
        "sample_csv": "train_100k_simv3_cosi_g002.csv",
        "original_dataset": "train_100k_simv3_cosi_g002",
        "regen_dataset": "train_100k_simv3_cosi_g002_regen_probe",
        "sample_id": 55813,
        "arm": "g002_control",
    },
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-index",
        type=int,
        help="1-based galaxy index in GALAXIES to re-render",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compare stored vs regen FITS and write HTML/JSON",
    )
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--fits-root", type=Path, default=FITS_ROOT)
    parser.add_argument("--output-html", type=Path, default=REPORT_HTML)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    return parser.parse_args(argv)


def part_number(sample_id: int) -> int:
    return int(sample_id) // PART_SIZE + 1


def load_row(sample_root: Path, sample_csv: str, sample_id: int) -> pd.Series:
    table = pd.read_csv(sample_root / sample_csv, float_precision="round_trip")
    if "ID" not in table.columns:
        raise ValueError(f"{sample_csv} has no ID column")
    matches = table.loc[table["ID"] == sample_id]
    if len(matches) != 1:
        raise ValueError(
            f"{sample_csv} ID={sample_id} matched {len(matches)} rows"
        )
    row = matches.iloc[0]
    if int(row.name) != int(sample_id):
        raise ValueError(
            f"{sample_csv} ID={sample_id} is not at iloc {sample_id} "
            f"(iloc={int(row.name)}); wrapper -i uses row index"
        )
    return row


def render_galaxy(galaxy: dict, *, sample_root: Path, fits_root: Path) -> Path:
    sample_id = int(galaxy["sample_id"])
    part = part_number(sample_id)
    load_row(sample_root, galaxy["sample_csv"], sample_id)
    original = simulator_v3_output_path(
        fits_root, galaxy["original_dataset"], part, sample_id
    )
    if not original.is_file():
        raise FileNotFoundError(f"Missing original FITS: {original}")
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_fits_wrapper.py"),
        f"-i={sample_id}",
        f"-j={sample_id + 1}",
        f"-n={part}",
        f"-s={galaxy['sample_csv']}",
        f"-d={galaxy['regen_dataset']}",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(SCRIPT_DIR))
    regen = simulator_v3_output_path(
        fits_root, galaxy["regen_dataset"], part, sample_id
    )
    if not regen.is_file():
        raise FileNotFoundError(f"Generator did not write {regen}")
    return regen


def residual_stats(difference: np.ndarray) -> dict[str, float]:
    flat = np.asarray(difference, dtype=np.float64).ravel()
    return {
        "maxabs": float(np.max(np.abs(flat))) if flat.size else float("nan"),
        "rms": float(np.sqrt(np.mean(np.square(flat)))) if flat.size else float("nan"),
    }


def load_product(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdus:
        if len(hdus) != SPECTRUM_HDU_COUNT + 2:
            raise ValueError(f"{path} has {len(hdus)} HDUs, expected 7")
        spectra = np.stack(
            [
                np.asarray(hdus[index].data, dtype=np.float64)
                for index in range(1, SPECTRUM_HDU_COUNT + 1)
            ]
        )
        positions = np.asarray(
            [
                (
                    float(hdus[index].header["FIBERDX"]),
                    float(hdus[index].header["FIBERDY"]),
                )
                for index in range(1, SPECTRUM_HDU_COUNT + 1)
            ],
            dtype=np.float64,
        )
        image = np.asarray(hdus[IMAGE_HDU].data, dtype=np.float64)
        primary = {
            key: hdus[0].header.get(key) for key in HEADER_KEYS if key in hdus[0].header
        }
        addnoise = [
            hdus[index].header.get("ADDNOISE")
            for index in range(1, SPECTRUM_HDU_COUNT + 2)
        ]
    return {
        "spectra": spectra,
        "positions": positions,
        "image": image,
        "primary_header": {key: str(value) for key, value in primary.items()},
        "addnoise": [str(value) for value in addnoise],
    }


def classify_product(product: dict, row: pd.Series) -> str:
    return classify_fiber_offset_sign(
        product["positions"],
        g1=float(row["g1"]),
        g2=float(row["g2"]),
        theta_int=float(row["theta_int"]),
        sini=float(row["sini"]),
    )


def compare_galaxy(galaxy: dict, *, sample_root: Path, fits_root: Path) -> dict:
    sample_id = int(galaxy["sample_id"])
    part = part_number(sample_id)
    row = load_row(sample_root, galaxy["sample_csv"], sample_id)
    original_path = simulator_v3_output_path(
        fits_root, galaxy["original_dataset"], part, sample_id
    )
    regen_path = simulator_v3_output_path(
        fits_root, galaxy["regen_dataset"], part, sample_id
    )
    stored = load_product(original_path)
    regen = load_product(regen_path)
    image_diff = residual_stats(stored["image"] - regen["image"])
    spec_stats = [
        residual_stats(stored["spectra"][index] - regen["spectra"][index])
        for index in range(SPECTRUM_HDU_COUNT)
    ]
    swapped_spectra, swapped_positions = swap_minor_axis_fibers(
        stored["spectra"], stored["positions"]
    )
    swapped_spec_stats = [
        residual_stats(swapped_spectra[index] - regen["spectra"][index])
        for index in range(SPECTRUM_HDU_COUNT)
    ]
    stored_sign = classify_product(stored, row)
    regen_sign = classify_product(regen, row)
    swapped_sign = classify_fiber_offset_sign(
        swapped_positions,
        g1=float(row["g1"]),
        g2=float(row["g2"]),
        theta_int=float(row["theta_int"]),
        sini=float(row["sini"]),
    )
    return {
        **galaxy,
        "part": part,
        "g1": float(row["g1"]),
        "g2": float(row["g2"]),
        "theta_int": float(row["theta_int"]),
        "sini": float(row["sini"]),
        "original_path": str(original_path),
        "regen_path": str(regen_path),
        "stored_sign": stored_sign,
        "regen_sign": regen_sign,
        "stored_after_swap_sign": swapped_sign,
        "stored_addnoise": stored["addnoise"],
        "regen_addnoise": regen["addnoise"],
        "stored_header": stored["primary_header"],
        "regen_header": regen["primary_header"],
        "image": image_diff,
        "spectra": spec_stats,
        "after_minor_swap": {
            "image": image_diff,
            "spectra": swapped_spec_stats,
        },
        "position_maxabs": float(
            np.max(np.abs(stored["positions"] - regen["positions"]))
        ),
        "position_maxabs_after_swap": float(
            np.max(np.abs(swapped_positions - regen["positions"]))
        ),
    }


def verdict(rows: list[dict]) -> str:
    image_floor = 1.0e-6
    spec_match_floor = 1.0e-6
    g01_xu3 = [row for row in rows if row["arm"] in {"valid_g01", "xu3"}]
    controls = [row for row in rows if row["arm"].endswith("_control")]
    g01_image_ok = all(row["image"]["maxabs"] < image_floor for row in g01_xu3)
    g01_swap = all(row["stored_sign"] == "swap_minor" for row in g01_xu3)
    g01_regen_match = all(row["regen_sign"] == "match" for row in g01_xu3)
    g01_after_swap_ok = all(
        max(item["maxabs"] for item in row["after_minor_swap"]["spectra"])
        < spec_match_floor
        for row in g01_xu3
    )
    control_ok = all(
        row["stored_sign"] == "match"
        and row["regen_sign"] == "match"
        and row["image"]["maxabs"] < image_floor
        and max(item["maxabs"] for item in row["spectra"]) < spec_match_floor
        for row in controls
    )
    if g01_image_ok and g01_swap and g01_regen_match and g01_after_swap_ok and control_ok:
        return (
            "Image and control residuals are ~0. Stored ±0.1/xu3 fibers are "
            "swap_minor; current regen is match. After swapping the stored "
            "minor-axis pair, spectra also match. Same renderer, different "
            "fiber gauge."
        )
    bit_match = all(
        row["image"]["maxabs"] < image_floor
        and max(item["maxabs"] for item in row["spectra"]) < spec_match_floor
        and row["stored_sign"] == row["regen_sign"] == "match"
        for row in rows
    )
    if bit_match:
        return "Current pipeline bit-matches every stored product, including fibers."
    g01_image_fail = any(row["image"]["maxabs"] >= image_floor for row in g01_xu3)
    if g01_image_fail:
        return (
            "Image residual is not ~0 on ±0.1/xu3. This is a real render "
            "change, not only the fiber gauge."
        )
    return (
        "Residuals do not match the swap_minor-only pattern. Inspect the "
        "per-galaxy table and FITS headers."
    )


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def write_html(rows: list[dict], path: Path, conclusion: str) -> None:
    def cell(value) -> str:
        return f"<td>{html.escape(str(value))}</td>"

    body_rows = []
    for row in rows:
        spec_max = max(item["maxabs"] for item in row["spectra"])
        swap_spec_max = max(
            item["maxabs"] for item in row["after_minor_swap"]["spectra"]
        )
        body_rows.append(
            "<tr>"
            + cell(row["label"])
            + cell(row["sample_id"])
            + cell(format_float(row["theta_int"]))
            + cell(row["stored_sign"])
            + cell(row["regen_sign"])
            + cell(format_float(row["image"]["maxabs"]))
            + cell(format_float(spec_max))
            + cell(format_float(swap_spec_max))
            + cell(format_float(row["position_maxabs"]))
            + cell(format_float(row["position_maxabs_after_swap"]))
            + "</tr>"
        )
    spec_detail = []
    for row in rows:
        fibers = "".join(
            f"<td>{format_float(item['maxabs'])}</td>" for item in row["spectra"]
        )
        spec_detail.append(
            f"<tr>{cell(row['label'])}{cell(row['sample_id'])}{fibers}</tr>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>simulator-v3 noiseless regen subtraction</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; max-width: 1100px; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.55rem; text-align: left; }}
th {{ background: #f4f4f4; }}
code {{ background: #f4f4f4; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>Noiseless regen subtraction</h1>
<p>{html.escape(conclusion)}</p>
<p>Stored FITS are already noiseless (<code>ADD_NOISE=False</code>).
Regen used current <code>generate_fits_wrapper.py</code> /
<code>compute_fiber_offsets</code>.</p>
<h2>Summary</h2>
<table>
<thead>
<tr>
<th>label</th><th>ID</th><th>theta_int</th>
<th>stored fibers</th><th>regen fibers</th>
<th>image max|Δ|</th><th>spec max|Δ|</th>
<th>spec max|Δ| after minor swap</th>
<th>fiber xy max|Δ|</th><th>fiber xy after swap</th>
</tr>
</thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
<h2>Per-fiber spectrum max|Δ| (HDU 1–5)</h2>
<table>
<thead>
<tr><th>label</th><th>ID</th>
<th>f0</th><th>f1</th><th>f2 center</th><th>f3 minor+</th><th>f4 minor-</th>
</tr>
</thead>
<tbody>
{''.join(spec_detail)}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.render_index is None and not args.report:
        raise SystemExit("pass --render-index N and/or --report")
    if args.render_index is not None:
        if args.render_index < 1 or args.render_index > len(GALAXIES):
            raise SystemExit(
                f"--render-index must be in 1..{len(GALAXIES)}"
            )
        galaxy = GALAXIES[args.render_index - 1]
        print(
            f"render {args.render_index}/{len(GALAXIES)} {galaxy['label']} "
            f"id={galaxy['sample_id']}",
            flush=True,
        )
        path = render_galaxy(
            galaxy, sample_root=args.sample_root, fits_root=args.fits_root
        )
        print(f"wrote {path}", flush=True)
    if args.report:
        rows = [
            compare_galaxy(
                galaxy, sample_root=args.sample_root, fits_root=args.fits_root
            )
            for galaxy in GALAXIES
        ]
        conclusion = verdict(rows)
        payload = {"conclusion": conclusion, "galaxies": rows}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_html(rows, args.output_html, conclusion)
        print(conclusion, flush=True)
        print(f"wrote {args.output_html}", flush=True)
        print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
