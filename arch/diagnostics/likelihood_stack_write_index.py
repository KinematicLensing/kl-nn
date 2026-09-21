#!/usr/bin/env python3
"""Write the likelihood-stack campaign index from stage JSON reports."""

from __future__ import annotations

from argparse import ArgumentParser
from html import escape
import json
from pathlib import Path

REPORT_ROOT = Path("/ocean/projects/phy250048p/shared/reports/likelihood-stack")
STAGES = (
    ("00_select", "Which galaxies the Mean pulls toward zero"),
    ("01_stack", "Shared-parameter stack versus inverse-variance Means"),
    ("02_coverage", "16–84% coverage inside versus outside |g| = 0.05"),
    ("03_nre", "Frozen-encoder NRE peak and posterior mean versus the Mean"),
    ("04_benchmark", "Fixed-shear five-estimator ensemble benchmark"),
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}f}"


def stage_status(folder: Path) -> str:
    html_path = folder / "report.html"
    payload = load_json(folder / "report.json")
    if payload and html_path.is_file():
        return "ready"
    if html_path.is_file():
        return "unparsed"
    return "pending"


def select_rows(payload: dict | None) -> list[str]:
    if not payload:
        return [
            "<tr><td>Select 128 shrunk Means</td><td colspan='5'>pending</td></tr>"
        ]
    parent = payload.get("parent_summary") or {}
    selected = payload.get("selected_summary") or {}
    rows = []
    for title, block, count in (
        ("Parent, |g| > 0.02", parent, parent.get("n")),
        ("Selected", selected, selected.get("n") or payload.get("n_selected")),
    ):
        abs_g = block.get("abs_g") or {}
        shrinkage = block.get("shrinkage") or {}
        rows.append(
            "<tr>"
            f"<td>{escape(title)}</td>"
            f"<td>{escape(str(count if count is not None else '—'))}</td>"
            f"<td>{escape(fmt(abs_g.get('median')))}</td>"
            f"<td>{escape(fmt(shrinkage.get('median')))}</td>"
            f"<td colspan='2'></td>"
            "</tr>"
        )
    return rows


def stack_rows(payload: dict | None) -> list[str]:
    if not payload:
        return [
            "<tr><td>Noise stack</td><td colspan='5'>pending</td></tr>"
        ]
    labels = {
        "mean_of_means": "mean of Means",
        "ivw": "inverse-variance Means",
        "stack_map": "shared-parameter MAP",
    }
    rows = []
    for row in payload.get("table") or []:
        if int(row.get("n_realizations") or 0) != 16:
            continue
        name = labels.get(row.get("estimator"), str(row.get("estimator")))
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>16</td>"
            f"<td>{escape(fmt(row.get('g1_m')))}</td>"
            f"<td>{escape(fmt(row.get('g2_m')))}</td>"
            f"<td>{escape(fmt(row.get('median_abs_residual'), 4))}</td>"
            f"<td></td>"
            "</tr>"
        )
    if not rows:
        return [
            "<tr><td>Noise stack</td><td colspan='5'>unparsed</td></tr>"
        ]
    return rows


def coverage_rows(payload: dict | None) -> list[str]:
    if not payload:
        return [
            "<tr><td>Inner/outer coverage</td><td colspan='5'>pending</td></tr>"
        ]
    labels = {
        ("proposal", "inner"): "training-prior, |g| < 0.05",
        ("proposal", "outer"): "training-prior, |g| > 0.05",
        ("tf", "inner"): "TF-weighted, |g| < 0.05",
        ("tf", "outer"): "TF-weighted, |g| > 0.05",
    }
    rows = []
    for row in payload.get("table") or []:
        key = (row.get("posterior"), row.get("slice"))
        if key not in labels:
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(labels[key])}</td>"
            f"<td>{escape(str(row.get('n', '—')))}</td>"
            f"<td>{escape(fmt(row.get('g1_coverage')))}</td>"
            f"<td>{escape(fmt(row.get('g2_coverage')))}</td>"
            f"<td colspan='2'></td>"
            "</tr>"
        )
    if not rows:
        return [
            "<tr><td>Inner/outer coverage</td><td colspan='5'>unparsed</td></tr>"
        ]
    return rows


def nre_rows(payload: dict | None) -> list[str]:
    if not payload:
        return [
            "<tr><td>NRE vs Mean</td><td colspan='5'>pending</td></tr>"
        ]
    labels = {
        ("nre_map", "full"): "NRE peak, full catalog",
        ("nre_mean", "full"): "NRE posterior mean, full catalog",
        ("npe_mean", "full"): "NPE Mean, full catalog",
        ("nre_stack_map", "outer"): "stacked NRE peak, |g| > 0.05, N = 128",
        ("nre_mean_of_means", "outer"): "mean of NRE posterior means, |g| > 0.05, N = 128",
        ("npe_mean_of_means", "outer"): "mean of Means, |g| > 0.05, N = 128",
    }
    rows = []
    for row in payload.get("galaxy_table") or []:
        key = (row.get("estimator"), row.get("slice"))
        if key not in labels:
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(labels[key])}</td>"
            f"<td>{escape(str(row.get('n', '—')))}</td>"
            f"<td>{escape(fmt(row.get('g1_m')))}</td>"
            f"<td>{escape(fmt(row.get('g2_m')))}</td>"
            f"<td colspan='2'></td>"
            "</tr>"
        )
    for row in payload.get("stack_table") or []:
        if int(row.get("stack_size") or 0) != 128:
            continue
        key = (row.get("estimator"), row.get("slice"))
        if key not in labels:
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(labels[key])}</td>"
            f"<td>{escape(str(row.get('n_groups', '—')))}</td>"
            f"<td>{escape(fmt(row.get('g1_m')))}</td>"
            f"<td>{escape(fmt(row.get('g2_m')))}</td>"
            f"<td colspan='2'></td>"
            "</tr>"
        )
    if not rows:
        return [
            "<tr><td>NRE vs Mean</td><td colspan='5'>unparsed</td></tr>"
        ]
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    root = args.report_root
    root.mkdir(parents=True, exist_ok=True)
    links = []
    for folder, title in STAGES:
        status = stage_status(root / folder)
        links.append(
            f"<li><a href='{escape(folder)}/report.html'>{escape(title)}</a> — {status}</li>"
        )
    select_payload = load_json(root / "00_select" / "report.json")
    stack_payload = load_json(root / "01_stack" / "report.json")
    coverage_payload = load_json(root / "02_coverage" / "report.json")
    nre_payload = load_json(root / "03_nre" / "report.json")
    benchmark_payload = load_json(root / "04_benchmark" / "report.json")
    takeaway = ""
    if benchmark_payload and benchmark_payload.get("takeaway"):
        takeaway = f"<p>{escape(str(benchmark_payload['takeaway']))}</p>"
    elif nre_payload and nre_payload.get("takeaway"):
        takeaway = f"<p>{escape(str(nre_payload['takeaway']))}</p>"
    elif coverage_payload and coverage_payload.get("takeaway"):
        takeaway = f"<p>{escape(str(coverage_payload['takeaway']))}</p>"
    elif stack_payload and stack_payload.get("takeaway"):
        takeaway = f"<p>{escape(str(stack_payload['takeaway']))}</p>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>likelihood-stack campaign</title>
<style>
body {{ font: 16px/1.5 Palatino, serif; margin: 2rem auto; max-width: 960px; }}
table {{ border-collapse: collapse; width: 100%; font: 14px/1.4 ui-sans-serif, sans-serif; }}
th, td {{ border-bottom: 1px solid #ccc; padding: .35rem .45rem; text-align: left; }}
</style></head><body>
<h1>likelihood-stack campaign</h1>
<p>Diagnostic only: redraw independent noise on 128 galaxies whose cached
Mean is already pulled toward zero, then compare a shared-parameter
likelihood stack to an inverse-variance mean of Means. A later page splits
the cached 16th–84th interval at |g| = 0.05. A third page trains a shear-only
ratio head on the frozen encoder and stacks it on xu3. This is not a catalog
shear estimator and does not replace the multiplicative-bias tables. Frozen
network trained out to shear 0.1; identity samples; no hopping.</p>
<ul>{''.join(links)}</ul>
<table><thead><tr>
<th>Stage</th><th>N</th><th>median |g| or m of g1 or coverage of g1</th>
<th>median shrinkage or m of g2 or coverage of g2</th>
<th>median |ĝ − g|</th><th></th>
</tr></thead><tbody>
{''.join(select_rows(select_payload))}
{''.join(stack_rows(stack_payload))}
{''.join(coverage_rows(coverage_payload))}
{''.join(nre_rows(nre_payload))}
</tbody></table>
{takeaway}
<p>See <a href="STATUS.txt">STATUS.txt</a> for job IDs.</p>
</body></html>
"""
    path = root / "index.html"
    path.write_text(document, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
