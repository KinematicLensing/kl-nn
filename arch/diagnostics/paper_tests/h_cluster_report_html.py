#!/usr/bin/env python3
"""H: assemble the cluster-test HTML report from a template, embedding the figures as data URIs.

usage: h_cluster_report_html.py [tag] [out.html] [template.html]
Only the {{img:key}} placeholders the template uses are filled.
"""
import base64
from pathlib import Path
import sys

S = Path("/ocean/projects/phy250048p/shared")
here = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "s3_r0"
base = S / "reports/paper-tests-1m/H_cluster" / tag
out = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "report.html"
template = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "h_cluster_report_template.html"
imgs = {"setup": base / "report_figs/setup.png", "cells": base / "report_figs/cells.png",
        "validation": base / "report_figs/validation.png", "tuning": base / "report_figs/tuning.png",
        "cellbias": base / "synthetic/cell_bias.png", "kappa": base / "synthetic/kappa_maps.png",
        "npe_cellbias": base / "npe/cell_bias.png", "npe_kappa": base / "npe/kappa_maps.png",
        "npe_shear": base / "npe/shear_maps.png", "syn_shear": base / "synthetic/shear_maps.png",
        "ens_cells": base.parent / "ensemble/cells.png", "ens_detection": base.parent / "ensemble/detection.png",
        "ens_diffmaps": base.parent / "ensemble/diffmaps.png"}
html = template.read_text()
for key, path in imgs.items():
    if "{{img:%s}}" % key in html:
        html = html.replace("{{img:%s}}" % key, "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode())
assert "{{img:" not in html, "unfilled image placeholder"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html)
print(out, f"{len(html) / 1e6:.2f} MB")
