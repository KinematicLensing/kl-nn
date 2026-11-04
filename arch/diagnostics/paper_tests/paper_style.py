"""Shared matplotlib / getdist style for the KL-NN paper figures.

Usage:
    from paper_style import apply_style, COL, DCOL, COLORS, savefig
    apply_style()
    fig, ax = plt.subplots(figsize=(COL, 0.75 * COL))
    ...
    savefig(fig, "F4_raw_bias")   # writes PDF + PNG to OUT
"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# KLNN_PAPER_TESTS_ROOT selects the results tree (default: the original paper-tests/).
OUT = Path(os.environ.get("KLNN_PAPER_TESTS_ROOT", "/ocean/projects/phy250048p/shared/reports/paper-tests")) / "paper_figs_final"
COL = 3.4    # inches, REVTeX single column
DCOL = 7.0   # inches, REVTeX double column

# One colour per estimator/role, used consistently in every figure.
COLORS = {
    "raw": "#1b6ca8",         # per-galaxy posterior mean, equal weights
    "weighted": "#d1495b",    # inverse-variance weighted
    "calibrated": "#2a9d8f",  # response calibrated
    "stacked": "#6a4c93",     # likelihood stacking
    "prediction": "#444444",  # theory curves
    "truth": "#e76f51",       # truth markers in posterior plots
    "g1": "#1b6ca8",
    "g2": "#d1495b",
    "snr_low": "#8ab6d6",
    "snr_high": "#1b3a5c",
}

# getdist smoothing (tuned for 10^4 TF-weighted samples).
GETDIST_SETTINGS = {
    "smooth_scale_1D": 0.4,
    "smooth_scale_2D": 0.4,
    "mult_bias_correction_order": 1,
    "fine_bins": 1024,
    "fine_bins_2D": 256,
}

LABELS = {
    "g1": r"g_1", "g2": r"g_2", "theta_int": r"\theta_{\rm int}", "cosi": r"\cos i",
    "v0": r"v_0\,[{\rm km\,s^{-1}}]", "vcirc": r"v_{\rm circ}\,[{\rm km\,s^{-1}}]",
    "rscale": r"R_{\rm vscale}\,[{\rm arcsec}]", "hlr": r"R_h\,[{\rm arcsec}]",
    "halpha_flux_true": r"\log_{10}F_{{\rm H}\alpha}",
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def savefig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
