#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_PATH = Path("/ocean/projects/phy250048p/shared/temp/rmag_snr_pv.npz")
FIT_PATH = Path("/jet/home/xwang30/kl-nn/arch/rmag_snr_fit.npz")
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = OUT_DIR / "rmag_snr_fit.png"


def _fit_line(log_snr: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * log_snr + b


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source calibration file: {SOURCE_PATH}")
    if not FIT_PATH.exists():
        raise FileNotFoundError(f"Missing fit artifact: {FIT_PATH}")

    with np.load(SOURCE_PATH) as data:
        snr = np.asarray(data["SNR"], dtype=float)
        rmag = np.asarray(data["rmag"], dtype=float)

    mask = np.isfinite(snr) & np.isfinite(rmag) & (snr > 0)
    log_snr = np.log10(snr[mask])
    rmag = rmag[mask]

    with np.load(FIT_PATH, allow_pickle=False) as fit:
        a = float(fit["a"])
        b = float(fit["b"])

    x_grid = np.linspace(log_snr.min(), log_snr.max(), 300)
    y_grid = _fit_line(x_grid, a, b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.scatter(log_snr, rmag, s=9, alpha=0.35, color="#4c78a8", edgecolors="none", label="DESI-PV Subset")
    ax.plot(x_grid, y_grid, color="#d62728", linewidth=2.5, label=f"Fit: rmag = {a:.3f} log10(SNR) + {b:.3f}")
    ax.set_xlabel("log10(SNR)")
    ax.set_ylabel("rmag")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()