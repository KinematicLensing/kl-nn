#!/usr/bin/env python3
"""Paper figures for the Xu3 test set (Sec. 3.3) and its shape noise (Sec. 5.1).

  fig_test_set          z, m_r, R_h and H-alpha SNR distributions of the sample-3 catalog rows the Xu3 test set is
                        drawn from (full-footprint catalog, H-alpha matched, training-range cuts; 2.32M rows,
                        unweighted, as sampled). Probability densities: the catalog has no single footprint area.
  fig_shape_noise_snr   median shape noise sigma_eps = sqrt(sigma_g1^2 + sigma_g2^2) of the TF-weighted posteriors
                        in bins of image SNR x H-alpha SNR, over the 10^5 Xu3 test galaxies, with contours enclosing
                        68% and 95% of the galaxies.
  fig_shape_noise_cosi_mag  the same median shape noise in bins of true cos i x m_r (cf. Fig. 4 of Xu et al. 2024,
                        which shows the shape noise vs m_r for cuts in sin i).
  fig_shape_noise_maps  the two maps above stacked vertically, each with its own colour bar (Sec. 5.3).

Run with KLNN_PAPER_TESTS_ROOT=$S/reports/paper-tests-1m; writes to paper_style.OUT and numbers to testset_numbers.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KLNN_PAPER_TESTS_ROOT", "/ocean/projects/phy250048p/shared/reports/paper-tests-1m")
from paper_style import COL, DCOL, COLORS, OUT, apply_style, savefig  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
CATALOG = S / "desi_cut_catalogs/xu_sample_3_fullfootprint.fits"
PER_GALAXY = S / "reports/paper-tests-1m/A_shear/per_galaxy.npz"
# eligibility of the Xu3 test set (samples/test_100k_simv3_cosi_xu3_tf_measaxis.manifest.json)
CUTS = {"rmag": (15.0, 23.4), "hlr": (0.1, 5.0), "img_snr": (10.0, 1000.0), "halpha_snr": (1.0, 150.0)}
MIN_PER_BIN = 30
N_BINS = 16
RHO_I_RANGE, RHO_S_RANGE = (50.0, 1000.0), (1.0, 150.0)
COSI_RANGE, MAG_RANGE = (0.0, 1.0), (16.8, 20.3)  # m_r < 16.8 holds 0.7% of the galaxies and is left out
NUM = {}


def catalog_rows():
    from astropy.io import fits
    data = fits.open(CATALOG, memmap=True)[1].data
    keep = {k: [] for k in ("z", "rmag", "hlr", "halpha_snr", "img_snr")}
    for a in range(0, len(data), 1_000_000):
        c = data[a:a + 1_000_000]
        cols = {k: np.asarray(c[k], float) for k in keep}
        ok = np.ones(len(c), bool)
        for k, (lo, hi) in CUTS.items():
            ok &= np.isfinite(cols[k]) & (cols[k] >= lo) & (cols[k] <= hi)
        for k in keep:
            keep[k].append(cols[k][ok])
    return {k: np.concatenate(v) for k, v in keep.items()}


def fig_test_set():
    d = catalog_rows()
    NUM["test_set_catalog_rows"] = int(len(d["rmag"]))
    panels = (("z", np.linspace(0, 0.6, 41), r"$z$"), ("rmag", np.linspace(15, 20.5, 45), r"$m_r$"),
              ("hlr", np.linspace(1, 5, 41), r"$R_h\,[{\rm arcsec}]$"), ("halpha_snr", np.linspace(0, 150, 41), r"H$\alpha$ SNR $\rho_S$"))
    fig, axes = plt.subplots(1, 4, figsize=(DCOL, 0.3 * DCOL))
    for ax, (k, bins, lab) in zip(axes, panels):
        x = d[k][np.isfinite(d[k])]
        ax.hist(x, bins=bins, density=True, histtype="step", color=COLORS["raw"], lw=1.2)
        ax.set_xlabel(lab); ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(bottom=0)
        NUM[f"test_set_median_{k}"] = float(np.median(x))
    axes[0].set_ylabel("probability density")
    fig.tight_layout(w_pad=1.0)
    savefig(fig, "fig_test_set")


def _shape_noise():
    z = np.load(PER_GALAXY)
    return z, np.sqrt(z["tf_var"][:, 0] + z["tf_var"][:, 1])


def _binned_median(x, y, sig, bx, by):
    ix = np.digitize(x, bx) - 1; jy = np.digitize(y, by) - 1
    med = np.full((len(bx) - 1, len(by) - 1), np.nan); cnt = np.zeros_like(med)
    for a in range(len(bx) - 1):
        for b in range(len(by) - 1):
            m = (ix == a) & (jy == b)
            cnt[a, b] = m.sum()
            if m.sum() >= MIN_PER_BIN:
                med[a, b] = np.median(sig[m])
    return med, cnt


def _colorbar(fig, ax, pc, extend):
    cb = fig.colorbar(pc, ax=ax, pad=0.02, extend=extend)
    cb.set_label(r"median shape noise $\sigma_\epsilon$")


def draw_snr_map(fig, ax):
    z, sig = _shape_noise()
    rho_i, rho_s = z["image_snr"], z["central_halpha_snr"]
    # the same number of log bins on both axes (square pixels); ranges chosen so that every edge row and column has
    # a bin with >= MIN_PER_BIN galaxies (rho_I < 50 holds 0.1% of the galaxies and is left out)
    bi = np.geomspace(RHO_I_RANGE[0], RHO_I_RANGE[1], N_BINS + 1); bs = np.geomspace(RHO_S_RANGE[0], RHO_S_RANGE[1], N_BINS + 1)
    med, cnt = _binned_median(rho_i, rho_s, sig, bi, bs)
    pc = ax.pcolormesh(bi, bs, med.T, cmap="viridis_r", shading="flat", vmin=0.04, vmax=0.08, rasterized=True)
    # contours enclosing 68% and 95% of the galaxies (number density in the same bins, lightly smoothed)
    from scipy.ndimage import gaussian_filter
    h = gaussian_filter(cnt, 0.8); flat = np.sort(h.ravel())[::-1]; cum = np.cumsum(flat) / flat.sum()
    levels = [flat[np.searchsorted(cum, q)] for q in (0.95, 0.68)]
    ci, cs = np.sqrt(bi[1:] * bi[:-1]), np.sqrt(bs[1:] * bs[:-1])
    import matplotlib.patheffects as pe
    cn = ax.contour(ci, cs, h.T, levels=levels, colors="w", linewidths=(0.7, 1.1), linestyles=("--", "-"))
    cn.set_path_effects([pe.withStroke(linewidth=1.8, foreground="k")])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(bi[0], bi[-1]); ax.set_ylim(bs[0], bs[-1]); ax.set_box_aspect(1)
    ax.set_xlabel(r"image SNR $\rho_I$"); ax.set_ylabel(r"H$\alpha$ SNR $\rho_S$")
    _colorbar(fig, ax, pc, "both")
    # numbers quoted in the text: trend along each axis at the median of the other
    lo_s, hi_s = np.percentile(rho_s, [16, 84]); lo_i, hi_i = np.percentile(rho_i, [16, 84])
    NUM.update(
        shape_noise_median=float(np.median(sig)), shape_noise_mean=float(np.mean(sig)),
        shape_noise_eff=float(np.mean(sig ** -2.0) ** -0.5),  # (mean of 1/sigma^2)^(-1/2), as in Xu et al. Sec. IV.D
        rho_s_16_84=[float(lo_s), float(hi_s)], rho_i_16_84=[float(lo_i), float(hi_i)],
        sig_at_low_rho_s=float(np.median(sig[rho_s < lo_s])), sig_at_high_rho_s=float(np.median(sig[rho_s > hi_s])),
        sig_at_low_rho_i=float(np.median(sig[rho_i < lo_i])), sig_at_high_rho_i=float(np.median(sig[rho_i > hi_i])),
        corr_log_rho_i_log_rho_s=float(np.corrcoef(np.log(rho_i), np.log(rho_s))[0, 1]),
        min_per_bin=MIN_PER_BIN, empty_bins=int(np.isnan(med).sum()), n_bins=N_BINS,
        n_outside_range=int(((rho_i < bi[0]) | (rho_i > bi[-1]) | (rho_s < bs[0]) | (rho_s > bs[-1])).sum()))


def draw_cosi_mag_map(fig, ax):
    z, sig = _shape_noise()
    cosi, mag = z["truth"][:, list(z["feature_names"]).index("cosi")], z["rmag_true"]
    bc = np.linspace(*COSI_RANGE, N_BINS + 1); bm = np.linspace(*MAG_RANGE, N_BINS + 1)
    med, _ = _binned_median(cosi, mag, sig, bc, bm)
    # face-on bins go down to ~0.013, well below the floor of the SNR map's colour scale, so this map gets its own
    pc = ax.pcolormesh(bc, bm, med.T, cmap="viridis_r", shading="flat", vmin=0.01, vmax=0.08, rasterized=True)
    ax.set_xlim(bc[0], bc[-1]); ax.set_ylim(bm[0], bm[-1]); ax.set_box_aspect(1)
    ax.set_xlabel(r"$\cos i$"); ax.set_ylabel(r"$m_r$")
    _colorbar(fig, ax, pc, "max")
    NUM.update(cosi_mag_empty_bins=int(np.isnan(med).sum()),
               cosi_mag_n_outside_range=int(((mag < bm[0]) | (mag > bm[-1])).sum()),
               cosi_mag_median_by_cosi_quarter=[float(np.median(sig[(cosi >= q) & (cosi < q + 0.25)])) for q in (0, 0.25, 0.5, 0.75)],
               cosi_mag_median_by_mag=[[float(lo), float(np.median(sig[(mag >= lo) & (mag < lo + 1)]))] for lo in (17, 18, 19)])


def fig_shape_noise_snr():
    fig, ax = plt.subplots(figsize=(COL, 0.86 * COL))
    draw_snr_map(fig, ax)
    savefig(fig, "fig_shape_noise_snr")


def fig_shape_noise_cosi_mag():
    fig, ax = plt.subplots(figsize=(COL, 0.86 * COL))
    draw_cosi_mag_map(fig, ax)
    savefig(fig, "fig_shape_noise_cosi_mag")


def fig_shape_noise_maps():
    """The two maps stacked vertically (paper Fig. shape-noise-maps), each with its own colour bar."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 1.72 * COL))
    draw_snr_map(fig, a)
    draw_cosi_mag_map(fig, b)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.text(0.03, 0.96, tag, transform=ax.transAxes, ha="left", va="top", color="k",
                bbox=dict(facecolor="w", edgecolor="none", pad=1.2, alpha=0.85))
    fig.subplots_adjust(hspace=0.28)
    savefig(fig, "fig_shape_noise_maps")

if __name__ == "__main__":
    apply_style()
    for name in sys.argv[1:] or ["shape_noise_maps", "test_set"]:
        globals()[f"fig_{name}"]()
        print("made", name, flush=True)
    path = OUT / "testset_numbers.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **NUM}, indent=1))
    print(json.dumps(NUM, indent=1))
