#!/usr/bin/env python3
"""Paper test group A (NPE validation): R1, R2a, R3, T6 report.

Reads the NPZ written by ``a_npe_stream.py`` (TF-weighted g1/g2 PIT ranks,
variances, recomputed TF ESS, corner-galaxy draws) plus the model loss CSV,
and writes report.html / report.json / PNGs to paper-tests/A_npe and the
paper-candidate figures to paper-tests/paper_figs (prefix ``A_npe_``).

Important cache limitation: the compact Xu3 test-set cache stores candidate
draws for (g1, g2) only.  The seven nuisance parameters are available only as
TF-weighted (16th, mean, 84th) summaries per galaxy, so their P-P curves are
known exactly at u = 0.16 and 0.84 only, and their 95% coverage is not
available without new inference.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
ARCH = DIAG.parent
for path in (ARCH, DIAG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_losses import load_losses  # noqa: E402
from shear_bias_report import (  # noqa: E402
    compose_shape_noise_regularized_weights,
    coverage_metrics,
    posterior_pp_curve,
    wrap_angle,
)
from likelihood_stack_core import G01_NPE  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
OUT = S / "reports/paper-tests/A_npe"
FIGS = S / "reports/paper-tests/paper_figs"
LOSS_CSV = S / f"models/losses/losses_{G01_NPE}.csv"
NPE = G01_NPE
CASE = None  # MODEL:CACHE_DATASET; None means likelihood_stack_core.G01_CASE

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

LABEL = {
    "g1": r"$g_1$",
    "g2": r"$g_2$",
    "theta_int": r"$\theta_{\rm int}$ [rad]",
    "cosi": r"$\cos i$",
    "v0": r"$v_0$ [km s$^{-1}$]",
    "vcirc": r"$v_{\rm circ}$ [km s$^{-1}$]",
    "rscale": r"$R_{\rm vscale}$ [arcsec]",
    "hlr": r"$R_{\rm h}$ [arcsec]",
    "halpha_flux_true": r"$\log_{10} F_{\rm H\alpha}$ [erg s$^{-1}$ cm$^{-2}$]",
}
SHORT = {
    "g1": r"$g_1$", "g2": r"$g_2$", "theta_int": r"$\theta_{\rm int}$",
    "cosi": r"$\cos i$", "v0": r"$v_0$", "vcirc": r"$v_{\rm circ}$",
    "rscale": r"$R_{\rm vscale}$", "hlr": r"$R_{\rm h}$",
    "halpha_flux_true": r"$\log_{10}F_{\rm H\alpha}$",
}
HTML_NAME = {
    "g1": "g<sub>1</sub>", "g2": "g<sub>2</sub>", "theta_int": "θ<sub>int</sub>",
    "cosi": "cos i", "v0": "v<sub>0</sub>", "vcirc": "v<sub>circ</sub>",
    "rscale": "R<sub>vscale</sub>", "hlr": "R<sub>h</sub>",
    "halpha_flux_true": "log<sub>10</sub> F<sub>Hα</sub>",
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, name: str, paper: str | None = None) -> str:
    path = OUT / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if paper is not None:
        for ext in ("png", "pdf"):
            target = FIGS / f"{paper}.{ext}"
            if target.exists():
                raise SystemExit(f"refusing to overwrite {target}")
            fig.savefig(target, dpi=300, bbox_inches="tight")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def normalized_weights(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - np.max(logw))
    return w / np.sum(w)


def weighted_quantile(values, quantiles, weight):
    order = np.argsort(values, kind="stable")
    values, weight = values[order], weight[order] / np.sum(weight[order])
    coords = np.cumsum(weight) - 0.5 * weight
    return np.interp(quantiles, coords, values)


# --------------------------------------------------------------------- R1
def r1_loss() -> tuple[dict, str]:
    train, valid = load_losses(str(LOSS_CSV))
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(valid))
    result = {
        "loss_csv": str(LOSS_CSV),
        "n_epochs": int(len(train)),
        "best_epoch_1indexed": best + 1,
        "train_at_best": float(train[best]),
        "valid_at_best": float(valid[best]),
        "generalization_gap_valid_minus_train": float(valid[best] - train[best]),
        "final_train": float(train[-1]),
        "final_valid": float(valid[-1]),
        "valid_improvement_last_10_epochs": float(valid[-11] - valid[-1]),
        "valid_improvement_last_50_epochs": float(valid[-51] - valid[-1]),
        "epoch1_train": float(train[0]),
        "epoch1_valid": float(valid[0]),
        "early_stopping_triggered": bool(best + 1 < len(train)),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(epochs, train, color=BLUE, lw=1.8, label="Training")
    ax.plot(epochs, valid, color=ORANGE, lw=1.8, label="Validation")
    ax.scatter([best + 1], [valid[best]], s=36, color=ORANGE, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood per galaxy")
    ax.set_xlim(0, len(train) + 2)
    ax.set_ylim(min(train.min(), valid.min()) - 0.6, 0.0)
    ax.legend(loc="upper right")
    inset = ax.inset_axes([0.42, 0.36, 0.53, 0.38])
    sel = epochs >= 100
    inset.plot(epochs[sel], train[sel], color=BLUE, lw=1.4)
    inset.plot(epochs[sel], valid[sel], color=ORANGE, lw=1.4)
    inset.tick_params(labelsize=7)
    inset.set_title("epochs 100–200", fontsize=7.5, color=INK2)
    fig.tight_layout()
    uri = save(fig, "r1_loss_curve.png", paper="A_npe_loss_curve")
    return result, uri


# --------------------------------------------------------------------- R2a
def pick_typical_width(z, names) -> int:
    """Same cos i / S/N window as the stream picks, plus a typical posterior width.

    Requires sigma_i = sqrt((Var g1 + Var g2)/2) within 10% of the sample median
    and a true |g| < 0.03 so the example is not dominated by the prior edge.
    """

    truth = z["truth"].astype(np.float64)
    cosi = truth[:, names.index("cosi")]
    g = np.hypot(truth[:, 0], truth[:, 1])
    sigma = np.sqrt(0.5 * (z["var_g1"] + z["var_g2"]))
    med = np.median(sigma)
    frac = z["ess_fraction_stored"]
    ok = ((cosi >= 0.45) & (cosi <= 0.65) & (g < 0.03) & (g > 0.005)
          & (np.abs(sigma / med - 1.0) < 0.10) & (frac > np.median(frac)))
    score = np.abs(np.log(z["image_snr"] / 170.0)) + np.abs(np.log(z["central_halpha_snr"] / 25.0))
    score = np.where(ok, score, np.inf)
    return int(np.argmin(score))


def load_draws_from_cache(index: int) -> tuple[np.ndarray, np.ndarray]:
    from cache_contract import load_cache_partitions
    from likelihood_stack_core import CACHE_ROOT, G01_CASE
    G01_CASE = CASE or G01_CASE

    model, dataset = G01_CASE.split(":", 1)
    parts = load_cache_partitions(CACHE_ROOT / model / dataset)
    for k, (start, end) in enumerate(parts.row_ranges):
        if start <= index < end:
            s = np.load(parts.files["shear_sample"][k], mmap_mode="r")
            w = np.load(parts.files["posterior_tf_log_weight"][k], mmap_mode="r")
            t = np.load(parts.files["truth"][k], mmap_mode="r")
            return (np.asarray(s[index - start], dtype=np.float64),
                    np.asarray(w[index - start], dtype=np.float64),
                    np.asarray(t[index - start], dtype=np.float64))
    raise IndexError(index)


def r2a_corner(z, names) -> tuple[dict, list[str]]:
    import corner

    truth = z["truth"]
    summ = z["tf_summary"]
    out = {"galaxies": []}
    uris = []
    typical = pick_typical_width(z, names)
    t_draws, t_logw, t_truth = load_draws_from_cache(typical)
    if not np.allclose(t_truth, truth[typical]):
        raise ValueError("cache truth row does not match stream truth")
    sigma_all = np.sqrt(0.5 * (z["var_g1"] + z["var_g2"]))
    out["median_sigma_i"] = float(np.median(sigma_all))
    picks = [("typical_width", typical, t_draws, t_logw)] + [
        ("snr_window", int(index), z[f"corner{rank}_draws"], z[f"corner{rank}_logw"])
        for rank, index in enumerate(z["corner_index"])
    ]
    for rank, (kind, index, draws, logw) in enumerate(picks):
        w = normalized_weights(logw)
        ess = 1.0 / np.sum(w * w)
        tvec = truth[index].astype(np.float64)
        info = {
            "index": index,
            "image_snr": float(z["image_snr"][index]),
            "central_halpha_snr": float(z["central_halpha_snr"][index]),
            "rmag_true": float(z["rmag_true"][index]),
            "tf_ess": float(ess),
            "tf_ess_fraction": float(ess / len(w)),
            "selection": kind,
            "sigma_i": float(sigma_all[index]),
            "sigma_i_percentile": float(100.0 * np.mean(sigma_all < sigma_all[index])),
            "truth": {},
            "tf_posterior": {},
        }
        for j, name in enumerate(names):
            lo, mean, hi = summ[index, :, j].astype(np.float64)
            t = tvec[j]
            if name == "halpha_flux_true":
                t, lo, mean, hi = np.log10(t), np.log10(lo), np.log10(mean), np.log10(hi)
            info["truth"][name] = float(t)
            info["tf_posterior"][name] = {"q16": float(lo), "mean": float(mean), "q84": float(hi)}
        for j, name in enumerate(("g1", "g2")):
            q = weighted_quantile(draws[:, j], np.array([0.025, 0.16, 0.5, 0.84, 0.975]), w)
            info["tf_posterior"][name].update(
                {"q2.5": float(q[0]), "median": float(q[2]), "q97.5": float(q[4]),
                 "sd": float(np.sqrt(np.sum(w * (draws[:, j] - np.sum(w * draws[:, j])) ** 2)))}
            )
        out["galaxies"].append(info)

        fig = plt.figure(figsize=(10.5, 4.6))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.25, 1.6], wspace=0.08, hspace=0.08)
        axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                         [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
        span = [
            (min(draws[:, k].min(), tvec[k]) - 0.003, max(draws[:, k].max(), tvec[k]) + 0.003)
            for k in range(2)
        ]
        span = [weighted_quantile(draws[:, k], np.array([0.001, 0.999]), w) for k in range(2)]
        span = [(min(s[0], tvec[k]) - 0.004, max(s[1], tvec[k]) + 0.004) for k, s in enumerate(span)]
        corner.corner(
            draws, weights=w, fig=fig, labels=[LABEL["g1"], LABEL["g2"]],
            truths=tvec[:2], truth_color=ORANGE, color=BLUE, bins=40,
            smooth=1.0, levels=(0.393, 0.865), range=span,
            plot_datapoints=False, fill_contours=True, max_n_ticks=4,
            hist_kwargs={"lw": 1.6},
        )
        for ax in fig.axes[:4]:
            ax.grid(False)
        # nuisance forest panel: (x - truth) / half-width of 16-84 interval
        axf = fig.add_subplot(gs[:, 3])
        for k, name in enumerate(names):
            post = info["tf_posterior"][name]
            t = info["truth"][name]
            half = 0.5 * (post["q84"] - post["q16"])
            if name == "theta_int":
                d_lo = wrap_angle(np.array(post["q16"] - post["mean"]))
                d_hi = wrap_angle(np.array(post["q84"] - post["mean"]))
                half = 0.5 * float(d_hi - d_lo)
                off_mean = float(wrap_angle(np.array(post["mean"] - t)))
                lo_n, hi_n = (off_mean + d_lo) / half, (off_mean + d_hi) / half
            else:
                off_mean = post["mean"] - t
                lo_n, hi_n = (post["q16"] - t) / half, (post["q84"] - t) / half
            y = len(names) - 1 - k
            axf.plot([lo_n, hi_n], [y, y], color=BLUE, lw=2.2, solid_capstyle="round")
            axf.scatter([off_mean / half], [y], color=BLUE, s=30, zorder=4,
                        edgecolor="white", linewidth=1.2)
        axf.axvline(0.0, color=ORANGE, lw=1.4)
        axf.set_yticks(range(len(names)))
        axf.set_yticklabels([SHORT[n] for n in names[::-1]])
        axf.yaxis.tick_right()
        axf.set_xlim(-3.5, 3.5)
        axf.set_xlabel(r"(posterior $-$ truth) / $\sigma_{68}$")
        axf.set_title("TF-weighted mean and 16–84% interval", fontsize=9, color=INK2)
        fig.suptitle(
            f"Xu3 galaxy {index}: cos i = {tvec[3]:.2f}, "
            f"image S/N = {info['image_snr']:.0f}, central Hα S/N = {info['central_halpha_snr']:.1f}",
            fontsize=10, y=0.99,
        )
        paper = f"A_npe_corner_g1g2_gal{index}" if rank == 0 else None
        uris.append(save(fig, f"r2a_corner_gal{index}.png", paper=paper))
    return out, uris


# --------------------------------------------------------------------- R3
def tail_fractions(truth, summary, name, weight):
    lo, mid, hi = summary[:, 0], summary[:, 1], summary[:, 2]
    t = truth
    if name == "theta_int":
        r_t = wrap_angle(t - mid)
        below = r_t < wrap_angle(lo - mid)
        above = r_t > wrap_angle(hi - mid)
    else:
        below = t < lo
        above = t > hi
    w = weight / np.sum(weight)
    return float(np.sum(w * below)), float(np.sum(w * above))


def pit_coverage(pit, weight, level):
    a = 0.5 * (1.0 - level)
    inside = (pit >= a) & (pit <= 1.0 - a)
    w = weight / np.sum(weight)
    cov = float(np.sum(w * inside))
    ess = 1.0 / np.sum(w * w)
    return cov, math.sqrt(cov * (1 - cov) / ess)


def r3_calibration(z, names) -> tuple[dict, list[str]]:
    truth = z["truth"].astype(np.float64)
    summ = z["tf_summary"].astype(np.float64)
    n = len(truth)
    equal = np.full(n, 1.0 / n)
    reg_w, reg_diag = compose_shape_noise_regularized_weights(
        equal, z["var_g1"], z["var_g2"]
    )
    weights = {"equal": equal, "shape_noise_regularized": reg_w}
    pits = {"g1": z["pit_g1"], "g2": z["pit_g2"]}
    result = {"n_galaxies": n, "regularized_weight_diagnostics": {
        k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
        for k, v in reg_diag.items()}, "rows": {}}
    for wname, w in weights.items():
        cov_rows = {r["parameter"]: r for r in coverage_metrics(truth, summ, w, names, wname)}
        rows = []
        for j, name in enumerate(names):
            below, above = tail_fractions(truth[:, j], summ[:, :, j], name, w)
            row = {
                "parameter": name,
                "cov68_summary": cov_rows[name]["coverage"],
                "cov68_summary_se": cov_rows[name]["coverage_se"],
                "frac_below_q16": below,
                "frac_above_q84": above,
                "galaxy_ess": cov_rows[name]["ess"],
            }
            if name in pits:
                for level in (0.5, 0.68, 0.9, 0.95, 0.99):
                    c, se = pit_coverage(pits[name], w, level)
                    row[f"cov{int(round(level * 100))}_pit"] = c
                    row[f"cov{int(round(level * 100))}_pit_se"] = se
                x, y, ks, ess = posterior_pp_curve(pits[name], w)
                row["ks_distance"] = ks
                row["ks_dkw95_band"] = math.sqrt(math.log(2 / 0.05) / (2 * ess))
                row["pit_mean"] = float(np.sum(w * pits[name]) / np.sum(w))
            rows.append(row)
        result["rows"][wname] = rows

    # --- P-P figure (equal weights = raw full sample)
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.2), sharex=True, sharey=True)
    grid = np.linspace(0, 1, 201)
    eps = math.sqrt(math.log(2 / 0.05) / (2 * n))
    for k, name in enumerate(names):
        ax = axes.flat[k]
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.9, ls="--")
        if name in pits:
            x, y, ks, _ = posterior_pp_curve(pits[name], equal)
            step = max(1, len(x) // 4000)
            ax.plot(np.r_[0, x[::step], 1], np.r_[0, y[::step], 1], color=BLUE, lw=1.8)
            ax.text(0.04, 0.9, f"KS = {ks:.4f}", transform=ax.transAxes, fontsize=8.5, color=INK2)
        else:
            row = result["rows"]["equal"][k]
            px = [0.0, 0.16, 0.84, 1.0]
            py = [0.0, row["frac_below_q16"], 1.0 - row["frac_above_q84"], 1.0]
            ax.plot(px, py, color=BLUE, lw=1.0, ls=":")
            ax.scatter(px[1:3], py[1:3], s=34, color=BLUE, zorder=4, edgecolor="white", linewidth=1.2)
            ax.text(0.04, 0.9, "16/84% only", transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_title(SHORT[name], fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("posterior quantile $u$")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of truths below $u$")
    fig.tight_layout()
    uri_pp = save(fig, "r3_pp_all9.png", paper="A_npe_pp_all9")

    # --- deviation figure for g1/g2 (what the P-P curve hides at N = 1e5)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name, color in (("g1", BLUE), ("g2", ORANGE)):
        x, y, ks, _ = posterior_pp_curve(pits[name], equal)
        step = max(1, len(x) // 4000)
        ax.plot(x[::step], (y - x)[::step], color=color, lw=1.6, label=f"{SHORT[name]} (KS = {ks:.4f})")
    ax.axhspan(-eps, eps, color=GRID, alpha=0.6, lw=0, label="DKW 95% band (no miscalibration)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("posterior quantile $u$")
    ax.set_ylabel(r"ECDF$(u) - u$")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    uri_dev = save(fig, "r3_pp_deviation_g1g2.png", paper="A_npe_pp_deviation_g1g2")

    # --- PIT histograms g1/g2
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, name in zip(axes, ("g1", "g2")):
        ax.hist(pits[name], bins=50, range=(0, 1), density=True, color=BLUE, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.axhline(1, color=INK2, ls="--", lw=0.9)
        ax.set_xlabel(f"PIT of {SHORT[name]}")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    uri_hist = save(fig, "r3_pit_hist_g1g2.png")
    return result, [uri_pp, uri_dev, uri_hist]


# --------------------------------------------------------------------- T6
def binned_quantiles(x, y, edges):
    centers, med, q16, q84, counts = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if np.count_nonzero(sel) < 20:
            continue
        centers.append(0.5 * (a + b))
        q = np.quantile(y[sel], [0.16, 0.5, 0.84])
        q16.append(q[0]); med.append(q[1]); q84.append(q[2])
        counts.append(int(np.count_nonzero(sel)))
    return [np.asarray(v) for v in (centers, med, q16, q84)] + [counts]


def t6_ess(z, names) -> tuple[dict, list[str]]:
    ess = z["ess_stored"].astype(np.float64)
    frac = z["ess_fraction_stored"].astype(np.float64)
    n_draw = int(round(np.median(ess / frac)))
    rec = z["ess_recomputed"]
    truth = z["truth"].astype(np.float64)
    cosi = truth[:, names.index("cosi")]
    vcirc = truth[:, names.index("vcirc")]
    rmag = z["rmag_true"].astype(np.float64)
    ha = z["central_halpha_snr"].astype(np.float64)
    img = z["image_snr"].astype(np.float64)
    maxw = z["max_weight_stored"].astype(np.float64)
    q = np.quantile(frac, [0.001, 0.01, 0.05, 0.16, 0.5, 0.84, 0.95])
    low = ess < 50
    result = {
        "n_galaxies": int(len(ess)),
        "draws_per_galaxy": n_draw,
        "stored_vs_recomputed_max_rel_diff": float(np.max(np.abs(rec - ess) / ess)),
        "ess_fraction_quantiles": {k: float(v) for k, v in zip(
            ["q0.1%", "q1%", "q5%", "q16%", "median", "q84%", "q95%"], q)},
        "ess_fraction_mean": float(np.mean(frac)),
        "ess_min": float(ess.min()),
        "ess_min_index": int(np.argmin(ess)),
        "count_ess_lt_50": int(np.count_nonzero(ess < 50)),
        "count_ess_lt_100": int(np.count_nonzero(ess < 100)),
        "count_ess_lt_500": int(np.count_nonzero(ess < 500)),
        "count_ess_lt_1000": int(np.count_nonzero(ess < 1000)),
        "max_weight_quantiles": {k: float(v) for k, v in zip(
            ["median", "q99%", "max"], np.quantile(maxw, [0.5, 0.99, 1.0]))},
        "count_max_weight_gt_0.1": int(np.count_nonzero(maxw > 0.1)),
        "spearman": {},
    }
    from scipy.stats import spearmanr
    for key, x in (("cosi", cosi), ("rmag_true", rmag), ("central_halpha_snr", ha),
                   ("image_snr", img), ("vcirc_true", vcirc)):
        result["spearman"][key] = float(spearmanr(x, frac).statistic)
    low_rows = []
    for i in np.flatnonzero(low)[np.argsort(ess[low])][:30]:
        low_rows.append({
            "index": int(i), "ess": float(ess[i]), "cosi": float(cosi[i]),
            "rmag_true": float(rmag[i]), "central_halpha_snr": float(ha[i]),
            "image_snr": float(img[i]), "vcirc_true": float(vcirc[i]),
            "max_weight": float(maxw[i]),
            "pit_g1": float(z["pit_g1"][i]), "pit_g2": float(z["pit_g2"][i]),
        })
    result["lowest_ess_galaxies"] = low_rows

    # shear calibration of low-ESS subsets (equal weights)
    subsets = {}
    for label, sel in (("all", np.ones_like(low)), ("ess<50", low),
                       ("ess<500", ess < 500), ("ess_frac<0.1", frac < 0.1),
                       ("ess_frac>=0.5", frac >= 0.5)):
        count = int(np.count_nonzero(sel))
        entry = {"n": count}
        if count >= 5:
            for comp in ("g1", "g2"):
                pit = z[f"pit_{comp}"][sel]
                entry[f"{comp}_cov68_pit"] = float(np.mean((pit >= 0.16) & (pit <= 0.84)))
                entry[f"{comp}_cov95_pit"] = float(np.mean((pit >= 0.025) & (pit <= 0.975)))
            entry["median_cosi"] = float(np.median(cosi[sel]))
        subsets[label] = entry
    result["subset_shear_coverage"] = subsets

    uris = []
    # distribution
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].hist(frac, bins=np.linspace(0, 1, 51), color=BLUE, edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel(r"TF importance ESS / $N_{\rm draw}$")
    axes[0].set_ylabel("galaxies")
    axes[1].hist(np.log10(ess), bins=60, color=BLUE, edgecolor="white", linewidth=0.6)
    axes[1].axvline(np.log10(50), color=RED, lw=1.4)
    axes[1].text(np.log10(50) + 0.03, axes[1].get_ylim()[1] * 0.85,
                 f"ESS = 50\n({result['count_ess_lt_50']} galaxies below)", fontsize=8, color=INK)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\log_{10}$ ESS")
    fig.tight_layout()
    uris.append(save(fig, "t6_ess_distribution.png"))

    # dependence
    panels = (
        ("cosi", cosi, np.linspace(0, 1, 21), r"true $\cos i$"),
        ("rmag_true", rmag, np.linspace(np.quantile(rmag, 0.001), rmag.max() + 1e-6, 21), r"$m_r$ (true)"),
        ("central_halpha_snr", ha, np.linspace(0, 150, 26), r"central H$\alpha$ S/N"),
        ("image_snr", img, np.geomspace(10, 1000, 21), "image S/N"),
        ("vcirc_true", vcirc, np.linspace(np.quantile(vcirc, 0.001), np.quantile(vcirc, 0.999), 21), r"true $v_{\rm circ}$ [km s$^{-1}$]"),
    )
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    for ax, (key, x, edges, xl) in zip(axes, panels):
        ax.hexbin(x, frac, gridsize=45, bins="log", cmap="Blues", mincnt=1,
                  xscale="log" if key == "image_snr" else "linear", linewidths=0)
        c, med, lo, hi, _ = binned_quantiles(x, frac, edges)
        ax.plot(c, med, color=ORANGE, lw=2)
        ax.plot(c, lo, color=ORANGE, lw=1, ls="--")
        ax.plot(c, hi, color=ORANGE, lw=1, ls="--")
        ax.scatter(x[low], frac[low], s=14, color=RED, zorder=5, label="ESS < 50")
        ax.set_xlabel(xl)
        ax.set_title(f"Spearman ρ = {result['spearman'][key]:+.2f}", fontsize=9, color=INK2)
        if key == "image_snr":
            ax.set_xscale("log")
    axes[0].set_ylabel("ESS fraction")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    uris.append(save(fig, "t6_ess_dependence.png"))
    return result, uris


# --------------------------------------------------------------------- HTML
STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;"
    "padding:0 1rem;color:#1b1b1b}section{margin:3rem 0;border-top:2px solid #ddd}"
    "table{border-collapse:collapse;width:100%;font-size:.88rem;margin:1rem 0}"
    "th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}"
    "th:first-child,td:first-child{text-align:left}th{background:#f3f3f3}"
    "img{display:block;max-width:100%;height:auto;margin:1rem auto}"
    "code{background:#f4f4f4;padding:.1rem .25rem}.warn{background:#fff4e0;"
    "border-left:4px solid #eda100;padding:.5rem .8rem}.ok{background:#eaf6ee;"
    "border-left:4px solid #0ca30c;padding:.5rem .8rem}"
)


def pct(v, se=None):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{100 * v:.2f}%" + (f" ± {100 * se:.2f}%" if se is not None else "")


def table(header, rows):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main(argv=None):
    global OUT, FIGS, NPE, CASE, LOSS_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=None)
    parser.add_argument("--case", default=None, help="MODEL:CACHE_DATASET (default: G01_CASE)")
    parser.add_argument("--out-dir", type=Path, default=None, help="testing override")
    parser.add_argument("--figs-dir", type=Path, default=None, help="testing override")
    args = parser.parse_args(argv)
    if args.case is not None:
        CASE = args.case
        NPE = CASE.split(":", 1)[0]
        LOSS_CSV = S / f"models/losses/losses_{NPE}.csv"
    if args.out_dir is not None:
        OUT = args.out_dir
    if args.figs_dir is not None:
        FIGS = args.figs_dir
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    z = np.load(args.stream if args.stream is not None else OUT / "a_npe_stream.npz")
    names = [str(n) for n in z["feature_names"]]
    meta = json.loads(str(z["meta_json"]))

    r1, uri_loss = r1_loss()
    r2, uri_corner = r2a_corner(z, names)
    r3, uri_r3 = r3_calibration(z, names)
    t6, uri_t6 = t6_ess(z, names)

    ref = {"g1": 0.6738, "g2": 0.6781}
    reg_rows = {r["parameter"]: r for r in r3["rows"]["shape_noise_regularized"]}
    eq_rows = {r["parameter"]: r for r in r3["rows"]["equal"]}
    validation = {
        c: {"reference": ref[c], "reproduced_regularized": reg_rows[c]["cov68_summary"],
            "equal_weight_summary": eq_rows[c]["cov68_summary"],
            "equal_weight_pit": eq_rows[c]["cov68_pit"]}
        for c in ("g1", "g2")
    }
    validation["pass"] = bool(all(
        abs(validation[c]["reproduced_regularized"] - ref[c]) < 5e-4 for c in ("g1", "g2")))
    report = {"meta": meta, "model": NPE, "R1": r1, "R2a": r2, "R3": r3, "T6": t6,
              "validation_vs_fiber_gauge_03": validation}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # --- HTML
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>A_npe NPE validation</title>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{STYLE}</style></head><body>"]
    parts.append(
        "<h1>Paper tests A_npe: NPE validation (R1, R2a, R3, T6)</h1>"
        f"<p>Model <code>{NPE}</code>; Xu3 test-set cache "
        f"<code>{html.escape(meta['case'].split(':')[1])}</code>; N = {meta['analyzed_size']:,} galaxies, "
        f"{t6['draws_per_galaxy']:,} identity+R90 draws per galaxy; all posterior summaries use the "
        "TF importance weights. No new inference.</p>"
        "<p class='warn'><b>Cache limitation.</b> The compact test-set cache stores candidate draws for "
        "(g<sub>1</sub>, g<sub>2</sub>) only (<code>stored_candidate_parameters = [g1, g2]</code>). "
        "The seven nuisances exist only as TF-weighted (16th, mean, 84th) summaries per galaxy. Consequences: "
        "(i) the R2a corner plot is a g<sub>1</sub>–g<sub>2</sub> corner plus a 9-parameter interval panel, "
        "not a full 9D corner; (ii) nuisance P–P curves are exact only at u = 0.16 and 0.84; "
        "(iii) nuisance 95% coverage is unavailable. A full 9D corner and 9D P–P need a short "
        "re-sampling run (GPU, seconds per galaxy) that stores all 9 columns.</p>"
    )
    # R1
    parts.append(
        "<section><h2>R1 · Loss curve</h2>"
        + table(["quantity", "value"], [
            ["epochs trained", r1["n_epochs"]],
            ["best (lowest-validation) epoch, 1-indexed", r1["best_epoch_1indexed"]],
            ["train NLL at best", f"{r1['train_at_best']:.3f}"],
            ["validation NLL at best", f"{r1['valid_at_best']:.3f}"],
            ["validation − train", f"{r1['generalization_gap_valid_minus_train']:+.3f}"],
            ["validation gain over last 10 / 50 epochs",
             f"{r1['valid_improvement_last_10_epochs']:.4f} / {r1['valid_improvement_last_50_epochs']:.3f}"],
            ["early stopping triggered (patience 20)", r1["early_stopping_triggered"]],
        ])
        + f"<img src='{uri_loss}' alt='loss curve'>"
        "<p>Losses are the per-galaxy flow NLL as logged by training (normalized target coordinates). "
        f"Epoch 1 train NLL ({r1['epoch1_train']:.2f}) lies above the plotted range. The best epoch is the final "
        "one (cosine schedule decays to the end); validation was still improving slightly.</p></section>"
    )
    # R2a
    rows = []
    for g in r2["galaxies"]:
        for name in names:
            p = g["tf_posterior"][name]
            rows.append([f"{g['index']} · {HTML_NAME[name]}", f"{g['truth'][name]:.4g}",
                         f"{p['mean']:.4g}", f"{p['q16']:.4g}", f"{p['q84']:.4g}",
                         f"{p.get('sd', float('nan')):.4g}" if "sd" in p else "—"])
    parts.append(
        "<section><h2>R2a · Example posteriors</h2>"
        "<p>All picks are ranked by |ln(ρ<sub>I</sub>/170)| + |ln(ρ<sub>S</sub>/25)| and require a TF "
        "ESS fraction above the median. <b>typical_width</b> (paper figure): true cos i in [0.45, 0.65], "
        "0.005 &lt; |g| &lt; 0.03, and σ<sub>i</sub> = √((Var g<sub>1</sub> + Var g<sub>2</sub>)/2) within 10% "
        f"of the sample median ({r2['median_sigma_i']:.4f}). <b>snr_window</b> (made by the stream pass): "
        "true cos i in [0.4, 0.7], 0.01 &lt; |g| &lt; 0.06, R<sub>h</sub> in its interquartile range; the "
        "second of these differs in cos i by &gt; 0.12. The first snr_window galaxy has a g<sub>2</sub> "
        "posterior piled against the +0.1 prior edge, which is why a width-controlled pick was added.</p>"
        + table(["galaxy", "selection", "image S/N", "central Hα S/N", "m<sub>r</sub>", "TF ESS (fraction)",
                 "σ<sub>i</sub> (percentile)"],
                [[g["index"], g["selection"], f"{g['image_snr']:.1f}", f"{g['central_halpha_snr']:.2f}",
                  f"{g['rmag_true']:.2f}", f"{g['tf_ess']:.0f} ({g['tf_ess_fraction']:.3f})",
                  f"{g['sigma_i']:.4f} ({g['sigma_i_percentile']:.0f}%)"]
                 for g in r2["galaxies"]])
        + "".join(f"<img src='{u}' alt='corner'>" for u in uri_corner)
        + "<p>Contours: 39.3% and 86.5% (1σ, 2σ in 2D) TF-weighted credible regions; orange lines: truth. "
        "Right panel: offset of the TF-weighted mean and 16–84% interval from truth, in units of the "
        "interval half-width (θ<sub>int</sub> handled circularly; F<sub>Hα</sub> in log10).</p>"
        + table(["galaxy · parameter", "truth", "TF mean", "q16", "q84", "sd (g only)"], rows)
        + "</section>"
    )
    # R3
    def r3_table(wname):
        out = []
        for r in r3["rows"][wname]:
            out.append([HTML_NAME[r["parameter"]],
                        pct(r.get("cov68_pit")) if "cov68_pit" in r else "—",
                        pct(r["cov68_summary"], r["cov68_summary_se"]),
                        pct(r.get("cov95_pit"), r.get("cov95_pit_se")) if "cov95_pit" in r else "n/a",
                        pct(r["frac_below_q16"]), pct(r["frac_above_q84"]),
                        f"{r['ks_distance']:.4f} (DKW {r['ks_dkw95_band']:.4f})" if "ks_distance" in r else "—"])
        return table(["parameter", "68% (PIT)", "68% (stored 16–84 interval)", "95% (PIT)",
                      "truth &lt; q16 (target 16%)", "truth &gt; q84 (target 16%)", "KS (95% band)"], out)
    parts.append(
        "<section><h2>R3 · Calibration (NPE only, TF-weighted ranks)</h2>"
        f"<p>Full Xu3 set, N = {r3['n_galaxies']:,}. PIT = TF-weighted fraction of posterior draws below "
        "the truth (midpoint ties), from <code>load_shear_posterior_diagnostics</code>. For θ<sub>int</sub> "
        "the stored interval is circular (centered on the circular mean), and truth is compared on the wrapped "
        "residual.</p>"
        "<h3>Equal galaxy weights (raw full sample) — paper table</h3>" + r3_table("equal")
        + "<h3>Shape-noise-regularized galaxy weights (reference-report convention)</h3>"
        + r3_table("shape_noise_regularized")
        + f"<p class='{'ok' if validation['pass'] else 'warn'}'><b>Validation.</b> With the "
        "fiber-gauge/03 convention (stored interpolated 16–84% interval, shape-noise-regularized weights, "
        f"ESS {r3['regularized_weight_diagnostics']['ess']:.1f}) g<sub>1</sub>/g<sub>2</sub> coverage is "
        f"{pct(validation['g1']['reproduced_regularized'])} / {pct(validation['g2']['reproduced_regularized'])} "
        "vs reference 67.38% / 67.81%. Equal-weight values: "
        f"{pct(validation['g1']['equal_weight_summary'])} / {pct(validation['g2']['equal_weight_summary'])} "
        f"(interval) and {pct(validation['g1']['equal_weight_pit'])} / {pct(validation['g2']['equal_weight_pit'])} (PIT).</p>"
        + "".join(f"<img src='{u}' alt='P-P'>" for u in uri_r3)
        + "<p>P–P panels use equal galaxy weights. Nuisance panels show the two exactly known points "
        "(u = 0.16, 0.84) from the stored summaries; the dotted line only connects them.</p></section>"
    )
    # T6
    sub_rows = [[k, v["n"], pct(v.get("g1_cov68_pit")), pct(v.get("g2_cov68_pit")),
                 pct(v.get("g1_cov95_pit")), pct(v.get("g2_cov95_pit")),
                 f"{v.get('median_cosi', float('nan')):.3f}"] for k, v in t6["subset_shear_coverage"].items()]
    low_rows = [[r["index"], f"{r['ess']:.1f}", f"{r['cosi']:.3f}", f"{r['rmag_true']:.2f}",
                 f"{r['central_halpha_snr']:.1f}", f"{r['image_snr']:.0f}", f"{r['vcirc_true']:.0f}",
                 f"{r['max_weight']:.3f}", f"{r['pit_g1']:.3f}", f"{r['pit_g2']:.3f}"]
                for r in t6["lowest_ess_galaxies"]]
    qf = t6["ess_fraction_quantiles"]
    parts.append(
        "<section><h2>T6 · TF importance-weight health</h2>"
        + table(["quantity", "value"], [
            ["draws per galaxy", f"{t6['draws_per_galaxy']:,}"],
            ["stored vs recomputed ESS, max relative difference", f"{t6['stored_vs_recomputed_max_rel_diff']:.1e}"],
            ["ESS fraction: mean / median", f"{t6['ess_fraction_mean']:.3f} / {qf['median']:.3f}"],
            ["ESS fraction quantiles 0.1% / 1% / 5% / 16% / 84% / 95%",
             " / ".join(f"{qf[k]:.4f}" for k in ["q0.1%", "q1%", "q5%", "q16%", "q84%", "q95%"])],
            ["min ESS (galaxy)", f"{t6['ess_min']:.1f} ({t6['ess_min_index']})"],
            ["galaxies with ESS &lt; 50 / 100 / 500 / 1000",
             f"{t6['count_ess_lt_50']} / {t6['count_ess_lt_100']} / {t6['count_ess_lt_500']} / {t6['count_ess_lt_1000']}"],
            ["max normalized weight: median / 99% / max",
             " / ".join(f"{v:.4f}" for v in t6["max_weight_quantiles"].values())],
            ["galaxies with max weight &gt; 0.1", t6["count_max_weight_gt_0.1"]],
            ["Spearman ρ(ESS fraction, ·)",
             ", ".join(f"{k} {v:+.3f}" for k, v in t6["spearman"].items())],
        ])
        + "".join(f"<img src='{u}' alt='ESS'>" for u in uri_t6)
        + "<p>Orange: binned median and 16/84% of the ESS fraction; red: ESS &lt; 50.</p>"
        "<h3>Shear coverage of low-ESS subsets (equal weights, PIT)</h3>"
        + table(["subset", "N", "g<sub>1</sub> 68%", "g<sub>2</sub> 68%", "g<sub>1</sub> 95%",
                 "g<sub>2</sub> 95%", "median cos i"], sub_rows)
        + "<h3>Lowest-ESS galaxies (up to 30 with ESS &lt; 50)</h3>"
        + table(["index", "ESS", "cos i", "m<sub>r</sub>", "Hα S/N", "image S/N", "v<sub>circ</sub>",
                 "max w", "PIT g<sub>1</sub>", "PIT g<sub>2</sub>"], low_rows)
        + "</section>"
    )
    parts.append(
        "<section><h2>Provenance</h2><p>Scripts: <code>kl-nn/arch/diagnostics/paper_tests/a_npe_stream.py</code> "
        "(SLURM, streaming pass) and <code>a_npe_report.py</code> (this report). Reused: "
        "<code>shear_bias_report.load_case</code>, <code>load_shear_posterior_diagnostics</code>, "
        "<code>coverage_metrics</code>, <code>compose_shape_noise_regularized_weights</code>, "
        "<code>posterior_pp_curve</code>; <code>plot_losses.load_losses</code>. "
        f"Stream pass took {meta['seconds']:.0f} s.</p></section></body></html>"
    )
    target = OUT / "report.html"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    target.write_text("".join(parts))
    print(json.dumps({"R1": r1, "validation": validation}, indent=1, default=float))


if __name__ == "__main__":
    main()
