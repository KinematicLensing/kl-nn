#!/usr/bin/env python3
"""Paper test T8 (Group C): report for NPE likelihood stacking at constant shear.

Reads the analysis npz written by ``c_stack_analysis.py`` (plus the per-cell
sample files for TF diagnostics) and writes report.html / report.json / PNGs
to the paper-tests C_stacking folder, and paper-candidate figures (prefix C_)
to paper-tests/paper_figs.
"""

from __future__ import annotations

from argparse import ArgumentParser
import base64
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ARCH_DIR = Path(__file__).resolve().parents[2]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))
from diagnostics.likelihood_stack_benchmark_core import fit_component_mc, rmse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
CACHE = S / "cache" / "paper_tests_C_stacking" / MODEL
REPORT = S / "reports" / "paper-tests" / "C_stacking"
FIGS = S / "reports" / "paper-tests" / "paper_figs"
BENCH_CSV = S / "samples" / "likelihood_stack_04_benchmark.csv"
XU3_CSV = S / "samples" / "test_100k_simv3_cosi_xu3_tf.csv"
N_LIST = (1, 2, 4, 8, 16, 32)
COL = {"mean": 0, "map": 4}
C1, C2, C3, C4, C5, C6, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                  "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"


def parse_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, default=CACHE / "analysis_v1.npz")
    p.add_argument("--samples", type=Path, default=CACHE / "likelihood_stack_04_benchmark_r90_8192")
    p.add_argument("--report-dir", type=Path, default=REPORT)
    p.add_argument("--figs-dir", type=Path, default=FIGS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default=MODEL, help="label only (report text and json)")
    p.add_argument("--dataset-name", default="likelihood_stack_04_benchmark", help="label only")
    return p.parse_args(argv)


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 150, "figure.dpi": 150,
    })


def fit(truth, est):
    f = fit_component_mc(truth, est)
    f.update(rmse(truth, est))
    return f


KEYS = ("g1_m", "g2_m", "g1_c", "g2_c", "combined_m", "combined_rmse")


def fit_rows(truth, rows):
    """rows (K, 25, 2): row 0 deterministic, rows 1.. bootstrap."""
    fits = [fit(truth, rows[k]) for k in range(rows.shape[0])]
    out = {"det": {k: fits[0][k] for k in KEYS}}
    if len(fits) > 1:
        bs = {k: np.array([f[k] for f in fits[1:]]) for k in KEYS}
        out["boot_median"] = {k: float(np.median(v)) for k, v in bs.items()}
        out["boot_p16"] = {k: float(np.percentile(v, 16)) for k, v in bs.items()}
        out["boot_p84"] = {k: float(np.percentile(v, 84)) for k, v in bs.items()}
        out["boot_std"] = {k: float(np.std(v)) for k, v in bs.items()}
    return out


def pergal_rows(mu, counts, rr):
    """mu (25, G, R, 2); counts (K, G) -> (K, 25, 2) averaged over first rr reals."""
    m = mu[:, :, :rr].mean(axis=2)  # (25, G, 2)
    return np.einsum("kg,cgd->kcd", counts, m) / counts.sum(1)[:, None, None]


def stack_rows(arr, setting_index, col):
    """arr (25, n_setting, K, 6) -> (K, 25, 2)."""
    return np.transpose(arr[:, setting_index, :, col:col + 2], (1, 0, 2))


def img_tag(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def main(argv=None):
    args = parse_args(argv)
    MODEL = args.model
    rd, fd = args.report_dir, args.figs_dir
    rd.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    if (rd / "report.html").exists() and not args.overwrite:
        raise FileExistsError(rd / "report.html")
    style()
    with np.load(args.analysis, allow_pickle=False) as _z:
        A = {k: _z[k] for k in _z.files}
    meta = json.loads(str(A["meta_json"]))
    truth = A["truth"]
    counts = {n: A[f"counts_N{n}"] for n in N_LIST}
    n_real = A["obs_train_mu"].shape[2]
    boot = counts[32].shape[0] - 1

    # ------------------------------------------------------------------ estimators
    est = {}  # name -> setting -> rows (K,25,2)
    labels = {}
    for w in ("train", "tf"):
        mu = A[f"obs_{w}_mu"]
        name = f"pergal_mean_{w}"
        labels[name] = f"per-galaxy posterior mean ({'TF-weighted' if w == 'tf' else 'unweighted'})"
        est[name] = {
            "Rcurve": [pergal_rows(mu, counts[32], rr) for rr in range(1, n_real + 1)],
            "N16": [pergal_rows(mu, counts[n], n_real) for n in N_LIST],
            "N1": [pergal_rows(mu, counts[n], 1) for n in N_LIST],
        }
        for cname in ("kde", "kde_bw0.5", "kde_bw2", "kde_a1e-4", "kde_a1e-2", "kde_a0", "gauss", "lapkde"):
            for stat in ("mean", "map"):
                key = f"{w}/{cname}/Rcurve"
                if key not in A:
                    continue
                name = f"stack_{cname}_{stat}_{w}"
                labels[name] = f"stacked {cname} posterior {stat} ({'TF' if w == 'tf' else 'train'} nuisance prior)"
                est[name] = {
                    "Rcurve": [stack_rows(A[f"{w}/{cname}/Rcurve"], i, COL[stat]) for i in range(n_real)],
                    "N16": [stack_rows(A[f"{w}/{cname}/N16"], i, COL[stat]) for i in range(len(N_LIST))],
                    "N1": [stack_rows(A[f"{w}/{cname}/N1"], i, COL[stat]) for i in range(len(N_LIST))],
                }
        for tag in ("m401", "m800"):
            key = f"{w}/kde_{tag}/Rcurve"
            if key in A:
                name = f"stack_kde_{tag}_mean_{w}"
                labels[name] = f"stacked kde mean on {tag[1:]} lattice ({w}) [grid check, det only]"
                est[name] = {
                    "Rcurve": [stack_rows(A[key], i, 0) for i in range(n_real)],
                    "N16": [stack_rows(A[f"{w}/kde_{tag}/N16"], i, 0) for i in range(len(N_LIST))],
                    "N1": [stack_rows(A[f"{w}/kde_{tag}/N1"], i, 0) for i in range(len(N_LIST))],
                }
    metrics = {}
    for name, sets in est.items():
        metrics[name] = {
            "vs_R_at_N32": {str(rr + 1): fit_rows(truth, rows) for rr, rows in enumerate(sets["Rcurve"])},
            f"vs_N_at_R{n_real}": {str(n): fit_rows(truth, rows) for n, rows in zip(N_LIST, sets["N16"])},
            "vs_N_at_R1": {str(n): fit_rows(truth, rows) for n, rows in zip(N_LIST, sets["N1"])},
        }
    RN = f"vs_N_at_R{n_real}"

    def get(name, curve, key, which="det"):
        return metrics[name][curve][key][which]

    # ------------------------------------------------------------------ grid checks
    checks = {}
    for w in ("train", "tf"):
        base = A[f"{w}/kde/Rcurve"][:, :, 0, :]  # (25, R, 6) det
        for tag in ("m401", "m800"):
            k = f"{w}/kde_{tag}/Rcurve"
            if k in A:
                d = A[k][:, :, 0, :2] - base[:, :, :2]
                checks[f"{w}_{tag}_minus_m400_mean_maxabs"] = float(np.max(np.abs(d)))
                checks[f"{w}_{tag}_minus_m400_mean_rms"] = float(np.sqrt(np.mean(d ** 2)))
        for rr in sorted({1, n_real}):
            k = f"{w}/kde_local/R{rr}"
            if k in A:
                loc = A[k]
                d_mean = loc[:, :2] - base[:, rr - 1, :2]
                d_map = loc[:, 4:6] - base[:, rr - 1, 4:6]
                checks[f"{w}_local_exact_R{rr}_mean_maxabs"] = float(np.max(np.abs(d_mean)))
                checks[f"{w}_local_exact_R{rr}_map_maxabs"] = float(np.max(np.abs(d_map)))
                checks[f"{w}_local_exact_R{rr}_mean_fit"] = fit(truth, loc[:, :2])
                checks[f"{w}_m400_R{rr}_mean_fit"] = fit(truth, base[:, rr - 1, :2])
                checks[f"{w}_stack_post_std_R{rr}_median"] = float(np.median(base[:, rr - 1, 2:4]))
        k = f"{w}/gauss/analytic_N32R16"
        if k in A:
            gl = A[f"{w}/gauss/Rcurve"][:, -1, 0, :2]
            checks[f"{w}_gauss_lattice_minus_analytic_maxabs"] = float(np.max(np.abs(gl - A[k])))
        # distance of the stacked posterior mean from the nearest lattice node, in node spacings
        m = base[:, -1, :2]
        frac = ((m - (-0.1)) / 0.0005) % 1.0
        checks[f"{w}_m400_R{n_real}_mean_node_phase"] = frac.ravel().round(3).tolist()
        # pulls with the stacked posterior width
        for ri in sorted({0, n_real - 1}):
            post = A[f"{w}/kde/Rcurve"][:, ri, 0, :]
            z = (post[:, :2] - truth) / post[:, 2:4]
            checks[f"{w}_pull_N32R{ri + 1}"] = {"mean": z.mean(0).tolist(), "std": z.std(0).tolist()}
            bstd = A[f"{w}/kde/Rcurve"][:, ri, 1:, :2].std(axis=1)
            checks[f"{w}_boot_std_over_post_std_N32R{ri + 1}_median"] = float(np.median(bstd / post[:, 2:4]))
        checks[f"{w}_bandwidth_or_sigma_guard_fraction"] = float(np.mean(A[f"obs_{w}_guarded"]))

    # ------------------------------------------------------------------ nuisance-prior diagnostics
    bench = pd.read_csv(BENCH_CSV)
    b0 = bench[bench.benchmark_cell == 0]
    z_bench = ((np.log10(b0.vcirc) - (36.0 - b0.rmag_true) / 7.22) / 0.1).to_numpy()
    xu3 = pd.read_csv(XU3_CSV, usecols=["vcirc", "rmag_true", "image_snr", "central_halpha_snr"])
    z_xu3 = ((np.log10(xu3.vcirc) - (36.0 - xu3.rmag_true) / 7.22) / 0.1).to_numpy()
    nsamp = int(json.loads((args.samples / "meta.json").read_text())["nsamples"]) if (args.samples / "meta.json").exists() else None
    ess = A["obs_tf_ess"]  # (25, G, R)
    ess_frac = ess / (nsamp or ess.max())
    ess_gal = np.median(ess_frac, axis=(0, 2))  # per latent galaxy
    prior = {
        "benchmark_source": "valid_100k_simv3_cosi (training-prior draws: vcirc uniform on [60,540], independent of rmag)",
        "tf_z_bench": z_bench.tolist(),
        "tf_z_bench_abs_gt3_fraction": float(np.mean(np.abs(z_bench) > 3)),
        "tf_z_bench_std": float(np.std(z_bench)),
        "tf_z_xu3_std": float(np.std(z_xu3)),
        "tf_ess_fraction_median": float(np.median(ess_frac)),
        "tf_ess_fraction_p05": float(np.percentile(ess_frac, 5)),
        "tf_ess_lt50_fraction": float(np.mean(ess < 50)),
        "tf_ess_fraction_per_galaxy_median": ess_gal.tolist(),
        "bench_image_snr_median": float(b0.image_snr.median()),
        "bench_halpha_snr_median": float(b0.central_halpha_snr.median()),
        "xu3_image_snr_median": float(xu3.image_snr.median()),
        "xu3_halpha_snr_median": float(xu3.central_halpha_snr.median()),
        "bench_rmag_range": [float(b0.rmag_true.min()), float(b0.rmag_true.max())],
    }
    obs = {
        w: {
            "std_g_median": float(np.median(A[f"obs_{w}_std"])),
            "sigma_eps_xu24_median": float(np.median(np.sqrt((A[f"obs_{w}_std"] ** 2).sum(-1)))),
            "sigma_eps_xu24_mean": float(np.mean(np.sqrt((A[f"obs_{w}_std"] ** 2).sum(-1)))),
            "sigma_eps_xu24_rms": float(np.sqrt(np.mean((A[f"obs_{w}_std"] ** 2).sum(-1)))),
            "std_g_p16_p84": np.percentile(A[f"obs_{w}_std"], [16, 84]).tolist(),
            "kde_h_median": float(np.median(A[f"obs_{w}_h"])),
            "n_eff_median": float(np.median(A[f"obs_{w}_neff"])),
            "lapkde_fit_ok_fraction": float(np.mean(A[f"obs_{w}_lapkde_ok"])),
            "kde_mass_in_box_min_max": [float(A[f"obs_{w}_kde_mass"].min()), float(A[f"obs_{w}_kde_mass"].max())],
            "abs_corr_g1g2_median": float(np.median(np.abs(A[f"obs_{w}_corr"]))),
        }
        for w in ("train", "tf")
    }

    # ------------------------------------------------------------------ figures
    figs = {}
    Ns = np.array(N_LIST)

    def errbar(ax, x, name, curve, comp, color, label, marker="o", dx=1.0, ls="-"):
        keys = [str(v) for v in x]
        d = metrics[name][curve]
        if "boot_median" in d[keys[0]]:
            y = np.array([d[k]["boot_median"][comp] for k in keys])
            lo = y - np.array([d[k]["boot_p16"][comp] for k in keys])
            hi = np.array([d[k]["boot_p84"][comp] for k in keys]) - y
            ax.errorbar(np.asarray(x) * dx, y, yerr=[lo, hi], color=color, marker=marker, ls=ls,
                        capsize=2, lw=1.6, label=label, markeredgecolor="white", markeredgewidth=0.8)
        else:
            y = np.array([d[k]["det"][comp] for k in keys])
            ax.plot(np.asarray(x) * dx, y, color=color, marker=marker, ls=ls, lw=1.6, label=label)

    def m_vs_n(fname, curve, series, title, paper=None):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
        for ax, comp, lab in zip(axes, ("g1_m", "g2_m"), ("$m_1$", "$m_2$")):
            k = len(series)
            for i, (name, color, label, marker) in enumerate(series):
                errbar(ax, Ns, name, curve, comp, color, label, marker, dx=2 ** (0.07 * (i - (k - 1) / 2)))
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(Ns, [str(n) for n in Ns])
            ax.set_xlabel("number of galaxies N")
            ax.set_ylabel(lab)
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(rd / fname)
        if paper:
            fig.savefig(fd / f"{paper}.png")
            fig.savefig(fd / f"{paper}.pdf")
        plt.close(fig)
        figs[fname] = title

    main_series = [
        ("pergal_mean_train", C1, "per-galaxy posterior mean", "o"),
        ("stack_kde_mean_train", C2, "stacked, KDE", "s"),
        ("stack_gauss_mean_train", C3, "stacked, moment Gaussian", "^"),
        ("stack_lapkde_mean_train", C4, "stacked, log-KDE quadratic", "D"),
    ]
    m_vs_n("m_vs_N_R16.png", RN, main_series,
           f"Multiplicative bias vs number of galaxies (R={n_real} noise realizations each; training nuisance prior)",
           paper="C_m_vs_N")
    m_vs_n("m_vs_N_R1.png", "vs_N_at_R1", main_series,
           "Multiplicative bias vs number of galaxies (R=1; training nuisance prior)")
    tf_series = [
        ("pergal_mean_train", C1, "per-galaxy mean, unweighted", "o"),
        ("pergal_mean_tf", C7, "per-galaxy mean, TF-weighted", "v"),
        ("stack_kde_mean_train", C2, "stacked KDE, training prior", "s"),
        ("stack_kde_mean_tf", C8, "stacked KDE, TF prior", "P"),
    ]
    m_vs_n("m_vs_N_tf.png", RN, tf_series,
           f"Training vs TF nuisance prior (R={n_real})")
    sens_series = [
        ("stack_kde_mean_train", C2, "KDE h x1, floor 1e-3", "s"),
        ("stack_kde_bw0.5_mean_train", C1, "KDE h x0.5", "o"),
        ("stack_kde_bw2_mean_train", C3, "KDE h x2", "^"),
        ("stack_kde_a0_mean_train", C8, "KDE no floor", "x"),
        ("stack_kde_a1e-2_mean_train", C4, "KDE floor 1e-2", "D"),
        ("stack_kde_map_train", C7, "KDE MAP (h x1)", "v"),
    ]
    m_vs_n("m_vs_N_sensitivity.png", RN, sens_series,
           f"Bandwidth / floor / estimator sensitivity of the stacked KDE (R={n_real}, training prior)")

    # m vs R at N=32
    Rs = np.arange(1, n_real + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, comp, lab in zip(axes, ("g1_m", "g2_m"), ("$m_1$", "$m_2$")):
        for i, (name, color, label, marker) in enumerate(main_series + [("stack_kde_mean_tf", C8, "stacked KDE, TF prior", "P")]):
            errbar(ax, Rs, name, "vs_R_at_N32", comp, color, label, marker, dx=1 + 0.02 * (i - 2))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("noise realizations per galaxy R  (N=32 galaxies)")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.55), ncol=2)
    fig.suptitle("Multiplicative bias vs number of noise realizations (32 x R observations per cell)", fontsize=10)
    fig.tight_layout()
    fig.savefig(rd / "m_vs_R.png")
    fig.savefig(fd / "C_m_vs_R.png")
    plt.close(fig)
    figs["m_vs_R.png"] = "m vs R"

    # residual vs truth (N=32, R=max)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, d in zip(axes, (0, 1)):
        for j, (name, color, label, marker, off) in enumerate((
                ("pergal_mean_train", C1, "per-galaxy posterior mean", "o", -0.0022),
                ("stack_kde_mean_train", C2, "stacked likelihood (KDE)", "s", 0.0022))):
            rows = est[name]["Rcurve"][-1]
            r = rows[0, :, d] - truth[:, d]
            err = rows[1:, :, d].std(0) if rows.shape[0] > 1 else None
            ax.errorbar(truth[:, d] + off + 0.0006 * (np.arange(25) % 5 - 2), r, yerr=err, fmt=marker, color=color,
                        ms=5, capsize=1.5, lw=1, label=label, markeredgecolor="white", markeredgewidth=0.6)
            f = metrics[name]["vs_R_at_N32"][str(n_real)]["det"]
            xx = np.linspace(-0.09, 0.09, 2)
            ax.plot(xx, f[f"g{d + 1}_m"] * xx + f[f"g{d + 1}_c"], color=color, lw=1.2, ls="--")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
        ax.set_xlabel(f"true $g_{d + 1}$")
        ax.set_ylabel(fr"$\hat g_{d + 1} - g_{d + 1}$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Cell residuals, N=32 galaxies x R={n_real} realizations per cell (bars: paired-bootstrap std over galaxies)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(rd / "residual_vs_truth.png")
    fig.savefig(fd / "C_residual_vs_truth.png")
    fig.savefig(fd / "C_residual_vs_truth.pdf")
    plt.close(fig)
    figs["residual_vs_truth.png"] = "residuals"

    # nuisance prior mismatch
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    bins = np.linspace(-10, 10, 41)
    axes[0].hist(z_xu3, bins=bins, density=True, color=C1, alpha=0.85, label="Xu3 test set (TF-conformed)")
    axes[0].hist(z_bench, bins=bins, density=True, histtype="step", color=C2, lw=2, label="benchmark 32 galaxies")
    axes[0].set_xlabel(r"TF residual $[\log_{10} v_{\rm circ} - \mathrm{TF}(m_r)]/0.1$ dex")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[1].scatter(np.abs(z_bench), ess_gal, color=C2, s=22, edgecolor="white", lw=0.6)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("|TF residual| of the benchmark galaxy (sigma)")
    axes[1].set_ylabel("median TF ESS fraction")
    fig.tight_layout()
    fig.savefig(rd / "tf_prior_mismatch.png")
    plt.close(fig)
    figs["tf_prior_mismatch.png"] = "tf"

    # ------------------------------------------------------------------ tables
    def fmt(x, nd=3):
        return "nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"

    def cell(name, curve, key, comp, nd=3):
        d = metrics[name][curve][key]
        if "boot_median" in d:
            return (f"{fmt(d['det'][comp], nd)} <span class='e'>[{fmt(d['boot_p16'][comp], nd)}, "
                    f"{fmt(d['boot_p84'][comp], nd)}]</span>")
        return fmt(d["det"][comp], nd)

    def table(names, curve, keys, keyname):
        head = (f"<tr><th>estimator</th><th>{keyname}</th><th>m<sub>1</sub></th><th>m<sub>2</sub></th>"
                "<th>c<sub>1</sub></th><th>c<sub>2</sub></th><th>RMSE</th></tr>")
        body = []
        for name in names:
            for k in keys:
                body.append(
                    f"<tr><td>{html.escape(labels[name])}</td><td>{k}</td>"
                    f"<td>{cell(name, curve, k, 'g1_m')}</td><td>{cell(name, curve, k, 'g2_m')}</td>"
                    f"<td>{cell(name, curve, k, 'g1_c', 4)}</td><td>{cell(name, curve, k, 'g2_c', 4)}</td>"
                    f"<td>{metrics[name][curve][k]['det']['combined_rmse']:.4f}</td></tr>")
        return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"

    nkeys = [str(n) for n in N_LIST]
    rkeys = [str(r) for r in (1, 2, 4, 8, 16) if r <= n_real]
    main_names = ["pergal_mean_train", "stack_kde_mean_train", "stack_kde_map_train",
                  "stack_gauss_mean_train", "stack_lapkde_mean_train"]
    tf_names = ["pergal_mean_tf", "stack_kde_mean_tf", "stack_gauss_mean_tf", "stack_lapkde_mean_tf"]
    sens_names = ["stack_kde_mean_train", "stack_kde_bw0.5_mean_train", "stack_kde_bw2_mean_train",
                  "stack_kde_a0_mean_train", "stack_kde_a1e-4_mean_train", "stack_kde_a1e-2_mean_train",
                  "stack_kde_mean_tf", "stack_kde_bw0.5_mean_tf", "stack_kde_bw2_mean_tf",
                  "stack_kde_a0_mean_tf", "stack_kde_a1e-2_mean_tf"]
    grid_names = [n for n in ("stack_kde_mean_train", "stack_kde_m401_mean_train", "stack_kde_m800_mean_train",
                              "stack_kde_mean_tf", "stack_kde_m401_mean_tf", "stack_kde_m800_mean_tf") if n in metrics]

    # headline numbers
    def hl(name, curve=RN, key="32", comp="g1_m"):
        return metrics[name][curve][key]["det"][comp]

    head = {
        name: {c: hl(name, comp=c) for c in ("g1_m", "g2_m", "g1_c", "g2_c")}
        for name in main_names + tf_names + sens_names[1:6]
    }

    def pm(name, comp):
        d = metrics[name][RN]["32"]
        s = d.get("boot_std", {}).get(comp, float("nan"))
        return f"{d['det'][comp]:+.3f} &plusmn; {s:.3f}"

    summary_rows = "".join(
        f"<tr><td>{html.escape(labels[n])}</td><td>{pm(n, 'g1_m')}</td><td>{pm(n, 'g2_m')}</td>"
        f"<td>{metrics[n][RN]['32']['det']['g1_c']:+.4f}</td><td>{metrics[n][RN]['32']['det']['g2_c']:+.4f}</td></tr>"
        for n in main_names + tf_names)

    chk_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v) if not isinstance(v, float) else f'{v:.3g}')}</td></tr>"
                       for k, v in checks.items() if "node_phase" not in k)

    fig_html = {k: img_tag(rd / k, v) for k, v in figs.items()}
    km = metrics["stack_kde_mean_train"][RN]["32"]["det"]
    pg = metrics["pergal_mean_train"][RN]["32"]["det"]
    kt = metrics["stack_kde_mean_tf"][RN]["32"]["det"]
    gm = metrics["stack_gauss_mean_train"][RN]["32"]["det"]
    lk = metrics["stack_lapkde_mean_train"][RN]["32"]["det"]
    b05 = metrics["stack_kde_bw0.5_mean_train"][RN]["32"]["det"]
    b2 = metrics["stack_kde_bw2_mean_train"][RN]["32"]["det"]

    payload = {
        "schema": "paper-tests-C-stacking-report-v1",
        "test": "T8: NPE likelihood stacking at constant shear (paper Sec. 6.5)",
        "model": MODEL,
        "dataset": str(S / "datasets" / args.dataset_name),
        "analysis_npz": str(args.analysis),
        "samples": str(args.samples),
        "n_samples_per_observation": nsamp,
        "n_realizations": n_real,
        "bootstrap": {"count": boot, "seed": meta.get("boot_seed"), "paired_across_cells": True,
                      "resampled_unit": "latent (nuisance) galaxy ID, with replacement",
                      "interval": "16th-84th percentile", "det": "first-N prefix / all galaxies"},
        "analysis_meta": meta,
        "headline_N32_Rmax": head,
        "metrics": metrics,
        "grid_checks": checks,
        "nuisance_prior": prior,
        "per_observation": obs,
        "figures": list(figs),
        "paper_figures": ["C_m_vs_N.png", "C_m_vs_N.pdf", "C_residual_vs_truth.png", "C_residual_vs_truth.pdf", "C_m_vs_R.png"],
    }
    (rd / "report.json").write_text(json.dumps(payload, indent=1, default=float))

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NPE likelihood stacking</title>
<script>MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']]}}, svg: {{fontCache: 'global'}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
body {{ font: 17px/1.55 Palatino, "Palatino Linotype", serif; margin: 2rem auto; max-width: 1180px; color: #1b1b1b; background: #fff; padding: 0 16px; }}
h1, h2, h3 {{ font-weight: 600; }} figure {{ margin: 1.4rem 0 2rem; }}
figcaption, .note {{ color: #444; }} img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; font-size: .86rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: .3rem .42rem; vertical-align: top; }}
th {{ background: #f4f4f4; text-align: left; }} .e {{ color: #666; font-size: .8rem; }}
.flag {{ border-left: 4px solid #d03b3b; padding: .4rem .8rem; background: #fbf1f1; }}
.ok {{ border-left: 4px solid #0ca30c; padding: .4rem .8rem; background: #f1f8f1; }}
.equation {{ overflow-x: auto; margin: 1rem 0; }}
</style></head><body>
<h1>T8: NPE likelihood stacking at constant shear</h1>
<p class="lead">Paper Sec. 6.5. NPE (<code>{MODEL}</code>), fixed-shear benchmark
<code>{args.dataset_name}</code>: 25 exact shear cells
\\(g_1,g_2\\in\\{{-0.08,-0.04,0,0.04,0.08\\}}\\) &times; 32 paired nuisance galaxies &times; {n_real}
independent noise realizations. {nsamp} identity+R90 posterior samples per observation. No NRE.</p>

<h2>Summary</h2>
<div class="ok"><p>With nuisances marginalized under the <b>training prior</b> (the prior the benchmark
nuisances were actually drawn from), multiplying the per-observation NPE shear likelihoods
\\(q(\\mathbf g|D_i)/\\pi_{{\\rm train}}(\\mathbf g)\\) removes the shrinkage of the per-galaxy
posterior mean. At N = 32 galaxies &times; R = {n_real}: per-galaxy posterior mean
m<sub>1</sub> = {pg['g1_m']:+.3f}, m<sub>2</sub> = {pg['g2_m']:+.3f}; stacked (KDE) posterior mean
m<sub>1</sub> = {km['g1_m']:+.3f}, m<sub>2</sub> = {km['g2_m']:+.3f}
(c<sub>1</sub> = {km['g1_c']:+.4f}, c<sub>2</sub> = {km['g2_c']:+.4f}).</p></div>
<p>Same N, R, other density models: moment-matched Gaussian m<sub>1</sub> = {gm['g1_m']:+.3f},
m<sub>2</sub> = {gm['g2_m']:+.3f}; log-KDE quadratic m<sub>1</sub> = {lk['g1_m']:+.3f},
m<sub>2</sub> = {lk['g2_m']:+.3f}. KDE bandwidth &times;0.5: ({b05['g1_m']:+.3f}, {b05['g2_m']:+.3f});
&times;2: ({b2['g1_m']:+.3f}, {b2['g2_m']:+.3f}). TF nuisance prior (mismatched to these galaxies):
stacked KDE m<sub>1</sub> = {kt['g1_m']:+.3f}, m<sub>2</sub> = {kt['g2_m']:+.3f}.</p>

<table><thead><tr><th>estimator (N=32, R={n_real})</th><th>m<sub>1</sub> &plusmn; boot std</th><th>m<sub>2</sub> &plusmn; boot std</th><th>c<sub>1</sub></th><th>c<sub>2</sub></th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<p class="note">Point values are the deterministic result (all 32 galaxies). &plusmn; is the standard deviation over {boot}
paired bootstrap resamples of the 32 galaxy IDs (the same resample is used for all 25 cells).
Fits: \\(\\hat g_\\alpha = (1+m_\\alpha) g_\\alpha + c_\\alpha\\) by least squares over the 25 cell estimates,
separately for each component (this is the fit of the old benchmark report; with equal numbers of observations
per cell it equals the pooled per-observation fit for the per-galaxy mean).</p>

<h2>Method</h2>
<h3>Likelihood from the NPE posterior</h3>
<p>The NPE was trained with a uniform shear prior \\(\\pi_{{\\rm train}}(\\mathbf g)=U(-0.1,0.1)^2\\) and nuisance
prior \\(\\pi_{{\\rm train}}(\\boldsymbol\\eta)\\). Its 2D shear marginal (obtained by ignoring the nuisance
coordinates of the samples) approximates</p>
<div class="equation">\\[ q(\\mathbf g|D_i) \\approx \\frac{{\\pi_{{\\rm train}}(\\mathbf g)\\int p(D_i|\\mathbf g,\\boldsymbol\\eta)\\,\\pi(\\boldsymbol\\eta)\\,d\\boldsymbol\\eta}}{{p(D_i)}}
\\;\\Rightarrow\\; L_i(\\mathbf g) \\propto \\frac{{q(\\mathbf g|D_i)}}{{\\pi_{{\\rm train}}(\\mathbf g)}} \\propto q(\\mathbf g|D_i)\\quad(\\mathbf g\\in{{\\rm box}}).\\]</div>
<p>For observations sharing one shear (Bernstein &amp; Armstrong 2014) the cell posterior is</p>
<div class="equation">\\[ \\log p(\\mathbf g|\\{{D_i\\}}) = \\log\\pi_{{\\rm train}}(\\mathbf g) + \\sum_{{i=1}}^{{N R}} \\log \\hat q_i(\\mathbf g) + {{\\rm const}}. \\]</div>
<p>The nuisance prior \\(\\pi(\\boldsymbol\\eta)\\) is the one the samples represent: <b>unweighted samples</b> give
the training prior (uniform v<sub>circ</sub> etc.); <b>TF-weighted samples</b> (importance weights
\\(p_{{\\rm TF}}(v_{{\\rm circ}}|m_r)/\\pi_{{\\rm train}}(v_{{\\rm circ}})\\)) give the TF prior on v<sub>circ</sub>.
Both are computed here; see the prior-mismatch section for which one matches this benchmark.</p>
<h3>Density estimate \\(\\hat q_i\\) (no grid snapping)</h3>
<ul>
<li><b>KDE</b> (primary): exact weighted Gaussian product-kernel KDE of the {nsamp} (g<sub>1</sub>,g<sub>2</sub>)
samples, with boundary reflection at all four box edges (the NPE posterior is bounded to the box; without
reflection the KDE is halved at the edge, which biases the product toward the center). Bandwidth
\\(h_d = f\\,\\sigma_d\\,n_{{\\rm eff}}^{{-1/6}}\\) (Scott = Silverman in 2D) with \\(f=1\\) (tests: 0.5, 2) and
weighted \\(n_{{\\rm eff}}=1/\\sum w^2\\). Floor: \\(\\hat q\\to(1-a)\\hat q + a/|{{\\rm box}}|\\) with \\(a=10^{{-3}}\\)
(tests: 0, 10<sup>-4</sup>, 10<sup>-2</sup>). Median per-observation, per-component posterior std (unweighted) =
{obs['train']['std_g_median']:.4f}, i.e. per-observation shape noise
\\(\\sigma_{{\\epsilon,i}}=\\sqrt{{\\sigma_{{g_1,i}}^2+\\sigma_{{g_2,i}}^2}}\\) (Xu+24 quadrature definition) median
{obs['train']['sigma_eps_xu24_median']:.4f}, mean {obs['train']['sigma_eps_xu24_mean']:.4f};
median bandwidth h = {obs['train']['kde_h_median']:.4f}.</li>
<li><b>Moment Gaussian</b>: weighted mean and covariance of the samples. The per-observation posterior
is truncated by the box, so its moments are shrunk; the product inherits this.</li>
<li><b>log-KDE quadratic</b>: weighted least-squares quadratic fit to \\(\\log\\hat q_{{\\rm KDE}}\\) over the KDE
core (KDE &ge; 5% of its peak), i.e. a Gaussian that follows the KDE inside the box but is not
truncated (falls back to the moment Gaussian if not negative definite:
{100 * (1 - obs['train']['lapkde_fit_ok_fraction']):.2f}% of observations).</li>
</ul>
<p>All \\(\\log\\hat q_i\\) are evaluated exactly at the nodes of a 400&times;400 lattice (spacing 0.0005 over
the box; the KDE itself is not binned), summed, and summarized by the posterior mean (lattice sum) and the
MAP (parabolic sub-node refinement). Median stacked-posterior std at N=32:
R=1 {checks.get('train_stack_post_std_R1_median', float('nan')):.4f}, R={n_real}
{checks.get(f'train_stack_post_std_R{n_real}_median', float('nan')):.4f}, i.e. several lattice spacings.</p>
<h3>Counting</h3>
<p>N = number of distinct nuisance galaxies (prefix of the 32 IDs for the point value; paired bootstrap
with replacement for the interval), each contributing R noise realizations, so a cell stacks N&times;R
observations. For the R curve N = 32. The per-galaxy posterior mean uses exactly the same observations
(average of the per-observation posterior means).</p>
<p class="note"><b>Repeated realizations of one galaxy.</b> The product over r of marginal likelihoods
\\(\\prod_r\\int p(D_{{nr}}|\\mathbf g,\\boldsymbol\\eta)\\pi(\\boldsymbol\\eta)d\\boldsymbol\\eta\\) is not the joint likelihood of
repeated exposures of one galaxy (which shares \\(\\boldsymbol\\eta\\)). It is the correct likelihood if the
N&times;R observations are treated as distinct galaxies. The benchmark re-uses 32 nuisance vectors, so R
adds noise realizations but not intrinsic-shape diversity: the galaxy bootstrap interval, not the
posterior width, is the relevant error (see checks).</p>

<h2>Bias vs number of galaxies</h2>
<figure>{fig_html['m_vs_N_R16.png']}<figcaption>Markers: bootstrap median; bars: 16&ndash;84% of {boot} paired galaxy
bootstrap resamples. Small horizontal offsets for legibility. Paper candidate:
<code>paper_figs/C_m_vs_N.png</code>.</figcaption></figure>
{table(main_names, RN, nkeys, 'N')}
<figure>{fig_html['m_vs_N_R1.png']}<figcaption>Same with one noise realization per galaxy (N observations per cell).</figcaption></figure>
{table(["pergal_mean_train", "stack_kde_mean_train", "stack_gauss_mean_train", "stack_lapkde_mean_train"], 'vs_N_at_R1', nkeys, 'N')}

<h2>Bias vs number of noise realizations (N = 32)</h2>
<figure>{fig_html['m_vs_R.png']}<figcaption>32&times;R observations per cell. Bars: paired galaxy bootstrap.</figcaption></figure>
{table(["pergal_mean_train", "stack_kde_mean_train", "stack_gauss_mean_train", "stack_kde_mean_tf"], 'vs_R_at_N32', rkeys, 'R')}

<h2>Residual vs true shear</h2>
<figure>{fig_html['residual_vs_truth.png']}<figcaption>Each point is one of the 25 cells (small jitter
separates the 5 cells sharing a true g<sub>&alpha;</sub>); dashed lines are the m, c fits. Paper candidate:
<code>paper_figs/C_residual_vs_truth.png</code>.</figcaption></figure>

<h2>Sensitivity: bandwidth, floor, MAP</h2>
<figure>{fig_html['m_vs_N_sensitivity.png']}</figure>
{table(sens_names, RN, ['1', '4', '32'], 'N')}

<h2>Nuisance prior: training vs TF</h2>
<figure>{fig_html['m_vs_N_tf.png']}</figure>
{table(tf_names, RN, nkeys, 'N')}
<figure>{fig_html['tf_prior_mismatch.png']}<figcaption>Left: TF residual of the true v<sub>circ</sub> for the 32
benchmark galaxies vs the Xu3 test set. Right: median TF-importance ESS fraction of each benchmark galaxy
vs its |TF residual|.</figcaption></figure>
<div class="flag"><p><b>Prior mismatch.</b> The benchmark galaxies were copied from
<code>valid_100k_simv3_cosi</code> (training-prior draws: v<sub>circ</sub> uniform on [60, 540] km/s,
independent of m<sub>r</sub>), not from the TF-conformed population. Their TF residuals have std
{prior['tf_z_bench_std']:.2f}&sigma; (Xu3: {prior['tf_z_xu3_std']:.2f}&sigma;) and
{100 * prior['tf_z_bench_abs_gt3_fraction']:.0f}% lie beyond 3&sigma;. The TF-weighted likelihood therefore
marginalizes v<sub>circ</sub> under a prior that is wrong for these galaxies, and the importance weights
degenerate for off-relation galaxies (median TF ESS fraction {prior['tf_ess_fraction_median']:.3f}, 5th
percentile {prior['tf_ess_fraction_p05']:.4f}; {100 * prior['tf_ess_lt50_fraction']:.1f}% of observations
have ESS &lt; 50). The self-consistent choice for this benchmark is the training prior (unweighted samples),
which is used for the headline numbers. For Xu3-like data (vcirc on the TF relation) the TF-weighted samples
would be the consistent choice; a TF-conformed fixed-shear benchmark would be needed to test that directly.
The benchmark is also brighter than Xu3: median image S/N {prior['bench_image_snr_median']:.0f} vs
{prior['xu3_image_snr_median']:.0f}, median central H&alpha; S/N {prior['bench_halpha_snr_median']:.0f} vs
{prior['xu3_halpha_snr_median']:.0f}, so its per-galaxy shrinkage (m &asymp; {pg['g1_m']:+.2f}) is smaller than Xu3's.</p></div>

<h2>Flags / caveats</h2>
<ul>
<li><b>Residual m<sub>1</sub>.</b> The stacked KDE mean leaves m<sub>1</sub> = {km['g1_m']:+.3f} &plusmn;
{metrics['stack_kde_mean_train'][RN]['32']['boot_std']['g1_m']:.3f} (about
{abs(km['g1_m']) / metrics['stack_kde_mean_train'][RN]['32']['boot_std']['g1_m']:.1f}&sigma; from zero) while
m<sub>2</sub> is consistent with zero. It does not move toward zero with smaller bandwidth, so it is not KDE
smoothing; it could be residual NPE likelihood miscalibration or a property of these 32 galaxies (the
per-galaxy mean is also more biased in g<sub>1</sub>). Treat stacking as removing the shrinkage to the ~1% level,
not exactly.</li>
<li><b>Stacked posterior too narrow at large R.</b> At N=32, R={n_real} the galaxy-bootstrap scatter of the stacked
mean is {checks.get(f'train_boot_std_over_post_std_N32R{n_real}_median', float('nan')):.1f}&times; the stacked
posterior std (pull std {checks[f'train_pull_N32R{n_real}']['std'][0]:.2f}, {checks[f'train_pull_N32R{n_real}']['std'][1]:.2f});
at R=1 the ratio is {checks.get('train_boot_std_over_post_std_N32R1_median', float('nan')):.1f} (pull std
{checks['train_pull_N32R1']['std'][0]:.2f}, {checks['train_pull_N32R1']['std'][1]:.2f}). Expected: the R noise realizations
of one galaxy share its intrinsic shape/orientation, so multiplying their marginal likelihoods over-counts
information. With distinct galaxies (R=1) the width is roughly right. The bias m is not affected by this.</li>
<li><b>Few galaxies.</b> With N &le; 2 the stacked posterior is still prior-dominated and m is noisy
(bootstrap std &ge; 0.2); stacking reaches |m| &lt; 0.04 by N = 4 (R = {n_real}) and N&times;R &asymp; 32 observations.</li>
<li><b>Moment Gaussian</b> is slightly worse than the KDE (m<sub>1</sub> = {gm['g1_m']:+.3f}) because the
per-observation posteriors are truncated by the box, so their moments are shrunk. The log-KDE quadratic fit was
not negative definite for {100 * (1 - obs['train']['lapkde_fit_ok_fraction']):.0f}% of observations (moment-Gaussian
fallback), so it is only a diagnostic.</li>
<li><b>Floor</b> a &isin; {{0, 10<sup>-4</sup>, 10<sup>-3</sup>, 10<sup>-2</sup>}} changes nothing at the 10<sup>-4</sup>
level: no observation's KDE reaches the floor near the stacked peak. The tail problem does not arise here because
every per-observation posterior covers the true shear with 8192 samples.</li>
<li><b>Guards.</b> Bandwidth h &ge; 0.001 and &sigma; &ge; 0.002 were enforced; they were active for
{100 * checks['train_bandwidth_or_sigma_guard_fraction']:.1f}% (unweighted) and
{100 * checks['tf_bandwidth_or_sigma_guard_fraction']:.1f}% (TF) of observations (very narrow posteriors, or TF
weights with ESS &asymp; 1). With TF weights the Gaussian-type estimators degrade badly (moment Gaussian
m<sub>1</sub> = {metrics['stack_gauss_mean_tf'][RN]['32']['det']['g1_m']:+.3f}, log-KDE quadratic
{metrics['stack_lapkde_mean_tf'][RN]['32']['det']['g1_m']:+.3f}), because degenerate weights give spuriously narrow
moments; the TF-weighted KDE survives (m<sub>1</sub> = {kt['g1_m']:+.3f}, m<sub>2</sub> = {kt['g2_m']:+.3f}) but its
stacked posterior is more over-confident (pull std {checks[f'tf_pull_N32R{n_real}']['std'][0]:.1f}).</li>
<li><b>Per-observation posteriors are strongly anisotropic</b> (median |corr(g<sub>1</sub>,g<sub>2</sub>)| =
{obs['train']['abs_corr_g1g2_median']:.2f}); the diagonal KDE kernel is still narrow compared with the narrow
posterior axis for most observations.</li>
</ul>

<h2>Grid and estimator checks</h2>
<p>The old grid-snapping problem is gone: changing the evaluation lattice (400&rarr;401 nodes, i.e. shifting nodes
relative to the true shears; or 400&rarr;800) moves the cell posterior means by at most
{checks['train_m401_minus_m400_mean_maxabs']:.1e} / {checks['train_m800_minus_m400_mean_maxabs']:.1e}, and an exact local
61&times;61 evaluation agrees to {checks[f'train_local_exact_R{n_real}_mean_maxabs']:.1e}; m changes by &lt; 10<sup>-4</sup>.
A lattice sum of a smooth posterior whose width (median {checks.get(f'train_stack_post_std_R{n_real}_median', float('nan')):.5f}
at R={n_real}) is comparable to or larger than the spacing is accurate far below the spacing.</p>
{table(grid_names, 'vs_R_at_N32', [str(n_real)], 'R') if grid_names else ''}
<table><thead><tr><th>check</th><th>value</th></tr></thead><tbody>{chk_rows}</tbody></table>
<p class="note">m401: lattice with 401 nodes per axis (spacing 0.000499), so node positions relative to the true
shears differ from the 400 lattice (on which every true shear sits exactly between two nodes).
m800: spacing 0.00025. local_exact: exact KDE evaluated on a 61&times;61 local lattice spanning &plusmn;5 posterior
std around the 400-lattice result. pull: (stacked mean &minus; truth)/stacked posterior std over 25 cells.
boot_std_over_post_std: galaxy-bootstrap scatter of the stacked mean divided by the stacked posterior std.</p>

<h2>Per-observation diagnostics</h2>
<pre>{html.escape(json.dumps(obs, indent=1))}</pre>

<h2>Provenance</h2>
<ul>
<li>Samples: <code>{html.escape(str(args.samples))}</code> (c_stack_infer.py; identity+R90 equal mixture, pre-noised
observations from <code>benchmark_noisy.npz</code>).</li>
<li>Analysis: <code>{html.escape(str(args.analysis))}</code> (c_stack_analysis.py).</li>
<li>Scripts: <code>kl-nn/arch/diagnostics/paper_tests/c_stack_infer.py, c_stack_analysis.py, c_stack_report.py</code>.</li>
<li>Old benchmark for comparison: <code>reports/likelihood-stack/04_benchmark</code> (identity-only, 1024 samples,
unweighted per-galaxy NPE mean m &asymp; &minus;0.39 at R=16; 21&times;21 grid).</li>
</ul>
</body></html>"""
    (rd / "report.html").write_text(body, encoding="utf-8")
    print("wrote", rd / "report.html")


if __name__ == "__main__":
    main()
