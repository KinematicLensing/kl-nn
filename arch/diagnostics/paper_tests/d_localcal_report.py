#!/usr/bin/env python3
"""D: figures and group report (report.html) for the local-calibration test.

Reads D_local_calibration/report.json (written by d_localcal_analysis.py) and
writes the PNGs and report.html into the same folder. Refuses to overwrite.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_localcal_analysis as A  # noqa: E402

OUT = A.OUT_DIR
if "--out-dir" in sys.argv:
    OUT = Path(sys.argv[sys.argv.index("--out-dir") + 1])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = {"self": "#2a78d6", "gauss": "#eb6834", "toy": "#1baf7a", "toy_alt": "#eda100"}
LAB = {"self": "NPE self-consistency (model-free)", "gauss": "Gaussian-prior formula",
       "toy": "uniform-prior toy (MC)", "toy_alt": "uniform toy, width match at mean"}
LS = {"self": "-", "gauss": "--", "toy": "-.", "toy_alt": ":"}
LEVEL_NOM = {"68": 0.68, "95": 0.95}
FIGS = {}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)


def save(fig, name):
    p = OUT / name
    if p.exists():
        raise SystemExit(f"{p} exists; refusing to overwrite")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = p


def coverage_panels(R, key_fmt, xlabel, name, preds=("self", "gauss", "toy"), resid_preds=("self", "toy")):
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 2, 1.2]})
    for c, comp in enumerate(A.COMPONENTS):
        for r, lev in enumerate(("68", "95")):
            rows = R["fine"][key_fmt.format(comp=comp, lev=lev)]["rows"]
            x = np.array([row["center"] for row in rows])
            m = np.array([row["measured"] for row in rows])
            e = np.array([row["measured_se"] for row in rows])
            ax, axr = axes[2 * r, c], axes[2 * r + 1, c]
            ax.axhline(LEVEL_NOM[lev], color=INK2, lw=0.8, ls=(0, (2, 2)))
            for p in preds:
                ax.plot(x, [row[f"pred_{p}"] for row in rows], color=COL[p], ls=LS[p], lw=2, label=LAB[p])
                if p in resid_preds:
                    axr.errorbar(x + (resid_preds.index(p) - 0.5) * 0.0015 * (x.max() - x.min()) / 0.2,
                                 [row[f"resid_{p}"] for row in rows], yerr=e, fmt="o", ms=4,
                                 color=COL[p], lw=1, capsize=0)
            ax.errorbar(x, m, yerr=e, fmt="o", ms=5, color=INK, lw=1, label="measured (NPE)", zorder=5)
            axr.axhline(0, color=INK2, lw=0.8)
            ax.set_ylabel(f"{lev}% coverage of ${comp[0]}_{comp[1]}$", fontsize=10)
            axr.set_ylabel("measured − pred.", fontsize=9)
            style(ax)
            style(axr)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=False, loc="lower center")
        axes[-1, c].set_xlabel(xlabel.format(k=comp[1]), fontsize=10)
    fig.tight_layout()
    save(fig, name)


def fig_theory():
    """Coverage vs true g for single-width populations: Gaussian formula vs uniform toy."""
    g = np.linspace(-0.1, 0.1, 201)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    shades = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for j, ratio in enumerate((0.3, 0.6, 0.8, 0.95)):
        sp = ratio * A.SIGMA_PI
        ax[0].plot(g, A.gaussian_prediction(g, np.full_like(g, sp), 0.68), color=shades[j], lw=2,
                   label=fr"$\sigma_{{\rm post}}/\sigma_\pi={ratio}$")
        # uniform toy: sigma_L giving the same Gaussian-model sigma_post
        sl = sp * A.SIGMA_PI / np.sqrt(A.SIGMA_PI**2 - sp**2)
        u = A.toy_pits(g, np.full_like(g, sl), 4000, rng)
        ax[1].plot(g, A.coverage_from_pits(u, 0.68), color=shades[j], lw=2,
                   label=fr"$\sigma_L={sl:.3f}$")
    for a, t in zip(ax, ("Gaussian prior $N(0,\\sigma_\\pi^2)$ (formula)",
                         "uniform prior $U(-0.1,0.1)$ (Monte Carlo)")):
        a.axhline(0.68, color=INK2, lw=0.8, ls=(0, (2, 2)))
        a.set_xlabel("true $g_k$")
        a.set_title(t, fontsize=10)
        a.legend(fontsize=8, frameon=False)
        style(a)
    ax[0].set_ylabel("P(truth inside 16–84% interval)")
    fig.tight_layout()
    save(fig, "theory_coverage_vs_g.png")


def fig_terciles(R):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for r, comp in enumerate(A.COMPONENTS):
        T = R["terciles"][comp]
        for j in range(3):
            rows = T["68_vs_gk_by_tercile"][j]["rows"]
            ax = axes[r, j]
            x = np.array([row["center"] for row in rows])
            ax.axhline(0.68, color=INK2, lw=0.8, ls=(0, (2, 2)))
            for p in ("self", "gauss", "toy"):
                ax.plot(x, [row[f"pred_{p}"] for row in rows], color=COL[p], ls=LS[p], lw=2, label=LAB[p])
            ax.errorbar(x, [row["measured"] for row in rows], yerr=[row["measured_se"] for row in rows],
                        fmt="o", ms=4, color=INK, lw=1, label="measured")
            lo, hi = T["sigma_edges"][j], T["sigma_edges"][j + 1]
            ax.set_title(f"${comp[0]}_{comp[1]}$, {['narrow', 'middle', 'wide'][j]} third: "
                         f"$\\sigma_{{\\rm post}}\\in[{lo:.3f},{hi:.3f}]$", fontsize=9)
            style(ax)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
            if r == 1:
                ax.set_xlabel("true $g_k$")
        axes[r, 0].set_ylabel("68% coverage")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower center")
    fig.tight_layout()
    save(fig, "coverage_vs_gk_by_width_tercile.png")


def fig_datacond(R):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for c, comp in enumerate(A.COMPONENTS):
        ax = axes[c]
        for lev, col in (("68", "#2a78d6"), ("95", "#eb6834")):
            rows = R["data_conditioned"][f"{comp}_{lev}_vs_ghat"]["rows"]
            x = np.array([row["center"] for row in rows])
            ax.errorbar(x, [row["resid_nominal"] for row in rows], yerr=[row["measured_se"] for row in rows],
                        fmt="o", ms=4, color=col, lw=1, label=f"{lev}% interval")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"posterior mean $\\hat g_{comp[1]}$ (data)")
        style(ax)
    axes[0].set_ylabel("coverage − nominal")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "coverage_vs_posterior_mean.png")


def fmtp(x, nd=1):
    return f"{100 * x:.{nd}f}"


def split_table(R):
    head = ("component", "level", "slice", "N", "measured (%)", "NPE self-consistency", "Gaussian formula",
            "uniform toy", "uniform toy (alt)")
    out = []
    for comp in A.COMPONENTS:
        for lev in ("68", "95"):
            for row in R["split"][f"{comp}_{lev}"]:
                cells = [comp, lev + "%", html.escape(row["label"]), str(row["n"]),
                         f"{fmtp(row['measured'], 2)} ± {fmtp(row['measured_se'], 2)}"]
                for p in ("self", "gauss", "toy", "toy_alt"):
                    cells.append(f"{fmtp(row['pred_' + p], 2)} ({100 * row['resid_' + p]:+.2f}, "
                                 f"{row['pull_' + p]:+.1f}σ)")
                out.append(cells)
    return head, out


def to_html_table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def embed(name, cap):
    import base64
    data = base64.b64encode(FIGS[name].read_bytes()).decode()
    return f"<figure><img src='data:image/png;base64,{data}' style='width:100%'><figcaption>{cap}</figcaption></figure>"


def main():
    R = json.loads((OUT / "report.json").read_text())
    if (OUT / "report.html").exists():
        raise SystemExit("report.html exists; refusing to overwrite")
    fig_theory()
    coverage_panels(R, "{comp}_{lev}_vs_gk", "true $g_{k}$", "coverage_vs_gk.png")
    coverage_panels(R, "{comp}_{lev}_vs_absg", "true $|g|$ (both components)", "coverage_vs_absg.png")
    fig_terciles(R)
    fig_datacond(R)

    chi_rows = []
    for comp in A.COMPONENTS:
        for lev in ("68", "95"):
            for kind in ("gk", "absg"):
                F = R["fine"][f"{comp}_{lev}_vs_{kind}"]
                chi_rows.append([comp, lev + "%", "g_k" if kind == "gk" else "|g|"] +
                                [f"{F['chi2_' + p]['chi2']:.0f} / {F['chi2_' + p]['nbins']}"
                                 for p in ("self", "gauss", "toy", "toy_alt")] +
                                [f"{F['chi2_self_null']['chi2']:.0f} / {F['chi2_self_null']['nbins']}"])
    terc_rows = []
    for comp in A.COMPONENTS:
        T = R["terciles"][comp]
        for row in T["68_overall"] + T["68_x_split"]:
            terc_rows.append([comp, html.escape(row["label"]), row["n"],
                              f"{fmtp(row['measured'], 2)} ± {fmtp(row['measured_se'], 2)}"] +
                             [f"{fmtp(row['pred_' + p], 2)} ({row['pull_' + p]:+.1f}σ)"
                              for p in ("self", "gauss", "toy")])
    dc = []
    for comp in A.COMPONENTS:
        for lev in ("68", "95"):
            c2 = R["data_conditioned"][f"{comp}_{lev}_vs_ghat"]["chi2_nominal"]
            dc.append([comp, lev + "%", f"{c2['chi2']:.0f} / {c2['nbins']}", f"{c2['max_abs_pull']:.1f}"])
    ti = R["toy_info"]
    rp = R["reproduction"]
    sh, sr = split_table(R)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>D: local calibration</title>
<style>body{{font:15px/1.5 ui-sans-serif,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:.6rem 0}}th,td{{border-bottom:1px solid #ccc;padding:.25rem .4rem;text-align:left}}
figcaption{{font-size:13px;color:#444}}code{{font-size:90%}}</style></head><body>
<h1>D: coverage conditional on the true shear</h1>
<p>Model <code>CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair</code>, Xu3 test set
({R['n_galaxies']} galaxies), TF-weighted posteriors, equal galaxy weights. Intervals: TF-weighted 16–84% and
2.5–97.5% quantiles. |g| = hypot(g1, g2) of the truth. σ_π = {R['sigma_pi']:.4f}.</p>
<h2>Reproduction of likelihood-stack/02_coverage</h2>
<p>|g|&lt;0.05: N = {rp['n_inner']} (ref {rp['reference_n_inner']}); g1 {fmtp(rp['g1_inner'], 2)}% (ref {fmtp(rp['g1_inner_reference'], 2)}),
g2 {fmtp(rp['g2_inner'], 2)}% (ref {fmtp(rp['g2_inner_reference'], 2)}); |g|&gt;0.05: g1 {fmtp(rp['g1_outer'], 2)}%, g2 {fmtp(rp['g2_outer'], 2)}%.
Max |difference| = {rp['max_abs_diff']:.2e}.</p>
<h2>(a) The |g| split</h2>
<p>Each prediction cell: predicted coverage (%), then (measured − predicted in percentage points, pull in units of the
measured binomial SE).</p>
{to_html_table(sh, sr)}
<h2>(b) Fine bins</h2>
{embed('coverage_vs_gk.png', 'Coverage vs true g_k (20 bins). Black: measured with binomial errors. Lines: predictions. Lower panels: measured − predicted for the self-consistency (blue) and uniform toy (green); the Gaussian residuals are large and can be read off the upper panels.')}
{embed('coverage_vs_absg.png', 'Same vs true |g| (14 bins).')}
<p>χ² of (measured − predicted) over bins, using the measured binomial SE; last column uses the exact-posterior null SE of the self-consistency residual.</p>
{to_html_table(['comp', 'level', 'binned in', 'self', 'Gaussian', 'toy', 'toy alt', 'self (null SE)'], chi_rows)}
<h2>(c) Width terciles (68%)</h2>
{to_html_table(['comp', 'subset', 'N', 'measured (%)', 'self', 'Gaussian', 'toy'], terc_rows)}
{embed('coverage_vs_gk_by_width_tercile.png', '68% coverage vs true g_k in thirds of the posterior width.')}
<h2>Model-free data-conditioned check</h2>
{embed('coverage_vs_posterior_mean.png', 'Coverage minus nominal in 20 bins of the posterior mean (a function of the data only).')}
{to_html_table(['comp', 'level', 'χ² vs nominal', 'max |pull|'], dc)}
<h2>Theory illustration</h2>
{embed('theory_coverage_vs_g.png', 'Coverage vs true g for single-width populations.')}
<h2>Toy bookkeeping</h2>
<pre>{html.escape(json.dumps(ti, indent=1))}</pre>
<p>Explanatory page: <code>paper-tests/tests/local-calibration.html</code> (if rendered). Scripts:
<code>{A.__file__}</code>, <code>d_localcal_stream.py</code>, <code>d_localcal_report.py</code>.</p>
</body></html>"""
    (OUT / "report.html").write_text(page)
    print("wrote", OUT / "report.html", list(FIGS))


if __name__ == "__main__":
    main()
