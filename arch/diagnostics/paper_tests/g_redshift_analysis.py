#!/usr/bin/env python3
"""G: how the NPE responds to the same galaxies rendered at other redshifts.

Reads the per-seed npz files from ``g_redshift_sample.py`` and writes
report.json, report.html and PNG figures to
$S/reports/paper-tests/G_redshift/<model>/.

All comparisons are paired: the same base galaxy, the same noise seed, and the
same flow seed at every redshift; z = 0.30 (the training redshift) is the
reference. Errors are from 1000 bootstrap resamples of the base galaxies.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis"
Z_REF = 0.30
NBOOT = 1000
LABELS = {"g1": r"$g_1$", "g2": r"$g_2$", "theta_int": r"$\theta_{\rm int}$",
          "cosi": r"$\cos i$", "v0": r"$v_0$", "vcirc": r"$v_{\rm circ}$",
          "rscale": r"$r_{\rm scale}$", "hlr": r"$r_{\rm hlr}$",
          "halpha_flux_true": r"$\log_{10}F_{\rm H\alpha}$"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1b1f24", "#5c6770"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#d9dee3", "grid.linewidth": 0.6, "savefig.dpi": 150})


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def load(cache):
    files = sorted(cache.glob("seed*.npz"), key=lambda p: int(p.stem[4:]))
    if not files:
        raise FileNotFoundError(f"no seed*.npz in {cache}")
    first = np.load(files[0])
    names = [str(n) for n in first["feature_names"]]
    z = first["redshift"]
    base = first["base_id"]
    zs = np.unique(z)
    nb = int(base.max()) + 1
    order = np.lexsort((base, z))  # (z, base) order
    if not (np.array_equal(z[order].reshape(len(zs), nb)[:, 0], zs)
            and np.array_equal(base[order].reshape(len(zs), nb)[0], np.arange(nb))):
        raise ValueError("rows do not form a full redshift x base grid")

    def grid(a):
        return a[order].reshape(len(zs), nb, *a.shape[1:])

    out = {k: [] for k in ("lo", "mu", "hi", "sd", "ess")}
    for f in files:
        d = np.load(f)
        if not np.array_equal(d["redshift"], z) or not np.array_equal(d["base_id"], base):
            raise ValueError(f"{f.name}: row layout differs from seed 0")
        out["lo"].append(grid(d["summary"][:, 0]))
        out["mu"].append(grid(d["summary"][:, 1]))
        out["hi"].append(grid(d["summary"][:, 2]))
        out["sd"].append(grid(d["sd"]))
        out["ess"].append(grid(d["ess_fraction"]))
        truth = grid(d["truth"])
    arrays = {k: np.stack(v) for k, v in out.items()}  # (seed, z, base, ...)
    arrays["truth"] = truth  # (z, base, param); identical across seeds
    if not np.allclose(truth, truth[:1]):
        raise ValueError("truth differs across redshifts for the same base")
    samples = dict(samples=grid(first["samples"]), log_weight=grid(first["log_weight"]),
                   kept=[str(n) for n in first["kept_names"]]) if "samples" in first else None
    return names, zs, arrays, samples, len(files)


def bootstrap(stat, nb, rng):
    """Value of stat(all bases) and its bootstrap sd over base resamples."""

    full = stat(np.arange(nb))
    draws = np.stack([stat(rng.integers(0, nb, nb)) for _ in range(NBOOT)])
    return full, draws.std(axis=0)


def analyse(names, zs, a, rng):
    nseed, nz, nb, npar = a["mu"].shape
    k0 = int(np.argmin(np.abs(zs - Z_REF)))
    th = names.index("theta_int")
    mu, sd, lo, hi, truth = a["mu"], a["sd"], a["lo"], a["hi"], a["truth"]

    diff = mu - mu[:, k0:k0 + 1]
    diff[..., th] = wrap(diff[..., th])
    shift_sig = diff / sd[:, k0:k0 + 1]  # paired shift in units of the z=0.30 sigma
    width = sd / sd[:, k0:k0 + 1]

    resid = mu - truth[None]
    resid[..., th] = wrap(resid[..., th])
    covered = (lo <= truth[None]) & (hi >= truth[None])
    # theta_int: compare offsets from the circular posterior centre.
    centre = mu[..., th]
    t_off = wrap(truth[None][..., th] - centre)
    covered[..., th] = (wrap(lo[..., th] - centre) <= t_off) & (t_off <= wrap(hi[..., th] - centre))

    res = {"redshifts": zs.tolist(), "reference_z": float(zs[k0]), "n_seeds": nseed,
           "n_bases": nb, "parameters": {}}
    for p, name in enumerate(names):
        s, e = bootstrap(lambda b: shift_sig[:, :, b, p].mean(axis=(0, 2)), nb, rng)
        w, we = bootstrap(lambda b: np.median(width[:, :, b, p], axis=(0, 2)), nb, rng)
        r, re = bootstrap(lambda b: (resid[:, :, b, p] / sd[:, :, b, p]).mean(axis=(0, 2)), nb, rng)
        c, ce = bootstrap(lambda b: covered[:, :, b, p].mean(axis=(0, 2)), nb, rng)
        # Signed shifts cancel for parameters symmetric about zero (v0, g), so
        # also report the RMS shift and the slope of mu(z) against mu(0.30).
        q, qe = bootstrap(lambda b: np.sqrt((shift_sig[:, :, b, p] ** 2).mean(axis=(0, 2))), nb, rng)

        def slope(b, p=p):
            x = mu[:, k0, b, p].ravel()
            return np.array([np.polyfit(x, mu[:, k, b, p].ravel(), 1)[0] for k in range(nz)])

        if p != th:
            sl, sle = bootstrap(slope, nb, rng)
        else:
            sl, sle = np.full(nz, np.nan), np.full(nz, np.nan)
        res["parameters"][name] = dict(
            paired_shift_sigma=s.tolist(), paired_shift_sigma_err=e.tolist(),
            width_ratio_median=w.tolist(), width_ratio_err=we.tolist(),
            bias_sigma=r.tolist(), bias_sigma_err=re.tolist(),
            coverage68=c.tolist(), coverage68_err=ce.tolist(),
            rms_shift_sigma=q.tolist(), rms_shift_sigma_err=qe.tolist(),
            slope_vs_ref=sl.tolist(), slope_vs_ref_err=sle.tolist())
    # Velocity scale: median of mu_v(z)/mu_v(0.30) against the naive (1+z)/1.3.
    v = names.index("vcirc")
    ratio = mu[..., v] / mu[:, k0:k0 + 1, :, v]
    vr, vre = bootstrap(lambda b: np.median(ratio[:, :, b], axis=(0, 2)), nb, rng)
    res["vcirc_ratio_median"], res["vcirc_ratio_err"] = vr.tolist(), vre.tolist()
    res["naive_pixel_scale"] = ((1 + zs) / (1 + zs[k0])).tolist()

    # Shear: ghat = (1+m) g + c per redshift, both components pooled.
    def mc(b):
        out = np.empty((nz, 2))
        for k in range(nz):
            g = np.concatenate([np.repeat(truth[k, b, j][None], nseed, 0).ravel() for j in (0, 1)])
            gh = np.concatenate([mu[:, k, b, j].ravel() for j in (0, 1)])
            slope, icpt = np.polyfit(g, gh, 1)
            out[k] = slope - 1.0, icpt
        return out

    full, err = bootstrap(mc, nb, rng)
    dm, dme = bootstrap(lambda b: mc(b)[:, 0] - mc(b)[k0, 0], nb, rng)
    res["shear"] = dict(m=full[:, 0].tolist(), m_err=err[:, 0].tolist(),
                        c=full[:, 1].tolist(), c_err=err[:, 1].tolist(),
                        delta_m=dm.tolist(), delta_m_err=dme.tolist())
    res["ess_fraction_median"] = np.median(a["ess"], axis=(0, 2)).tolist()
    return res, k0


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grid(res, names, key, err, ylabel, ref, path):
    zs = np.array(res["redshifts"])
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.5), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        d = res["parameters"][name]
        ax.errorbar(zs, d[key], yerr=d[err], fmt="o-", color=C1, ms=4, lw=1, capsize=2)
        ax.axhline(ref, color=MUTED, lw=0.8, ls="--")
        ax.axvline(res["reference_z"], color=C2, lw=0.6, alpha=0.5)
        if key == "slope_vs_ref":
            ax.plot(zs, res["naive_pixel_scale"], color=C2, lw=0.8, ls=":")
        ax.set_title(LABELS.get(name, name))
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("source redshift $z$")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return savefig(fig, path)


def plot_velocity(res, path):
    zs = np.array(res["redshifts"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.errorbar(zs, res["vcirc_ratio_median"], yerr=res["vcirc_ratio_err"], fmt="o-",
                color=C1, ms=4, label=r"median $\hat v_{\rm circ}(z)/\hat v_{\rm circ}(0.3)$")
    ax.plot(zs, res["naive_pixel_scale"], color=C2, ls="--",
            label=r"pixel-scale prediction $(1+z)/1.3$")
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("source redshift $z$")
    ax.set_ylabel(r"$v_{\rm circ}$ ratio")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return savefig(fig, path)


def plot_shear(res, path):
    zs = np.array(res["redshifts"])
    sh = res["shear"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].errorbar(zs, sh["delta_m"], yerr=sh["delta_m_err"], fmt="o-", color=C1, ms=4)
    axes[0].set_ylabel(r"$m(z) - m(0.3)$ (paired)")
    axes[1].errorbar(zs, sh["m"], yerr=sh["m_err"], fmt="o-", color=C1, ms=4)
    axes[1].set_ylabel(r"$m$ (posterior mean)")
    axes[2].errorbar(zs, sh["c"], yerr=sh["c_err"], fmt="o-", color=C1, ms=4)
    axes[2].set_ylabel(r"$c$")
    for ax in axes:
        ax.axhline(0.0, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("source redshift $z$")
    fig.tight_layout()
    return savefig(fig, path)


def contour(ax, x, y, w, color, extent):
    h, xe, ye = np.histogram2d(x, y, bins=60, range=extent, weights=w)
    h = gaussian_filter(h, 1.2)
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    levels = sorted({flat[np.searchsorted(cum, q)] for q in (0.95, 0.68)})
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, h.T, levels=levels, colors=color, linewidths=1.2)


def plot_example(samples, zs, truth, names, base, path):
    kept = samples["kept"]
    show = [int(np.argmin(np.abs(zs - z))) for z in (zs.min(), Z_REF, zs.max())]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for (ix, iy), ax in zip(((0, 1), (2, 3)), axes):
        allx = np.concatenate([samples["samples"][k, base, :, ix] for k in show])
        ally = np.concatenate([samples["samples"][k, base, :, iy] for k in show])
        extent = [np.percentile(allx, [0.5, 99.5]), np.percentile(ally, [0.5, 99.5])]
        for k, color in zip(show, (C1, INK, C2)):
            lw = samples["log_weight"][k, base]
            w = np.exp(lw - lw.max())
            contour(ax, samples["samples"][k, base, :, ix], samples["samples"][k, base, :, iy],
                    w, color, extent)
            ax.plot([], [], color=color, label=f"z = {zs[k]:.2f}")
        ax.plot(truth[base, names.index(kept[ix])], truth[base, names.index(kept[iy])],
                "k+", ms=10, mew=1.5)
        ax.set_xlabel(LABELS[kept[ix]])
        ax.set_ylabel(LABELS[kept[iy]])
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return savefig(fig, path)


def img(path):
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def table(res, names, key, err, fmt):
    zs = res["redshifts"]
    head = "".join(f"<th>{z:.2f}</th>" for z in zs)
    rows = []
    for name in names:
        d = res["parameters"][name]
        cells = "".join(f"<td>{fmt.format(v)}&plusmn;{fmt.format(e)}</td>"
                        for v, e in zip(d[key], d[err]))
        rows.append(f"<tr><th>{html.escape(name)}</th>{cells}</tr>")
    return f"<table><tr><th>z</th>{head}</tr>{''.join(rows)}</table>"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--example-base", type=int, default=0)
    p.add_argument("--cache", type=Path, default=None, help="default: $S/cache/paper_tests_G_redshift/<model>")
    p.add_argument("--out", type=Path, default=None, help="default: $S/reports/paper-tests/G_redshift/<model>")
    args = p.parse_args()
    cache = args.cache or S / "cache/paper_tests_G_redshift" / args.model
    out = args.out or S / "reports/paper-tests/G_redshift" / args.model
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)

    names, zs, arrays, samples, nseed = load(cache)
    res, k0 = analyse(names, zs, arrays, rng)
    res["model"] = args.model
    figs = {
        "shift": plot_grid(res, names, "paired_shift_sigma", "paired_shift_sigma_err",
                           r"$\langle\Delta\hat\mu\rangle/\sigma_{0.3}$", 0.0, out / "paired_shift.png"),
        "width": plot_grid(res, names, "width_ratio_median", "width_ratio_err",
                           r"$\sigma(z)/\sigma(0.3)$", 1.0, out / "width_ratio.png"),
        "coverage": plot_grid(res, names, "coverage68", "coverage68_err",
                              "68% coverage", 0.68, out / "coverage.png"),
        "bias": plot_grid(res, names, "bias_sigma", "bias_sigma_err",
                          r"$\langle(\hat\mu-\theta)/\sigma\rangle$", 0.0, out / "bias.png"),
        "slope": plot_grid(res, [n for n in names if n != "theta_int"], "slope_vs_ref", "slope_vs_ref_err",
                           r"slope of $\hat\mu(z)$ vs $\hat\mu(0.3)$", 1.0, out / "slope.png"),
        "rms": plot_grid(res, names, "rms_shift_sigma", "rms_shift_sigma_err",
                         r"RMS $\Delta\hat\mu/\sigma_{0.3}$", 0.0, out / "rms_shift.png"),
        "velocity": plot_velocity(res, out / "vcirc_ratio.png"),
        "shear": plot_shear(res, out / "shear.png"),
    }
    if samples is not None:
        figs["example"] = plot_example(samples, zs, arrays["truth"][0], names,
                                       args.example_base, out / "example_posteriors.png")
    (out / "report.json").write_text(json.dumps(res, indent=1))

    sh = res["shear"]
    shear_rows = "".join(
        f"<tr><th>{z:.2f}</th><td>{dm:+.4f}&plusmn;{dme:.4f}</td><td>{m:+.4f}&plusmn;{me:.4f}</td>"
        f"<td>{c:+.5f}&plusmn;{ce:.5f}</td></tr>"
        for z, dm, dme, m, me, c, ce in zip(zs, sh["delta_m"], sh["delta_m_err"], sh["m"],
                                           sh["m_err"], sh["c"], sh["c_err"]))
    body = f"""
<h1>Redshift response of the NPE</h1>
<p class="meta">Model <code>{html.escape(args.model)}</code>. {res['n_bases']} Xu3 galaxies
(hlr &ge; 1&Prime;, TF v<sub>circ</sub>, measured-axis fibers), each rendered at
{len(zs)} redshifts with every other input fixed; {nseed} paired noise realizations;
10<sup>4</sup> identity+R90 samples, TF-weighted. Reference: z = {zs[k0]:.2f}, the
training redshift. Errors: bootstrap over galaxies.</p>
<p>What changes with z: the H&alpha; window moves with the line, so velocities and the
line width (fixed 0.05 nm rest frame) span (1+z)/1.3 times as many pixels, and the
continuum under H&alpha; changes because m<sub>r</sub> is fixed. The image and fiber
positions are identical at every z.</p>
<h2>Velocity scale</h2>{img(figs['velocity'])}
<h2>Shear</h2>{img(figs['shear'])}
<table><tr><th>z</th><th>&Delta;m (paired)</th><th>m</th><th>c</th></tr>{shear_rows}</table>
<h2>Paired shift of the posterior mean (units of the z = 0.30 posterior sd)</h2>
{img(figs['shift'])}{table(res, names, 'paired_shift_sigma', 'paired_shift_sigma_err', '{:+.2f}')}
<h2>Slope of the posterior mean against z = 0.30 (1 = no change; (1+z)/1.3 = pixel-scale misreading)</h2>
{img(figs['slope'])}{table(res, [n for n in names if n != 'theta_int'], 'slope_vs_ref', 'slope_vs_ref_err', '{:.3f}')}
<h2>RMS paired shift (units of the z = 0.30 posterior sd)</h2>
{img(figs['rms'])}{table(res, names, 'rms_shift_sigma', 'rms_shift_sigma_err', '{:.2f}')}
<h2>Posterior width ratio &sigma;(z)/&sigma;(0.30) (median)</h2>
{img(figs['width'])}{table(res, names, 'width_ratio_median', 'width_ratio_err', '{:.2f}')}
<h2>Bias against truth (units of posterior sd)</h2>{img(figs['bias'])}
<h2>68% coverage</h2>
{img(figs['coverage'])}{table(res, names, 'coverage68', 'coverage68_err', '{:.2f}')}
"""
    if "example" in figs:
        body += f"<h2>Example galaxy (base {args.example_base}, noise seed 0)</h2>{img(figs['example'])}"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Redshift response</title><style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1b1f24;background:#fff}}
img{{max-width:100%}} table{{border-collapse:collapse;font-size:12px;margin:8px 0 20px;display:block;overflow-x:auto}}
td,th{{border:1px solid #d9dee3;padding:3px 6px;text-align:right}} .meta{{color:#5c6770}}
</style></head><body>{body}</body></html>"""
    (out / "report.html").write_text(page)
    print(f"wrote {out}/report.html")
    print(json.dumps({"shear_delta_m": sh["delta_m"], "vcirc_ratio": res["vcirc_ratio_median"]}))


if __name__ == "__main__":
    main()
