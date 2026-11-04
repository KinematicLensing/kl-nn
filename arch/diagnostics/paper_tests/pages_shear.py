"""Explanatory pages for the Sec. 6 shear-estimation tests (Group A shear).

Seven pages, rendered by ``write_test_pages.py``. Every number comes from the
Group-A JSON files through ``J``; nothing is retyped. Prose blocks that contain
LaTeX are written as ``string.Template`` strings (``$name`` placeholders), so
LaTeX braces need no escaping.
"""
from __future__ import annotations

import html
from string import Template

import numpy as np
from scipy.stats import chi2 as _chi2

from test_pages_common import J, SCRIPTS, fig, page, pct, table
from test_pages_common import f as _f_common


def f(x, nd=3, sign=False):
    """Like the common ``f`` but stays in fixed-point unless |x| < 10^-(nd+1) (e.g. m = 0.0004 -> +0.000)."""
    x = float(x)
    if x != 0 and abs(x) < 10 ** (-(nd + 1)):
        return _f_common(x, nd, sign)
    s = f"{x:.{nd}f}" if abs(x) < 1e5 else f"{x:.{max(nd - 1, 1)}e}"
    if sign and x >= 0:
        s = "+" + s
    return s.replace("-", "−")


def pm(x, e, nd=3, sign=False):
    return f"{f(x, nd, sign)} ± {f(e, nd)}"

SECTION = "Sec. 6: Shear estimation"
DIAG = SCRIPTS.rsplit("/", 1)[0]
REPORT_LINK = "<a href='../A_shear/report.html'>Group A shear report (raw agent report)</a>"


# ----------------------------------------------------------------- helpers
def T(s: str, **kw) -> str:
    """Fill ``$name`` placeholders; raises KeyError on a missing one."""
    return Template(s).substitute(**kw)


def R(*k):
    return J("shear", *k)


def S(*k):
    return J("supp", *k)


def S2(*k):
    return J("supp2", *k)


def e4(x, nd=2, sign=False):
    """Format a value in units of 1e-4 (always fixed-point)."""
    s = f"{1e4 * float(x):{'+' if sign else ''}.{nd}f}"
    return s.replace("-", "−")


def pm4(x, e, nd=2):
    return f"{e4(x, nd, True)} ± {e4(e, nd)}"


def lnk(slug, text):
    return f"<a href='{slug}.html'>{text}</a>"


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def ol(items):
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def jobs_for(*scripts):
    """SLURM job lines whose description names one of the given scripts."""
    jobs = J("shear_report", "provenance", "jobs")
    out = [f"<code>{jid}</code>: {html.escape(desc)}" for jid, desc in jobs.items()
           if any(desc.split(":")[0] == s + ".slurm" for s in scripts)]
    if not out:
        raise KeyError(f"no job found for {scripts}")
    return out


def provenance(scripts, jobs, keys, extra=""):
    sc = [f"<code>{s if s.startswith('/') else SCRIPTS + '/' + s}</code>" for s in scripts]
    return (
        "<p><b>Scripts.</b></p>" + ul(sc)
        + "<p><b>SLURM jobs</b> (Bridges-2 RM-shared, CPU):</p>" + ul(jobs_for(*jobs))
        + "<p><b>JSON keys</b> (files in <code>/ocean/projects/phy250048p/shared/reports/paper-tests/A_shear/</code>):</p>"
        + ul([f"<code>{html.escape(k)}</code>" for k in keys])
        + extra
        + f"<p>Raw report with all tables: {REPORT_LINK}.</p>"
    )


def est(kind, thr=None):
    """Find a T5 estimator entry by kind ('raw'/'weighted') and cut threshold (None = no cut)."""
    for name, val in R("T5", "estimators").items():
        if val["kind"] != kind:
            continue
        if thr is None and name == kind:
            return name, val
        if thr is not None and "<" in name and abs(float(name.split("<")[1]) - thr) < 1e-9:
            return name, val
    raise KeyError(f"T5 estimator {kind} {thr}")


def est_cut_value(name):
    return name.split("<")[1] if "<" in name else None


def r6row(t, rows=None):
    rows = R("R6", "rows") if rows is None else rows
    for r in rows:
        if abs(r["threshold"] - t) < 1e-9:
            return r
    raise KeyError(f"R6 threshold {t}")


def chi_fmt(c):
    return f"{f(c['chi2'], 1)} / {c['ndof']}"


def pval(c):
    p = float(_chi2.sf(c["chi2"], c["ndof"]))
    if p < 1e-6:
        return "p &lt; 10<sup>−6</sup>"
    return f"p = {p:.1e}".replace("-", "−") if p < 1e-2 else f"p = {p:.2f}"


# Symbols used in prose
SYM = dict(
    SE=r"\(\sigma_{\epsilon,i}\)",
    SPI=r"\(\sigma_\pi\)",
    gh=r"\(\hat g\)",
)

COMP_LABEL = {"g1": r"\(g_1\)", "g2": r"\(g_2\)", "g+": r"\(g_+\)", "gx": r"\(g_\times\)"}
M_LABEL = {"g1": r"\(m_1\)", "g2": r"\(m_2\)", "g+": r"\(m_+\)", "gx": r"\(m_\times\)"}
COMPS = ["g1", "g2", "g+", "gx"]

LOWG = R("low_g")
SIGPI = R("sigma_pi")
NGAL = R("n_galaxies")
NDRAW = J("shear_report", "provenance", "extract", "n_draws")

GALFRAME_DEF = r"""
<p><b>Image frame and galaxy frame.</b> \((g_1, g_2)\) are measured relative to the
pixel grid (the image frame). For diagnostics it helps to rotate them into a frame
tied to each galaxy's true intrinsic position angle \(\theta_{\rm int}\) (known in the
simulation):
\[ g_+ = g_1\cos 2\theta_{\rm int} - g_2 \sin 2\theta_{\rm int}, \qquad
   g_\times = g_1 \sin 2\theta_{\rm int} + g_2 \cos 2\theta_{\rm int} \]
(with the simulator's clockwise sign convention for \(\theta_{\rm int}\); code:
<code>utils.img_to_gal_axis</code>). \(g_+\) stretches or squashes the disk along its own
major and minor axes, and \(g_\times\) shears it at 45° to them. In the usual KL picture the
two are constrained differently: \(g_+\) mainly changes the apparent axis ratio, which is
degenerate with inclination until the Tully–Fisher relation breaks it, while \(g_\times\)
twists the velocity field relative to the image. The galaxy frame needs the true
\(\theta_{\rm int}\), so it is a diagnostic only, not something one could measure on real data.
Because \(\theta_{\rm int}\) is uniformly distributed, a galaxy-frame response
\(\mathrm{diag}(1+m_+, 1+m_\times)\) averages in the image frame to
\(m_1 \approx m_2 \approx (m_+ + m_\times)/2\).</p>"""


# =================================================================== page 1
def page_raw():
    raw = {c: R("R_raw", c) for c in COMPS}
    _, rawb = est("raw")
    rawb = rawb["components"]
    se = R("R_raw", "sigma_eps")
    p = se["percentiles_5_16_50_84_95"]
    t5raw = est("raw")[1]
    dpx = raw["g+"]["low_m"] - raw["gx"]["low_m"]
    dpx_se = float(np.hypot(raw["g+"]["low_m_se"], raw["gx"]["low_m_se"]))
    avg_px = 0.5 * (raw["g+"]["low_m"] + raw["gx"]["low_m"])
    checks = J("shear_report", "provenance", "extract", "checks")

    rows = []
    for c in COMPS:
        r, b = raw[c], rawb[c]
        rows.append([COMP_LABEL[c], pm(r["low_m"], r["low_m_se"], 3, True), f(b["m_boot_se"], 3),
                     pm4(r["low_c"], r["low_c_se"]), e4(b["c_boot_se"]),
                     pm4(r["c"], r["c_se"]), f"{r['n_low']:,}"])
    tab1 = table(["component", r"\(m\) ± fit error", "bootstrap error on \\(m\\)",
                  r"\(10^4\,c\) (intercept) ± fit error", r"bootstrap error on \(10^4 c\)",
                  r"\(10^4\,\langle\hat g - g\rangle\) over all galaxies", "galaxies in fit window"], rows)
    tab2 = table(["quantity", "value"], [
        [r"mean of \(\sigma_{\epsilon,i}\)", pm(se["mean"], t5raw["sigma_eps_mean_boot_se"], 5) + " (bootstrap error)"],
        [r"RMS of \(\sigma_{\epsilon,i}\)", pm(se["rms"], t5raw["sigma_eps_rms_boot_se"], 5)],
        [r"median of \(\sigma_{\epsilon,i}\)", f(se["median"], 4)],
        ["5 / 16 / 50 / 84 / 95 percentiles", " / ".join(f(v, 4) for v in p)],
        [r"mean per-component widths \(\sigma_{g_1}\), \(\sigma_{g_2}\)",
         f"{f(se['per_component_mean_sigma_g1'], 4)}, {f(se['per_component_mean_sigma_g2'], 4)}"],
        [r"prior limit \(\sqrt2\,\sigma_\pi\)", f(se["prior_sigma_eps_equiv_sqrt2_sigma_pi"], 4)],
        [r"fraction of galaxies with \(\sigma_{\epsilon,i} > 0.075\)", pct(se["frac_sigma_eps_gt_0.075"])],
    ])

    question = T(r"""<p>If we take each galaxy's posterior mean of \((g_1, g_2)\) at face value,
with every galaxy weighted equally, the full test set, and no cuts, weighting, calibration or
stacking, what multiplicative bias \(m\) and additive bias \(c\) do we get? And how wide are the
per-galaxy posteriors, i.e. what is the per-galaxy shape noise $SE and the ensemble shape
noise?</p>""", **SYM)

    why = r"""<p>This is the opening result of Sec. 6 (Sec. 6.1). It is what the NPE gives straight
out of inference, and every mitigation later in the section (weighting, width cuts, response
calibration, stacking) is judged against it. The distribution of per-galaxy widths also sets
how much those mitigations can achieve, because the bias turns out to be controlled by the
width (next page).</p>"""

    theory = T(r"""
<p><b>Estimator.</b> For galaxy \(i\) the NPE gives $ND posterior draws \(g_{ik}\) of the
shear. Each draw carries an importance weight \(w_{ik}\) that converts the NPE's training prior
into the Tully–Fisher (TF) prior. The shear estimate is the weighted posterior mean
\[ \hat g_i = \frac{\sum_k w_{ik}\, g_{ik}}{\sum_k w_{ik}}, \]
computed separately for \(g_1\) and \(g_2\).</p>
<p><b>Bias model.</b> The standard linear model for shear bias is
\[ \hat g - g = m\,g + c , \]
where \(g\) is the true shear, \(m\) the multiplicative bias (the estimate responds to a
fraction \(1+m\) of the true shear) and \(c\) the additive bias. We fit it by least squares
over galaxies whose true shear component satisfies \(|g| < $LOWG\). The window is applied to
the component being fit: for \(m_1\) it keeps galaxies with \(|g_1| < $LOWG\), whatever
\(g_2\) is. Why a small window is used and whether it matters is the subject of
$LIN. The intercept of this fit is \(c\). As a second estimate of \(c\) we also give the
mean residual \(\langle \hat g - g\rangle\) over all galaxies; the true shears are symmetric
about zero, so the \(m\,g\) term averages out and this mean is also an estimate of \(c\).</p>
<p><b>Why a posterior mean should be biased low.</b> A posterior mean is pulled toward where
the prior puts its weight (here \(g=0\), the centre of the prior box) by an amount that grows
as the data get less informative. A galaxy whose data say almost nothing about \(g\) returns
roughly the prior mean, 0, whatever its true shear, i.e. \(m \approx -1\) for that galaxy.
The quantitative version of this "shrinkage" is on $SHR.</p>
$GF
<p><b>Shape noise.</b> Following Xu et al. (2024), the per-galaxy shape noise is the width of
the shear posterior,
\[ \sigma_{\epsilon,i} = \sqrt{\sigma_{g_1,i}^2 + \sigma_{g_2,i}^2}, \]
where \(\sigma_{g_1,i}\) and \(\sigma_{g_2,i}\) are the TF-weighted standard deviations of the
two 1D marginals. The ensemble shape noise is the mean (or the RMS; we give both) of
\(\sigma_{\epsilon,i}\) over galaxies. The prior on each component is uniform on
\([-0.1, 0.1]\), whose standard deviation is \(\sigma_\pi = 0.2/\sqrt{12} = $SPIV\). A galaxy
whose posterior is just the prior therefore has \(\sigma_{\epsilon,i} = \sqrt2\,\sigma_\pi =
$SQ2\). That value is the upper end of the useful range: a galaxy near it carries almost no
shear information.</p>""",
        ND=f"{NDRAW:,}", LOWG=f(LOWG, 2), LIN=lnk("bias-fit-linearity", "the linearity page"),
        SHR=lnk("gaussian-shrinkage", "the shrinkage page"), GF=GALFRAME_DEF,
        SPIV=f(SIGPI, 4), SQ2=f(se["prior_sigma_eps_equiv_sqrt2_sigma_pi"], 4))

    procedure = T(r"""<ol>
<li>One pass over the posterior cache (<code>a_shear_extract.py</code>) reduced the $ND
draws per galaxy for all $NG test galaxies to TF-weighted summaries of \((g_1,g_2)\): the
mean, both variances, the covariance and five weighted quantiles. The weights were normalized
within each galaxy. As a check, on $NC galaxies the variances and quantiles were recomputed
with the existing helper functions in <code>shear_bias_report.py</code>; the largest
differences were $VC (variance) and $QC (quantiles), and the means matched the cached TF means
to $MC.</li>
<li>For each component, fit \(\hat g - g = m g + c\) on the galaxies inside the
\(|g| < $LOWG\) window with equal weights, using the existing
<code>shear_bias_report.component_metrics</code>. Its error bars are "sandwich" errors (see
$BOOT).</li>
<li>Rotate the truth and the estimate into the galaxy frame with the true \(\theta_{\rm int}\),
and repeat the fit for \(g_+\) and \(g_\times\).</li>
<li>Get bootstrap errors by refitting on $NB resamples of the galaxies (details on $BOOT).</li>
<li>For the figure, bin \(\hat g - g\) in true \(g\) (bins of width 0.01) and plot the bin
means with their standard errors.</li>
<li>Histogram \(\sigma_{\epsilon,i}\) and take its mean, RMS and percentiles.</li>
</ol>""", ND=f"{NDRAW:,}", NG=f"{NGAL:,}", NC=f"{checks['n']:,}", VC=f(checks["var_maxabs"], 1),
        QC=f(checks["q_maxabs"], 1), MC=f(checks["mean_vs_cache_maxabs"], 1), LOWG=f(LOWG, 2),
        BOOT=lnk("error-bars-bootstrap", "the error-bar page"), NB=f"{R('T5', 'n_boot'):,}")

    results = (
        "<p><b>Bias of the raw posterior means</b> (equal weights, all "
        f"{NGAL:,} galaxies):</p>" + tab1
        + fig("A_shear/Rraw_binned_residual.png",
              r"Raw \(\hat g - g\) against the true shear, image frame (left) and galaxy frame (right). "
              r"Points are bin means ± standard error (bins of width 0.01). Lines are the linear fits made "
              r"inside the grey band \(|g| < 0.02\), extended over the whole range. The galaxy-frame components "
              r"reach \(|g| \approx 0.14\) because the square prior box \(|g_1|, |g_2| < 0.1\) has corners.")
        + "<p><b>Per-galaxy shape noise</b> \\(\\sigma_{\\epsilon,i}\\):</p>" + tab2
        + fig("A_shear/Rraw_sigma_hist.png",
              r"Histogram of the per-galaxy \(\sigma_{\epsilon,i}=\sqrt{\sigma_{g_1}^2+\sigma_{g_2}^2}\). "
              r"Vertical lines: the mean, the RMS, and the prior limit \(\sqrt2\,\sigma_\pi\) reached by a "
              r"galaxy whose posterior equals the prior.")
    )

    interpretation = T(r"""
<p><b>The raw estimates respond to only about $RESP of the true shear.</b> \(m_1 = $M1\) and
\(m_2 = $M2\) (bootstrap errors). The binned residuals in the figure lie on straight lines
across the whole prior range, so this is a clean multiplicative bias, not a nonlinearity (see
also $LIN).</p>
<p><b>The additive bias is consistent with zero</b> within errors of about $CERR:
\(10^4 c_1 = $C1\), \(10^4 c_2 = $C2\).</p>
<p><b>The galaxy frame shows an anisotropy.</b> \(m_+ = $MP\) and \(m_\times = $MX\), a
difference of $DPX. Treating the two errors as independent this is $NSIG standard deviations;
the same galaxies enter both fits, so the true significance may differ somewhat. The response
to shear along the galaxy's axes is weaker than to shear at 45°. In the image frame this averages
out: \((m_+ + m_\times)/2 = $AVG\), close to \(m_1\) and \(m_2\). The shrinkage model on the
next page explains \(m_1\) and \(m_2\) but not this split.</p>
<p><b>Many galaxies are close to prior-dominated.</b> The distribution of $SE piles up just
below the prior limit $SQ2: the median is $MED and $FR of galaxies have
\(\sigma_{\epsilon,i} > 0.075\). These galaxies carry little shear information but pull the
posterior means strongly toward zero, which is what drives \(m\) so negative. The ensemble
shape noise is $SMEAN (mean) or $SRMS (RMS).</p>""",
        RESP=pct(1 + raw["g1"]["low_m"], 0),
        M1=pm(raw["g1"]["low_m"], rawb["g1"]["m_boot_se"], 3, True),
        M2=pm(raw["g2"]["low_m"], rawb["g2"]["m_boot_se"], 3, True),
        LIN=lnk("bias-fit-linearity", "the linearity page"),
        C1=pm4(raw["g1"]["low_c"], rawb["g1"]["c_boot_se"]), C2=pm4(raw["g2"]["low_c"], rawb["g2"]["c_boot_se"]),
        CERR=r"\(" + e4(max(rawb["g1"]["c_boot_se"], rawb["g2"]["c_boot_se"]), 1) + r"\times10^{-4}\)",
        MP=pm(raw["g+"]["low_m"], rawb["g+"]["m_boot_se"], 3, True),
        MX=pm(raw["gx"]["low_m"], rawb["gx"]["m_boot_se"], 3, True),
        DPX=pm(dpx, dpx_se, 3, True), NSIG=f(abs(dpx) / dpx_se, 1), AVG=f(avg_px, 3, True),
        SE=SYM["SE"], SQ2=f(se["prior_sigma_eps_equiv_sqrt2_sigma_pi"], 4), MED=f(se["median"], 4),
        FR=pct(se["frac_sigma_eps_gt_0.075"], 0), SMEAN=f(se["mean"], 4), SRMS=f(se["rms"], 4))

    caveats = ul([
        "These numbers are for the current 100k-trained model. The paper model will be trained on 1M "
        "galaxies, so all values will change.",
        "Only the Xu3 (DESI BGS Any) test set is used.",
        r"The galaxy frame uses the true \(\theta_{\rm int}\); it is a diagnostic, not an observable estimator.",
        "The errors are statistical errors for this fixed set of test galaxies. They assume galaxies are "
        "independent and do not include systematic effects.",
        r"The \(m_+\) vs \(m_\times\) difference is not yet explained (see the shrinkage page).",
    ])

    prov = provenance(
        ["a_shear_extract.py", "a_shear_analysis.py (functions test_rraw, full_metrics)", "a_shear_report.py (re-renders the residual figure)"],
        ["a_shear_extract", "a_shear_analysis"],
        ["results.json: R_raw.{g1,g2,g+,gx}.{low_m, low_m_se, low_c, low_c_se, c, c_se, n_low}",
         "results.json: R_raw.sigma_eps.*", "results.json: R_raw.binned (figure data)",
         "results.json: T5.estimators.raw.components.*.{m_boot_se, c_boot_se}",
         "results.json: T5.estimators.raw.{sigma_eps_mean_boot_se, sigma_eps_rms_boot_se}",
         "report.json: provenance.extract.checks"],
        extra="<p>Per-galaxy table: <code>A_shear/per_galaxy.npz</code> (+ <code>.json</code> metadata).</p>")

    answer = (f"The raw posterior means respond to only about {pct(1 + raw['g1']['low_m'], 0)} of the true shear "
              f"(m1 = {pm(raw['g1']['low_m'], rawb['g1']['m_boot_se'], 3, True)}, "
              f"m2 = {pm(raw['g2']['low_m'], rawb['g2']['m_boot_se'], 3, True)}) with c consistent with zero, "
              f"and {pct(se['frac_sigma_eps_gt_0.075'], 0)} of galaxies have nearly prior-dominated posteriors.")
    return page(slug="raw-shear-bias", title="Raw shear bias of per-galaxy posterior means", code="R-raw",
                section=SECTION, answer=answer, question=question, why=why, theory=theory,
                procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 2
def page_shrinkage():
    raw = {c: R("R_raw", c) for c in COMPS}
    fs = R("T3", "full_sample_predictions")
    chi3, chi4 = R("T3", "chi2"), S("S4_chi2")
    bins3, bins4 = R("T3", "bins"), S("S4_T3_predicted_sigma_bins")
    s3 = S("S3_mechanism")
    ntoy = len(R("T3", "toy_grid", "s_lik"))
    nb3, nb4 = len(bins3), len(bins4)
    b30, b40 = bins3[0], bins4[0]

    theory = T(r"""
<p><b>The Gaussian model.</b> Take one shear component and suppose the data for a galaxy give
a Gaussian likelihood centred on a noisy value \(g_{\rm obs}\) with width \(\sigma_L\),
and the prior is a Gaussian of width \(\sigma_\pi\) centred on 0:
\[ p(g \mid {\rm data}) \propto
   \exp\!\left[-\frac{(g-g_{\rm obs})^2}{2\sigma_L^2}\right]
   \exp\!\left[-\frac{g^2}{2\sigma_\pi^2}\right]. \]
The product of two Gaussians is a Gaussian, with
\[ \text{posterior mean} = g_{\rm obs}\,\frac{\sigma_\pi^2}{\sigma_\pi^2+\sigma_L^2}, \qquad
   \sigma_{\rm post}^2 = \frac{\sigma_L^2\,\sigma_\pi^2}{\sigma_L^2+\sigma_\pi^2}. \]
If the likelihood is unbiased, then averaged over noise \(\langle g_{\rm obs}\rangle = g\), so the
average posterior mean is \(g\,\sigma_\pi^2/(\sigma_\pi^2+\sigma_L^2)\). The response is
therefore
\[ 1+m = \frac{\sigma_\pi^2}{\sigma_\pi^2+\sigma_L^2}
       = 1 - \frac{\sigma_L^2}{\sigma_L^2+\sigma_\pi^2}
       = 1 - \frac{\sigma_{\rm post}^2}{\sigma_\pi^2}, \]
where the last step uses the formula for \(\sigma_{\rm post}^2\). Hence
\[ m = -\frac{\sigma_L^2}{\sigma_L^2+\sigma_\pi^2} = -\frac{\sigma_{\rm post}^2}{\sigma_\pi^2}. \]
A well-measured galaxy (\(\sigma_L \ll \sigma_\pi\)) has \(m \to 0\); a galaxy whose data say
nothing (\(\sigma_L \gg \sigma_\pi\), so \(\sigma_{\rm post} \to \sigma_\pi\)) has \(m \to -1\).
For a sample, the fitted \(m\) is (to first order) the average of the per-galaxy values,
\(m \approx -\langle\sigma_{{\rm post},i}^2\rangle/\sigma_\pi^2\). The same algebra in two
dimensions, with an isotropic prior, gives the response matrix
\(\mathbf{1} - \Sigma_{\rm post}/\sigma_\pi^2\), so each component (image or galaxy frame)
should obey the 1D formula with its own posterior variance.</p>
<p><b>Which width goes in the formula.</b> The two forms of \(m\) above are equal, but they
use different widths. \(-\sigma^2/(\sigma^2+\sigma_\pi^2)\), the formula in the original test
plan, is right only if \(\sigma\) is the <i>likelihood</i> width \(\sigma_L\). What we have for
each galaxy is the <i>posterior</i> width. Plugging \(\sigma_{\rm post}\) into that formula
underestimates \(|m|\): since \(\sigma_{\rm post} \le \sigma_\pi\), it can never go below
\(-1/2\), while the correct \(m = -\sigma_{\rm post}^2/\sigma_\pi^2\) reaches \(-1\). The
data below settle which one is right.</p>
<p><b>The prior width.</b> The true prior on each component is uniform on \([-0.1, 0.1]\).
We approximate it by a Gaussian with the same variance, \(\sigma_\pi^2 = 0.2^2/12\), i.e.
\(\sigma_\pi = $SPIV\).</p>
<p><b>A uniform-prior toy model.</b> A uniform prior behaves differently from a Gaussian one:
it does not pull at all on a narrow likelihood far from the box edges, and it pulls only
through truncation when the likelihood reaches an edge. To see how much this matters, the
script also runs a 1D toy with the real prior shape: true \(g\) uniform in the fit window
\((-$LOWG, $LOWG)\), \(g_{\rm obs} = g + \) Gaussian noise of width \(s\), and the posterior is
that Gaussian truncated to \([-0.1, 0.1]\), whose mean and variance have closed forms. For each
of $NTOY values of \(s\) (log-spaced from 0.001 to 1) it records the average posterior
standard deviation and the fitted slope \(m\). This gives a curve \(m(\sigma_{\rm post})\), and
each real galaxy is assigned the toy \(m\) at its own posterior standard deviation. The toy
predicts \(m \approx 0\) for narrow posteriors, unlike the Gaussian formula.</p>
<p><b>Why binning on the observed width is biased.</b> The obvious test bins galaxies by their
observed $SE and compares \(m\) in each bin with the prediction. But $SE depends partly on each
galaxy's noise realization. At fixed true shear, noise that makes a galaxy look more strongly
sheared also narrows its posterior (at fixed true \(|g|\), the regression slope of
\(\ln\sigma_{\epsilon,i}\) on \(|\hat g|\) is $S3; see $WVG). A bin of small observed $SE is
therefore enriched in galaxies whose noise pushed \(\hat g\) away from zero. That inflates the
fitted response, and can even make \(m > 0\), which no amount of shrinkage can produce.</p>
<p><b>Selection-free binning.</b> To avoid this, the follow-up (supplement S4) bins on a
<i>predicted</i> width instead. A gradient-boosted regression tree model (GBM) predicts
\(\ln\sigma_{\epsilon,i}\) from truth-level quantities that do not depend on the shear or on the
noise realization (true \(\cos i\), the image and H\(\alpha\) SNRs, magnitude, rotation
speed, sizes, flux, \(v_0\), and \(\cos 2\theta_{\rm int}\), \(\sin 2\theta_{\rm int}\)). It is
"cross-fitted": galaxies are split into five groups and each group's prediction comes from a
model trained on the other four, so the prediction cannot memorize a galaxy's own noise. It
explains \(R^2 = $R2\) of the variance of \(\ln\sigma_{\epsilon,i}\). Bin membership then cannot
depend on the noise, while galaxies in a bin still have similar widths.</p>""",
        SPIV=f(SIGPI, 4), LOWG=f(LOWG, 2), NTOY=ntoy, SE=SYM["SE"],
        S3=pm(s3["dresid_dabsghat_at_fixed_absg"], s3["se_absghat"], 2, True),
        WVG=lnk("width-vs-true-shear", "the width-vs-shear page"), R2=f(S("gbm_r2"), 2))

    procedure = T(r"""<ol>
<li>For every galaxy take the TF-weighted posterior variance of each component: \(\sigma_{g_1}^2\),
\(\sigma_{g_2}^2\), and for the galaxy frame
\(\sigma_{g_+}^2 = c^2\sigma_{g_1}^2 + s^2\sigma_{g_2}^2 - 2cs\,C_{12}\) and
\(\sigma_{g_\times}^2 = s^2\sigma_{g_1}^2 + c^2\sigma_{g_2}^2 + 2cs\,C_{12}\), with
\(c = \cos 2\theta_{\rm int}\), \(s = \sin 2\theta_{\rm int}\) and \(C_{12}\) the posterior
covariance.</li>
<li>Binning A (original plan): $NB3 equal-count bins in the observed $SE ($N3 galaxies each).
Binning B (selection-free): $NB4 equal-count bins in the GBM-predicted width ($N4 galaxies each).</li>
<li>In each bin fit \(m\) from \(\hat g - g = mg + c\) on \(|g| &lt; $LOWG\) with equal weights,
for each component.</li>
<li>In each bin compute three predictions from the same galaxies' per-component variances:
(a) the plan formula \(-\langle \sigma^2/(\sigma^2+\sigma_\pi^2)\rangle\) with \(\sigma\) the
posterior width; (b) the Gaussian result \(-\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2\);
(c) the uniform-prior toy.</li>
<li>Score each prediction with
\(\chi^2 = \sum_{\rm bins} \big[(m_{\rm meas} - m_{\rm pred})/\sigma_m\big]^2\), where
\(\sigma_m\) is the fit error of the measured \(m\). No parameters are fitted and the bins
contain disjoint galaxies, so a correct prediction gives \(\chi^2 \approx\) number of bins
(± \(\sqrt{2\times\text{bins}}\)).</li>
<li>Also compare the full-sample predictions with the raw \(m\).</li>
</ol>""", NB3=nb3, NB4=nb4, N3=f"{b30['n']:,}", N4=f"{b40['n']:,}", SE=SYM["SE"], LOWG=f(LOWG, 2))

    # chi2 table
    rows = []
    for key, lab in (("plan_formula", r"plan: \(-\langle\sigma^2/(\sigma^2+\sigma_\pi^2)\rangle\)"),
                     ("gauss_posterior", r"Gaussian: \(-\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2\)"),
                     ("uniform_toy", "uniform-prior toy")):
        rows.append([lab] + [f"{chi_fmt(chi4[key + ':' + c])} ({pval(chi4[key + ':' + c])})" for c in COMPS]
                    + [f"{chi_fmt(chi3[key + ':' + c])}" for c in ("g1", "g2")])
    chitab = table(["prediction"] + [f"{COMP_LABEL[c]}, selection-free bins" for c in COMPS]
                   + [f"{COMP_LABEL[c]}, observed-σ bins" for c in ("g1", "g2")], rows)

    # S4 per-bin table
    rows = []
    for r in bins4:
        rows.append([f(r["mean_sigma_eps_pred"], 4), f(r["mean_sigma_eps_obs"], 4)]
                    + [f"{pm(r[c]['m'], r[c]['m_se'], 3, True)} | {f(r[c]['gauss_posterior'], 3, True)} | "
                       f"{f(r[c]['uniform_toy'], 3, True)} | {f(r[c]['plan_formula'], 3, True)}" for c in COMPS])
    s4tab = table([r"mean predicted \(\sigma\)", r"mean observed \(\sigma_{\epsilon,i}\)"]
                  + [f"{COMP_LABEL[c]}: measured | Gaussian | toy | plan" for c in COMPS], rows)

    # full sample
    rows = []
    for c in COMPS:
        rows.append([COMP_LABEL[c], pm(raw[c]["low_m"], raw[c]["low_m_se"], 3, True),
                     f(fs[c]["gauss_posterior"], 3, True), f(fs[c]["uniform_toy"], 3, True),
                     f(fs[c]["plan_formula"], 3, True)])
    avg_meas = 0.5 * (raw["g+"]["low_m"] + raw["gx"]["low_m"])
    avg_pred = 0.5 * (fs["g+"]["gauss_posterior"] + fs["gx"]["gauss_posterior"])
    rows.append([r"\((m_+ + m_\times)/2\)", f(avg_meas, 3, True), f(avg_pred, 3, True), "", ""])
    fstab = table(["component", r"measured \(m\) (all galaxies)", "Gaussian", "uniform toy", "plan formula"], rows)

    results = (
        "<p><b>χ² of each prediction against the measured m</b> (value / number of bins; the p-value is "
        "the chance of a χ² this large if the prediction were exact):</p>" + chitab
        + fig("A_shear/S4_shrinkage_selection_free.png",
              r"Selection-free test (recommended for the paper). Measured \(m_1\), \(m_2\) in "
              f"{nb4} bins of the predicted width, plotted at each bin's mean observed "
              r"\(\sigma_{\epsilon,i}\). Dashed: the Gaussian prediction \(-\langle\sigma_g^2\rangle/\sigma_\pi^2\). "
              r"Solid: the uniform-prior toy. Note the x-axis is \(\sigma_{\epsilon,i}\approx\sqrt2\,\sigma_g\), "
              r"so \(m=-1\) is reached near \(\sqrt2\,\sigma_\pi\).")
        + "<p><b>Per-bin values, selection-free binning</b> (measured ± fit error | predictions):</p>" + s4tab
        + fig("A_shear/T3_shrinkage.png",
              f"The same test in {nb3} bins of the <i>observed</i> " + r"\(\sigma_{\epsilon,i}\) "
              r"(original plan), image frame (left: \(m_1\), \(m_2\)) and galaxy frame (right: blue \(m_+\), "
              r"orange \(m_\times\); the legend labels are only drawn on the left). Dotted: the plan formula, "
              r"which flattens out at \(-1/2\). The lowest bins lie above zero, which shrinkage cannot produce; "
              r"this is the selection effect described above.")
        + "<p><b>Whole sample</b> (predictions averaged over all galaxies):</p>" + fstab
    )

    interpretation = T(r"""
<p><b>Shrinkage explains the image-frame bias quantitatively, provided the formula uses the
posterior width.</b> With selection-free bins, \(m = -\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2\)
gives \(\chi^2 = $G1\) for \(g_1\) and $G2 for \(g_2\), consistent with the errors. Over the whole
sample it predicts $FS1 and $FS2, against measured \(m_1 = $M1\) and \(m_2 = $M2\).</p>
<p><b>The plan formula is rejected.</b> \(-\langle\sigma^2/(\sigma^2+\sigma_\pi^2)\rangle\) with the
posterior width gives \(\chi^2 = $P1\) and $P2. Over the whole sample it predicts only $FP1, about
half the measured value. It flattens at \(-1/2\) while the most prior-dominated bins reach
\(m\approx -1\). Sec. 6.2 should write \(m = -\sigma_{\rm post}^2/\sigma_\pi^2\), or say clearly that
the \(\sigma\) in \(\sigma^2/(\sigma^2+\sigma_\pi^2)\) is the likelihood width.</p>
<p><b>The uniform toy does not do better</b> despite using the true prior shape: $U1 and $U2 on the
selection-free bins. It predicts too little shrinkage at intermediate widths. Why the Gaussian
approximation fits the NPE better than the toy with the real box prior is not understood.</p>
<p><b>The per-component prediction fails in the galaxy frame</b> ($GP for \(g_+\), $GX for
\(g_\times\)). \(m_+\) is more negative than its own posterior width predicts ($MP measured vs $PP
predicted), and \(m_\times\) less negative ($MX vs $PX). The two errors cancel in the average,
$AM measured vs $AP predicted, which is why the image frame works: image-frame components mix
\(g_+\) and \(g_\times\) over uniformly distributed position angles. In the Gaussian model the
galaxy-frame formula should hold as well, so the galaxy-frame posteriors must depart from that
model in some way that is not yet identified.</p>
<p><b>Observed-width bins are biased by selection,</b> as expected. In the lowest observed-width bin
the measured \(m_\times = $LX\), significantly positive. Binned by predicted width instead, the lowest
bin has \(m\) slightly negative (\(m_1 = $L1\)). With observed-width bins the uniform
toy even looks better than the Gaussian ($UO1 vs $GO1 for \(g_1\)), because its \(m\approx0\) at
small width happens to sit closer to the selection-inflated points. This comparison should
therefore not be used. The selection effect itself is quantified on $SEL.</p>""",
        G1=chi_fmt(chi4["gauss_posterior:g1"]), G2=chi_fmt(chi4["gauss_posterior:g2"]),
        FS1=f(fs["g1"]["gauss_posterior"], 3, True), FS2=f(fs["g2"]["gauss_posterior"], 3, True),
        M1=pm(raw["g1"]["low_m"], raw["g1"]["low_m_se"], 3, True), M2=pm(raw["g2"]["low_m"], raw["g2"]["low_m_se"], 3, True),
        P1=chi_fmt(chi4["plan_formula:g1"]), P2=chi_fmt(chi4["plan_formula:g2"]),
        FP1=f(fs["g1"]["plan_formula"], 3, True),
        U1=chi_fmt(chi4["uniform_toy:g1"]), U2=chi_fmt(chi4["uniform_toy:g2"]),
        GP=chi_fmt(chi4["gauss_posterior:g+"]), GX=chi_fmt(chi4["gauss_posterior:gx"]),
        MP=f(raw["g+"]["low_m"], 3, True), PP=f(fs["g+"]["gauss_posterior"], 3, True),
        MX=f(raw["gx"]["low_m"], 3, True), PX=f(fs["gx"]["gauss_posterior"], 3, True),
        AM=f(avg_meas, 3, True), AP=f(avg_pred, 3, True),
        LX=pm(b30["gx"]["m"], b30["gx"]["m_se"], 3, True), L1=pm(b40["g1"]["m"], b40["g1"]["m_se"], 3, True),
        UO1=chi_fmt(chi3["uniform_toy:g1"]), GO1=chi_fmt(chi3["gauss_posterior:g1"]),
        SEL=lnk("width-cut-selection-bias", "the selection-bias page"))

    caveats = ul([
        r"\(\sigma_\pi\) treats the uniform prior as a Gaussian of equal variance. The toy with the real "
        "box shape fits worse, and we do not know why. Would be good to understand this before writing the "
        "shrinkage derivation as the explanation in Sec. 6.2 or the appendix.",
        "The galaxy-frame failure is unexplained. The paper can stay in the image frame, but the "
        r"\(m_+\) vs \(m_\times\) split should be mentioned if galaxy-frame results are shown.",
        r"Each bin's \(\chi^2\) uses only the fit error of the measured \(m\). The predictions come from the "
        "observed posterior variances and carry negligible statistical noise.",
        "The GBM predictor is only as shear-independent as its inputs. The SNR columns are treated as "
        "shear-independent truths.",
        "The 100k model and Xu3 test set only.",
    ])

    prov = provenance(
        ["a_shear_analysis.py (test_t3, shrink_predictions, uniform_prior_toy, gbm_predicted_log_sigma)",
         "a_shear_supplement.py (S3 mechanism, S4 selection-free binning)"],
        ["a_shear_analysis", "a_shear_supplement"],
        ["results.json: T3.bins[*].{g1,g2,g+,gx}.{m, m_se, plan_formula, gauss_posterior, uniform_toy}",
         "results.json: T3.chi2, T3.full_sample_predictions, T3.toy_grid, sigma_pi",
         "supplement.json: S4_T3_predicted_sigma_bins, S4_chi2, gbm_r2",
         "supplement.json: S3_mechanism.dresid_dabsghat_at_fixed_absg"],
        extra="<p>Paper-figure copies: <code>paper-tests/paper_figs/A_D1_shrinkage_selection_free</code> "
              "(recommended) and <code>A_D1_shrinkage</code> (observed-width bins).</p>")

    answer = (f"Yes in the image frame, if the formula uses the posterior width: m = −σ_post²/σ_π² fits the "
              f"selection-free bins (χ² = {chi_fmt(chi4['gauss_posterior:g1'])} for g1, "
              f"{chi_fmt(chi4['gauss_posterior:g2'])} for g2), while −σ²/(σ²+σ_π²) is rejected "
              f"(χ² = {f(chi4['plan_formula:g1']['chi2'], 0)}); the per-component prediction fails in the galaxy frame.")
    return page(slug="gaussian-shrinkage", title="Does Bayesian shrinkage explain the bias?", code="T3 (+ S4)",
                section=SECTION, answer=answer, question=T(r"""<p>Can the raw multiplicative bias be
predicted from the per-galaxy posterior widths alone, using the shrinkage of a Bayesian posterior
mean toward the prior? And which formula is the right one?</p>"""),
                why=r"""<p>Sec. 6.2 explains why posterior means are biased. Eason's scope decision is
that shrinkage belongs in the Discussion as the explanation of the bias, not as a reason to accept
it. The explanation is only convincing if it is quantitative: the predicted \(m\) should match the
measured \(m\) as a function of posterior width. It also motivates the mitigations: if \(m\) is set
by the width, then weighting or cutting on the width should reduce \(|m|\).</p>""",
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 3
def page_linearity():
    T4 = R("T4")
    raw = {c: R("R_raw", c) for c in COMPS}
    wins = [w["max_abs_g"] for w in T4["g1"]["linear_window"]]
    rows = []
    for c in COMPS:
        lw = T4[c]["linear_window"]
        cf, cq = T4[c]["cubic_full"], T4[c]["cubic_full_with_quadratic"]
        rows.append([COMP_LABEL[c]] + [f"{pm(w['m'], w['m_se'], 3, True)}<br><small>N = {w['n']:,}</small>" for w in lw]
                    + [pm(cf["m"], cf["m_se"], 3, True), pm(cf["q"], cf["q_se"], 1, True), pm(cq["q2"], cq["q2_se"], 3, True)])
    tab = table(["component"] + [rf"linear, \(|g| &lt; {f(w, 3).rstrip('0')}\)" for w in wins]
                + [r"cubic fit: \(m\)", r"cubic fit: \(q\)", r"\(g^2\) coefficient (if added)"], rows)

    # max deviation of window m from the |g|<LOWG value, image frame
    dev = []
    for c in ("g1", "g2"):
        lw = T4[c]["linear_window"]
        ref = [w for w in lw if abs(w["max_abs_g"] - LOWG) < 1e-9][0]["m"]
        dev += [abs(w["m"] - ref) for w in lw]
    maxdev = max(dev)
    narrow_se = T4["g1"]["linear_window"][0]["m_se"]
    cub_dev = max(abs(T4[c]["cubic_full"]["m"] - raw[c]["low_m"]) for c in ("g1", "g2"))
    q1 = T4["g1"]["cubic_full"]["q"]
    edge = 0.1
    q_edge = q1 * edge ** 3
    m_edge = raw["g1"]["low_m"] * edge

    theory = T(r"""
<p><b>What the fit measures.</b> The multiplicative bias is meant to be the response of the
estimator to a small shear: \(1 + m = d\langle\hat g\rangle/dg\) at \(g = 0\). Cosmic shears are
of order a few per cent, so this small-shear slope is what matters in practice. Fitting only
\(|g| &lt; $LOWG\) measures that slope directly. Fitting a wider range would mix in any curvature
of \(\langle\hat g\rangle(g)\) at larger shears.</p>
<p><b>Why curvature is expected near the prior edge.</b> The prior restricts each component to
\([-0.1, 0.1]\). For a galaxy whose true shear is close to an edge, part of its likelihood lies
outside the box and is cut off. The remaining posterior sits inside the box, so its mean is
pushed back toward zero, on top of the ordinary shrinkage. The truncation also narrows the
posterior (see $WVG), which by itself would mean <i>less</i> shrinkage. Which effect wins is an
empirical question.</p>
<p><b>How the test detects it.</b> Two checks: (1) refit the linear model in windows
\(|g| &lt; g_{\max}\) from 0.01 up to the full range and see whether \(m\) drifts; (2) fit the whole
range with a cubic, \(\hat g - g = c + m g + q g^3\). A symmetric pull toward zero near both edges
is an odd function of \(g\), so it shows up as the \(g^3\) term \(q\). If \(q\) is small, the
linear model holds over the whole range. Adding a \(g^2\) term tests for an asymmetric (even)
distortion, which should be zero by symmetry.</p>""", LOWG=f(LOWG, 2),
        WVG=lnk("width-vs-true-shear", "the width-vs-shear page"))

    procedure = T(r"""<ol>
<li>For each component, fit \(\hat g - g = m g + c\) (equal weights, all galaxies) in windows
\(|g| &lt; g_{\max}\) with \(g_{\max}\) = $WINS. For \(g_1, g_2\) the windows from 0.1 up cover the
full range; the galaxy-frame components extend to about 0.14.</li>
<li>Fit the cubic \(c + m g + q g^3\) over all galaxies (<code>component_metrics</code>), and repeat
with an added \(g^2\) term.</li>
<li>Plot the binned mean \(\hat g\) against true \(g\) over the full range with the \(|g| &lt; $LOWG\)
fit extended.</li>
</ol>""", WINS=", ".join(f(w, 3).rstrip("0") for w in wins), LOWG=f(LOWG, 2))

    results = (tab + fig("A_shear/T4_linearity.png",
                         r"Left: binned mean \(\hat g\) against true \(g\) over the full prior range, with the "
                         r"\(|g| &lt; 0.02\) linear fits extended (lines) and \(\hat g = g\) (dotted). Right: the "
                         r"linear-fit \(m\) against the fit window, all four components. Equal weights."))

    interpretation = T(r"""
<p><b>The low-shear linear fit is valid.</b> For \(g_1\) and \(g_2\), \(m\) changes by at most $MD
across all windows, about the size of the error of the narrowest window ($NSE). The cubic fit's
\(m\) agrees with the \(|g| &lt; $LOWG\) value to within $CD.</p>
<p><b>A prior-edge term is detected but small.</b> For \(g_1\), \(q = $Q1\): a significant extra pull
toward zero near the edges. At \(|g| = $EDGE\) it shifts \(\hat g\) by \(q g^3 = $QE\), compared with
the linear bias \(m g = $ME\). This is why \(m\) from the widest windows is slightly more negative.
No \(g^2\) term is detected (\(g_1\): $Q2).</p>
<p><b>The galaxy frame behaves differently</b>: \(q_+ = $QP\) has the opposite sign and
\(q_\times = $QX\) is larger. The galaxy-frame components are not bounded by a fixed box (the box is
rotated differently for each galaxy), so edge effects mix. This is not needed for the paper, which
fits in the image frame.</p>
<p>The paper can quote the \(|g| &lt; $LOWG\) fit and mention that a cubic fit over the full range
gives the same \(m\).</p>""",
        MD=f(maxdev, 3), NSE=f(narrow_se, 3), LOWG=f(LOWG, 2), CD=f(cub_dev, 3),
        Q1=pm(q1, T4["g1"]["cubic_full"]["q_se"], 2, True), EDGE=f(edge, 1), QE=f(q_edge, 4, True),
        ME=f(m_edge, 3, True), Q2=pm(T4["g1"]["cubic_full_with_quadratic"]["q2"], T4["g1"]["cubic_full_with_quadratic"]["q2_se"], 3, True),
        QP=pm(T4["g+"]["cubic_full"]["q"], T4["g+"]["cubic_full"]["q_se"], 1, True),
        QX=pm(T4["gx"]["cubic_full"]["q"], T4["gx"]["cubic_full"]["q_se"], 1, True))

    caveats = ul([
        "The window test uses equal weights and the full sample only. Weighted and cut estimators were "
        "not checked for curvature.",
        r"The errors of different windows are strongly correlated (wider windows contain the narrower ones), "
        "so the drift of m across windows should not be judged window by window.",
        "The sign flip of q in the galaxy frame is not understood.",
    ])
    prov = provenance(["a_shear_analysis.py (test_t4)"], ["a_shear_analysis"],
                      ["results.json: T4.{g1,g2,g+,gx}.linear_window[*].{max_abs_g, n, m, m_se, c, c_se}",
                       "results.json: T4.*.cubic_full.{m, m_se, q, q_se}",
                       "results.json: T4.*.cubic_full_with_quadratic.{q2, q2_se, q3, q3_se}",
                       "results.json: R_raw.*.low_m"])
    answer = (f"Yes: for g1 and g2, m changes by at most {f(maxdev, 3)} across fit windows from 0.01 to the full "
              f"range and a cubic fit agrees to within {f(cub_dev, 3)}; the prior-edge (cubic) term is detected but "
              f"shifts ĝ by only {f(abs(q_edge), 4)} at |g| = {f(edge, 1)}.")
    return page(slug="bias-fit-linearity", title="Is the linear bias fit valid?", code="T4", section=SECTION,
                answer=answer,
                question=T(r"""<p>All \(m\) values in this campaign come from a linear fit restricted to
\(|g| &lt; $LOWG\). Does that fit describe the estimator's response well, or does the response curve
at larger shears, especially near the prior edges at \(|g| \to 0.1\)?</p>""", LOWG=f(LOWG, 2)),
                why=r"""<p>The paper quotes \(m\) from the low-shear fit. A referee may ask whether the
answer depends on the window, and whether the prior box distorts the response near its edges. If
it did, the quoted \(m\) would depend on an arbitrary choice.</p>""",
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 4
def page_bootstrap():
    ests = R("T5", "estimators")
    nboot, seed = R("T5", "n_boot"), R("T5", "seed")
    ratios_m = [v["components"][c]["m_boot_se"] / v["components"][c]["m_fit_se"] for v in ests.values() for c in COMPS]
    ratios_c = [v["components"][c]["c_boot_se"] / v["components"][c]["c_fit_se"] for v in ests.values() for c in COMPS]
    rows = []
    for name, v in ests.items():
        cc = v["components"]
        rows.append([html.escape(name), pct(v["frac_retained"], 1)]
                    + [f"{f(cc[c]['m'], 3, True)}<br><small>{f(cc[c]['m_fit_se'], 4)} / {f(cc[c]['m_boot_se'], 4)}</small>" for c in COMPS]
                    + [f"{e4(cc[c]['c'], 1, True)}<br><small>{e4(cc[c]['c_fit_se'], 1)} / {e4(cc[c]['c_boot_se'], 1)}</small>" for c in ("g1", "g2")]
                    + [f"{f(v['sigma_eps_mean'], 4)} ± {f(v['sigma_eps_mean_boot_se'], 5)}"])
    tab = table(["estimator", "kept"] + [f"{M_LABEL[c]}<br><small>fit / bootstrap error</small>" for c in COMPS]
                + [r"\(10^4 c_1\)<br><small>fit / bootstrap</small>", r"\(10^4 c_2\)<br><small>fit / bootstrap</small>",
                   r"ensemble \(\sigma_\epsilon\) (mean) ± bootstrap error"], rows)
    _, rawv = est("raw")
    b1 = rawv["components"]["g1"]
    half = 0.5 * (b1["m_boot_p16_p84"][1] - b1["m_boot_p16_p84"][0])
    nmax = R("T7", "matched_fraction", len(R("T7", "matched_fraction")) - 1, "std:raw", "threshold")

    theory = T(r"""
<p><b>Fit (sandwich) errors.</b> The fit \(\hat g - g = m g + c\) is a weighted least-squares
regression. The textbook error bars for least squares assume every galaxy's residual has the same
scatter. That is false here: a prior-dominated galaxy returns \(\hat g \approx 0\), so its residual
\(\hat g - g \approx -g\) hardly scatters, while a well-measured galaxy's residual scatters by its noise. The existing
<code>fit_design</code> function therefore uses the "sandwich" (heteroskedasticity-robust)
covariance
\[ \mathrm{Cov}(\hat\beta) = (X^\top W X)^{-1}\Big(\sum_i w_i^2 r_i^2\, x_i x_i^\top\Big)(X^\top W X)^{-1}
   \times \frac{N_{\rm eff}}{N_{\rm eff} - p}, \]
where \(x_i = (1, g_i)\) is the design row, \(w_i\) the normalized galaxy weight, \(r_i\) the fit
residual, \(p = 2\) the number of fitted parameters and \(N_{\rm eff} = (\sum w_i)^2/\sum w_i^2\)
the effective number of galaxies. It uses each galaxy's own residual instead of a common scatter.
It still assumes galaxies are independent and the sample is large.</p>
<p><b>Bootstrap errors.</b> The bootstrap makes almost no assumptions. Draw a new sample of
\(N\) galaxies from the \(N\) test galaxies <i>with replacement</i> (some appear twice, some not
at all), redo the whole estimate, and repeat many times. The spread (standard deviation) of the
resampled estimates approximates the error bar one would see from independent test sets of the
same kind. Crucially, it reruns every step of the estimator, including the steps that the
sandwich formula ignores: the shape-noise floor of the weights is recomputed in each resample.</p>
<p><b>Why compare.</b> If the two agree, the cheap fit errors can be quoted in the paper and used
everywhere (e.g. in the threshold scans, where bootstrapping every point would be costly). If they
disagree, the bootstrap wins.</p>""")

    procedure = T(r"""<ol>
<li>Estimators: raw (equal weights) and weighted (inverse-variance weights with a shape-noise floor;
see $WC), each with no cut and with cuts \(\sigma_{\epsilon,i} &lt; t\) for five thresholds
\(t\).</li>
<li>$NB bootstrap resamples (random seed $SEED). Each resample is drawn as multinomial counts over the
galaxies, i.e. how many times each galaxy is picked. The counts multiply the estimator's galaxy
weights, so a cut keeps the same galaxies, but each is counted as often as it was drawn. For
weighted estimators the floor is recomputed from the resampled, retained galaxies.</li>
<li>In each resample refit \(m\) and \(c\) for all four components, and recompute the ensemble
\(\sigma_\epsilon\) (mean and RMS of \(\sigma_{\epsilon,i}\) with the estimator's galaxy weights).</li>
<li>The bootstrap error is the standard deviation over resamples; compare it with the sandwich
error from the fit on the original sample.</li>
</ol>""", WC=lnk("weighting-and-width-cuts", "the weighting page"), NB=f"{nboot:,}", SEED=seed)

    results = (
        T(r"""<p>Ratio of bootstrap to fit error across all $NE estimators × 4 components:
<b>$RM1–$RM2 for \(m\)</b> and <b>$RC1–$RC2 for \(c\)</b>.</p>""",
          NE=len(ests), RM1=f(min(ratios_m), 2), RM2=f(max(ratios_m), 2), RC1=f(min(ratios_c), 2), RC2=f(max(ratios_c), 2))
        + tab
        + T(r"""<p>The row "\(\sigma_{\epsilon,i} &lt; 0.1\)" keeps every galaxy (the largest
\(\sigma_{\epsilon,i}\) is $NMAX), so it repeats the uncut row. For the raw \(m_1\), the bootstrap
distribution's 16–84% half-range is $HALF against a standard deviation of $BSE, i.e. it is close to
Gaussian.</p>""", NMAX=f(nmax, 4), HALF=f(half, 4), BSE=f(b1["m_boot_se"], 4))
    )

    interpretation = T(r"""
<p><b>The fit errors are reliable</b> for every estimator, including the weighted ones where the
floor is re-estimated, and the cut ones with few galaxies. The paper can quote the sandwich errors
directly; bootstrap errors are available for the tables if wanted. The ensemble shape noise is
known to about $SEB (raw), so it is effectively exact at this sample size.</p>
<p>Note that these errors describe the scatter from a finite test set only. They do not include the
selection bias of width cuts ($SEL), or the variation between trained models (e.g. training seeds),
which have not been measured.</p>""",
        SEB=f(rawv["sigma_eps_mean_boot_se"], 5), SEL=lnk("width-cut-selection-bias", "see the selection-bias page"))

    caveats = ul([
        "The bootstrap assumes galaxies are independent. In this simulated test set each galaxy has its own "
        "shear and noise, so that holds.",
        "The cut thresholds are fixed (not re-optimized per resample), as they would be in a real analysis.",
        "Errors for the threshold scan on the weighting page are fit errors only; given this test, that is fine.",
    ])
    prov = provenance(["a_shear_analysis.py (test_t5, estimator_list, estimator_weights, m_c_fast)",
                       f"{DIAG}/shear_bias_report.py (fit_design, compose_shape_noise_regularized_weights)"],
                      ["a_shear_analysis"],
                      ["results.json: T5.n_boot, T5.seed",
                       "results.json: T5.estimators.<name>.components.<comp>.{m, m_fit_se, m_boot_se, c, c_fit_se, c_boot_se, m_boot_p16_p84}",
                       "results.json: T5.estimators.<name>.{frac_retained, sigma_eps_mean, sigma_eps_mean_boot_se, sigma_eps_rms, sigma_eps_rms_boot_se}"])
    answer = (f"Yes: across {len(ests)} estimators and 4 components the bootstrap errors ({nboot:,} resamples) agree "
              f"with the sandwich fit errors to a ratio of {f(min(ratios_m), 2)}–{f(max(ratios_m), 2)} for m and "
              f"{f(min(ratios_c), 2)}–{f(max(ratios_c), 2)} for c, so the fit errors can be quoted.")
    return page(slug="error-bars-bootstrap", title="Are the fit error bars reliable?", code="T5", section=SECTION,
                answer=answer,
                question=r"""<p>Are the error bars that the least-squares fit reports for \(m\), \(c\) and the
ensemble shape noise correct, for the raw estimator, the weighted estimator and the width-cut
estimators?</p>""",
                why=r"""<p>Every comparison in Sec. 6 (e.g. "weighting shifts \(m\)", "\(m\) is
consistent with zero after a cut") rests on these error bars. Weighted and cut estimators involve
steps (a data-dependent floor, a data-dependent selection) that the fit's error formula does not
see, so it needs an independent check.</p>""",
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 5
def page_width_vs_shear():
    T1 = R("T1")
    s1 = S("S1_interior_slopes")
    s3 = S("S3_mechanism")
    rows = []
    for name, lab in (("|g|", r"\(|g|\)"), ("g+", r"\(g_+\)"), ("gx", r"\(g_\times\)")):
        r = T1[name]
        rows.append([lab, pm(r["dsigma_dg"], r["dsigma_dg_se"], 4, True), pm(r["dlnsigma_dg"], r["dlnsigma_dg_se"], 3, True),
                     pm(r["dlnsigma_dg_residualized"], r["dlnsigma_dg_residualized_se"], 3, True),
                     pm(r["quad_residualized"]["lin"], r["quad_residualized"]["lin_se"], 3, True),
                     pm(r["quad_residualized"]["quad"], r["quad_residualized"]["quad_se"], 2, True)])
    tab1 = table(["true shear variable", r"\(d\sigma_{\epsilon,i}/dg\)", r"\(d\ln\sigma_{\epsilon,i}/dg\)",
                  r"\(d\ln\sigma/dg\), residualized", "residualized, linear + quadratic fit: linear term", "quadratic term"], rows)
    rows = []
    for r in T1["by_cosi"]:
        rows.append([f"[{r['cosi_lo']}, {r['cosi_hi']}]", f"{r['n']:,}", f(r["mean_sigma"], 4)]
                    + [pm(r[k]["residualized"], r[k]["residualized_se"], 3, True) for k in ("|g|", "g+", "gx")])
    tab2 = table([r"true \(\cos i\)", "galaxies", r"mean \(\sigma_{\epsilon,i}\)", r"\(d\ln\sigma/d|g|\)",
                  r"\(d\ln\sigma/dg_+\)", r"\(d\ln\sigma/dg_\times\)"], rows)
    rows = []
    for r in s1:
        rows.append([f(r["box_max_abs_g1_g2"], 2), f"{r['n']:,}", pm(r["|g|"]["dlnsigma_dg"], r["|g|"]["se"], 3, True),
                     pm(r["g+"]["quad_lin"], r["g+"]["quad_lin_se"], 3, True), pm(r["g+"]["quad"], r["g+"]["quad_se"], 1, True),
                     pm(r["gx"]["quad"], r["gx"]["quad_se"], 1, True)])
    tab3 = table([r"box: \(|g_1|, |g_2| &lt;\)", "galaxies", r"\(d\ln\sigma/d|g|\)", r"linear term in \(g_+\)",
                  r"quadratic term in \(g_+\)", r"quadratic term in \(g_\times\)"], rows)

    # illustration: fraction of low-|g1| galaxies kept by a sigma cut vs overall fraction
    rows = []
    n_low_all = est("raw")[1]["components"]["g1"]["ess_low"]
    for name, v in R("T5", "estimators").items():
        if v["kind"] != "raw" or "<" not in name or v["frac_retained"] >= 1:
            continue
        kept_low = v["components"]["g1"]["ess_low"] / n_low_all
        rows.append([html.escape(est_cut_value(name)), pct(v["frac_retained"], 1), pct(kept_low, 1)])
    tab4 = table([r"cut \(\sigma_{\epsilon,i} &lt;\)", "fraction of all galaxies kept",
                  r"fraction of galaxies with \(|g_1| &lt; 0.02\) kept"], rows)

    g_abs = T1["|g|"]
    win_change = g_abs["dlnsigma_dg_residualized"] * LOWG
    full_change = g_abs["dsigma_dg"] * 0.1 / g_abs["intercept"]
    box05 = [r for r in s1 if abs(r["box_max_abs_g1_g2"] - 0.05) < 1e-9][0]
    box03 = [r for r in s1 if abs(r["box_max_abs_g1_g2"] - 0.03) < 1e-9][0]
    cos_lo, cos_hi = T1["by_cosi"][0], T1["by_cosi"][-1]

    theory = T(r"""
<p><b>Why it matters.</b> The posterior width $SE is what we weight by and cut on. Any quantity used
to weight or select galaxies should not itself depend on the shear. If it does, the weights or the
selection change with the shear, and the sample's average response picks up an extra term: this is
the "selection response" of metacalibration. There are two ways the width can depend on the shear:</p>
<ol>
<li><b>On the true shear</b> (this page). If galaxies with larger \(|g|\) have narrower posteriors,
a width cut keeps relatively more high-\(|g|\) galaxies. A dependence on \(|g|\) is even in \(g\):
the same fraction is kept at \(+g\) and \(-g\), so to first order it does not bias the fitted slope
inside the small fit window. A dependence that is odd in \(g\) (e.g. linear in \(g_+\)) would.</li>
<li><b>On the noise-driven apparent shear.</b> Noise that makes a galaxy look more sheared can also
narrow its posterior. Then a width cut keeps galaxies whose noise pushed \(\hat g\) outward, which
biases \(m\) directly. This is tested on $SEL; the mechanism is closely related to what is measured
here.</li>
</ol>
<p><b>Two reasons the width could depend on the shear.</b></p>
<ul>
<li><b>Prior truncation.</b> The prior box \(|g_1|, |g_2| &lt; 0.1\) cuts off any part of the
posterior that would lie outside it. For a galaxy with true shear near an edge, the truncated
posterior is narrower. This is an artefact of the bounded prior. It should matter most for galaxies
with moderately wide posteriors near the edges. It should matter least for very well-measured
galaxies (whose posterior rarely reaches the edge) and for prior-dominated ones (whose posterior is
the whole box wherever the true shear is).</li>
<li><b>Physical dependence.</b> Shear changes the galaxy's apparent shape and velocity field, and a
strongly sheared galaxy may be easier or harder to measure. In particular shear along the major
axis (\(g_+ &gt; 0\)) makes the galaxy look more elongated and \(g_+ &lt; 0\) rounder, so a
dependence linear in \(g_+\) is physically plausible.</li>
</ul>
<p>The prior box treats \(g_+\) and \(g_\times\) the same once averaged over the uniformly
distributed position angles. A difference between the \(g_+\) and \(g_\times\) dependence therefore
points to a physical cause.</p>
<p><b>Residualizing.</b> Most of the galaxy-to-galaxy scatter in \(\ln\sigma_{\epsilon,i}\) comes
from inclination, SNR and size, not from the shear. To measure a small shear dependence precisely,
we first subtract a prediction of \(\ln\sigma_{\epsilon,i}\) made from shear-independent truths only
(the cross-fitted GBM described on $SHR, \(R^2 = $R2\)). The prediction does not use \(g\), and in
the simulation \(g\) is independent of those truths, so subtracting it removes scatter without
changing the expected slope.</p>""",
        SE=SYM["SE"], SEL=lnk("width-cut-selection-bias", "the selection-bias page"),
        SHR=lnk("gaussian-shrinkage", "the shrinkage page"), R2=f(T1["gbm_r2_log_sigma"], 2))

    procedure = T(r"""<ol>
<li>Regress \(\sigma_{\epsilon,i}\), \(\ln\sigma_{\epsilon,i}\) and the residualized \(\ln\sigma_{\epsilon,i}\)
on the true \(|g|\), \(g_+\) and \(g_\times\) (one at a time; least squares with sandwich errors).
Also fit linear + quadratic terms to the residualized values.</li>
<li>Repeat in four bins of true \(\cos i\).</li>
<li>Supplement S1: repeat using only galaxies inside smaller boxes \(|g_1|, |g_2| &lt; b\), to see
how much of the dependence survives away from the prior edges.</li>
<li>Plot binned means against each shear variable.</li>
</ol>""")

    results = (
        "<p><b>All galaxies.</b> Slopes of the posterior width against the true shear (per unit shear):</p>" + tab1
        + fig("A_shear/T1_sigma_vs_shear.png",
              r"Residualized \(\ln\sigma_{\epsilon,i}\) (observed minus the shear-free prediction) against true "
              r"\(|g|\), \(g_+\) and \(g_\times\), in four bins of true \(\cos i\) and for all galaxies (black). "
              r"Bin means ± standard error. Panel titles give the linear slope for all galaxies.")
        + fig("A_shear/T1_sigma_raw_vs_gplus.png",
              r"Mean \(\sigma_{\epsilon,i}\) (not residualized) against true \(g_+\) and \(g_\times\).", width="60%")
        + "<p><b>By inclination</b> (residualized linear slopes):</p>" + tab2
        + "<p><b>Away from the prior edges</b> (supplement S1; residualized):</p>" + tab3
        + "<p><b>A direct consequence.</b> Because small-|g| galaxies have wider posteriors, a width cut keeps "
          "fewer of them than of the sample as a whole (raw estimators):</p>" + tab4
    )

    interpretation = T(r"""
<p><b>The width is not independent of the shear.</b> $SE falls with \(|g|\):
\(d\ln\sigma/d|g| = $SL\). That means $SE is about $FULL smaller at \(|g| = 0.1\) than at \(g=0\) and
about $WIN smaller across the \(|g| &lt; $LOWG\) fit window.</p>
<p><b>Much of it is prior truncation.</b> The slope shrinks as galaxies near the edges are excluded:
$B05 inside \(|g_1|,|g_2| &lt; 0.05\) and $B03 inside 0.03. It is also smallest for nearly edge-on
galaxies (\(\cos i &lt; 0.25\): $CLO), whose posteriors are nearly prior-dominated (mean width $CLOS)
and so fill the box regardless of the shear. It is steepest for nearly face-on galaxies
(\(\cos i &gt; 0.75\): $CHI), whose narrower posteriors (mean width $CHIS) get truncated when the
true shear is near an edge. The dependence does not vanish in the interior, but with posterior
widths of a few per cent even a box of 0.03 is not far from the edges, so this test cannot fully
separate truncation from physics.</p>
<p><b>There is also a physical part.</b> The curvature is much stronger in \(g_+\) than in
\(g_\times\) (quadratic terms $QP vs $QX), also inside the 0.05 box ($QP5 vs $QX5). The box would
treat the two alike, so shear along the galaxy's axes changes the posterior width more than shear
at 45°. There is also a small odd term in \(g_+\) ($LP): galaxies stretched along their major axis
have slightly narrower posteriors. Over the fit window this odd term changes \(\ln\sigma\) by only
about $LPW, so by itself it biases \(m\) very little.</p>
<p><b>What this means for weighting and cuts.</b> The dependence on the <i>true</i> shear is mostly
even in \(g\) and small inside the fit window, so on its own it would bias \(m\) little. The same
physics, though, makes the width respond to the noise-driven apparent shear: at fixed true
\(|g|\), \(\ln\sigma_{\epsilon,i}\) falls with the estimated \(|\hat g|\) (slope $S3). That is what
makes width cuts select galaxies scattered outward ($SEL). The paper should say that the posterior
width depends on the shear and is therefore not a shear-independent selection variable.</p>""",
        SE=SYM["SE"], SL=pm(g_abs["dlnsigma_dg_residualized"], g_abs["dlnsigma_dg_residualized_se"], 2, True),
        FULL=pct(-full_change, 0), WIN=pct(-win_change, 0), LOWG=f(LOWG, 2),
        B05=pm(box05["|g|"]["dlnsigma_dg"], box05["|g|"]["se"], 2, True),
        B03=pm(box03["|g|"]["dlnsigma_dg"], box03["|g|"]["se"], 2, True),
        CLO=pm(cos_lo["|g|"]["residualized"], cos_lo["|g|"]["residualized_se"], 2, True), CLOS=f(cos_lo["mean_sigma"], 3),
        CHI=pm(cos_hi["|g|"]["residualized"], cos_hi["|g|"]["residualized_se"], 2, True), CHIS=f(cos_hi["mean_sigma"], 3),
        QP=pm(T1["g+"]["quad_residualized"]["quad"], T1["g+"]["quad_residualized"]["quad_se"], 1, True),
        QX=pm(T1["gx"]["quad_residualized"]["quad"], T1["gx"]["quad_residualized"]["quad_se"], 1, True),
        QP5=pm(box05["g+"]["quad"], box05["g+"]["quad_se"], 1, True), QX5=pm(box05["gx"]["quad"], box05["gx"]["quad_se"], 1, True),
        LP=pm(T1["g+"]["dlnsigma_dg_residualized"], T1["g+"]["dlnsigma_dg_residualized_se"], 3, True),
        LPW=pct(abs(T1["g+"]["dlnsigma_dg_residualized"]) * LOWG, 2),
        S3=pm(s3["dresid_dabsghat_at_fixed_absg"], s3["se_absghat"], 2, True),
        SEL=lnk("width-cut-selection-bias", "see the selection-bias page"))

    caveats = ul([
        "Prior truncation and physical dependence are not cleanly separated. A test with a wider prior box "
        "(or a posterior that is not truncated) would be needed to do so.",
        "The residualization relies on the GBM inputs being independent of the shear. The SNR columns are treated as such.",
        r"Why face-on galaxies show the steepest \(|g|\) dependence has not been checked beyond the truncation argument above.",
        r"The figure's \(|g|\) axis extends to 0.14 (the corners of the square box), where few galaxies live.",
    ])
    prov = provenance(["a_shear_analysis.py (test_t1, gbm_predicted_log_sigma, slope_fit)",
                       "a_shear_supplement.py (S1 interior slopes, S3 mechanism)"],
                      ["a_shear_analysis", "a_shear_supplement"],
                      ["results.json: T1.{|g|,g+,gx}.{dsigma_dg, dlnsigma_dg, dlnsigma_dg_residualized, quad_residualized}",
                       "results.json: T1.by_cosi[*], T1.gbm_r2_log_sigma",
                       "supplement.json: S1_interior_slopes[*]",
                       "supplement.json: S3_mechanism.{dresid_dabsghat_at_fixed_absg, se_absghat}",
                       "results.json: T5.estimators.<raw cuts>.components.g1.ess_low (kept low-|g1| count)"])
    answer = (f"Yes: the posterior width falls with the true |g| (d ln σ/d|g| = "
              f"{pm(g_abs['dlnsigma_dg_residualized'], g_abs['dlnsigma_dg_residualized_se'], 2, True)}), much of it from "
              f"the prior box truncating posteriors near its edges but partly physical (stronger along g+ than g×), "
              f"so σ_ε,i is not a shear-independent selection variable.")
    return page(slug="width-vs-true-shear", title="Does posterior width depend on the true shear?", code="T1 (+ S1)",
                section=SECTION, answer=answer,
                question=T(r"""<p>Does the per-galaxy posterior width $SE change with the galaxy's true
shear, as a function of \(|g|\) and of the galaxy-frame components \(g_+\) and \(g_\times\)? If so,
how much of that comes from the bounded prior and how much is physical?</p>""", **SYM),
                why=T(r"""<p>Sec. 6.3 weights galaxies by, and cuts on, $SE. Both are only safe if $SE
does not itself respond to the shear. This test checks the most direct way it could.</p>""", **SYM),
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 6
def page_weighting():
    ref = R("reference_check")
    _, rawv = est("raw")
    _, wv = est("weighted")
    floor = ref["got"]["floor"]
    sig_max = R("T7", "matched_fraction", len(R("T7", "matched_fraction")) - 1, "std:raw", "threshold")
    ratio = (0.5 * sig_max ** 2 + floor ** 2) / floor ** 2
    sq2 = float(np.sqrt(2.0))

    # weighting table (no cut)
    rows = []
    for lab, v in (("raw (equal weights)", rawv), ("weighted", wv)):
        cc = v["components"]
        rows.append([lab] + [pm(cc[c]["m"], cc[c]["m_fit_se"], 3, True) for c in COMPS]
                    + [f(v["sigma_eps_mean"], 4), f(v["sigma_eps_rms"], 4)])
    tabw = table(["estimator"] + [M_LABEL[c] for c in COMPS]
                 + [r"ensemble \(\sigma_\epsilon\), mean", r"ensemble \(\sigma_\epsilon\), RMS"], rows)
    tabs5 = table(["quantity", r"\((m_1+m_2)/2\) or difference"], [
        ["raw", f(S2("raw", "m_avg"), 3, True)],
        ["weighted, weights from observed posterior variances (the existing code)", f(S2("weighted_observed", "m_avg"), 3, True)],
        ["weighted, weights from the shear-free predicted width", f(S2("weighted_gbm", "m_avg"), 3, True)],
        ["weighted − raw", pm(S2("delta_obs_minus_raw"), S2("delta_obs_minus_raw_se"), 3, True)],
        ["  … of which not explained by shrinkage (excess over the toy)", pm(S2("excess_obs_minus_raw"), S2("excess_obs_minus_raw_se"), 3, True)],
        ["observed-variance − predicted-width weights", pm(S2("delta_obs_minus_gbm"), S2("delta_obs_minus_gbm_se"), 3, True)],
        ["  … of which not explained by shrinkage", pm(S2("excess_obs_minus_gbm"), S2("excess_obs_minus_gbm_se"), 3, True)],
    ])
    tabconv = table(["quantity", r"existing-code convention \(\sigma_{{\rm gal},i}=\sqrt{(\sigma_{g_1}^2+\sigma_{g_2}^2)/2}\)",
                     r"Xu et al. (2024) convention \(\sigma_{\epsilon,i}=\sqrt{\sigma_{g_1}^2+\sigma_{g_2}^2}\)"], [
        ["floor (unweighted mean width, all galaxies)", f(floor, 4), f(sq2 * floor, 4)],
        ["reported shape noise (weighted mean width)", f(ref["got"]["sigma_shape_weighted"], 4), f(wv["sigma_eps_mean"], 4)],
    ])
    tabref = table(["quantity", "earlier report", "this campaign"], [
        [r"\(m_1\)", pm(ref["expected"]["m1"], ref["expected"]["m1_se"], 5, True), pm(ref["got"]["m1"], ref["got"]["m1_se"], 5, True)],
        [r"\(m_2\)", pm(ref["expected"]["m2"], ref["expected"]["m2_se"], 5, True), pm(ref["got"]["m2"], ref["got"]["m2_se"], 5, True)],
        [r"weighted \(\sigma_{\rm gal}\)", f(ref["expected"]["sigma_shape"], 5), f(ref["got"]["sigma_shape_weighted"], 5)],
        ["effective number of galaxies", f(ref["expected"]["ess"], 1), f(ref["got"]["ess"], 1)],
    ])

    # width-cut scan table
    rows = []
    for t in (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 1.0):
        r = r6row(t)
        rows.append([("no cut" if t == 1.0 else f(r["threshold"], 4).rstrip("0")), pct(r["frac_retained"], 1), f"{r['n_retained']:,}",
                     pm(r["raw"]["g1"]["m"], r["raw"]["g1"]["m_se"], 3, True), pm(r["raw"]["g2"]["m"], r["raw"]["g2"]["m_se"], 3, True),
                     f(r["raw"]["sigma_eps_mean"], 4),
                     pm(r["weighted"]["g1"]["m"], r["weighted"]["g1"]["m_se"], 3, True), pm(r["weighted"]["g2"]["m"], r["weighted"]["g2"]["m_se"], 3, True),
                     f(r["weighted"]["sigma_eps_mean"], 4)])
    tabcut = table([r"cut \(\sigma_{\epsilon,i} &lt;\)", "kept", "galaxies kept", r"raw \(m_1\)", r"raw \(m_2\)",
                    r"raw \(\sigma_\epsilon\) (mean)", r"weighted \(m_1\)", r"weighted \(m_2\)", r"weighted \(\sigma_\epsilon\) (mean)"], rows)
    r045, r035 = r6row(0.045), r6row(0.035)
    rmax = max(R("R6", "rows"), key=lambda r: 0.5 * (r["raw"]["g1"]["m"] + r["raw"]["g2"]["m"]))
    _, c05r = est("raw", 0.05)
    _, c05w = est("weighted", 0.05)
    nocut = r6row(1.0)
    thr_all = sorted(R("R6", "thresholds"))

    theory = T(r"""
<p><b>Inverse-variance weighting.</b> If each galaxy gave an unbiased shear estimate with noise
\(\sigma_i\), the most precise combination would weight each galaxy by \(1/\sigma_i^2\). Weak lensing
analyses usually add the intrinsic shape noise to each galaxy's measurement noise, so the weights
do not become extreme. KL removes most of the intrinsic shape noise, so the existing code adds a
"shape-noise floor" instead:
\[ w_i \propto \frac{1}{\sigma_{{\rm gal},i}^2 + \sigma_{\rm floor}^2}, \qquad
   \sigma_{{\rm gal},i} = \sqrt{\frac{\sigma_{g_1,i}^2 + \sigma_{g_2,i}^2}{2}}, \qquad
   \sigma_{\rm floor} = \langle \sigma_{{\rm gal},i}\rangle , \]
where the average for the floor is over the galaxies in the sample (recomputed after any cut)
(<code>shear_bias_report.compose_shape_noise_regularized_weights</code>). The floor limits how much
a single narrow posterior can dominate. Here the heaviest and lightest weights differ by at most a
factor of about $RATIO (a galaxy with zero width vs the widest one, $SMAX in $SE).</p>
<p><b>Two width conventions.</b> The existing code uses the per-component RMS width
\(\sigma_{{\rm gal},i} = \sigma_{\epsilon,i}/\sqrt2\). This campaign uses the Xu et al. (2024)
\(\sigma_{\epsilon,i}\) everywhere else. The weights are identical in the two conventions
(\(1/(\sigma_{\rm gal}^2+\sigma_{\rm floor}^2) = 2/(\sigma_\epsilon^2 + \langle\sigma_\epsilon\rangle^2)\)),
but the numbers quoted as "shape noise" differ by \(\sqrt2\):</p>
$TABCONV
<p><b>Why weighting reduces \(|m|\).</b> The shrinkage model (see $SHR) gives each galaxy a bias
\(m_i \approx -\sigma_{{\rm post},i}^2/\sigma_\pi^2\). Weighting by inverse width gives less weight to
the galaxies with the widest posteriors, which are exactly the most shrunk ones, so the weighted
average \(m\) moves toward zero. This is a change in which galaxies dominate, not a correction of the
bias.</p>
<p><b>Why it does not simply reduce the noise on \(m\).</b> The \(1/\sigma^2\) optimality argument
assumes unbiased per-galaxy estimates. Posterior means are not: a prior-dominated galaxy returns
\(\hat g \approx 0\) with very little scatter. Down-weighting such galaxies does not necessarily
shrink the error on \(m\). Weighting does lower the ensemble shape noise (the weighted mean of
$SE), which is the quantity the paper reports.</p>
<p><b>Posterior-width cuts.</b> Keep only galaxies with \(\sigma_{\epsilon,i} &lt; t\). In the shrinkage
model the sample's \(m \approx -\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2\) goes to 0 as \(t\)
decreases, at the cost of throwing galaxies away. Shrinkage alone can never make \(m\) positive,
since \(m_i \le 0\) for every galaxy. A positive \(m\) after a cut is therefore a sign of something
else: the cut selects on the noise (see $WVG). The full selection-bias analysis is on $SEL and is
not repeated here.</p>
<p><b>Selection check for the weights.</b> The weights also depend on the observed widths, so they
could in principle carry the same noise-driven selection. Supplement S5 builds the same weights from
the shear-free predicted width (the cross-fitted GBM prediction; see $SHR), which cannot respond to
noise, and compares. It also subtracts the difference in \(m\) that the uniform-prior shrinkage toy
predicts from the change in weights alone. Whatever is left ("excess") would be selection.</p>""",
        RATIO=f(ratio, 1), SMAX=f(sig_max, 4), SE=SYM["SE"], TABCONV=tabconv,
        SHR=lnk("gaussian-shrinkage", "the shrinkage page"), WVG=lnk("width-vs-true-shear", "the width-vs-shear page"),
        SEL=lnk("width-cut-selection-bias", "the selection-bias page"))

    procedure = T(r"""<ol>
<li>Reference check: the weighted estimator on the full sample must reproduce the number in the earlier
report on this model (weighted \(m_1, m_2\) and shape noise). It does, to all printed digits, which
validates the loading of the cache and the TF weights.</li>
<li>Weighting without cuts: compare raw and weighted \(m\) for all four components (fit errors; the
bootstrap on $BOOT gives the same errors).</li>
<li>Supplement S5: compare weights from observed variances with weights from the predicted width, and
both with raw; $NB5 paired bootstrap resamples for the differences.</li>
<li>Threshold scan: for \(t\) from $TLO to $THI in steps of $TST (plus no cut), apply the cut, then
compute raw and weighted \(m_1, m_2\) (|g| &lt; $LOWG fit, fit errors), the ensemble \(\sigma_\epsilon\) of the
kept galaxies (weighted by the estimator's weights), and the kept fraction. For the weighted estimator
the floor is recomputed on the kept galaxies.</li>
</ol>""", BOOT=lnk("error-bars-bootstrap", "the error-bar page"), NB5=S2("n_boot"), LOWG=f(LOWG, 2),
        TLO=f(thr_all[0], 3), THI=f(max(t for t in thr_all if t < 0.1), 2), TST=f(thr_all[1] - thr_all[0], 4))

    results = (
        "<p><b>Reference reproduction</b> (weighted, full sample):</p>" + tabref
        + "<p><b>Weighting, no cut</b> (± fit errors):</p>" + tabw
        + "<p><b>Is the weighting gain just shrinkage?</b> (supplement S5, paired bootstrap):</p>" + tabs5
        + "<p><b>Width-cut scan</b> (selected thresholds; ± fit errors):</p>" + tabcut
        + fig("A_shear/R6_width_cut.png",
              r"Left: \(m_1\), \(m_2\) (\(|g|&lt;0.02\) fit) against the cut threshold on \(\sigma_{\epsilon,i}\); solid "
              r"lines raw, dashed lines weighted, bands ±1σ fit errors; markers on the right are the uncut values. "
              r"Middle: ensemble \(\sigma_\epsilon\) (mean and RMS of the kept galaxies' \(\sigma_{\epsilon,i}\)). "
              r"Right: kept fraction. The dotted 'low-|g1| galaxy count' line on the right axis is a plotting "
              r"artefact: it counts all galaxies with \(|g_1|&lt;0.02\), not only the kept ones.")
    )

    interpretation = T(r"""
<p><b>Weighting lowers \(|m|\) by about $DWS and the ensemble shape noise by $DSN, and this is all
shrinkage.</b> Weighted \(m_1 = $WM1\) and \(m_2 = $WM2\) against raw $RM1 and $RM2, with the
ensemble shape noise dropping from $RS to $WS. The weighted − raw shift of $DW is matched by the
shrinkage toy to within $EW, and weights built from the noise-free predicted width give almost the
same \(m\) (excess $EG). So the weighting shows no detectable noise-driven selection at this
precision. The error on \(m\) does not improve for the full sample ($RE raw vs $WE weighted); it does
for cut samples (e.g. \(\sigma_{\epsilon,i}&lt;0.05\): $CRE raw vs $CWE weighted), where fewer galaxies are
prior-dominated.</p>
<p><b>Width cuts drive \(m\) to zero, then past it.</b> \(|m|\) falls steadily as the threshold
tightens. At \(\sigma_{\epsilon,i} &lt; $T45\) the raw \(m_1, m_2 = $A45, $B45\) with $F45 of galaxies kept
and ensemble \(\sigma_\epsilon = $S45\). Tightening further, \(m\) <b>overshoots to positive values</b>,
peaking at $MMAX near \(t = $TMAX\) ($FMAX kept). The shrinkage model cannot produce \(m &gt; 0\), so the
zero crossing is a cancellation between shrinkage (negative) and a positive selection bias of the cut.
It is not evidence that the cut has removed the bias. See $SEL for the size of that selection term.</p>
<p><b>The trade-off.</b> Every tightening throws away galaxies. At $T45 about $F45 remain and the error
on \(m\) grows from $E0 to $E45, although the kept galaxies are much better measured (ensemble
\(\sigma_\epsilon\) from $RS to $S45). With weighting the curve sits higher (less negative) when little is
cut and converges with the raw curve for tight cuts, where the kept galaxies have similar widths.</p>
<p><b>For the paper</b> (Sec. 6.3): present weighting as a mild, selection-safe improvement, and the
width-cut curve as a trade-off whose \(m\) includes a selection response. It should not be described as
"the cut removes the bias".</p>""",
        WM1=pm(wv["components"]["g1"]["m"], wv["components"]["g1"]["m_fit_se"], 3, True),
        WM2=pm(wv["components"]["g2"]["m"], wv["components"]["g2"]["m_fit_se"], 3, True),
        RM1=f(rawv["components"]["g1"]["m"], 3, True), RM2=f(rawv["components"]["g2"]["m"], 3, True),
        RS=f(rawv["sigma_eps_mean"], 4), WS=f(wv["sigma_eps_mean"], 4),
        DWS=f(S2("delta_obs_minus_raw"), 2), DSN=pct(1 - wv["sigma_eps_mean"] / rawv["sigma_eps_mean"], 0),
        DW=pm(S2("delta_obs_minus_raw"), S2("delta_obs_minus_raw_se"), 3, True),
        EW=pm(S2("excess_obs_minus_raw"), S2("excess_obs_minus_raw_se"), 3, True),
        EG=pm(S2("excess_obs_minus_gbm"), S2("excess_obs_minus_gbm_se"), 3, True),
        RE=f(rawv["components"]["g1"]["m_fit_se"], 4), WE=f(wv["components"]["g1"]["m_fit_se"], 4),
        CRE=f(c05r["components"]["g1"]["m_fit_se"], 4), CWE=f(c05w["components"]["g1"]["m_fit_se"], 4),
        T45=f(r045["threshold"], 3), A45=f(r045["raw"]["g1"]["m"], 3, True), B45=f(r045["raw"]["g2"]["m"], 3, True),
        F45=pct(r045["frac_retained"], 0), S45=f(r045["raw"]["sigma_eps_mean"], 4),
        MMAX=f(0.5 * (rmax["raw"]["g1"]["m"] + rmax["raw"]["g2"]["m"]), 3, True) + r" in \((m_1+m_2)/2\)",
        TMAX=f(rmax["threshold"], 4), FMAX=pct(rmax["frac_retained"], 0),
        SEL=lnk("width-cut-selection-bias", "the selection-bias page"),
        E0=f(nocut["raw"]["g1"]["m_se"], 3), E45=f(r045["raw"]["g1"]["m_se"], 3))

    caveats = ul([
        "The floor is fixed to the mean width of the sample (the existing code's choice). Other floors were "
        "not tried.",
        r"Weighting and cuts use the same observed widths that respond to the noise-driven apparent shear "
        "(see the width-vs-shear page). For the weights this has no detectable effect; for cuts it does.",
        "The threshold scan uses fit errors only (justified by the bootstrap page). Neighbouring thresholds "
        "share most galaxies, so the wiggles of the curve are strongly correlated.",
        "A response calibration (Sec. 6.4) would be needed before quoting any cut result as unbiased.",
        "100k model, Xu3 test set.",
    ])
    prov = provenance(["a_shear_analysis.py (reference_check, estimator_weights, test_r6, r6_curve)",
                       "a_shear_supplement2.py (S5 weighting selection check)",
                       f"{DIAG}/shear_bias_report.py (compose_shape_noise_regularized_weights)"],
                      ["a_shear_analysis", "a_shear_supplement2"],
                      ["results.json: reference_check.{expected, got}",
                       "results.json: T5.estimators.{raw, weighted, …}.components.*.{m, m_fit_se}",
                       "results.json: R6.rows[*].{threshold, frac_retained, n_retained, raw, weighted}",
                       "results.json: T7.matched_fraction[-1].std:raw.threshold (largest σ_ε,i)",
                       "supplement2.json: raw, weighted_observed, weighted_gbm, delta_*, excess_*"],
                      extra="<p>Paper-figure copy: <code>paper-tests/paper_figs/A_R6_width_cut</code>. "
                            "The reference numbers come from the earlier report on this model in "
                            "<code>fiber-gauge/03_g01_retrain</code>.</p>")
    answer = (f"Weighting moves (m1+m2)/2 from {f(S2('raw', 'm_avg'), 2, True)} to {f(S2('weighted_observed', 'm_avg'), 2, True)}, "
              f"entirely through down-weighting shrunk galaxies (no detectable selection); width cuts take m through zero "
              f"near σ_ε,i < {f(r045['threshold'], 3)} with {pct(r045['frac_retained'], 0)} of galaxies kept, but then overshoot to "
              f"m ≈ {f(0.5 * (rmax['raw']['g1']['m'] + rmax['raw']['g2']['m']), 2, True)}, a sign of selection bias.")
    return page(slug="weighting-and-width-cuts", title="Inverse-variance weighting and posterior-width cuts",
                code="R6 (+ S5)", section=SECTION, answer=answer,
                question=T(r"""<p>How much do the two simplest mitigations reduce the bias and the shape noise?
The first weights galaxies by the inverse of their posterior variance (with a floor). The second keeps
only galaxies with a narrow posterior, \(\sigma_{\epsilon,i} &lt; t\). How do \(m\), the ensemble shape
noise and the number of kept galaxies trade off as the threshold \(t\) changes?</p>"""),
                why=r"""<p>This is Sec. 6.3. Eason's scope note says the results must show under which cuts
\(m\) is acceptable, with the posterior-width cut the most promising. The page also has to establish
<i>why</i> each mitigation changes \(m\). A cut that reaches \(m \approx 0\) by cancelling two biases is
not a calibrated estimator.</p>""",
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


# =================================================================== page 7
def page_convention():
    T7 = R("T7")
    p1, p2 = T7["halfwidth_over_std_g1_percentiles_5_16_50_84_95"], T7["halfwidth_over_std_g2_percentiles_5_16_50_84_95"]
    mf = T7["matched_fraction"]
    rows = []
    diffs = []
    for r in mf:
        rows.append([pct(r["frac"], 0), f"{f(r['std:raw']['threshold'], 4)} / {f(r['halfwidth:raw']['threshold'], 4)}"]
                    + [f"{f(r[k]['m1'], 3, True)} / {f(r[k]['m2'], 3, True)}" for k in ("std:raw", "halfwidth:raw", "std:weighted", "halfwidth:weighted")]
                    + [pct(r["overlap_fraction"], 1)])
        if r["frac"] < 1:
            for kind in ("raw", "weighted"):
                for mm in ("m1", "m2"):
                    diffs.append((abs(r[f"std:{kind}"][mm] - r[f"halfwidth:{kind}"][mm]), r["frac"], kind, mm))
    tabmf = table(["kept fraction", "threshold: std / half-width", r"raw \(m_1/m_2\), std", r"raw \(m_1/m_2\), half-width",
                   r"weighted \(m_1/m_2\), std", r"weighted \(m_1/m_2\), half-width", "overlap of kept sets"], rows)
    tab = table(["quantity", "standard deviation (Xu et al. 2024)", "16–84% half-width"], [
        [r"ensemble \(\sigma_\epsilon\), mean", f(T7["sigma_eps_std"]["mean"], 4), f(T7["sigma_eps_halfwidth"]["mean"], 4)],
        [r"ensemble \(\sigma_\epsilon\), RMS", f(T7["sigma_eps_std"]["rms"], 4), f(T7["sigma_eps_halfwidth"]["rms"], 4)],
        [r"half-width ÷ std per galaxy, \(g_1\): 5/16/50/84/95 percentiles", "", " / ".join(f(v, 3) for v in p1)],
        [r"half-width ÷ std per galaxy, \(g_2\): 5/16/50/84/95 percentiles", "", " / ".join(f(v, 3) for v in p2)],
        ["Spearman rank correlation between the two", f(T7["spearman_sig_vs_sigq"], 4), ""],
    ])
    ratio_mean = T7["sigma_eps_halfwidth"]["mean"] / T7["sigma_eps_std"]["mean"]
    thr_shift = [r["halfwidth:raw"]["threshold"] / r["std:raw"]["threshold"] - 1 for r in mf if r["frac"] < 1]
    biggest = max(diffs)
    small = max(d for d in diffs if d[1] >= 0.2 - 1e-9)
    errs = [r[f"std:{k}"][f"{mm}_se"] for r in mf if r["frac"] < 1 for k in ("raw", "weighted") for mm in ("m1", "m2")]
    unif = 0.68 * np.sqrt(3.0)  # (q84-q16)/2 over std for a uniform distribution: 0.68 a / (a/sqrt3)
    ovl = [r["overlap_fraction"] for r in mf if r["frac"] < 1]

    theory = T(r"""
<p><b>Two ways to measure a width.</b> Xu et al. (2024) define the per-galaxy shape noise from the
posterior standard deviation. A common alternative is the half-width of the central 68% interval,
\(h = (q_{84} - q_{16})/2\), where \(q_{16}\) and \(q_{84}\) are the 16th and 84th percentiles of the
1D marginal. The half-width analogue of the shape noise is
\(\sigma^{(q)}_{\epsilon,i} = \sqrt{h_{1,i}^2 + h_{2,i}^2}\).</p>
<p><b>When they differ.</b> For a Gaussian the two are equal. For a flat-topped distribution the
half-width is larger: a uniform distribution on \([-a, a]\) has \(h = 0.68a\) and standard deviation
\(a/\sqrt3\), a ratio of $UNIF. For a distribution with heavy tails the half-width is smaller, because
the standard deviation is inflated by the tails. The shear posteriors here are cut off by the prior box,
so many are flat-topped, and we expect ratios above 1.</p>
<p><b>Why it matters.</b> The paper quotes a shape noise and cut thresholds. If the conclusions
depended on the convention, the paper would need to justify the choice. The std is the natural choice
for the other formulas in Sec. 6: the shrinkage relation \(m = -\sigma_{\rm post}^2/\sigma_\pi^2\) and
inverse-variance weights are both written in terms of the variance.</p>""", UNIF=f(unif, 2))

    procedure = T(r"""<ol>
<li>From the TF-weighted quantiles of each galaxy's \(g_1\) and \(g_2\) marginals (computed in the
extraction pass), form \(h_1\), \(h_2\) and \(\sigma^{(q)}_{\epsilon,i}\).</li>
<li>Compare the per-galaxy ratio \(h/\sigma\), the ensemble values and the rank correlation.</li>
<li>Rerun the width-cut scan with the threshold on \(\sigma^{(q)}_{\epsilon,i}\). For the weighted
estimator, use \(h^2\) in place of the variance in the weights.</li>
<li>Because a given threshold means different things in the two conventions, compare at matched
<i>kept fraction</i>: keep the best fraction \(f\) of galaxies by each convention and compare \(m\).</li>
</ol>""")

    results = (
        tab + fig("A_shear/T7_convention.png",
                  r"Left: per-galaxy ratio of the 16–84% half-width to the standard deviation, for \(g_1\) and "
                  r"\(g_2\). Right: \((m_1+m_2)/2\) against the kept fraction for cuts in each convention, raw "
                  r"(solid) and weighted (dashed).")
        + "<p><b>At matched kept fraction</b>:</p>" + tabmf
        + fig("A_shear/T7_R6_halfwidth.png",
              r"The width-cut scan of the weighting page repeated with the threshold on the half-width "
              r"\(\sigma^{(q)}_{\epsilon,i}\). The middle panel still shows the std-convention ensemble "
              r"\(\sigma_\epsilon\) of the kept galaxies. As on the weighting page, the flat dotted line on the "
              r"right axis is a plotting artefact.")
    )

    interpretation = T(r"""
<p><b>The posteriors are flat-topped, so the half-width is larger.</b> The median ratio
\(h/\sigma\) is $MED1 (\(g_1\)) and $MED2 (\(g_2\)), between the Gaussian value 1 and the uniform value
$UNIF, as expected from truncation by the prior box. The ensemble shape noise is about $RAT larger in
the half-width convention ($HW vs $SD), and the threshold that keeps a given fraction (below 100%) is larger by
$TS1–$TS2.</p>
<p><b>The ranking of galaxies hardly changes</b> (Spearman $SP; the kept sets overlap by $OV1–$OV2),
so neither do the results. At matched kept fraction the largest difference in \(m\) is $BIG (at
\(f = $BF\), $BK \(m_{$BM}\)); for \(f \ge 0.2\) all differences are below $SML. For comparison, the
statistical error of each \(m\) is $EL–$EH; the error on the <i>difference</i> is smaller, because the two
kept sets share most galaxies, so the \(f = $BF\) difference may be real, but it does not change any
conclusion.</p>
<p><b>Recommendation.</b> Use the standard deviation (Xu et al. 2024), and state the convention wherever a
shape noise or a threshold is quoted, since the numbers themselves shift by about $RAT.</p>""",
        MED1=f(p1[2], 3), MED2=f(p2[2], 3), UNIF=f(unif, 2), RAT=pct(ratio_mean - 1, 0),
        HW=f(T7["sigma_eps_halfwidth"]["mean"], 4), SD=f(T7["sigma_eps_std"]["mean"], 4),
        TS1=pct(min(thr_shift), 0), TS2=pct(max(thr_shift), 0), SP=f(T7["spearman_sig_vs_sigq"], 4),
        OV1=pct(min(ovl), 0), OV2=pct(max(ovl), 0), BIG=f(biggest[0], 3), BF=f(biggest[1], 1), BK=biggest[2],
        BM=biggest[3][-1], SML=f(small[0], 3), EL=f(min(errs), 3), EH=f(max(errs), 3))

    caveats = ul([
        f"The half-width uses weighted quantiles of {NDRAW:,} importance-weighted draws. For galaxies with a low "
        "TF effective sample size these quantiles are noisier than the standard deviation.",
        "Only the 16–84% interval was tried. The quoted numbers depend on the convention even though the "
        "conclusions do not.",
    ])
    prov = provenance(["a_shear_extract.py (weighted quantiles)", "a_shear_analysis.py (test_t7, r6_curve, matched_fraction_cut)"],
                      ["a_shear_extract", "a_shear_analysis"],
                      ["results.json: T7.{sigma_eps_std, sigma_eps_halfwidth, spearman_sig_vs_sigq}",
                       "results.json: T7.halfwidth_over_std_{g1,g2}_percentiles_5_16_50_84_95",
                       "results.json: T7.matched_fraction[*]", "results.json: T7.R6_halfwidth_rows[*]"])
    answer = (f"The choice changes no conclusion: the half-width is about {pct(ratio_mean - 1, 0)} larger than the std "
              f"for these flat-topped posteriors, but the rankings are almost identical (Spearman "
              f"{f(T7['spearman_sig_vs_sigq'], 3)}) and m at matched kept fraction differs by at most {f(biggest[0], 3)}; "
              f"use the std and state it.")
    return page(slug="width-convention", title="Standard deviation vs 16–84% half-width", code="T7", section=SECTION,
                answer=answer,
                question=r"""<p>Should the per-galaxy width be the posterior standard deviation or the half-width of
the central 68% interval? Does the choice change the shape noise, the width-cut curve, or any
conclusion?</p>""",
                why=r"""<p>The paper's scope note leaves this choice open. The shape noise and every threshold
quoted in Sec. 6.3 depend on it, so we need to know whether it matters and which one to use.</p>""",
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=prov)


PAGES = [
    page_raw(),
    page_shrinkage(),
    page_linearity(),
    page_bootstrap(),
    page_width_vs_shear(),
    page_weighting(),
    page_convention(),
]
