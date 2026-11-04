"""Explanatory pages for Sec. 6 (shear estimation): width-cut selection bias,
the matched-shear stencil / response calibration, and NPE likelihood stacking.

Imported by ``write_test_pages.py``. Every number is looked up with ``J(...)``
(campaign JSONs) or, where a JSON lacks it, with the local helpers below that
read the old benchmark JSON, a group ``report.html`` or a ``STATUS.txt``.
"""
from __future__ import annotations

import html as _html
import json as _json
import math
import re as _re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

from test_pages_common import ROOT, SCRIPTS, J, f, fig, page, pct, pm, table

SEC = "Sec. 6: Shear estimation"
OLD_BENCH = Path("/ocean/projects/phy250048p/shared/reports/likelihood-stack/04_benchmark/report.json")


# --------------------------------------------------------------------------- helpers
def T(s: str, **kw) -> str:
    """Fill <<name>> placeholders (lets LaTeX braces stay single in raw strings)."""
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", str(v))
    left = _re.findall(r"<<[A-Za-z0-9_]+>>", s)
    if left:
        raise KeyError(f"unfilled placeholders {left}")
    return s


def fx(x, nd=3, sign=True):
    """Fixed-point formatting (no scientific notation), unicode minus."""
    s = f"{float(x):+.{nd}f}" if sign else f"{float(x):.{nd}f}"
    return s.replace("-", "−")


def fxpm(x, e, nd=3, sign=True):
    return f"{fx(x, nd, sign)} ± {fx(e, nd, False)}"


def fint(x):
    return f"{int(round(float(x))):,}"


@lru_cache(maxsize=None)
def _old():
    return _json.loads(OLD_BENCH.read_text())


def JO(*keys):
    """Lookup in the old likelihood-stack benchmark report.json; raise on a missing key."""
    node = _old()
    for k in keys:
        if isinstance(node, dict):
            if k not in node:
                raise KeyError(f"missing key old_bench/{'/'.join(map(str, keys))}")
            node = node[k]
        else:
            node = node[int(k)]
    return node


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True
        if tag in ("p", "tr", "h2", "h3", "li", "div", "figcaption"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


@lru_cache(maxsize=None)
def _report_text(rel: str) -> str:
    p = _Text()
    p.feed((ROOT / rel).read_text())
    return "".join(p.out)


def H(rel: str, pattern: str):
    """Numbers that exist only in a group report's text (regex groups, as printed there)."""
    m = _re.search(pattern, _report_text(rel))
    if not m:
        raise KeyError(f"pattern not found in {rel}: {pattern}")
    return m.groups()


def status_block(dirname: str) -> str:
    """The SLURM-job block of a group's STATUS.txt, verbatim."""
    lines = (ROOT / dirname / "STATUS.txt").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("SLURM jobs"))
    out = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip() or not l[0].isspace():
            break
        out.append(l)
    return f"<pre style='font-size:12px;overflow-x:auto'>{_html.escape(chr(10).join(out))}</pre>"


def ul(*items) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def P(*paras) -> str:
    return "".join(f"<p>{p}</p>" for p in paras)


def code(s):
    return f"<code>{_html.escape(s)}</code>"


def sc(name):
    return code(f"{SCRIPTS}/{name}")


# ============================================================================ page 1
OBS = "σ_ε,i (observed)"
GBM = "GBM σ from shear-free truths"
COSI = "true cos i (σ-ranked)"
RHOI = "ρ_I image SNR (σ-ranked)"
RHOS = "ρ_S Hα SNR (σ-ranked)"


def _curve(name, i, key):
    return J("shear", "T2a", "curves", name, i, key)


def _sel(src, variant, i, *keys):
    return J(src, variant, "selection", i, *keys)


def _rtot_diag(src, variant, i):
    r = _sel(src, variant, i, "value", "R_tot")
    return 0.5 * (r[0][0] + r[1][1])


def _r6_row(thr):
    for r in J("shear", "R6", "rows"):
        if math.isclose(r["threshold"], thr):
            return r
    raise KeyError(f"R6 threshold {thr}")


def page_selection():
    fr = J("shear", "T2a", "fractions")
    s2 = lambda kind, i, k: J("supp", "S2_paired_selection_excess", kind, i, k)
    s2fr = [s2("raw", i, "frac") for i in range(7)]
    i02 = s2fr.index(0.2)
    i07 = s2fr.index(0.7)
    i01 = s2fr.index(0.1)
    i05 = s2fr.index(0.5)
    X = ("stencil_xu3", "tf_equal")
    PR = ("stencil_prop", "proposal")
    thr = [_sel(*X, i, "threshold") for i in range(4)]
    i05t = thr.index(0.05)
    i075t = thr.index(0.075)
    r6 = _r6_row(0.045)
    sig_pi = J("shear", "sigma_pi")

    answer = T(
        "Yes. Cutting on the observed posterior width adds a positive selection bias: "
        "about <<a>> in m when ~20% of galaxies are kept (Xu3 cache, relative to a shear-free cut), and a "
        "directly measured selection response of <<b>> (+<<c>> of the shear response) at a cut "
        "keeping <<d>> of a matched-shear stencil. Loose cuts (keeping ≥70%) and the regularized "
        "inverse-variance weights show no detectable excess (<<e>>).",
        a=fxpm(s2("raw", i02, "selection_excess"), s2("raw", i02, "selection_excess_se")),
        b=fxpm(_sel(*X, i05t, "R_sel_diag"), _sel(*X, i05t, "R_sel_diag_se")),
        c=pct(_sel(*X, i05t, "R_sel_over_R_gamma"), 0),
        d=pct(_sel(*X, i05t, "value", "frac_by_state")[0], 0),
        e=fxpm(J("supp2", "excess_obs_minus_gbm"), J("supp2", "excess_obs_minus_gbm_se")),
    )

    question = P(
        T(r"""Keeping only galaxies with a narrow shear posterior (small \(\sigma_{\epsilon,i}\)) is
the most promising way to reduce the shrinkage bias of the posterior-mean shear estimator
(see <a href='weighting-and-width-cuts.html'>weighting and width cuts</a>). On the Xu3 test set the
raw \(m\) passes through zero near \(\sigma_{\epsilon,i} < <<thr>>\) (kept fraction <<kf>>,
\(m_1 = <<m1>>\), \(m_2 = <<m2>>\)). This page asks whether that number can be trusted, or whether
part of it is a <b>selection bias</b>: the cut variable \(\sigma_{\epsilon,i}\) is itself
computed from noisy data and might prefer galaxies whose noise happens to push \(\hat g\) along
the true shear.""",
          thr=f(r6["threshold"], 3), kf=pct(r6["frac_retained"], 0),
          m1=pm(r6["raw"]["g1"]["m"], r6["raw"]["g1"]["m_se"]),
          m2=pm(r6["raw"]["g2"]["m"], r6["raw"]["g2"]["m_se"])),
        r"""Two independent tests answer it: (1) on the Xu3 posterior cache, compare the bias after
cutting on the observed \(\sigma_{\epsilon,i}\) with the bias after cutting, at the same kept
fraction, on a prediction of \(\sigma_{\epsilon,i}\) built only from shear-free true galaxy
properties; (2) on a matched-shear stencil (the same galaxy and the same noise rendered at
slightly different shears), measure the selection response of the cut directly, as done in
metacalibration.""")

    why = P(
        r"""In Sec. 6 of the paper the width cut is presented as a mitigation of shrinkage. If the cut
also creates a selection bias, then the "\(m \approx 0\)" point on the width-cut curve is a
cancellation between two effects of opposite sign (shrinkage pulls \(m\) negative, selection
pushes it positive), not a sign that the narrow-posterior galaxies are unbiased. That changes
how the curve may be presented, and it means a calibration of a cut sample needs a selection
term, not only the shear response.""",
        r"""The same question matters for inverse-variance weighting, which also uses the observed
posterior widths.""")

    theory = (
        "<h3>Where a selection bias comes from</h3>"
        + P(
            r"""Write the estimate of galaxy \(i\) as \(\hat g_i = R_i\, g + n_i\), where \(g\) is the
true shear, \(R_i\) is the galaxy's response (\(R_i = 1\) for an unbiased estimator; shrinkage makes
it smaller) and \(n_i\) is the error caused by pixel and spectral noise (and by the unknown intrinsic
shape). Averaging over a sample \(S\):
\[ \langle \hat g\rangle_S = \langle R\rangle_S\, g + \langle n\rangle_S . \]
If membership of \(S\) does not depend on the noise (e.g. a cut on a true, shear-free galaxy
property), then \(\langle n\rangle_S = 0\) and the slope is just the mean response of the
selected galaxies. If membership depends on a quantity that is computed from the same noisy data,
and that quantity is correlated with the part of \(n_i\) that points along \(g\), then
\(\langle n\rangle_S\) is no longer zero: it is proportional to \(g\) and adds to the slope. That
extra slope is the selection bias. It has nothing to do with the estimator being biased
galaxy-by-galaxy; it comes from which galaxies we keep.""",
            r"""The mechanism found here: a noise realization that makes a galaxy look <i>more</i>
sheared (larger \(|\hat g|\)) also makes its posterior <i>narrower</i>. The network has learned that
the shear posterior narrows as \(|g|\) grows (on the Xu3 set \(d\ln\sigma_{\epsilon,i}/d|g|\)
at fixed galaxy properties is """ + pm(J("shear", "T1", "|g|", "dlnsigma_dg_residualized"),
                                       J("shear", "T1", "|g|", "dlnsigma_dg_residualized_se"))
            + r""", see <a href='width-vs-true-shear.html'>width vs true shear</a>; part of this is the
truncation of the posterior by the \(\pm 0.1\) prior box). Noise that mimics extra shear triggers the
same narrowing. A cut \(\sigma_{\epsilon,i} < t\) therefore preferentially keeps galaxies scattered
outward along the true shear, and \(\langle n\rangle_S\) is positive along \(g\): \(m\) goes up.""",
        )
        + "<h3>Test 1: matched-fraction comparison with a shear-free proxy (Xu3 cache)</h3>"
        + P(
            r"""Cutting on anything that correlates with posterior width changes \(m\) even without
selection bias, simply because narrow posteriors shrink less: for a Gaussian model
\(m \approx -\sigma_{\rm post}^2/\sigma_\pi^2\), with \(\sigma_\pi = 0.2/\sqrt{12} = """
            + f(sig_pi, 4) + r"""\) the standard deviation of the uniform shear prior on \([-0.1, 0.1]\)
(see <a href='gaussian-shrinkage.html'>Gaussian shrinkage</a>). So "\(m\) changes after the cut"
proves nothing. The test compares two cuts that keep the <i>same fraction</i> \(f\) of galaxies:""",
        )
        + ul(
            r"""<b>observed cut:</b> rank galaxies by the observed \(\sigma_{\epsilon,i}\) and keep the
narrowest fraction \(f\);""",
            T(r"""<b>shear-free cut:</b> rank galaxies by a <i>prediction</i> of
\(\ln\sigma_{\epsilon,i}\) computed only from true, shear-independent galaxy properties
(\(\cos i\), catalog image S/N \(\rho_I\) and H\(\alpha\) S/N \(\rho_S\), \(r\)-band magnitude,
\(v_{\rm circ}\), half-light radius, disk scale radius, H\(\alpha\) flux, \(v_0\), and
\(\cos 2\theta_{\rm int}, \sin 2\theta_{\rm int}\)). The prediction is a gradient-boosted regression
tree model (scikit-learn <code>HistGradientBoostingRegressor</code>) trained with 5-fold
cross-fitting: each galaxy's prediction comes from a model that never saw that galaxy, so it cannot
memorize the galaxy's own noise. It explains \(R^2 = <<r2>>\) of the variance of
\(\ln\sigma_{\epsilon,i}\) and ranks galaxies almost like the observed width (Spearman
<<sp>>), but by construction it cannot respond to the shear or the noise.""",
              r2=f(J("supp2", "gbm_r2"), 2),
              sp=f(J("shear", "T2a", "proxy_spearman_with_sigma", GBM), 2)),
        )
        + P(
            r"""Because the proxy is imperfect, the galaxies it keeps have somewhat broader posteriors
than those kept by the observed cut, so part of the difference \(\Delta m = m_{\rm obs\,cut} -
m_{\rm proxy\,cut}\) is ordinary shrinkage. This expected part, \(\Delta m_{\rm toy}\), is computed
with a 1D toy: \(g \sim U(-0.02, 0.02)\), a Gaussian likelihood of width \(s\), and the
\(U(-0.1, 0.1)\) prior give a truncated-normal posterior whose mean has a known slope
\(m(s)\) and a known width; each galaxy's per-component posterior standard deviation is mapped
to the toy \(m\), and the retained galaxies are averaged with the estimator's weights. The
<b>selection excess</b> is
\[ \text{excess}(f) = \big[m_{\rm obs\,cut}(f) - m_{\rm proxy\,cut}(f)\big] - \big[m_{\rm toy,obs}(f)
- m_{\rm toy,proxy}(f)\big]. \]
It is zero if the observed cut acts like a shear-free cut of the same narrowness. The toy is only
used for a difference between two similar samples, so its known absolute error (it under-predicts
\(|m|\) somewhat, see the shrinkage page) largely cancels.""",
            r"""Errors come from a <b>paired bootstrap</b>: galaxies are resampled once and both cuts are
evaluated on the same resample. The two kept samples overlap strongly, so the error on their
difference is much smaller than the two individual errors combined.""",
            r"""As a direct check of the mechanism, the residual \(\ln\sigma_{\epsilon,i} -
\widehat{\ln\sigma}_{\rm proxy}\) (the part of the width not explained by true galaxy properties)
is correlated with the <b>radial error</b> \((\hat{\mathbf g} - \mathbf g)\cdot\mathbf g/|\mathbf g|\),
the component of the estimation error along the true shear direction, at fixed \(|\mathbf g|\).""",
        )
        + "<h3>Test 2: metacalibration-style selection response (matched-shear stencil)</h3>"
        + P(
            r"""A matched-shear stencil renders each of 1000 base galaxies five times, at applied shears
\((0,0)\), \((\pm\Delta g, 0)\) and \((0, \pm\Delta g)\) with \(\Delta g = """
            + f(J("stencil_xu3", "delta_g"), 2) + r"""\), with <i>identical</i> intrinsic parameters and the
<i>same</i> pixel and spectral noise pattern (details on the
<a href='response-calibration.html'>response-calibration page</a>). For each base galaxy \(b\) the
shear response is the finite difference
\(R_{ij,b} = [\hat g_i(+\Delta g\,\mathbf e_j) - \hat g_i(-\Delta g\,\mathbf e_j)]/(2\Delta g)\).
Let \(S_s\) be the set of bases that pass the cut \(\sigma_{\epsilon} < t\) when \(\sigma_\epsilon\) is
measured on the version sheared to state \(s\) (same noise, sheared image). Following
metacalibration (Huff &amp; Mandelbaum 2017; Sheldon &amp; Huff 2017), the script computes""",
        )
        + r"""\[ R_{\gamma,ij} = \big\langle R_{ij,b}\big\rangle_{S_0}, \qquad
R_{{\rm sel},ij} = \frac{\langle \hat g_i(0)\rangle_{S_{+j}} - \langle \hat g_i(0)\rangle_{S_{-j}}}{2\Delta g}, \qquad
R_{{\rm tot},ij} = \frac{\langle \hat g_i(+j)\rangle_{S_{+j}} - \langle \hat g_i(-j)\rangle_{S_{-j}}}{2\Delta g}. \]"""
        + P(
            r"""\(R_\gamma\) is the ordinary response of the galaxies that pass the cut at zero shear.
\(R_{\rm sel}\) keeps the estimates fixed at their zero-shear values and changes only <i>which</i>
galaxies are selected: it asks how much the mean estimate would move with the shear purely because
the selected set moves. To first order in \(\Delta g\), \(R_{\rm tot} = R_\gamma + R_{\rm sel}\), and
the mean estimate of the cut sample is \(\langle\hat g\rangle_S \approx (R_\gamma + R_{\rm sel})\,g + c\).
Calibrating a cut sample by dividing by \(R_\gamma\) alone gives
\(\langle\hat g\rangle_S/R_\gamma \approx (1 + R_{\rm sel}/R_\gamma)\, g\), i.e. a residual
multiplicative bias \(m \approx R_{\rm sel}/R_\gamma\). The shared noise is what makes this
measurable with only 1000 galaxies: the \(\pm\) versions differ only by the shear, so a galaxy
changes its selection between them only if the shear itself moves its width across the threshold.
(The posterior samples are drawn independently for each version, so Monte Carlo noise in
\(\sigma_\epsilon\) also flips a few galaxies; that adds scatter but no bias, because it is
symmetric between the \(+\) and \(-\) versions.)""",
            r"""The script also records the change of the selected fraction with shear,
\(df/dg = [f(S_{+j}) - f(S_{-j})]/(2\Delta g)\), and the number of "flips" (bases selected in one
of the \(\pm\) versions but not the other).""",
        )
        + "<h3>Why weighting can behave differently</h3>"
        + P(
            r"""The regularized weights \(w_i \propto 1/(\sigma_{{\rm gal},i}^2 + \sigma_{\rm floor}^2)\)
(with \(\sigma_{{\rm gal},i} = \sigma_{\epsilon,i}/\sqrt 2\) the per-component width and the floor
equal to the sample mean of \(\sigma_{\rm gal}\)) depend on the same noisy widths, so the same
mechanism is possible in principle. The same comparison is therefore made with weights built from
the shear-free predicted width. A plausible reason why the effect is much weaker for weights (this
is a heuristic; the measurement below is what establishes it): noise moves \(\sigma_{\epsilon,i}\)
by a few percent, which changes a smooth, floor-limited weight by a few percent, whereas a hard
threshold turns the same small shift into all-or-nothing membership for every galaxy near the
threshold. When only a small fraction is kept, a large share of the kept galaxies lies close to
the threshold, which is why the excess grows as the cut tightens.""")
    )

    procedure = ul(
        T(r"""<b>Input.</b> The Xu3 posterior cache (<<n>> galaxies, \(10^4\) posterior samples each,
TF-weighted), reduced to per-galaxy TF-weighted means and standard deviations by
<code>a_shear_extract.py</code>.""", n=fint(J("shear", "n_galaxies"))),
        r"""<b>Matched-fraction cuts.</b> For \(f = 0.1, 0.2, \dots, 1.0\) keep the \(f\) narrowest
galaxies according to each ranking variable: the observed \(\sigma_{\epsilon,i}\); true \(\cos i\),
\(\rho_I\), \(\rho_S\) (each ranked by the mean observed \(\sigma_{\epsilon,i}\) in 50 quantile
bins of that variable, which handles the non-monotonic \(\cos i\) dependence); and the cross-fitted
shear-free prediction. Fit \(\hat g - g = m g + c\) on \(|g| < 0.02\) for \(g_1\) and \(g_2\) with
equal weights and report \((m_1 + m_2)/2\) (<code>test_t2a</code> in <code>a_shear_analysis.py</code>).""",
        T(r"""<b>Selection excess.</b> For \(f\) = 0.1–0.9, compute \(\Delta m\), \(\Delta m_{\rm toy}\) and
the excess for the observed-width cut minus the shear-free cut, with equal weights ("raw") and
with the regularized weights recomputed on the kept galaxies ("weighted"); <<nb>> paired bootstrap
resamples (<code>a_shear_supplement.py</code>, S2).""", nb=J("supp", "S2_paired_selection_excess", "n_boot")),
        r"""<b>Mechanism.</b> Regress the width residual on \(|\mathbf g|\) and the radial error; compare the
mean radial error of low-\(|g|\) galaxies kept by the two cuts (<code>a_shear_supplement.py</code>, S3).""",
        T(r"""<b>Weights without a cut.</b> Same comparison for the regularized weights built from the
observed variances vs from the predicted width; <<nb>> paired bootstrap resamples
(<code>a_shear_supplement2.py</code>, S5).""", nb=J("supp2", "n_boot")),
        T(r"""<b>Stencil selection response.</b> On the Xu3-like stencil (<<nb>> base galaxies × 5 shears,
TF-weighted posterior means, equal base weights) and on the training-proposal stencil (unweighted
posterior means), compute \(R_\gamma\), \(R_{\rm sel}\), \(R_{\rm tot}\), \(df/dg\) and flips for
\(t = 0.03, 0.05, 0.075, 0.1\); errors from 2000 bootstrap resamples of base galaxies (all five
versions of a base move together) (<code>selection_response</code> in <code>b_t2b_analysis.py</code>).""",
          nb=J("stencil_xu3", "nbase")),
    )

    # ---------------- results
    rows = []
    for i in (0, 1, 2, 4, 6, 9):
        rows.append([pct(fr[i], 0)]
                    + [x for nm in (OBS, GBM, COSI) for x in (f(_curve(nm, i, "mean_sigma_eps"), 4),
                                                           fxpm(_curve(nm, i, "m_avg"), _curve(nm, i, "m_avg_se")))])
    t_curves = table(["kept fraction", "observed cut: mean σ<sub>ε,i</sub>", "observed cut: (m₁+m₂)/2",
                      "shear-free prediction cut: mean σ<sub>ε,i</sub>", "shear-free cut: (m₁+m₂)/2",
                      "true cos i cut: mean σ<sub>ε,i</sub>", "cos i cut: (m₁+m₂)/2"], rows)
    rows = []
    for i in range(7):
        rows.append([pct(s2("raw", i, "frac"), 0), fx(s2("raw", i, "m_sigma_cut")), fx(s2("raw", i, "m_gbm_cut")),
                     fxpm(s2("raw", i, "delta_m"), s2("raw", i, "delta_m_se")), fx(s2("raw", i, "delta_toy")),
                     "<b>" + fxpm(s2("raw", i, "selection_excess"), s2("raw", i, "selection_excess_se")) + "</b>",
                     fxpm(s2("weighted", i, "delta_m"), s2("weighted", i, "delta_m_se")),
                     "<b>" + fxpm(s2("weighted", i, "selection_excess"), s2("weighted", i, "selection_excess_se")) + "</b>"])
    t_s2 = table(["kept fraction f", "m, observed cut", "m, shear-free cut", "Δm", "Δm<sub>toy</sub>",
                  "selection excess (equal weights)", "Δm (weighted)", "selection excess (weighted)"], rows)
    S3 = lambda *k: J("supp", "S3_mechanism", *k)
    rows = [[pct(S3("radial_resid_by_cut", i, "frac"), 0),
             fxpm(1e3 * S3("radial_resid_by_cut", i, "mean_radial_resid_sigma_cut_lowg"),
                  1e3 * S3("radial_resid_by_cut", i, "se_sigma_cut"), 2),
             fxpm(1e3 * S3("radial_resid_by_cut", i, "mean_radial_resid_gbm_cut_lowg"),
                  1e3 * S3("radial_resid_by_cut", i, "se_gbm_cut"), 2)] for i in range(4)]
    t_s3 = table(["kept fraction", "mean radial error ×10³, observed-width cut",
                  "mean radial error ×10³, shear-free cut"], rows)
    q = lambda k: J("supp2", k)
    t_s5 = table(["estimator (no cut)", "(m₁+m₂)/2"], [
        ["equal weights (raw)", fx(q("raw")["m_avg"])],
        ["regularized weights from the observed variances (existing)", fx(q("weighted_observed")["m_avg"])],
        ["same weights from the shear-free predicted width", fx(q("weighted_gbm")["m_avg"])],
        ["observed − predicted weights: Δm", fxpm(q("delta_obs_minus_gbm"), q("delta_obs_minus_gbm_se"))],
        ["observed − predicted weights: selection excess (Δm − Δm<sub>toy</sub>)",
         "<b>" + fxpm(q("excess_obs_minus_gbm"), q("excess_obs_minus_gbm_se")) + "</b>"],
        ["weighted − raw: Δm", fxpm(q("delta_obs_minus_raw"), q("delta_obs_minus_raw_se"))],
        ["weighted − raw: excess over the toy", fxpm(q("excess_obs_minus_raw"), q("excess_obs_minus_raw_se"))],
    ])

    def sel_rows(src, variant, label):
        out = []
        for i in range(3):
            g = lambda *k: _sel(src, variant, i, *k)
            fl = g("n_flips")
            out.append([label, f(g("threshold"), 3), pct(g("value", "frac_by_state")[0], 1),
                        f"{fxpm(g('value', 'dfrac_dg')[0], g('se', 'dfrac_dg')[0], 2)} / "
                        f"{fxpm(g('value', 'dfrac_dg')[1], g('se', 'dfrac_dg')[1], 2)}",
                        f(g("R_gamma_diag"), 3),
                        "<b>" + fxpm(g("R_sel_diag"), g("R_sel_diag_se")) + "</b>",
                        fx(g("R_sel_significance_sigma"), 1, sign=False),
                        f(g("R_gamma_diag") + g("R_sel_diag"), 3), f(_rtot_diag(src, variant, i), 3),
                        pct(g("R_sel_over_R_gamma"), 1),
                        f"{fl['g1_plus_vs_g1_minus']} / {fl['g2_plus_vs_g2_minus']}"])
        return out

    t_sel = table(["stencil, estimator", "cut t", "kept at g = 0", "df/dg₁ / df/dg₂", "R<sub>γ</sub> (diag. mean)",
                   "R<sub>sel</sub> (diag. mean)", "R<sub>sel</sub>/error", "R<sub>γ</sub> + R<sub>sel</sub>",
                   "R<sub>tot</sub> (direct)", "R<sub>sel</sub>/R<sub>γ</sub>", "flips g₁ / g₂"],
                  sel_rows(*X, "Xu3-like, TF-weighted mean") + sel_rows(*PR, "training-proposal, unweighted mean"))
    t_cmp = table(["cut", "stencil R<sub>sel</sub> (Xu3-like)", "Xu3-cache selection excess (equal weights)"], [
        [T("t = <<t>> (kept <<k>> of stencil) vs f = <<f>>", t=f(thr[i05t], 3),
           k=pct(_sel(*X, i05t, "value", "frac_by_state")[0], 0), f=pct(s2fr[i02], 0)),
         fxpm(_sel(*X, i05t, "R_sel_diag"), _sel(*X, i05t, "R_sel_diag_se")),
         fxpm(s2("raw", i02, "selection_excess"), s2("raw", i02, "selection_excess_se"))],
        [T("t = <<t>> (kept <<k>> of stencil) vs f = <<f>>", t=f(thr[i075t], 3),
           k=pct(_sel(*X, i075t, "value", "frac_by_state")[0], 0), f=pct(s2fr[i07], 0)),
         fxpm(_sel(*X, i075t, "R_sel_diag"), _sel(*X, i075t, "R_sel_diag_se")),
         fxpm(s2("raw", i07, "selection_excess"), s2("raw", i07, "selection_excess_se"))],
    ])
    maxsig = J("stencil_xu3", "sigma_eps_zero_pct")[4]

    results = (
        "<h3>Test 1a: bias against kept fraction for each ranking</h3>" + t_curves
        + P(r"""Equal weights, low-\(|g|\) linear fit, errors are fit errors. At the same kept fraction the
observed-width cut keeps narrower posteriors than any shear-free ranking (it is the only ranking
that uses the actual width), so the columns cannot be compared row by row without the shrinkage
correction below. The \(\rho_I\) and \(\rho_S\) rankings are too weakly related to the width (Spearman """
            + f(J("shear", "T2a", "proxy_spearman_with_sigma", RHOI), 2) + " and "
            + f(J("shear", "T2a", "proxy_spearman_with_sigma", RHOS), 2) + r""") to reach narrow samples;
they are in the figure and the JSON.""")
        + fig("A_shear/T2a_selection.png",
              r"""Left: \((m_1+m_2)/2\) against the mean \(\sigma_{\epsilon,i}\) of the kept sample, for cuts keeping
10–100% of Xu3 galaxies ranked by the observed width and by shear-independent true quantities (\(\cos i\),
\(\rho_I\), \(\rho_S\), each ranked by its bin-mean width, and the cross-fitted shear-free prediction).
Right: \(m\) minus the uniform-prior toy prediction for the same kept galaxies, against the kept
fraction. The shear-free rankings follow one common curve; the observed-width cut leaves it when it keeps
fewer than about half of the galaxies.""")
        + "<h3>Test 1b: selection excess at matched kept fraction</h3>" + t_s2
        + P(T(r"""Errors are paired-bootstrap standard deviations. The toy correction removes only a minority of
\(\Delta m\) at small \(f\) (e.g. \(\Delta m = <<dm>>\) vs \(\Delta m_{\rm toy} = <<dt>>\) at
\(f = <<f>>\)), so the conclusion does not hinge on the toy.""",
              dm=fx(s2("raw", i01, "delta_m")), dt=fx(s2("raw", i01, "delta_toy")), f=pct(s2fr[i01], 0)))
        + fig("A_shear/S2_S4_selection_and_shrinkage.png",
              r"""Left panel (this test): selection excess of the observed-width cut relative to the shear-free
cut, with paired-bootstrap \(\pm1\sigma\) bars, for equal weights (solid) and regularized weights (dashed);
grey lines are \(\Delta m\) before the toy correction. The right panel belongs to the
<a href='gaussian-shrinkage.html'>shrinkage test</a> (bias in bins of the shear-free predicted width).""")
        + "<h3>Test 1c: mechanism</h3>"
        + P(T(r"""At fixed \(|\mathbf g|\), the width residual falls with the radial error with slope
<<s1>> (per unit radial error; with \(|\hat{\mathbf g}|\) instead of the radial error the slope is <<s2>>).
Spearman rank correlations: <<r1>> between the width residual and the radial error, <<r2>> between the width
residual and \(|\hat{\mathbf g}|\) (the latter is not conditioned on \(|\mathbf g|\), so it also contains the
true-shear dependence of the width). Among low-\(|g|\) galaxies (\(|\mathbf g| < 0.02\)), the kept sample of
the observed cut has a more positive mean radial error than the shear-free cut:""",
              s1=pm(S3("dresid_dradial_at_fixed_absg"), S3("se"), 2),
              s2=pm(S3("dresid_dabsghat_at_fixed_absg"), S3("se_absghat"), 2),
              r1=f(S3("spearman_resid_vs_radial"), 3), r2=f(S3("spearman_resid_vs_absghat"), 3)))
        + t_s3
        + P(r"""The radial error is negative on average for every sample because of shrinkage (the estimates sit
inside the true shear); what matters is the difference between the two columns, which is largest for the
tightest cut.""")
        + "<h3>Test 1d: inverse-variance weights without a cut</h3>" + t_s5
        + "<h3>Test 2: selection response on the matched-shear stencil</h3>" + t_sel
        + P(T(r"""Diagonal means are \((R_{11}+R_{22})/2\). "\(R_{\rm sel}\)/error" is the diagonal \(R_{\rm sel}\)
divided by its bootstrap error. At \(t = 0.1\) nothing is cut (the largest widths are
\(\approx <<mx>>\), the 95th percentile of the zero-shear widths on the Xu3-like stencil; a posterior
equal to the prior has \(\sigma_\epsilon = \sqrt2\,\sigma_\pi = <<ceil>>\)), so \(R_{\rm sel} = 0\) there. At
\(t = 0.03\) the Xu3-like stencil keeps only <<n03>> bases, too few to measure \(R_{\rm sel}\).
\(R_\gamma + R_{\rm sel}\) agrees with the directly computed \(R_{\rm tot}\), as it should to first order.
Every \(df/dg\) is within about one error of zero: the shear does not change <i>how many</i> galaxies pass,
it changes <i>which</i> ones (galaxies whose estimate lines up with the applied shear become narrower and
enter the cut).""",
              mx=f(maxsig, 3), ceil=f(math.sqrt(2) * sig_pi, 3), n03=_sel(*X, 0, "n_selected_zero")))
        + fig("B_inference/t2b_selection_response.png",
              r"""Left: diagonal selection response \(R_{\rm sel}\) against the cut threshold \(t\) (bootstrap
\(\pm1\sigma\)). Middle: \(R_\gamma\) (solid) and \(R_{\rm tot}\) (dashed). Right: fraction of stencil bases
kept at zero shear. Blue: Xu3-like stencil, TF-weighted posterior mean (the production estimator). Orange:
training-proposal stencil, unweighted posterior mean. Green: training-proposal stencil, TF-weighted mean
with TF population weights on the bases (effective sample size """ + f(J("stencil_prop", "tf_population_ess"), 0)
              + " of " + str(J("stencil_prop", "nbase")) + ").")
        + "<h3>The two tests agree where they overlap</h3>" + t_cmp
        + P(r"""Both numbers are the extra response (i.e. extra \(m\)) caused by the selection, for the same
estimator (TF-weighted posterior mean, equal galaxy weights). The kept fractions are close but not
identical, and the stencil population differs from Xu3 (see caveats).""")
    )

    interpretation = P(
        T(r"""<b>Width cuts are not selection-free.</b> Cutting on the observed \(\sigma_{\epsilon,i}\) raises
\(m\) by a selection term that grows as the cut tightens: from consistent with zero when ≥70% of galaxies are
kept to <<e05>> at 50%, <<e02>> at 20% and <<e01>> at 10% (equal weights). The stencil measures the same
effect directly (<<rs>> at a cut keeping <<k>>, i.e. <<ratio>> of \(R_\gamma\)). The "\(m \approx 0\)" of the
width-cut curve near \(\sigma_{\epsilon,i} \approx <<thr>>\) is therefore a cancellation: shrinkage (a
shear-free cut keeping the same <<f02>> of galaxies has \(m = <<mg>>\)) plus a positive selection term of the size
listed above.""",
          e05=fxpm(s2("raw", i05, "selection_excess"), s2("raw", i05, "selection_excess_se")),
          e02=fxpm(s2("raw", i02, "selection_excess"), s2("raw", i02, "selection_excess_se")),
          e01=fxpm(s2("raw", i01, "selection_excess"), s2("raw", i01, "selection_excess_se")),
          rs=fxpm(_sel(*X, i05t, "R_sel_diag"), _sel(*X, i05t, "R_sel_diag_se")),
          k=pct(_sel(*X, i05t, "value", "frac_by_state")[0], 0),
          ratio=pct(_sel(*X, i05t, "R_sel_over_R_gamma"), 0), thr=f(r6["threshold"], 3),
          mg=fx(s2("raw", i02, "m_gbm_cut")), f02=pct(s2fr[i02], 0)),
        T(r"""<b>What this means for calibration.</b> A cut sample calibrated with its shear response alone
(\(R_\gamma\), or the \(R(\sigma)\) model of the <a href='response-calibration.html'>next page</a>) keeps a
residual \(m \approx R_{\rm sel}/R_\gamma\), about <<r05>> at \(t = <<t05>>\) and <<r075>> at
\(t = <<t075>>\) on the Xu3-like stencil. The correction needs the metacalibration-style \(R_{\rm sel}\) as
well, measured on a stencil that resembles the survey.""",
          r05=pct(_sel(*X, i05t, "R_sel_over_R_gamma"), 0), t05=f(thr[i05t], 2),
          r075=pct(_sel(*X, i075t, "R_sel_over_R_gamma"), 0), t075=f(thr[i075t], 3)),
        T(r"""<b>Weighting is selection-safe at this precision.</b> The regularized inverse-variance weights built
from the observed widths and from the shear-free prediction differ by \(\Delta m = <<d>>\), fully accounted for
by shrinkage (excess <<e>>). The whole gain from weighting over equal weights (<<dw>>) is what shrinkage
predicts from giving narrow posteriors more weight (excess <<ew>>).""",
          d=fxpm(q("delta_obs_minus_gbm"), q("delta_obs_minus_gbm_se")),
          e=fxpm(q("excess_obs_minus_gbm"), q("excess_obs_minus_gbm_se")),
          dw=fxpm(q("delta_obs_minus_raw"), q("delta_obs_minus_raw_se")),
          ew=fxpm(q("excess_obs_minus_raw"), q("excess_obs_minus_raw_se"))),
        r"""<b>For the paper (Sec. 6.3).</b> Present weighting as safe; present width cuts as a trade-off curve
whose \(m\) includes a selection response, quote \(R_{\rm sel}\), or restrict cuts to loose ones (kept
fraction ≳ 70%). Do not write that the width cut "removes" the bias.""",
    )

    caveats = ul(
        r"""The cache test relies on the proxy being shear-free (true by construction: it uses only true,
shear-independent inputs and cross-fitting) and on the toy correction for the remaining width mismatch. The toy
is an approximation, but at small kept fractions it removes only a minority of \(\Delta m\).""",
        T(r"""At the loosest cut where both tests measure something they differ mildly: the stencil gives
\(R_{\rm sel} = <<a>>\) at \(t = <<t>>\) (kept <<k>>), the cache comparison <<b>> at \(f = <<f>>\). Possible
reasons: the stencil measures the response at \(|g| \le 0.01\) while the cache fit uses \(|g| < 0.02\);
the Xu3-like stencil lacks Xu3's narrowest posteriors; the kept fractions are not identical. Neither changes
the conclusion for tight cuts.""",
          a=fxpm(_sel(*X, i075t, "R_sel_diag"), _sel(*X, i075t, "R_sel_diag_se")), t=f(thr[i075t], 3),
          k=pct(_sel(*X, i075t, "value", "frac_by_state")[0], 0),
          b=fxpm(s2("raw", i07, "selection_excess"), s2("raw", i07, "selection_excess_se")), f=pct(s2fr[i07], 0)),
        r"""The stencil has 1000 base galaxies; the tightest cut (\(t = 0.03\)) keeps too few of them to measure
\(R_{\rm sel}\). The narrowest stencil galaxies also show a suspicious response \(R > 1\) (face-on disks; see
the <a href='response-calibration.html'>response-calibration page</a>), which may affect tight cuts.""",
        r"""The weighting result covers the existing regularized weights (with the floor). Other weighting
schemes, e.g. pure inverse variance without a floor, were not tested.""",
        r"""In real data the truth-level proxy is not available. A cut on noisy observables (a fitted
\(\cos i\), measured S/N) would itself carry some selection response, so the proxy cut is a diagnostic here,
not a ready-made alternative.""",
        r"""Thresholds refer to the standard-deviation convention for \(\sigma_{\epsilon,i}\); the 16–84%
half-width shifts thresholds but not the conclusions (see <a href='width-convention.html'>width
convention</a>). All numbers are from the 100k-trained model; the paper model is trained on 1M galaxies.""",
    )

    provenance = (
        ul(
            "Group reports: <a href='../A_shear/report.html'>Xu3-cache shear tests</a> (sections T2a, S2, S3, S5) "
            "and <a href='../B_inference/report.html'>matched-shear stencil</a> (section 6).",
            "Scripts: " + ", ".join(sc(s) for s in ("a_shear_extract.py", "a_shear_analysis.py",
                                                     "a_shear_supplement.py", "a_shear_supplement2.py",
                                                     "b_t2b_analysis.py", "b_make_report.py")) + ".",
            "Cache-test functions: <code>gbm_predicted_log_sigma</code>, <code>uniform_prior_toy</code>, "
            "<code>toy_m_from_post_std</code>, <code>matched_fraction_cut</code>, <code>test_t2a</code> "
            "(a_shear_analysis.py); S2/S3 blocks of a_shear_supplement.py; <code>evaluate</code> in "
            "a_shear_supplement2.py. Stencil: <code>selection_response</code> in b_t2b_analysis.py.",
            "JSON keys: <code>A_shear/results.json</code> → <code>T2a</code> (<code>curves</code>, "
            "<code>proxy_spearman_with_sigma</code>, <code>matched_sigma_comparison</code>), <code>T1/|g|</code>, "
            "<code>R6/rows</code>; <code>A_shear/supplement.json</code> → <code>S2_paired_selection_excess</code>, "
            "<code>S3_mechanism</code>; <code>A_shear/supplement2.json</code> (S5); "
            "<code>B_inference/t2b_xu3like.json</code> → <code>tf_equal/selection</code>; "
            "<code>B_inference/t2b_fibrepair.json</code> → <code>proposal/selection</code>.",
            "Figures: <code>A_shear/T2a_selection.png</code>, <code>A_shear/S2_S4_selection_and_shrinkage.png</code>, "
            "<code>B_inference/t2b_selection_response.png</code>.",
        )
        + "<p>SLURM jobs, Xu3-cache tests (from <code>A_shear/STATUS.txt</code>):</p>" + status_block("A_shear")
        + "<p>SLURM jobs, stencil (from <code>B_inference/STATUS.txt</code>; the selection analysis itself ran on "
          "the login node):</p>" + status_block("B_inference")
    )
    return page(slug="width-cut-selection-bias", title="Do posterior-width cuts cause selection bias?",
                code="T2a (+ S2, S3), T2b selection", section=SEC, answer=answer, question=question, why=why,
                theory=theory, procedure=procedure, results=results, interpretation=interpretation,
                caveats=caveats, provenance=provenance)


# ============================================================================ page 2
def page_response():
    XU = lambda *k: J("stencil_xu3", *k)
    PRS = lambda *k: J("stencil_prop", *k)
    CONS = lambda *k: J("stencil", "T2b_xu3_consistency", *k)
    sig_pi = J("shear", "sigma_pi")
    fitx = XU("tf_equal", "fit_R_sigma")
    ov = XU("tf_equal", "overall")
    raw1 = 1 + CONS("fit_from_xu3like_tf_equal", "g1", "raw_low_m")
    raw2 = 1 + CONS("fit_from_xu3like_tf_equal", "g2", "raw_low_m")
    rw = lambda which, comp: (CONS(which, comp, "m_after_resp_weighted_calibration"), CONS(which, comp, "se"))
    B = "B_inference/report.html"
    med, p90 = H(B, r"median of the per-base trace response ([\d.]+); 90th percentile ([\d.]+)")
    (cos_thr,) = H(B, r"All but one have cos i > (\d+\.\d+)")
    ess_thr, ess_prop, ess_xu = H(B, r"TF ESS fraction < ([\d.]+) for (\d+)% of rows, versus ([\d.]+)% on the Xu3-like")
    nar_st, nar_xu3 = H(B, r"deficit below 0\.03 \(([\d.]+)% vs ([\d.]+)% of Xu3\)")
    seed, = H(B, r"were drawn \(seed (\d+)\)")
    nrm, rhos_cut = H(B, r"removing the (\d+) rows with ρS > (\d+)")
    bs0 = XU("tf_equal", "by_sigma", 0)

    answer = T(
        "Measured directly by shearing the same galaxy with the same noise, the response of the posterior-mean "
        "estimator on a Xu3-like stencil is R ≈ (<<r1>>, <<r2>>), close to the Xu3 1+m ≈ (<<x1>>, <<x2>>). Per "
        "component it follows R ≈ a + bσ² with b = <<b>> (Gaussian shrinkage predicts −300). Calibrating the Xu3 "
        "ensemble with this R(σ) leaves m = (<<m1>>, <<m2>>). Dividing each galaxy by its own R fails, and the "
        "narrowest, face-on galaxies show a suspicious R > 1.",
        r1=f(ov["R"][0][0], 2), r2=f(ov["R"][1][1], 2), x1=f(raw1, 2), x2=f(raw2, 2),
        b=pm(fitx["per_component"]["b"], fitx["per_component"]["b_se"], 0),
        m1=fxpm(*rw("resp_weighted_xu3like", "g1")), m2=fxpm(*rw("resp_weighted_xu3like", "g2")))

    question = P(
        r"""The raw bias \(m\) on Xu3 is measured by regressing \(\hat g\) on the true \(g\) over many different
galaxies. This page measures the <b>shear response</b> \(R = \partial\langle\hat g\rangle/\partial g\)
galaxy by galaxy, by rendering the same galaxy, with the same noise, at slightly different applied shears.
It then asks: (1) does \(R\) agree with the Xu3 \(1 + m\)? (2) how does \(R\) depend on the posterior width
\(\sigma\) and on the inclination \(\cos i\), and does it follow the Gaussian-shrinkage prediction
\(R = 1 - \sigma^2/\sigma_\pi^2\)? (3) can a fitted \(R(\sigma)\) be used to calibrate the Xu3 shear
estimates (paper Sec. 6.4)?""")

    why = P(
        r"""Response calibration is one of the mitigations of shrinkage discussed in Sec. 6. It is only credible
if the response is measured in a way that does not rely on the same galaxies' true shears (as metacalibration
does for real data), if it transfers from the calibration sample to the survey sample, and if the per-galaxy
model of \(R\) is stable. The stencil also provides the selection response used on the
<a href='width-cut-selection-bias.html'>selection-bias page</a>.""")

    theory = (
        "<h3>The matched-shear stencil</h3>"
        + P(
            T(r"""<code>make_shear_response_samples.py</code> draws <<nb>> base galaxies at random from a source
catalog. For each base it writes five rows that copy every intrinsic parameter (\(\theta_{\rm int}\),
\(\sin i\), \(v_0\), \(v_{\rm circ}\), disk scale radius, half-light radius) and every observation setting
(\(r\)-band magnitude, H\(\alpha\) flux, image S/N, central H\(\alpha\) S/N, fiber layout type), and set only
the shear: \((0,0)\), \((+\Delta g, 0)\), \((-\Delta g, 0)\), \((0, +\Delta g)\), \((0, -\Delta g)\) with
\(\Delta g = <<dg>>\). The simulator renders the noiseless image and spectra of each row. Noise is added at
inference time (<code>cache_posteriors.py --matched-group-size 5</code>, which calls
<code>sample_density</code> in <code>arch/train.py</code>): all five rows of a base use the same random seed,
and the noise amplitude is set from the zero-shear version's signal, so the five versions carry exactly the
same pixel and spectral noise pattern. Each row keeps the fiber positions computed by the simulator for its
own sheared galaxy.""", nb=XU("nbase"), dg=f(XU("delta_g"), 2)),
        )
        + "<h3>Response by finite differences, and why the shared noise matters</h3>"
        + P(
            r"""With \(\hat g_i(\mathbf s)\) the TF-weighted posterior mean of component \(i\) for the version sheared
by \(\mathbf s\), the response matrix of base \(b\) is the central difference
\[ R_{ij,b} = \frac{\hat g_i(+\Delta g\,\mathbf e_j) - \hat g_i(-\Delta g\,\mathbf e_j)}{2\Delta g}, \qquad
i, j \in \{1, 2\}. \]
Write \(\hat g = F(D)\) for the estimator applied to the data \(D(\mathbf g, \boldsymbol\eta, n)\), which
depend on the shear, the intrinsic parameters \(\boldsymbol\eta\) and the noise \(n\). With the same \(n\)
in both versions, \(R_{ij,b}\) is the derivative of \(F\) along the shear for that fixed noise
realization; the noise enters only through where the derivative is evaluated. With independent noise in the
two versions it would instead contain the term \([\epsilon(n^+) - \epsilon(n^-)]/(2\Delta g)\), where
\(\epsilon\) is the noise-driven error in \(\hat g\). That error is comparable to the posterior width
(several \(\times10^{-2}\)), so with \(\Delta g = 0.01\) each galaxy's \(R\) would scatter by order unity and
1000 galaxies would not suffice. The shared noise removes this term. What remains is the Monte Carlo noise of
the posterior means (samples are drawn independently for each version), which adds a small scatter to each
\(R_{ij,b}\) and no bias. The central difference cancels even terms in \(g\); the cubic term of the Xu3 bias
fit, \(q\,g^3\) with \(q = """ + pm(J("shear", "R_raw", "g1", "cubic_q"), J("shear", "R_raw", "g1", "cubic_q_se"), 1)
            + r"""\) (see <a href='bias-fit-linearity.html'>bias-fit linearity</a>), changes \(R\) by only
\(q\,\Delta g^2 \approx """ + fx(J("shear", "R_raw", "g1", "cubic_q") * XU("delta_g") ** 2, 4) + r"""\).
The stencil mean of \(\hat g(0)\) gives the additive bias \(c\) at zero shear.""",
        )
        + "<h3>What Gaussian shrinkage predicts</h3>"
        + P(
            r"""Take one shear component with a Gaussian prior of variance \(\sigma_\pi^2\) and a Gaussian likelihood
of width \(\sigma_L\) centred on the noisy measurement \(g_{\rm obs} = g + n\). The posterior mean is
\(\hat g = g_{\rm obs}\,\sigma_\pi^2/(\sigma_\pi^2 + \sigma_L^2)\) and the posterior variance is
\(\sigma_{\rm post}^2 = \sigma_L^2\sigma_\pi^2/(\sigma_L^2 + \sigma_\pi^2)\), so
\[ R = \frac{\partial\langle\hat g\rangle}{\partial g} = \frac{\sigma_\pi^2}{\sigma_\pi^2 + \sigma_L^2}
= 1 - \frac{\sigma_{\rm post}^2}{\sigma_\pi^2}. \]
Here \(\sigma_{\rm post}\) is the <i>per-component</i> posterior width, which is what the NPE gives us.
Using the variance of the uniform training prior, \(\sigma_\pi^2 = 0.2^2/12 = 1/300\)
(\(\sigma_\pi = """ + f(sig_pi, 4) + r"""\)), the model \(R_{kk} = a + b\,\sigma_{g_k}^2\) should have
\(a = 1\) and \(b = -1/\sigma_\pi^2 = -300\). Written against the quadrature width
\(\sigma_{\epsilon,i}^2 = \sigma_{g_1}^2 + \sigma_{g_2}^2\), the mean diagonal response is
\((R_{11} + R_{22})/2 = 1 - \sigma_{\epsilon,i}^2/(2\sigma_\pi^2)\), so the slope is
\(b = -1/(2\sigma_\pi^2) = -150\). The two fits below test these predictions. The uniform prior is not
Gaussian, so the model is only approximate: for posteriors much narrower than the box a uniform prior shrinks
less than a Gaussian one.""",
            r"""<b>Inclination.</b> Nearly edge-on disks respond little. The direct reason is empirical: their
shear posteriors are almost as wide as the prior (see the \(\cos i\) table below), so by the formula above their
\(R\) is close to 0. One intuition for why: an edge-on disk looks like a thin line, and the ellipticity of an
already very elongated image hardly changes under a small shear (along the major axis the ellipticity
\(\varepsilon\) responds as \(1 - |\varepsilon|^2\), which vanishes as \(|\varepsilon| \to 1\)). Face-on disks give
the narrowest posteriors and the largest response.""",
        )
        + "<h3>Using \\(R(\\sigma)\\) to calibrate an ensemble</h3>"
        + P(
            r"""Given a fitted \(R(\sigma)\), there are two ways to correct the Xu3 estimates.
<b>Per-galaxy division</b>, \(\hat g_i/R(\sigma_i)\), is unbiased only if each \(R(\sigma_i)\) is right, and it
is unstable: where \(R\) is small (posteriors nearly as wide as the prior) it multiplies the noise in
\(\hat g_i\) by \(1/R\), and any error in the predicted \(R\) becomes a large relative error in that galaxy's
corrected estimate. <b>Ensemble calibration</b> instead fits
\[ \hat g_{k,i} = c + s\,\big[R(\sigma_{k,i})\,g_{k,i}\big] \]
over the low-\(|g|\) Xu3 galaxies, and reports \(m = s - 1\). This is the analogue of dividing the mean
estimate by the mean response, \(\langle\hat g\rangle/\langle R\rangle\), as metacalibration does; it is
equivalent (up to the free intercept) to
weighting each galaxy's \(\hat g_i/R_i\) by \(R_i^2\), so galaxies with a small, uncertain
response carry little weight. (This is a post-hoc calibration of the estimator; it is unrelated to the
"R(σ)" training option that exists in the code base.)""",
        )
        + "<h3>Why two stencils</h3>"
        + P(
            T(r"""The existing stencil takes its bases from the training proposal: image S/N uniform on
\([10, 1000]\) and \(v_{\rm circ}\) drawn independently of the magnitude, i.e. not on the Tully–Fisher (TF)
relation. The production estimator reweights the posterior samples by the TF prior,
\(w \propto p_{\rm TF}(v_{\rm circ}\mid m_r)/\pi_{\rm train}(v_{\rm circ})\). For a galaxy far off the TF
relation the posterior samples sit where the TF prior is tiny, a handful of samples take all the weight
(effective sample size ≈ 1), and the TF-weighted mean is noisy and not the estimator we care about. On the
proposal stencil the TF effective-sample fraction is below <<et>> for <<ep>>% of rows (<<ex>>% on the
Xu3-like stencil), and weighting the bases to a TF population leaves an effective <<pe>> of 1000. A second,
<b>Xu3-like stencil</b> was therefore built: <<nb>> bases drawn (seed <<seed>>) from the TF-conformed,
catalog-S/N sibling of the Xu3 catalog (<code>test_100k_simv3_xu3_tf.csv</code>, not the paper test set),
after removing the <<nrm>> rows with \(\rho_S > <<rc>>\) (outside the model's training support), and rendered
with the current simulator.""",
              et=ess_thr, ep=ess_prop, ex=ess_xu, pe=f(PRS("tf_population_ess"), 0), nb=XU("nbase"), seed=seed,
              nrm=nrm, rc=rhos_cut),
        )
        + "<h3>A suspicious response above 1</h3>"
        + P(
            r"""In the Gaussian model \(R \le 1\) always; \(R > 1\) means the estimate moves <i>more</i> than the
applied shear. The narrowest Xu3-like bases do exceed it (results below). <b>Hypothesis (untested):</b> the
simulator computes the fiber positions from the <i>noiseless true</i> sheared galaxy axes
(<code>compute_fiber_offsets(g1, g2, theta_int, sini)</code> in
<code>kl-nn/data_generate/observation_schema.py</code>, called by <code>generate_fits.py</code> with the true
shear). For a nearly face-on disk the intrinsic shape is almost round, so the orientation of its observed axes,
and hence of the fiber layout, is set mostly by the shear. The network is given these positions, so it can read
the shear orientation from the fiber layout without any noise. Between the \(\pm\Delta g\) versions the
layout of a nearly round galaxy can rotate substantially, which could produce large finite-difference responses.
Real observations would place fibers from the noisy observed image, so if this is the cause it is a simulation
artifact that flatters exactly the galaxies a width cut keeps.""")
    )

    procedure = ul(
        T(r"""<b>Stencils.</b> Xu3-like: <<nb>> bases × 5 shears, generated and ingested for this campaign.
Training-proposal: the existing 1000-base stencil (<code>shear_response_simv3_cosi_5k</code>), copied unchanged
(a dry run showed its fiber convention already matches the current model). Inference with the canonical model,
\(10^4\) identity + 90°-rotation posterior samples per row, shared noise within each group of five.""",
          nb=XU("nbase")),
        r"""<b>Estimators.</b> Primary: TF-weighted posterior mean, equal weight per base galaxy. Also: the
unweighted posterior mean (samples under the training prior); for the proposal stencil, the TF-weighted mean
with the bases weighted to the TF population ("TF target", as in the earlier shear-response report).""",
        r"""<b>Response.</b> Per-base \(R_{ij,b}\) by central differences; mean over bases overall, in bins of the
zero-shear \(\sigma_{\epsilon,i}\) and in bins of true \(\cos i\); errors from bootstrap resampling of base
galaxies (1000 resamples for the binned blocks, 2000 for the fits).""",
        r"""<b>\(R(\sigma)\) fits.</b> Ordinary least squares with bootstrap errors, widths from the zero-shear
version: pooled \(R_{11}\) vs \(\sigma_{g_1}^2\) and \(R_{22}\) vs \(\sigma_{g_2}^2\) ("per component", prediction
\(b = -300\)); \((R_{11}+R_{22})/2\) vs \(\sigma_{\epsilon,i}^2\) ("trace", prediction \(b = -150\)); a
one-parameter version with \(a\) fixed to 1 (<code>fit_R_sigma</code> in <code>b_t2b_analysis.py</code>).""",
        r"""<b>Transfer to Xu3.</b> Per-galaxy TF-weighted means and widths of the Xu3 cache
(<code>b_xu3_sigma_summary.py</code>). Per-galaxy division with \(R\) clipped at 0.05 (<code>xu3_consistency</code>
in <code>b_make_report.py</code>); ensemble calibration by regressing \(\hat g_k\) on \(R(\sigma_k)\,g_k\) for
\(|g_k| < 0.02\) with \(R\) clipped at 0, with a nested bootstrap (500 resamples; the stencil fit is redone on
resampled bases and the Xu3 galaxies are resampled) (<code>resp_weighted_calibration</code>).""",
    )

    # ------------- results
    def ovrow(label, blk, ess):
        o = blk["overall"]
        return [label, pm(o["R"][0][0], o["R_se"][0][0]), pm(o["R"][1][1], o["R_se"][1][1]),
                pm(o["R"][0][1], o["R_se"][0][1]), pm(o["R"][1][0], o["R_se"][1][0]),
                fxpm(1e3 * o["c"][0], 1e3 * o["c_se"][0], 2), fxpm(1e3 * o["c"][1], 1e3 * o["c_se"][1], 2),
                f(ess, 0)]

    t_ov = table(["stencil, estimator", "R₁₁", "R₂₂", "R₁₂", "R₂₁", "c₁ ×10³", "c₂ ×10³", "effective no. of bases"], [
        ovrow("Xu3-like, TF-weighted mean (primary)", XU("tf_equal"), XU("tf_equal", "overall", "weight_ess")),
        ovrow("Xu3-like, unweighted mean", XU("proposal"), XU("proposal", "overall", "weight_ess")),
        ovrow("training-proposal, unweighted mean", PRS("proposal"), PRS("proposal", "overall", "weight_ess")),
        ovrow("training-proposal, TF mean, TF-population base weights", PRS("tf_target"),
              PRS("tf_target", "overall", "weight_ess")),
        ovrow("training-proposal, TF mean, equal base weights (TF weights degenerate)", PRS("tf_equal"),
              PRS("tf_equal", "overall", "weight_ess")),
    ])
    se1 = math.hypot(ov["R_se"][0][0], CONS("fit_from_xu3like_tf_equal", "g1", "raw_low_m_se"))
    se2 = math.hypot(ov["R_se"][1][1], CONS("fit_from_xu3like_tf_equal", "g2", "raw_low_m_se"))
    z1 = (ov["R"][0][0] - raw1) / se1
    z2 = (ov["R"][1][1] - raw2) / se2

    def binrows(blk, key, label_fmt):
        out = []
        for b in blk[key]:
            if "R" not in b:
                continue
            gauss = 1 - b["mean_sigma_comp_sq"] / sig_pi ** 2
            out.append([label_fmt(b), b["n"], pm(b["R"][0][0], b["R_se"][0][0]), pm(b["R"][1][1], b["R_se"][1][1]),
                        f(b["mean_sigma_eps"], 4), f(gauss, 3)])
        return out

    sig_lab = lambda b: f"[{f(b['lo'], 3)}, {f(b['hi'], 3)})"
    cos_lab = lambda b: f"[{f(b['lo'], 1)}, {f(b['hi'], 1)})"
    hdr = lambda v: [v, "n", "R₁₁", "R₂₂", "mean σ<sub>ε,i</sub>", "Gaussian 1 − ⟨σ<sub>g<sub>k</sub></sub>²⟩/σ<sub>π</sub>²"]
    t_sig = table(hdr("σ<sub>ε,i</sub> bin (Xu3-like, TF mean)"), binrows(XU("tf_equal"), "by_sigma", sig_lab))
    t_sigp = table(hdr("σ<sub>ε,i</sub> bin (proposal, unweighted)"), binrows(PRS("proposal"), "by_sigma", sig_lab))
    t_cos = table(hdr("true cos i bin (Xu3-like, TF mean)"), binrows(XU("tf_equal"), "by_cosi", cos_lab))
    t_cosp = table(hdr("true cos i bin (proposal, unweighted)"), binrows(PRS("proposal"), "by_cosi", cos_lab))

    def fitrow(label, fr):
        pc, tr = fr["per_component"], fr["trace_vs_sigma_eps_sq"]
        return [label, pm(pc["a"], pc["a_se"]), pm(pc["b"], pc["b_se"], 0), f(pc["b_fixed_a1"], 0),
                pm(tr["a"], tr["a_se"]), pm(tr["b"], tr["b_se"], 0), f(tr["residual_rms"], 2)]

    t_fit = table(["sample", "a (per comp.)", "b (per comp.; predicted −300)", "b with a ≡ 1", "a (trace)",
                   "b (trace vs σ<sub>ε,i</sub>²; predicted −150)", "per-base residual rms"], [
        fitrow("Xu3-like, TF-weighted mean (primary)", XU("tf_equal", "fit_R_sigma")),
        fitrow("Xu3-like, unweighted mean", XU("proposal", "fit_R_sigma")),
        fitrow("training-proposal, unweighted mean", PRS("proposal", "fit_R_sigma")),
        fitrow("training-proposal, TF mean, TF-population weights", PRS("tf_target", "fit_R_sigma")),
        fitrow("training-proposal, TF mean, equal weights (misspecified)", PRS("tf_equal", "fit_R_sigma")),
    ])
    r11, r22 = fitx["R11_vs_sigma_g1_sq"], fitx["R22_vs_sigma_g2_sq"]

    def calrow(label, which, rwkey):
        c1, c2 = CONS(which, "g1"), CONS(which, "g2")
        return [label,
                f"{f(c1['predicted_1pm_from_stencil_fit'], 3)} / {f(c2['predicted_1pm_from_stencil_fit'], 3)}",
                f"{fxpm(c1['calibrated_low_m'], c1['calibrated_low_m_se'])} / "
                f"{fxpm(c2['calibrated_low_m'], c2['calibrated_low_m_se'])}",
                f"{pct(c1['frac_R_clipped'])} / {pct(c2['frac_R_clipped'])}",
                "<b>" + f"{fxpm(*rw(rwkey, 'g1'))} / {fxpm(*rw(rwkey, 'g2'))}" + "</b>"]

    c1x = CONS("fit_from_xu3like_tf_equal", "g1")
    c2x = CONS("fit_from_xu3like_tf_equal", "g2")
    t_cal = table(["R(σ) fit from", "mean R(σ<sub>i</sub>) over Xu3 low-|g| (g₁ / g₂)",
                   "m after per-galaxy division (g₁ / g₂)", "Xu3 galaxies with R(σ) < 0.05 (clipped)",
                   "m after ensemble calibration (g₁ / g₂)"], [
        calrow("Xu3-like stencil, TF-weighted mean", "fit_from_xu3like_tf_equal", "resp_weighted_xu3like"),
        calrow("training-proposal stencil, unweighted mean", "fit_from_proposal_stencil_unweighted",
               "resp_weighted_proposal_stencil"),
    ])

    results = (
        "<h3>Overall response</h3>" + t_ov
        + P(T(r"""\(c\) is the mean estimate at zero shear. For comparison, the Xu3 test set with the same estimator
(equal weights, low-\(|g|\) fit) has \(1 + m = (<<a>>, <<b>>)\). The Xu3-like stencil differs by
<<z1>>\(\sigma\) (\(g_1\)) and <<z2>>\(\sigma\) (\(g_2\)). The Gaussian formula evaluated on the Xu3 widths
gives \((<<g1>>, <<g2>>)\). The training-proposal stencil responds more (<<rp>>), because its galaxies are on
average brighter and narrower than Xu3 (zero-shear median \(\sigma_{\epsilon,i}\) of <<mp>> vs <<mx>> on the
Xu3-like stencil). With degenerate TF weights (last row) the off-diagonal \(R_{21}\) becomes significant,
another sign that that estimator is misspecified on training-proposal galaxies.""",
              a=pm(raw1, CONS("fit_from_xu3like_tf_equal", "g1", "raw_low_m_se")),
              b=pm(raw2, CONS("fit_from_xu3like_tf_equal", "g2", "raw_low_m_se")),
              z1=f(z1, 1), z2=f(z2, 1), g1=f(c1x["predicted_1pm_gaussian"], 3), g2=f(c2x["predicted_1pm_gaussian"], 3),
              mp=f(PRS("sigma_eps_zero_pct")[2], 3), mx=f(XU("sigma_eps_zero_pct")[2], 3),
              rp=f(PRS("proposal", "overall", "R_diag"), 2)))
        + fig("B_inference/t2b_sigma_hist.png",
              T(r"""Distribution of the zero-shear \(\sigma_{\epsilon,i}\): Xu3 test set (100k galaxies), Xu3-like stencil
bases and training-proposal stencil bases (TF-weighted widths). Dotted lines mark the cut thresholds used on the
selection page. The Xu3-like stencil follows Xu3 except for a deficit of very narrow posteriors
(<<a>>% vs <<b>>% below 0.03, from the group report). The stencil sits at \(|g| \le 0.01\) while Xu3 spans
\(|g| \le 0.1\), and widths shrink with \(|g|\), which may explain part of it.""", a=nar_st, b=nar_xu3), "75%")
        + "<h3>Response against posterior width</h3>" + t_sig + t_sigp
        + P(T(r"""The response falls steeply with width, roughly as Gaussian shrinkage predicts, but the narrowest bin
of the Xu3-like stencil has \(R \approx <<r>>\), <b>well above 1</b> (<<n>> bases; the group report gives a
median per-base diagonal response of <<med>> and a 90th percentile of <<p90>>, and all but one of these bases
have \(\cos i > <<ct>>\)). The proposal stencil, whose narrow bases are narrow mostly because they are bright,
shows no such excess.""",
              r=f(bs0["R_diag"], 2), n=bs0["n"], med=med, p90=p90, ct=cos_thr))
        + "<h3>Response against inclination</h3>" + t_cos + t_cosp
        + fig("B_inference/t2b_R_vs_cosi.png",
              r"""Diagonal response \(R_{11}\) (circles) and \(R_{22}\) (squares) in bins of true \(\cos i\), with
bootstrap errors. Blue: Xu3-like stencil, TF-weighted mean. Orange: training-proposal stencil, unweighted mean.
Green: training-proposal stencil, TF mean with TF-population weights. Edge-on disks (\(\cos i \to 0\)) barely
respond; face-on disks respond most.""", "70%")
        + "<h3>The \\(R(\\sigma) = a + b\\sigma^2\\) fit</h3>" + t_fit
        + P(T(r"""Separate fits for the Xu3-like stencil: \(R_{11}\) vs \(\sigma_{g_1}^2\) gives \(b = <<b1>>\),
\(R_{22}\) vs \(\sigma_{g_2}^2\) gives \(b = <<b2>>\). The correlation between the per-base trace response and
\(\sigma_{\epsilon,i}^2\) is <<corr>>.""",
              b1=pm(r11["b"], r11["b_se"], 0), b2=pm(r22["b"], r22["b_se"], 0),
              corr=f(fitx["trace_vs_sigma_eps_sq"]["corr"], 2)))
        + fig("B_inference/t2b_R_vs_sigma.png",
              r"""Pooled \(R_{11}\) and \(R_{22}\) of the Xu3-like stencil (TF-weighted mean) in deciles of the
per-component posterior variance \(\sigma_{g_k}^2\) (zero-shear version), with the linear fit (orange), the fit
to the training-proposal stencil (unweighted mean, dash-dotted) and the Gaussian-shrinkage line
\(1 - \sigma^2/\sigma_\pi^2\) (grey dashed). The dotted vertical line marks \(\sigma_\pi^2\).""", "75%")
        + "<h3>Calibrating the Xu3 test set</h3>" + t_cal
        + P(T(r"""Raw Xu3 bias for reference: \(m = (<<m1>>, <<m2>>)\). Ensemble-calibration errors come from the
nested bootstrap.""",
              m1=fxpm(c1x["raw_low_m"], c1x["raw_low_m_se"]), m2=fxpm(c2x["raw_low_m"], c2x["raw_low_m_se"])))
    )

    interpretation = P(
        T(r"""<b>The stencil response matches the Xu3 bias.</b> Measured without using the Xu3 truths, the response
of the production estimator on a Xu3-like population is \(R \approx (<<r1>>, <<r2>>)\), close to the Xu3
\(1 + m\). The shear response is essentially diagonal (the off-diagonal terms are <<z12>> and <<z21>> times
their errors), and the additive bias at zero shear is small (\(|c| \le <<cm>>\)).""", r1=f(ov["R"][0][0], 3),
          r2=f(ov["R"][1][1], 3), z12=f(abs(ov["R"][0][1] / ov["R_se"][0][1]), 1),
          z21=f(abs(ov["R"][1][0] / ov["R_se"][1][0]), 1), cm=f(max(abs(ov["c"][0]), abs(ov["c"][1])), 4)),
        T(r"""<b>Shrinkage explains the width and inclination dependence.</b> The slope of \(R\) against the
per-component posterior variance, <<b>>, is close to the Gaussian value \(-1/\sigma_\pi^2 = -300\) (with the
intercept fixed to 1 it is <<bf>>). The intercept above 1 comes from the \(R > 1\) face-on points and the
non-Gaussian uniform prior. Edge-on disks have posteriors nearly as wide as the prior and respond little
(\(R \approx <<re>>\) for \(\cos i < 0.2\)).""",
          b=pm(fitx["per_component"]["b"], fitx["per_component"]["b_se"], 0), bf=f(fitx["per_component"]["b_fixed_a1"], 0),
          re=f(XU("tf_equal", "by_cosi", 0, "R_diag"), 2)),
        T(r"""<b>Ensemble calibration works to a few percent; per-galaxy division does not.</b> Calibrating the Xu3
ensemble with the Xu3-like \(R(\sigma)\) leaves \(m = (<<a>>, <<b>>)\); with the training-proposal fit it
leaves \((<<c>>, <<d>>)\). The spread between the two calibration samples (up to <<sp>> in \(m\)) is the real
systematic of the method: \(R(\sigma)\) depends on the population it is measured on, so the stencil must
resemble the survey. Per-galaxy division leaves \(m \approx <<e>>\).""",
          a=fxpm(*rw("resp_weighted_xu3like", "g1")), b=fxpm(*rw("resp_weighted_xu3like", "g2")),
          c=fxpm(*rw("resp_weighted_proposal_stencil", "g1")), d=fxpm(*rw("resp_weighted_proposal_stencil", "g2")),
          e=fx(c1x["calibrated_low_m"], 2),
          sp=f(max(abs(rw("resp_weighted_xu3like", c)[0] - rw("resp_weighted_proposal_stencil", c)[0])
                   for c in ("g1", "g2")), 2)),
        r"""<b>For the paper (Sec. 6.4).</b> Use the Xu3-like stencil and the ensemble form of the calibration;
state the population dependence; for cut samples add the selection response
(<a href='width-cut-selection-bias.html'>previous page</a>); flag the \(R > 1\) face-on galaxies.""",
    )

    caveats = ul(
        r"""The stencil measures the response at zero shear (\(|g| \le 0.01\)); the Xu3 bias is fitted over
\(|g| < 0.02\). The Xu3-like stencil lacks Xu3's tail of very narrow posteriors; the group report attributes the
excess of the mean predicted \(R\) over Xu3 (above the measured \(1 + m\)) to the linear \(R(\sigma)\)
over-predicting \(R\) in that tail.""",
        r"""The \(R > 1\) anomaly of face-on, narrow-posterior galaxies is not understood. The fiber-position
explanation is a hypothesis. A direct test would re-render the face-on bases with fiber offsets computed from
the zero-shear (or intrinsic) axes for all five versions and compare \(R\) (not done).""",
        T(r"""\(R(\sigma)\) is a phenomenological two-parameter fit with a per-base residual scatter of <<rr>>; it
does not include the selection response of a cut, and it depends on the calibration population.""",
          rr=f(fitx["trace_vs_sigma_eps_sq"]["residual_rms"], 2)),
        r"""The legacy report on the training-proposal stencil used an older model evaluated in a different fiber
convention from the one it was trained with; those older numbers should not be compared with these.""",
        r"""1000 bases per stencil; all numbers are from the 100k-trained model.""",
    )

    provenance = (
        ul(
            "Group report: <a href='../B_inference/report.html'>matched-shear stencil</a> (sections 3–5 and 7).",
            "Scripts: " + ", ".join(sc(s) for s in ("b_t2b_analysis.py", "b_make_report.py", "b_write_html.py",
                                                     "b_xu3_sigma_summary.py", "b_regen_probe.py",
                                                     "b_xu3like_generate.slurm", "b_xu3like_database.slurm",
                                                     "b_t2b_inference_cpu.slurm",
                                                     "b_t2b_xu3like_inference_cpu.slurm"))
            + "; stencil builder <code>/jet/home/xwang30/kl-nn/data_generate/make_shear_response_samples.py</code>; "
              "inference <code>arch/cache_posteriors.py --matched-group-size 5</code>.",
            "JSON keys: <code>B_inference/t2b_xu3like.json</code> and <code>t2b_fibrepair.json</code> → "
            "<code>&lt;variant&gt;/overall</code>, <code>by_sigma</code>, <code>by_cosi</code>, <code>fit_R_sigma</code> "
            "(variants: <code>tf_equal</code> = TF-weighted mean, equal base weights; <code>proposal</code> = "
            "unweighted posterior mean; <code>tf_target</code> = TF mean with TF-population base weights); "
            "<code>B_inference/report.json</code> → <code>T2b_xu3_consistency</code>. "
            "Values quoted \"from the group report\" (face-on anomaly details, TF effective-sample fractions, the "
            "σ<sub>ε,i</sub> &lt; 0.03 fractions, the Xu3-like stencil seed and ρ<sub>S</sub> cut) are read from "
            "the text of <code>B_inference/report.html</code>.",
            "Figures: <code>B_inference/t2b_sigma_hist.png</code>, <code>t2b_R_vs_cosi.png</code>, "
            "<code>t2b_R_vs_sigma.png</code> (paper candidate <code>paper_figs/B_t2b_R_vs_sigma</code>).",
        )
        + "<p>SLURM jobs (from <code>B_inference/STATUS.txt</code>; the analysis and report ran on the login "
          "node):</p>" + status_block("B_inference")
    )
    return page(slug="response-calibration",
                title="Shear response on a matched-shear stencil and response calibration", code="T2b",
                section=SEC, answer=answer, question=question, why=why, theory=theory, procedure=procedure,
                results=results, interpretation=interpretation, caveats=caveats, provenance=provenance)


# ============================================================================ page 3
def page_stack():
    M = lambda name, curve, key, which, comp: J("stack", "metrics", name, curve, key, which, comp)
    RN = "vs_N_at_R16"
    NR = J("stack", "n_realizations")
    NL = J("stack", "analysis_meta", "N_list")
    GC = lambda k: J("stack", "grid_checks", k)
    NP = lambda k: J("stack", "nuisance_prior", k)
    PO = lambda w, k: J("stack", "per_observation", w, k)
    cfg = J("stack", "analysis_meta", "configs")
    NS = J("stack", "n_samples_per_observation")
    NB = J("stack", "bootstrap", "count")

    def det(name, comp, curve=RN, key="32"):
        return M(name, curve, key, "det", comp)

    def dstd(name, comp, curve=RN, key="32"):
        return M(name, curve, key, "boot_std", comp)

    def cellv(name, curve, key, comp, nd=3):
        d = J("stack", "metrics", name, curve, key)
        s = fx(d["det"][comp], nd)
        if "boot_p16" in d:
            s += f" <span style='color:#666'>[{fx(d['boot_p16'][comp], nd)}, {fx(d['boot_p84'][comp], nd)}]</span>"
        return s

    kde, pg = "stack_kde_mean_train", "pergal_mean_train"
    answer = T(
        "Yes. Multiplying the per-galaxy NPE shear likelihoods (posterior divided by the uniform training shear "
        "prior) at constant shear removes the shrinkage: for <<n>> galaxies × <<r>> noise realizations per cell, "
        "m goes from (<<p1>>, <<p2>>) for averaged per-galaxy posterior means to (<<k1>>, <<k2>>) for the stacked "
        "posterior. The stacked posterior is too narrow when noise realizations of the same galaxy are stacked, "
        "and this benchmark tests the training nuisance prior, not the TF prior.",
        n=NL[-1], r=NR, p1=fx(det(pg, "g1_m")), p2=fx(det(pg, "g2_m")),
        k1=fxpm(det(kde, "g1_m"), dstd(kde, "g1_m")), k2=fxpm(det(kde, "g2_m"), dstd(kde, "g2_m")))

    question = P(
        r"""Averaging per-galaxy posterior means keeps the shrinkage of each mean: every posterior is pulled toward
the centre of the prior, and averaging many pulled estimates does not undo the pull. If all galaxies in a patch
share one shear, the statistically correct combination is instead to multiply their <i>likelihoods</i>
(Bernstein &amp; Armstrong 2014, constant-shear case). This page asks whether that works with likelihoods
obtained from the NPE, and how the bias depends on the number of galaxies \(N\) and the number of noise
realizations \(R\) per galaxy.""")

    why = P(
        r"""Stacking is the mitigation in Sec. 6 that addresses the cause of the shrinkage (the prior entering once
per galaxy) rather than correcting for it afterwards. It needs no calibration stencil and no width cut, so it
avoids the selection and population issues of the other two mitigations. It also shows that the NPE carries
unbiased shear information even though its posterior means are biased.""")

    theory = (
        "<h3>Combining galaxies that share one shear</h3>"
        + P(
            r"""Let \(D_i\) be the data of galaxy \(i\) and \(\boldsymbol\eta_i\) its nuisance parameters (orientation,
inclination, kinematics, size, flux), drawn independently from a population prior \(\pi(\boldsymbol\eta)\).
For galaxies sharing the shear \(\mathbf g\),
\[ p(\mathbf g\mid D_1, \dots, D_N) \propto \pi(\mathbf g)\prod_{i=1}^{N} L_i(\mathbf g), \qquad
L_i(\mathbf g) = p(D_i\mid\mathbf g) = \int p(D_i\mid\mathbf g, \boldsymbol\eta)\,\pi(\boldsymbol\eta)\,d\boldsymbol\eta . \]
The prior on the shear appears <b>once</b>. The per-galaxy posterior mean instead uses
\(\pi(\mathbf g)\,L_i(\mathbf g)\) for every galaxy, so the prior pulls every estimate toward its centre and the
average keeps that pull. In the product, the likelihoods sharpen around the true shear as \(N\) grows and the
prior's influence falls like \(1/N\). A Gaussian illustration: with \(L_i \propto
\exp[-(g_{{\rm obs},i} - g)^2/2\sigma_L^2]\) and a Gaussian prior of width \(\sigma_\pi\), the stacked posterior
mean has response \(\sigma_\pi^2/(\sigma_\pi^2 + \sigma_L^2/N) \to 1\), while each per-galaxy mean has response
\(\sigma_\pi^2/(\sigma_\pi^2 + \sigma_L^2)\) regardless of \(N\).""",
        )
        + "<h3>A likelihood from the NPE</h3>"
        + P(
            r"""The NPE was trained on simulations with shear drawn from a uniform prior
\(\pi_{\rm train}(\mathbf g) = U(-0.1, 0.1)^2\) and nuisances from the training prior
\(\pi_{\rm train}(\boldsymbol\eta)\). Its shear marginal (the \((g_1, g_2)\) columns of its samples, which
marginalizes the nuisances) therefore approximates
\[ q(\mathbf g\mid D_i) \approx \frac{\pi_{\rm train}(\mathbf g)\int p(D_i\mid\mathbf g,\boldsymbol\eta)\,
\pi_{\rm train}(\boldsymbol\eta)\,d\boldsymbol\eta}{p(D_i)}
\quad\Rightarrow\quad L_i(\mathbf g) \propto \frac{q(\mathbf g\mid D_i)}{\pi_{\rm train}(\mathbf g)}
\propto q(\mathbf g\mid D_i)\ \ \text{inside the box}. \]
With the uniform training prior, the division is a constant, and the cell posterior is
\(\log p(\mathbf g\mid\{D_i\}) = \sum_i \log q(\mathbf g\mid D_i) + {\rm const}\) on the box. The nuisances are
marginalized under whatever prior the samples represent: unweighted samples give the training prior; samples
reweighted by \(p_{\rm TF}(v_{\rm circ}\mid m_r)/\pi_{\rm train}(v_{\rm circ})\) give the Tully–Fisher (TF)
prior on \(v_{\rm circ}\). The right choice is the prior the galaxies were actually drawn from.""",
        )
        + "<h3>Estimating each \\(q(\\mathbf g\\mid D_i)\\) from samples</h3>"
        + ul(
            T(r"""<b>Kernel density estimate (primary).</b> Each observation has <<ns>> posterior samples (half from the
data, half from the data rotated by 90° and the samples rotated back, the canonical ensemble). The density is an
exact (unbinned) sum of Gaussian product kernels with per-axis bandwidth \(h_d = f\,\sigma_d\,n_{\rm eff}^{-1/6}\)
(Scott's rule in two dimensions), \(\sigma_d\) the sample standard deviation along axis \(d\),
\(n_{\rm eff} = 1/\sum_s w_s^2\) the effective number of samples, \(f = 1\) (tests: 0.5 and 2), and
\(h_d \ge 0.001\). Median bandwidth: <<h>>.""", ns=NS, h=f(PO("train", "kde_h_median"), 4)),
            r"""<b>Reflection at the box edges.</b> The NPE posterior lives inside \([-0.1, 0.1]^2\). A plain KDE
spills half of its kernel mass outside near an edge, underestimating \(q\) there and biasing the product toward
the centre. Each sample is therefore mirrored across both edges of each axis
(\(x \to 2(-0.1) - x\) and \(x \to 2(0.1) - x\)); in the product kernel this also mirrors across the corners.
With reflection the KDE mass inside the box lies between """ + f(min(PO("train", "kde_mass_in_box_min_max")[0],
    PO("tf", "kde_mass_in_box_min_max")[0]), 4) + r""" and 1 for every observation.""",
            T(r"""<b>Floor.</b> \(q \to (1 - a)\,q + a/|{\rm box}|\) with \(a = <<a>>\) (tests: 0, \(10^{-4}\),
\(10^{-2}\)). Without a floor, one observation whose samples happen to miss the region of the true shear would
set the product to zero there and veto the true value.""", a=cfg["kde"][2]),
            r"""<b>Moment Gaussian (comparison).</b> A 2D Gaussian with the samples' mean and covariance. Each posterior
is truncated by the box, so its mean sits closer to the centre and its variance is smaller than those of the
underlying likelihood; the Gaussian built from those moments carries part of the box prior into every factor of
the product, i.e. a little shrinkage per galaxy survives.""",
            T(r"""<b>Log-KDE quadratic (diagnostic).</b> A quadratic fit to \(\log q_{\rm KDE}\) over its core (density
≥ 5% of the peak): a Gaussian that follows the KDE but is not truncated. It falls back to the moment Gaussian when
the fit is not a proper (negative-definite) quadratic, which happened for <<fb>> of observations.""",
              fb=pct(1 - PO("train", "lapkde_fit_ok_fraction"), 0)),
        )
        + "<h3>Evaluation lattice, and the old grid snapping</h3>"
        + P(
            r"""The summed \(\log q_i\) is evaluated at the nodes of a \(400\times400\) lattice with spacing 0.0005,
node centres at \(-0.1 + 0.0005(k + \tfrac12)\), so every true cell shear (a multiple of 0.04) lies exactly
<i>between</i> two nodes. The posterior mean is a lattice sum, the MAP a parabolic refinement around the best node.
Only the evaluation is on the lattice; the KDE itself is exact. The earlier version of this benchmark evaluated
stacked surfaces on a \(21\times21\) grid (spacing 0.01) on which every true shear <i>is</i> a node. A stacked
posterior much narrower than 0.01 then collapses onto the nearest node, which is the truth whenever the estimate
is within 0.005 of it: biases of that size become invisible. Here the lattice is finer than the stacked posterior
width at \(R = 1\), comparable to it at \(R = 16\), and the results are checked on a shifted (401-node) lattice, a
finer (800-node) lattice and an exact local \(61\times61\) evaluation around the peak.""",
        )
        + "<h3>Repeated noise realizations of the same galaxy over-count information</h3>"
        + P(
            r"""The benchmark has \(R\) independent noise realizations of each galaxy, all sharing one \(\boldsymbol\eta\).
The stack treats each realization as a separate galaxy: it multiplies \(\prod_r\int p(D_{r}\mid\mathbf g,\boldsymbol\eta)
\pi(\boldsymbol\eta)\,d\boldsymbol\eta\), whereas the joint likelihood of repeated observations of one galaxy is
\(\int\prod_r p(D_r\mid\mathbf g,\boldsymbol\eta)\,\pi(\boldsymbol\eta)\,d\boldsymbol\eta\). In a Gaussian
picture, \(g_{{\rm obs},r} = g + e + n_r\), with \(e\) the error due to the unknown intrinsic shape (shared) and
\(n_r\) the pixel noise (independent): the product of marginals has variance \((\sigma_e^2 + \sigma_n^2)/R\), the
correct joint likelihood \(\sigma_e^2 + \sigma_n^2/R\). The product is therefore too narrow, increasingly so with
\(R\), but it is still centred correctly, so the bias is unaffected. With \(R = 1\) (one observation per galaxy, as
in a survey) the stack is the correct likelihood. The honest error for this benchmark is the scatter over
resampled galaxies, not the stacked posterior width.""",
        )
    )

    procedure = ul(
        T(r"""<b>Benchmark.</b> The existing fixed-shear benchmark (<code>likelihood_stack_04_benchmark</code>): 25 cells
with exact shears \(g_1, g_2 \in \{-0.08, -0.04, 0, 0.04, 0.08\}\), the same <<ng>> nuisance galaxies in every cell
(only the shear changes), <<nr>> pre-computed noise realizations per galaxy. The nuisance galaxies are 32 draws
from the training-prior validation catalog.""", ng=NL[-1], nr=NR),
        T(r"""<b>Posterior samples.</b> <<ns>> identity + 90°-rotation samples per observation with the canonical model,
storing \((g_1, g_2)\), \(v_{\rm circ}\) and normalized TF log-weights (<code>c_stack_infer.py</code>).""", ns=NS),
        r"""<b>Stacking.</b> For each cell, sum \(\log q_i\) over the first \(N\) galaxies (\(N = 1, 2, 4, 8, 16, 32\)) and
the first \(R\) realizations (\(R = 1\dots16\)); summarize by the posterior mean and MAP. Two nuisance priors:
"training" (unweighted samples) and "TF" (TF-weighted samples). Density models: KDE (with bandwidth and floor
variations), moment Gaussian, log-KDE quadratic (<code>c_stack_analysis.py</code>).""",
        r"""<b>Per-galaxy comparison.</b> The average of the per-observation posterior means over exactly the same
\(N \times R\) observations.""",
        T(r"""<b>Bias fit.</b> \(\hat g_\alpha = (1 + m_\alpha)\,g_\alpha + c_\alpha\) by least squares over the 25 cell
estimates, per component. Note that this differs from the conventions box: fits are over cells, not galaxies, and
the headline uses unweighted samples. Point values use all galaxies (or the first \(N\)); intervals are the
16th–84th percentiles of <<nb>> bootstrap resamples of the galaxy IDs with replacement, the same resample for all
25 cells (<code>c_stack_report.py</code>). Figures show the bootstrap median.""", nb=NB),
    )

    # ---------------- results
    main_names = [(pg, "per-galaxy posterior mean"), (kde, "stacked, KDE (posterior mean)"),
                  ("stack_kde_map_train", "stacked, KDE (MAP)"), ("stack_gauss_mean_train", "stacked, moment Gaussian"),
                  ("stack_lapkde_mean_train", "stacked, log-KDE quadratic")]
    rows = []
    for name, lab in main_names:
        for n in (1, 4, 32):
            rows.append([lab, n, cellv(name, RN, str(n), "g1_m"), cellv(name, RN, str(n), "g2_m"),
                         cellv(name, RN, str(n), "g1_c", 4), cellv(name, RN, str(n), "g2_c", 4)])
    t_n = table(["estimator (training nuisance prior)", "N", "m₁", "m₂", "c₁", "c₂"], rows)
    rows = []
    for n in NL:
        rows.append([n, cellv(pg, RN, str(n), "g1_m"), cellv(kde, RN, str(n), "g1_m"),
                     cellv(pg, RN, str(n), "g2_m"), cellv(kde, RN, str(n), "g2_m"),
                     cellv(kde, "vs_N_at_R1", str(n), "g1_m"), cellv(kde, "vs_N_at_R1", str(n), "g2_m")])
    t_nfull = table(["N", f"m₁ per-galaxy (R={NR})", f"m₁ stacked KDE (R={NR})", f"m₂ per-galaxy (R={NR})",
                     f"m₂ stacked KDE (R={NR})", "m₁ stacked KDE (R=1)", "m₂ stacked KDE (R=1)"], rows)
    rows = []
    for r in (1, 2, 4, 8, 16):
        rows.append([r] + [cellv(nm, "vs_R_at_N32", str(r), c) for nm in (pg, kde, "stack_kde_mean_tf")
                           for c in ("g1_m", "g2_m")])
    t_r = table(["R (N = 32)", "m₁ per-galaxy", "m₂ per-galaxy", "m₁ stacked KDE", "m₂ stacked KDE",
                 "m₁ stacked KDE, TF prior", "m₂ stacked KDE, TF prior"], rows)
    sens = [(kde, "KDE, h × 1, floor 10⁻³ (primary)"), ("stack_kde_bw0.5_mean_train", "KDE, h × 0.5"),
            ("stack_kde_bw2_mean_train", "KDE, h × 2"), ("stack_kde_a0_mean_train", "KDE, no floor"),
            ("stack_kde_a1e-4_mean_train", "KDE, floor 10⁻⁴"), ("stack_kde_a1e-2_mean_train", "KDE, floor 10⁻²"),
            ("stack_kde_map_train", "KDE MAP")]
    t_sens = table([f"variant (N = 32, R = {NR}, training prior)", "m₁", "m₂"],
                   [[lab, cellv(nm, RN, "32", "g1_m"), cellv(nm, RN, "32", "g2_m")] for nm, lab in sens])
    tfn = [("pergal_mean_tf", "per-galaxy posterior mean, TF-weighted"), ("stack_kde_mean_tf", "stacked KDE, TF prior"),
           ("stack_gauss_mean_tf", "stacked moment Gaussian, TF prior"),
           ("stack_lapkde_mean_tf", "stacked log-KDE quadratic, TF prior")]
    t_tf = table([f"estimator (N = 32, R = {NR})", "m₁", "m₂", "c₁", "c₂"],
                 [[lab, cellv(nm, RN, "32", "g1_m"), cellv(nm, RN, "32", "g2_m"), cellv(nm, RN, "32", "g1_c", 4),
                   cellv(nm, RN, "32", "g2_c", 4)] for nm, lab in tfn])
    pull1, pull16 = GC("train_pull_N32R1"), GC(f"train_pull_N32R{NR}")
    t_chk = table(["check (training prior, KDE)", "value"], [
        ["401-node lattice − 400-node lattice: max |Δ cell mean|", f(GC("train_m401_minus_m400_mean_maxabs"), 1)],
        ["800-node lattice − 400-node lattice: max |Δ cell mean|", f(GC("train_m800_minus_m400_mean_maxabs"), 1)],
        [f"exact local 61×61 evaluation − lattice, R = {NR}: max |Δ cell mean|",
         f(GC(f"train_local_exact_R{NR}_mean_maxabs"), 1)],
        ["median stacked posterior std, N = 32, R = 1 / R = " + str(NR),
         f"{f(GC('train_stack_post_std_R1_median'), 4)} / {f(GC(f'train_stack_post_std_R{NR}_median'), 5)}"],
        ["pull std (g₁, g₂), R = 1", f"{f(pull1['std'][0], 2)}, {f(pull1['std'][1], 2)}"],
        [f"pull std (g₁, g₂), R = {NR}", f"{f(pull16['std'][0], 2)}, {f(pull16['std'][1], 2)}"],
        ["bootstrap scatter / stacked posterior std (median), R = 1",
         f(GC("train_boot_std_over_post_std_N32R1_median"), 2)],
        [f"bootstrap scatter / stacked posterior std (median), R = {NR}",
         f(GC(f"train_boot_std_over_post_std_N32R{NR}_median"), 2)],
        ["observations with the bandwidth / width guards active (training / TF)",
         f"{pct(GC('train_bandwidth_or_sigma_guard_fraction'))} / {pct(GC('tf_bandwidth_or_sigma_guard_fraction'))}"],
    ])
    t_obs = table(["per-observation quantity", "unweighted (training prior)", "TF-weighted"], [
        ["median per-component posterior std", f(PO("train", "std_g_median"), 4), f(PO("tf", "std_g_median"), 4)],
        ["median σ<sub>ε,i</sub> (quadrature)", f(PO("train", "sigma_eps_xu24_median"), 4),
         f(PO("tf", "sigma_eps_xu24_median"), 4)],
        ["median effective number of samples", f(PO("train", "n_eff_median"), 0), f(PO("tf", "n_eff_median"), 0)],
        ["median KDE bandwidth", f(PO("train", "kde_h_median"), 4), f(PO("tf", "kde_h_median"), 4)],
        ["median |corr(g₁, g₂)|", f(PO("train", "abs_corr_g1g2_median"), 2), f(PO("tf", "abs_corr_g1g2_median"), 2)],
    ])
    zsig = lambda comp: det(kde, comp) / dstd(kde, comp)

    results = (
        T(r"""<h3>Bias against the number of galaxies</h3><p>Point values use the first \(N\) galaxy IDs (all
<<r>> realizations each); brackets are the bootstrap 16–84% range. At \(N = 1\) the bracket spans most of the
range because single galaxies differ a lot.</p>""", r=NR) + t_n
        + fig("C_stacking/m_vs_N_R16.png",
              T(r"""\(m_1\) (left) and \(m_2\) (right) against the number of galaxies per cell, each with <<r>> noise
realizations, training nuisance prior. Markers: bootstrap median; bars: 16–84% of <<nb>> paired galaxy bootstrap
resamples. Blue: average of per-galaxy posterior means. Orange: stacked KDE likelihood. Green: stacked moment
Gaussian. Yellow: stacked log-KDE quadratic. (Paper candidate <code>paper_figs/C_m_vs_N</code>.)""", r=NR, nb=NB))
        + t_nfull
        + "<h3>Bias against the number of noise realizations (N = 32)</h3>" + t_r
        + fig("C_stacking/m_vs_R.png",
              r"""\(m_1\), \(m_2\) against the number of noise realizations per galaxy, with 32 galaxies per cell.
The per-galaxy average stays near \(m_1 \approx """ + fx(det(pg, "g1_m", "vs_R_at_N32", "1"), 2) + r"""\) however many
realizations are added; the stacks are at the percent level from \(R = 1\).""")
        + fig("C_stacking/residual_vs_truth.png",
              T(r"""Residual \(\hat g - g\) of each of the 25 cells against the true shear (small jitter separates cells
that share a true component), 32 galaxies × <<r>> realizations. Blue: per-galaxy posterior mean; orange: stacked KDE
likelihood. Bars: bootstrap standard deviation over galaxies; dashed lines: the \(m, c\) fits.""", r=NR))
        + "<h3>Sensitivity to the density estimate</h3>" + t_sens
        + "<h3>Checks: lattice, posterior width, guards</h3>" + t_chk
        + P(r"""A pull is (stacked mean − truth)/(stacked posterior std) over the 25 cells; a pull standard deviation of 1
and a bootstrap/posterior ratio of 1 mean the stacked width is right.""")
        + "<h3>Per-observation posteriors</h3>" + t_obs
        + "<h3>Training vs TF nuisance prior</h3>" + t_tf
        + P(T(r"""The benchmark galaxies were drawn from the training prior (\(v_{\rm circ}\) uniform on \([60, 540]\) km/s,
independent of \(m_r\)), not from the TF-conformed population. Their TF residuals
\([\log_{10} v_{\rm circ} - {\rm TF}(m_r)]/0.1\,{\rm dex}\) have a standard deviation of <<zb>> (Xu3: <<zx>>),
and <<zf>> lie beyond \(3\sigma\). The TF importance weights degenerate for off-relation galaxies: median TF
effective-sample fraction <<em>>, 5th percentile <<e5>>; <<e50>> of observations have fewer than 50 effective
samples. The benchmark is also brighter than Xu3 (median image S/N <<bi>> vs <<xi>>, median central H\(\alpha\)
S/N <<bh>> vs <<xh>>), so its per-galaxy shrinkage (\(m_1 = <<mb>>\)) is smaller than Xu3's
(raw Xu3 \(m_1 = <<mx>>\) with the TF-weighted mean, see <a href='raw-shear-bias.html'>raw shear bias</a>).""",
              zb=f(NP("tf_z_bench_std"), 2), zx=f(NP("tf_z_xu3_std"), 2), zf=pct(NP("tf_z_bench_abs_gt3_fraction"), 0),
              em=f(NP("tf_ess_fraction_median"), 3), e5=f(NP("tf_ess_fraction_p05"), 4),
              e50=pct(NP("tf_ess_lt50_fraction"), 1), bi=f(NP("bench_image_snr_median"), 0),
              xi=f(NP("xu3_image_snr_median"), 0), bh=f(NP("bench_halpha_snr_median"), 0),
              xh=f(NP("xu3_halpha_snr_median"), 0), mb=fx(det(pg, "g1_m"), 2),
              mx=fx(J("shear", "R_raw", "g1", "low_m"), 2)))
        + fig("C_stacking/tf_prior_mismatch.png",
              r"""Left: TF residual of the true \(v_{\rm circ}\) for the 32 benchmark galaxies (orange) and the Xu3 test
set (blue). Right: median TF effective-sample fraction of each benchmark galaxy against its |TF residual|.""", "85%")
        + "<h3>The earlier benchmark, for comparison</h3>"
        + P(T(r"""The earlier analysis of the same benchmark (<code>reports/likelihood-stack/04_benchmark</code>) used
<<ns>> identity-only samples per observation and a <<gn>>×<<gn>> evaluation grid. Its per-galaxy NPE mean had
combined \(m = <<pm>>\) at \(R = 16\), consistent with the per-galaxy row above. Its NPE-based stack combined
importance-weighted 9D posterior candidates per galaxy, whose weights degenerate to an effective sample size of
about one (as noted in the test plan), and did not remove the shrinkage (combined \(m = <<p9>>\)). The present
analysis marginalizes the nuisances through the 2D shear marginal instead.""",
              ns=JO("npe_samples"), gn=JO("grid_n"), pm=fx(JO("takeaway_data", "npe_mean", "16", "combined_m")),
              p9=fx(JO("takeaway_data", "npe_9d_stack_mean", "16", "combined_m"))))
    )

    interpretation = P(
        T(r"""<b>Stacking removes the shrinkage to the percent level.</b> With the nuisance prior that matches how the
galaxies were drawn, the stacked KDE posterior has \(m = (<<k1>>, <<k2>>)\) at \(N = 32\), \(R = <<r>>\), against
\((<<p1>>, <<p2>>)\) for the average of per-galaxy posterior means on exactly the same observations. With one
realization per galaxy (the survey-like case) it is \((<<q1>>, <<q2>>)\). From \(N = 4\) on (with \(R = <<r>>\))
\(|m| \le <<m4>>\), i.e. once the likelihood product dominates the prior. It is insensitive to the bandwidth
(factor 0.5 to 2) and to the floor, and the lattice checks move cell means by at most <<lc>>, so the old grid
snapping is gone.""",
          k1=fxpm(det(kde, "g1_m"), dstd(kde, "g1_m")), k2=fxpm(det(kde, "g2_m"), dstd(kde, "g2_m")), r=NR,
          p1=fx(det(pg, "g1_m")), p2=fx(det(pg, "g2_m")),
          q1=fx(det(kde, "g1_m", "vs_N_at_R1")), q2=fx(det(kde, "g2_m", "vs_N_at_R1")),
          m4=f(max(abs(det(kde, c, RN, str(n))) for n in NL if n >= 4 for c in ("g1_m", "g2_m")), 3),
          lc=f(max(GC("train_m401_minus_m400_mean_maxabs"), GC("train_m800_minus_m400_mean_maxabs"),
                   GC(f"train_local_exact_R{NR}_mean_maxabs")), 1)),
        T(r"""<b>A small residual in \(m_1\).</b> \(m_1 = <<k1>>\) is about <<z>> bootstrap standard deviations from zero
and does not shrink with smaller bandwidth, so it is not KDE smoothing. It may be a residual error in the learned
likelihood or a property of these 32 galaxies (the per-galaxy mean is also more biased in \(g_1\)). Stacking removes
the prior-driven shrinkage; it does not remove errors of the learned likelihood itself, which enter every factor
of the product coherently.""", k1=fx(det(kde, "g1_m")), z=f(abs(zsig("g1_m")), 1)),
        T(r"""<b>The Gaussian approximation is slightly worse</b> (\(m_1 = <<g>>\)) because it inherits the box truncation of
each posterior, as explained above.""", g=fx(det("stack_gauss_mean_train", "g1_m"))),
        T(r"""<b>Stacking realizations of the same galaxy gives over-confident widths</b>, as expected: at \(R = <<r>>\) the
galaxy-bootstrap scatter is <<b16>>× the stacked posterior width, at \(R = 1\) it is <<b1>>×. For a survey (one
observation per galaxy) the product is the right likelihood; this benchmark must quote bootstrap errors.""",
          r=NR, b16=f(GC(f"train_boot_std_over_post_std_N32R{NR}_median"), 1),
          b1=f(GC("train_boot_std_over_post_std_N32R1_median"), 2)),
        T(r"""<b>The TF-prior configuration is not tested by this benchmark.</b> With TF weights the stacked KDE still gives
\(m = (<<a>>, <<b>>)\), but here the TF prior is the wrong prior for the galaxies, the weights are degenerate, the
Gaussian-type estimators degrade (moment Gaussian \(m_1 = <<g>>\)), and the TF-weighted stack is more
over-confident. For Xu3-like data (on the TF relation) the TF-weighted samples would be the consistent choice; a
TF-conformed fixed-shear benchmark is needed to test the production configuration.""",
          a=fx(det("stack_kde_mean_tf", "g1_m")), b=fx(det("stack_kde_mean_tf", "g2_m")),
          g=fx(det("stack_gauss_mean_tf", "g1_m"))),
        r"""<b>For the paper (Sec. 6.5).</b> Show \(m\) against \(N\) for per-galaxy means vs the stack, state that the
likelihood is the NPE shear marginal divided by the uniform training prior, and state the benchmark's limitations
(training-prior galaxies, brighter than Xu3, constant shear only).""",
    )

    caveats = ul(
        r"""The benchmark reuses 32 nuisance galaxies in all 25 cells and is brighter than Xu3; it tests the
stacking machinery with a matched nuisance prior, not the production setting (Xu3-like galaxies, TF prior).""",
        r"""Constant shear only. Non-constant shear (Bernstein &amp; Armstrong 2014, Sec. II.2) would need the local
derivatives of each likelihood, not a product at a common shear.""",
        r"""The result requires \(q/\pi_{\rm train}\) to be a good likelihood. NPE miscalibration does not average
down in the product the way noise does (see <a href='posterior-calibration.html'>posterior calibration</a> for how
well the NPE marginals are calibrated); the residual \(m_1\) is the size of effect this caveat covers.""",
        T(r"""The diagonal KDE kernel is set from the marginal standard deviations while the posteriors are strongly
correlated in \((g_1, g_2)\) (median \(|{\rm corr}| = <<cr>>\)); halving or doubling the bandwidth changes \(m\) at
\(N = 32\) by at most <<bw>>.""", cr=f(PO("train", "abs_corr_g1g2_median"), 2),
          bw=f(max(abs(det(v, c) - det(kde, c)) for v in ("stack_kde_bw0.5_mean_train", "stack_kde_bw2_mean_train")
                   for c in ("g1_m", "g2_m")), 3)),
        T(r"""For TF-weighted samples the bandwidth and width guards were active for <<gf>> of observations; the TF
results are not meaningful for this benchmark beyond showing the machinery runs.""",
          gf=pct(GC("tf_bandwidth_or_sigma_guard_fraction"))),
        r"""All numbers are from the 100k-trained model.""",
    )

    provenance = (
        ul(
            "Group report: <a href='../C_stacking/report.html'>NPE likelihood stacking</a>; earlier benchmark: "
            "<code>/ocean/projects/phy250048p/shared/reports/likelihood-stack/04_benchmark/report.html</code>.",
            "Scripts: " + ", ".join(sc(s) for s in ("c_stack_infer.py", "c_stack_analysis.py", "c_stack_report.py",
                                                     "c_stack_pipeline.slurm")) + ".",
            "Key functions: <code>_sample</code> (c_stack_infer.py); <code>kde_density</code>, <code>_axis_kernel</code> "
            "(reflection), <code>gauss_logq</code>, <code>lapkde_fit</code>, <code>summarize</code>, "
            "<code>run_settings</code>, <code>boot_counts</code> (c_stack_analysis.py); <code>fit_rows</code> "
            "(c_stack_report.py).",
            "JSON keys: <code>C_stacking/report.json</code> → <code>metrics/&lt;estimator&gt;/vs_N_at_R16</code>, "
            "<code>vs_N_at_R1</code>, <code>vs_R_at_N32</code> (each with <code>det</code>, <code>boot_p16</code>, "
            "<code>boot_p84</code>, <code>boot_std</code>), <code>grid_checks</code>, <code>nuisance_prior</code>, "
            "<code>per_observation</code>, <code>analysis_meta/configs</code>; earlier benchmark "
            "<code>likelihood-stack/04_benchmark/report.json</code> → <code>takeaway_data</code>, <code>grid_n</code>, "
            "<code>npe_samples</code>.",
            "Samples and analysis cache: <code>/ocean/projects/phy250048p/shared/cache/paper_tests_C_stacking/</code>.",
            "Figures: <code>C_stacking/m_vs_N_R16.png</code> (paper <code>C_m_vs_N</code>), <code>m_vs_R.png</code>, "
            "<code>residual_vs_truth.png</code>, <code>tf_prior_mismatch.png</code>.",
        )
        + "<p>SLURM jobs (from <code>C_stacking/STATUS.txt</code>):</p>" + status_block("C_stacking")
    )
    return page(slug="npe-likelihood-stacking", title="Stacking NPE likelihoods at constant shear", code="T8",
                section=SEC, answer=answer, question=question, why=why, theory=theory, procedure=procedure,
                results=results, interpretation=interpretation, caveats=caveats, provenance=provenance)


PAGES = [page_selection(), page_response(), page_stack()]
for _p in PAGES:
    for _k, _v in _p.items():
        if not isinstance(_v, str):
            raise TypeError(f"{_p['slug']}: field {_k} is {type(_v).__name__}, not str")
