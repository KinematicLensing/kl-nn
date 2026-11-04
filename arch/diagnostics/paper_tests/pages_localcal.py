"""Explanatory page for test D: shear coverage conditional on the true shear.

One page (slug ``local-calibration``). Every number comes from
``paper-tests/D_local_calibration/report.json`` through the local loader ``D``
(the shared ``J`` only knows the campaign JSON files registered in
``test_pages_common``; this module does not edit that file). Figures are the
PNGs written by ``d_localcal_report.py``. Prose with LaTeX uses
``string.Template`` placeholders (``$name``).

Not yet wired into ``write_test_pages.py`` (ORDER / imports); to include it,
add ``pages_localcal`` there. Test-render with ``python pages_localcal.py OUTDIR``.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from string import Template

from test_pages_common import ROOT, SCRIPTS, fig, page, render, table
from test_pages_common import f

SECTION = "Sec. 6: Shear estimation"
DDIR = ROOT / "D_local_calibration"
COMPS = ("g1", "g2")
CL = {"g1": r"\(g_1\)", "g2": r"\(g_2\)"}
PRED = {"self": "NPE self-consistency", "gauss": "Gaussian formula", "toy": "uniform-prior toy"}


@lru_cache(maxsize=None)
def _d():
    return json.loads((DDIR / "report.json").read_text())


def D(*keys):
    """Look up a value in D_local_calibration/report.json; raise on a missing key."""
    node = _d()
    path = ["D"]
    for key in keys:
        path.append(str(key))
        if isinstance(node, dict):
            if key not in node:
                raise KeyError(f"missing key {'/'.join(path)}")
            node = node[key]
        elif isinstance(node, list):
            node = node[int(key)]
        else:
            raise KeyError(f"cannot index {'/'.join(path)}")
    return node


def T(s: str, **kw) -> str:
    return Template(s).substitute(**kw)


def P(x, nd=1):
    """Percent, fixed decimals."""
    return f"{100 * float(x):.{nd}f}%".replace("-", "−")


def PP(x, nd=1):
    """Percentage points with sign."""
    return f"{100 * float(x):+.{nd}f}".replace("-", "−")


def SG(x):
    """Signed pull, one decimal."""
    return f"{float(x):+.1f}".replace("-", "−")


def row(comp, lev, sl):
    """Row of the |g|-split table: sl in 0 (full), 1 (|g|<0.05), 2 (|g|>0.05)."""
    return D("split", f"{comp}_{lev}", sl)


def fine(comp, lev, kind):
    return D("fine", f"{comp}_{lev}_vs_{kind}")


def chi(entry, name):
    c = entry[f"chi2_{name}"]
    return f"{f(c['chi2'], 0)} / {c['nbins']}"


def build():
    sig_pi = D("sigma_pi")
    rp = D("reproduction")
    ti = D("toy_info")

    # ------------------------------------------------------------ question
    question = T(r"""<p>Over the whole test set the \(g_1, g_2\) posteriors are well calibrated: the
16–84% credible interval contains the true shear for $G1 / $G2 of galaxies. But an earlier report
split the galaxies by the size of their true shear, \(|g| = \sqrt{g_1^2+g_2^2}\), and found
coverage of $IN1 for \(|g| &lt; 0.05\) and $OUT1 for \(|g| &gt; 0.05\) (for \(g_1\)). This page asks
whether that "local miscalibration" is what a <i>correct</i> posterior must show once it is shrunk
toward the centre of the bounded shear prior \(U(-0.1, 0.1)\), or whether it points to a real error
in the learned posterior.</p>""", G1=P(rp["g1_full"]), G2=P(rp["g2_full"]), IN1=P(rp["g1_inner"]),
                 OUT1=P(rp["g1_outer"]))

    why = r"""<p>Sec. 5.2 of the paper quotes the global coverage as evidence that the NPE posterior is
right. A reader (or referee) who sees that coverage depends strongly on the true shear could read
it as a sign that the posterior is wrong exactly where it matters for shear estimation. If the
pattern is the normal behaviour of a correct Bayesian posterior under a bounded prior, the paper
can say so in one sentence and point to the shrinkage discussion in Sec. 6.2. If it is not, the
calibration claim needs a caveat. The test also connects the two sides of the story: the same
shrinkage that makes the raw posterior mean biased (\(m \approx -\sigma_{\rm post}^2/\sigma_\pi^2\),
see the shrinkage page) is what makes the coverage depend on the true shear.</p>"""

    # ------------------------------------------------------------ theory
    theory = T(r"""
<p><b>Symbols.</b> For one shear component \(g\) (either \(g_1\) or \(g_2\)) of one galaxy, \(D\) is
the galaxy's data, \(p(g\mid D)\) the posterior, and \(I(D)\) its central credible interval at level
\(L\) (the 16th to 84th percentile for \(L = 68\%\), 2.5th to 97.5th for 95%). The prior of the
training and test sets is \(\pi(g) = U(-0.1, 0.1)\) for each component; its standard deviation is
\(\sigma_\pi = 0.2/\sqrt{12} = $SPI\). \(\sigma_{\rm post}\) is the posterior standard deviation,
\(\hat g\) the posterior mean.</p>

<p><b>Two kinds of coverage.</b> The <i>global</i> coverage is the fraction of all test galaxies
whose interval contains the truth. The <i>conditional</i> coverage at true shear \(g\) is
\(C(g) = P\big(g \in I(D) \mid g\big)\): fix the true shear, draw many noisy data sets, and count how
often the interval catches it. The global coverage is the prior average,
\(\int \pi(g)\, C(g)\, dg\). A correct posterior guarantees only this average (and averages over
any subset chosen by the data, see below). It does <i>not</i> guarantee \(C(g) = L\) at each \(g\):
Bayesian credible intervals are not frequentist confidence intervals.</p>

<p><b>Why \(C(g)\) depends on \(g\): the Gaussian model.</b> Suppose the data give a Gaussian
likelihood \(N(g_{\rm obs};\, g, \sigma_L^2)\): a noisy measurement \(g_{\rm obs} = g + n\) with
\(n \sim N(0, \sigma_L^2)\). Approximate the prior by \(N(0, \sigma_\pi^2)\). The posterior is
Gaussian with
\[ \sigma_{\rm post}^2 = \frac{\sigma_L^2\sigma_\pi^2}{\sigma_L^2+\sigma_\pi^2}, \qquad
   \hat g = k\, g_{\rm obs}, \qquad k = \frac{\sigma_\pi^2}{\sigma_\pi^2+\sigma_L^2} = 1 - \frac{\sigma_{\rm post}^2}{\sigma_\pi^2}. \]
\(k\) is the <i>shrink factor</i>: \(k = 1\) means the data dominate, \(k = 0\) means the posterior is
just the prior. The error of the posterior mean is \(\hat g - g = -(1-k)\,g + k\,n\), so at fixed
true \(g\)
\[ \hat g - g \sim N(\beta,\; s^2), \qquad \beta = -g\,\frac{\sigma_{\rm post}^2}{\sigma_\pi^2}, \qquad
   s^2 = k^2\sigma_L^2 = k\,\sigma_{\rm post}^2 . \]
The mean offset \(\beta\) is the shrinkage bias toward zero; it grows with \(|g|\). The interval is
\(\hat g \pm z\,\sigma_{\rm post}\) (\(z = 0.994\) for 16–84%, 1.960 for 95%), so it contains \(g\) when
\(|\hat g - g| \le z\sigma_{\rm post}\), which happens with probability
\[ C(g) = \Phi\!\left(\frac{z\sigma_{\rm post} - \beta}{s}\right) - \Phi\!\left(\frac{-z\sigma_{\rm post} - \beta}{s}\right), \]
with \(\Phi\) the standard normal CDF. Two things happen:</p>
<ul>
<li>At \(g = 0\) there is no offset, and the interval half-width \(z\sigma_{\rm post}\) is wider than the
scatter \(s = \sqrt{k}\,\sigma_{\rm post}\) of \(\hat g\): <b>over-coverage</b>,
\(C(0) = 2\Phi(z/\sqrt k) - 1 &gt; L\).</li>
<li>At large \(|g|\) the offset \(\beta\) pushes the truth into the outer tail of the interval:
<b>under-coverage</b>.</li>
</ul>
<p>Averaged over the Gaussian prior, \(\hat g - g\) has variance
\(\sigma_{\rm post}^4/\sigma_\pi^2 + k\sigma_{\rm post}^2 = \sigma_{\rm post}^2\), so the average coverage is
exactly \(L\). The shrinkage trades coverage at small \(|g|\) against coverage at large \(|g|\)
without changing the average. The effect is large only when \(k\) is well below 1, i.e. when
\(\sigma_{\rm post}\) is comparable to \(\sigma_\pi\), which is the case for many of our galaxies
(median \(\sigma_{\rm post} = $MEDSIG\) for \(g_1\), against \(\sigma_\pi = $SPI\)).</p>

<p><b>Prior-dominated galaxies.</b> In the Gaussian model \(\sigma_{\rm post} &lt; \sigma_\pi\) always.
Real posteriors can be wider ($FRACWIDE of \(g_1\) posteriors have \(\sigma_{\rm post} \ge \sigma_\pi\)),
because the true prior is a box and the posterior need not be Gaussian. For those we set
\(\sigma_{\rm post} = \sigma_\pi\), the limit where the posterior equals the prior: \(k = 0\),
\(\hat g = 0\) and the interval is \(0 \pm z\sigma_\pi\), which contains \(g\) exactly when
\(|g| &lt; z\sigma_\pi\) (\(C\) becomes 1 or 0).</p>

<p><b>The real prior is a box: a Monte Carlo toy.</b> The uniform prior behaves differently: it
does not pull at all on a narrow likelihood far from the edges, and it pulls through truncation
when the likelihood reaches an edge. The toy keeps the Gaussian likelihood but uses the true box:
the posterior is \(N(g_{\rm obs}, \sigma_L^2)\) truncated to \([-0.1, 0.1]\). For each galaxy and
component we find the likelihood (location \(\mu_L\) and width \(\sigma_L\)) whose truncated posterior
has the same mean and variance as the measured NPE posterior. Where no truncated Gaussian is that
wide at that mean, we instead fix the location at \(\hat g\) and match only the variance; where even
that fails (\(\sigma_{\rm post} \ge \sigma_\pi\)), we set \(\sigma_L = \infty\) (posterior = prior). Then
we draw \(g_{\rm obs} \sim N(g_{\rm true}, \sigma_L^2)\) many times for that galaxy's true shear and
record how often the truncated posterior's interval contains \(g_{\rm true}\). Averaged over truths
drawn from the box prior, this toy gives $TOYGLOB global coverage (target 68%), as it should.</p>

<p><b>A model-free prediction: the NPE's own self-consistency.</b> Both models above assume a
likelihood shape. There is a prediction that assumes nothing except that the posterior is correct
and that the test-set truths are drawn from the prior (true here, with TF weighting). By Bayes'
theorem \(p(D\mid g) = p(D)\,p(g\mid D)/\pi(g)\), so for a truth bin \(B\)
\[ P\big(g \in I(D) \mid g \in B\big)
   = \frac{\big\langle \int_B p(g\mid D)\,\mathbf 1[g \in I(D)]\,dg \big\rangle_D}
          {\big\langle \int_B p(g\mid D)\,dg \big\rangle_D}, \]
where \(\langle\cdot\rangle_D\) averages over the test galaxies. For each galaxy the numerator is the
posterior mass that lies both in \(B\) and inside that galaxy's own interval, and the denominator is
its posterior mass in \(B\); both are TF-weighted sums over its posterior draws. So the NPE's
posteriors themselves predict what the truth-conditioned coverage must be if they are right. If
the measured coverage matches this prediction, the local pattern is fully accounted for by the
posteriors' own shape, including shrinkage, box truncation and non-Gaussian tails. (The
denominator summed over galaxies should also equal the number of truths in \(B\); we check that
too.) The statistical error of measured − predicted, if the posteriors are exact, comes from the
randomness of each truth given its data; we compute it from the same posterior masses.</p>

<p><b>What a real error looks like: conditioning on the data.</b> For a correct posterior,
coverage is nominal within any subset chosen by the data alone, such as bins of the posterior
mean \(\hat g\) or thirds of the posterior width \(\sigma_{\rm post}\), because given the data the
truth is distributed exactly as the posterior. Departures there cannot be blamed on shrinkage.</p>""",
                SPI=f(sig_pi, 4), MEDSIG=f(ti["g1"]["median_sigma_post"], 4),
                FRACWIDE=P(ti["g1"]["frac_sigma_post_ge_sigma_pi"]),
                TOYGLOB=P(ti["g1"]["toy_global_68_truths_from_prior"]))

    # ------------------------------------------------------------ procedure
    procedure = T(r"""<ol>
<li><b>Data.</b> The per-galaxy table extracted earlier from the Xu3 cache
(<code>A_shear/per_galaxy.npz</code>): for each of the $N galaxies, the TF-weighted posterior mean,
variance and 2.5/16/50/84/97.5% quantiles of \(g_1\) and \(g_2\), and the true values. The measured
coverage uses the 16–84% and 2.5–97.5% quantile intervals with equal galaxy weights.</li>
<li><b>Reproduce the earlier split.</b> \(|g| = \sqrt{g_1^2 + g_2^2}\) of the truth, split at 0.05, equal
weights, TF-weighted 16–84% interval (the definitions of <code>likelihood_stack_coverage.py</code>).</li>
<li><b>Gaussian prediction</b> for each galaxy and component from its own \(\sigma_{\rm post}\) and
true \(g\), with \(\sigma_{\rm post}\) clipped at \(\sigma_\pi\).</li>
<li><b>Uniform-prior toy</b> with $NMC draws of \(g_{\rm obs}\) per galaxy and component.</li>
<li><b>Self-consistency prediction.</b> One streaming pass over the cache (10,000 TF-weighted draws
per galaxy) stores each galaxy's posterior mass, and the part of it inside its own interval, in each
truth bin (<code>d_localcal_stream.py</code>).</li>
<li><b>Bins.</b> (a) \(|g| &lt; 0.05\) / \(&gt; 0.05\); (b) 20 bins of the true \(g_k\) across
\([-0.1, 0.1]\) and 14 bins of the true \(|g|\) on \([0, 0.141]\); (c) thirds of \(\sigma_{\rm post}\)
(per component), crossed with (a) and (b); and, as the data-conditioned check, 20 bins of the
posterior mean \(\hat g_k\).</li>
<li><b>Residual</b> = measured − predicted, with the binomial error of the measured coverage,
\(\sqrt{C(1-C)/N_{\rm bin}}\). Across bins we quote \(\chi^2 = \sum ({\rm residual}/{\rm error})^2\) against
the number of bins.</li>
</ol>""", N=f"{D('n_galaxies'):,}", NMC=D("n_mc"))

    # ------------------------------------------------------------ results
    head = ["component", "level", "subset", "N", "measured", "self-consistency", "Gaussian formula",
            "uniform toy"]
    rows = []
    for c in COMPS:
        for lev in ("68", "95"):
            for sl in (0, 1, 2):
                r = row(c, lev, sl)
                cells = [CL[c], lev + "%", ["all", r"\(|g| &lt; 0.05\)", r"\(|g| &gt; 0.05\)"][sl], f"{r['n']:,}",
                         f"{P(r['measured'], 2)} ± {100 * r['measured_se']:.2f}"]
                for p in ("self", "gauss", "toy"):
                    cells.append(f"{P(r['pred_' + p], 2)}<br><small>resid {PP(r['resid_' + p], 2)} "
                                 f"({SG(r['pull_' + p])}σ)</small>")
                rows.append(cells)
    tab_split = table(head, rows)

    chi_rows = []
    for c in COMPS:
        for lev in ("68", "95"):
            for kind, lab in (("gk", r"true \(g_k\), 20 bins"), ("absg", r"true \(|g|\), 14 bins")):
                F = fine(c, lev, kind)
                chi_rows.append([CL[c], lev + "%", lab] + [chi(F, p) for p in ("self", "gauss", "toy")]
                                + [f(max(abs(x["resid_self"]) for x in F["rows"]) * 100, 2)])
    tab_chi = table(["component", "level", "binned in", "χ² self-consistency", "χ² Gaussian", "χ² uniform toy",
                     "max |resid| self (pp)"], chi_rows)

    terc_rows = []
    for c in COMPS:
        TT = D("terciles", c)
        for r in TT["68_overall"] + TT["68_x_split"]:
            lab = r["label"].replace("T1", "narrow").replace("T2", "middle").replace("T3", "wide")
            lab = lab.replace(" |g|<0.05", r", \(|g| &lt; 0.05\)").replace(" |g|>0.05", r", \(|g| &gt; 0.05\)")
            terc_rows.append([CL[c], lab, f"{r['n']:,}", f"{P(r['measured'], 2)} ± {100 * r['measured_se']:.2f}"]
                             + [f"{P(r['pred_' + p], 2)} ({SG(r['pull_' + p])}σ)"
                                for p in ("self", "gauss", "toy")])
    tab_terc = table(["component", "subset (third of σ_post)", "N", "measured 68%", "self-consistency",
                      "Gaussian", "uniform toy"], terc_rows)
    tedges = D("terciles", "g1", "sigma_edges")

    dc_rows = []
    for c in COMPS:
        for lev in ("68", "95"):
            e = D("data_conditioned", f"{c}_{lev}_vs_ghat")
            dc_rows.append([CL[c], lev + "%", f"{f(e['chi2_nominal']['chi2'], 0)} / {e['chi2_nominal']['nbins']}",
                            f(e["chi2_nominal"]["max_abs_pull"], 1),
                            f(max(abs(x["resid_nominal"]) for x in e["rows"]) * 100, 2)])
    tab_dc = table(["component", "level", "χ² vs nominal", "max |pull|", "max |coverage − nominal| (pp)"], dc_rows)

    results = T(r"""
<p><b>Reproduction.</b> With \(|g| = \sqrt{g_1^2+g_2^2}\), equal weights and the TF-weighted 16–84%
interval, the split has $NIN galaxies at \(|g| &lt; 0.05\) and $NOUT above, and the coverages match the
earlier report to $DIFF (largest difference).</p>
<p><b>(a) The \(|g|\) split.</b> Measured coverage and the three predictions; "resid" is measured −
predicted in percentage points, followed by its size in units of the measured binomial error. For
"all", the self-consistency prediction is 68% (95%) by construction: each galaxy's own interval
holds that fraction of its posterior.</p>
$TAB_SPLIT
$FIG_THEORY
<p><b>(b) Fine bins.</b></p>
$FIG_GK
$FIG_ABS
<p>\(\chi^2\) of the residuals across bins (errors: measured binomial):</p>
$TAB_CHI
<p><b>(c) Thirds of the posterior width.</b> Thresholds for \(g_1\): narrow
\(\sigma_{\rm post} &lt; $T1\), middle $T1–$T2, wide \(&gt; $T2\). The first three rows per component are
data-conditioned subsets (their global coverage should be 68% for a correct posterior); the rest
split each third by \(|g|\).</p>
$TAB_TERC
$FIG_TERC
<p><b>Data-conditioned check.</b> Coverage in 20 bins of the posterior mean \(\hat g_k\):</p>
$TAB_DC
$FIG_DC""",
                NIN=f"{rp['n_inner']:,}", NOUT=f"{rp['n_outer']:,}", DIFF=f(rp["max_abs_diff"], 3),
                TAB_SPLIT=tab_split, TAB_CHI=tab_chi, TAB_TERC=tab_terc, TAB_DC=tab_dc,
                T1=f(tedges[1], 4), T2=f(tedges[2], 4),
                FIG_THEORY=fig(DDIR / "theory_coverage_vs_g.png",
                               r"Theory: coverage of the 16–84% interval as a function of the true shear for "
                               r"populations with a single posterior width. Left: the Gaussian-prior formula "
                               r"above. Right: the uniform-prior toy with the matching likelihood width. Both "
                               r"over-cover near \(g = 0\) and under-cover at large \(|g|\); the box prior makes the "
                               r"drop sharper and concentrates it near the edges."),
                FIG_GK=fig(DDIR / "coverage_vs_gk.png",
                           r"Coverage against the true \(g_k\) (20 bins). Black points: measured, with binomial "
                           r"errors. Blue solid: NPE self-consistency; orange dashed: Gaussian formula; green "
                           r"dash-dot: uniform-prior toy. Grey dotted: nominal level. The small panels show "
                           r"measured − predicted for the self-consistency and toy predictions (same colours); "
                           r"the Gaussian residuals are large and can be read off the upper panels."),
                FIG_ABS=fig(DDIR / "coverage_vs_absg.png",
                            r"The same against the true \(|g| = \sqrt{g_1^2+g_2^2}\) (14 bins). Above \(|g| = 0.1\) "
                            r"only the box corners remain, so bins there are small."),
                FIG_TERC=fig(DDIR / "coverage_vs_gk_by_width_tercile.png",
                             r"68% coverage against the true \(g_k\) in each third of the posterior width. "
                             r"Narrow posteriors (left) stay within about 10 points of nominal except at the box "
                             r"edges; the strong dependence on \(g\) comes from the wide, prior-dominated posteriors "
                             r"(right). The self-consistency prediction (blue) follows the measurement in all three."),
                FIG_DC=fig(DDIR / "coverage_vs_posterior_mean.png",
                           r"Coverage minus nominal in bins of the posterior mean \(\hat g_k\), a function of "
                           r"the data only. A correct posterior gives zero in every bin."))

    interpretation = INTERPRETATION(rp, ti)
    caveats = CAVEATS(ti)
    provenance = T(r"""<ul>
<li>Scripts in <code>$SCRIPTS/</code>: <code>d_localcal_stream.py</code> (+ <code>.slurm</code>; posterior
masses per truth bin, streaming the cache), <code>d_localcal_analysis.py</code> (reproduction, Gaussian
formula, uniform toy, self-consistency, all bins; writes <code>report.json</code>),
<code>d_localcal_report.py</code> (figures, group report).</li>
<li>SLURM job (RM-shared, CPU): see <code>D_local_calibration/STATUS.txt</code>.</li>
<li>Inputs (read-only): <code>A_shear/per_galaxy.npz</code>; the Xu3 cache of the canonical model;
<code>likelihood-stack/02_coverage/report.json</code> (reference numbers).</li>
<li>JSON: <code>D_local_calibration/report.json</code> → <code>reproduction</code>, <code>split</code>,
<code>fine</code>, <code>terciles</code>, <code>data_conditioned</code>, <code>toy_info</code>.</li>
<li>Group report: <a href='../D_local_calibration/report.html'>D_local_calibration/report.html</a>.</li>
</ul>""", SCRIPTS=SCRIPTS)

    r1 = row("g1", "68", 1)
    r2 = row("g1", "68", 2)
    answer = ANSWER(r1, r2)
    return page(slug="local-calibration", title="Coverage conditional on the true shear", code="D",
                section=SECTION, answer=answer, question=question, why=why, theory=theory,
                procedure=procedure, results=results, interpretation=interpretation, caveats=caveats,
                provenance=provenance)


# ------------------------------------------------------------ outcome-dependent text
def ANSWER(r1, r2):
    return (f"Yes, it is shrinkage: if the NPE posteriors are correct they predict {P(r1['pred_self'])} / "
            f"{P(r2['pred_self'])} coverage of g1 for |g|<0.05 / >0.05, against {P(r1['measured'])} / "
            f"{P(r2['measured'])} measured. The residual errors are ≲1 point here and a few points near the "
            f"prior edges.")


def _explained(r):
    return (r["pred_self"] - 0.68) / (r["measured"] - 0.68)


def INTERPRETATION(rp, ti):
    r1i, r1o = row("g1", "68", 1), row("g1", "68", 2)
    r2i, r2o = row("g2", "68", 1), row("g2", "68", 2)
    F1, F2 = fine("g1", "68", "gk"), fine("g2", "68", "gk")
    A1, A2 = fine("g1", "68", "absg"), fine("g2", "68", "absg")
    corner1 = A1["rows"][-1]
    t1, t2 = D("terciles", "g1", "68_overall"), D("terciles", "g2", "68_overall")
    dc1 = D("data_conditioned", "g1_68_vs_ghat")
    dc2 = D("data_conditioned", "g2_68_vs_ghat")
    dcmax = max(abs(x["resid_nominal"]) for x in dc1["rows"] + dc2["rows"])
    dc_inner = [x for x in dc1["rows"] + dc2["rows"] if abs(x["center"]) < 0.05]
    narrow = [x["measured"] for t in (D("terciles", "g1", "68_vs_gk_by_tercile", 0, "rows"),
                                      D("terciles", "g2", "68_vs_gk_by_tercile", 0, "rows"))
              for x in t[2:-2]]
    r1f, r2f = row("g1", "68", 0), row("g2", "68", 0)
    dc_inner_max = max(abs(x["resid_nominal"]) for x in dc_inner)
    gk1 = F1["rows"]
    frac = [x["expected_n_self_frac_diff"] for x in gk1]
    n_bin = gk1[0]["n"]
    edge_bin = gk1[-1]
    next_bin = gk1[-2]
    mid_bins = [x for x in gk1 if abs(x["center"]) < 0.04]
    mid_frac = sum(x["expected_n_self"] for x in mid_bins) / sum(x["n"] for x in mid_bins) - 1
    return T(r"""
<p><b>The local pattern is what a correct posterior must show.</b> The model-free prediction (the
NPE's own posteriors, assuming they are right) gives $S1I for \(|g| &lt; 0.05\) and $S1O for
\(|g| &gt; 0.05\) (\(g_1\)), against measured $M1I and $M1O; for \(g_2\), $S2I / $S2O against
$M2I / $M2O. The departures from 68% are $D1I and $D1O points; the self-consistency prediction
accounts for $E1I and $E1O of them. So a correct posterior shrunk toward the centre of the box
prior would show essentially the same over-coverage at small \(|g|\) and under-coverage at large
\(|g|\). The same holds in fine bins: against the true \(g_k\) the residuals are at most
$MAXR1 points (\(g_1\)) and $MAXR2 points (\(g_2\)), while the coverage itself runs from about 16% at
the edges to 88% at the centre.</p>

<p><b>The width thirds show where it comes from.</b> For narrow posteriors (lowest third of
\(\sigma_{\rm post}\)) the coverage stays between $NLO and $NHI except in the two outermost bins,
where the box edge truncates the posterior; the strong dependence on \(g\) comes from
the wide, prior-dominated posteriors, whose intervals are close to the prior's own 16–84% range
\([-0.068, 0.068]\): they contain almost every truth with \(|g_k| &lt; 0.05\) and almost none near the
edges. This is the shrinkage picture of the Gaussian model, with \(k \to 0\).</p>

<p><b>The simple models get the sign and most of the size, but not the details.</b> The
Gaussian formula predicts $G1I / $G1O and the uniform-prior toy $U1I / $U1O (\(g_1\), inner / outer);
both are rejected in the fine bins (\(\chi^2\) $CG and $CU for 20 bins, against $CS for the
self-consistency prediction). The Gaussian prior has long tails where the true box has none, so the
formula under-predicts the coverage of wide posteriors near the edges and over-predicts the global
coverage loss. The toy has the right prior but assumes a Gaussian likelihood whose width is inferred
from the posterior moments, which misplaces the coverage of the middle third of widths. They are
useful to explain the effect in words, not to test the NPE.</p>

<p><b>Small genuine errors remain, mostly near the prior edges.</b> Conditioning on the data
removes shrinkage from the question, so any departure there is a real calibration error:</p>
<ul>
<li>By width: the middle third of \(g_1\) widths covers $TM1 (\(g_2\): $TM2) instead of 68%, and the
narrow third $TN1; the wide third is at $TW1.</li>
<li>By posterior mean: within \(|\hat g_k| &lt; 0.05\) the coverage is within $DCIN points of 68%; the
largest departures ($DCMAX points) are in the outermost bins, \(|\hat g_k| \gtrsim 0.08\), where few
galaxies are.</li>
<li>Summed posterior mass against the number of truths: the test set has $NBIN truths in each
0.01-wide bin of \(g_k\). Added over galaxies, the NPE posteriors put $EDGE too much mass in the
outermost bin (\(0.09 &lt; |g_1| &lt; 0.1\)), $NEXT too little in the next one inward, and $MID too much in
\(|g_1| &lt; 0.04\). For a correct posterior these would agree. The corners of the box
(\(|g| &gt; 0.13\)) also cover more often than predicted ($CORM measured vs $CORP).</li>
<li>The known global \(g_1\) deficit ($G1FULL at 68%, posteriors about 2% too narrow; see the
calibration page) is the same effect: \(g_1\) residuals in the split are $R1F (all), $R1I and $R1O
points, those of \(g_2\) $R2F, $R2I and $R2O.</li>
</ul>
<p>These are percent-level errors on top of a 20-point effect that is fully expected. For the paper:
Sec. 5.2 can say that coverage conditioned on the true shear is not expected to be nominal under the
bounded prior, and that the NPE's own posteriors reproduce the measured dependence to about a
percentage point away from the prior edges.</p>""",
             S1I=P(r1i["pred_self"]), S1O=P(r1o["pred_self"]), M1I=P(r1i["measured"]), M1O=P(r1o["measured"]),
             S2I=P(r2i["pred_self"]), S2O=P(r2o["pred_self"]), M2I=P(r2i["measured"]), M2O=P(r2o["measured"]),
             D1I=PP(r1i["measured"] - 0.68), D1O=PP(r1o["measured"] - 0.68),
             E1I=P(_explained(r1i), 0), E1O=P(_explained(r1o), 0),
             MAXR1=f(100 * max(abs(x["resid_self"]) for x in F1["rows"]), 1),
             MAXR2=f(100 * max(abs(x["resid_self"]) for x in F2["rows"]), 1),
             G1I=P(r1i["pred_gauss"]), G1O=P(r1o["pred_gauss"]), U1I=P(r1i["pred_toy"]), U1O=P(r1o["pred_toy"]),
             CG=chi(F1, "gauss"), CU=chi(F1, "toy"), CS=chi(F1, "self"),
             TM1=P(t1[1]["measured"]), TM2=P(t2[1]["measured"]), TN1=P(t1[0]["measured"]), TW1=P(t1[2]["measured"]),
             DCIN=f(100 * dc_inner_max, 1), DCMAX=f(100 * dcmax, 1), NBIN=f"{n_bin:,}",
             EDGE=P(edge_bin["expected_n_self"] / edge_bin["n"] - 1, 0),
             NEXT=P(1 - next_bin["expected_n_self"] / next_bin["n"], 0), MID=P(mid_frac, 0),
             CORM=P(corner1["measured"]), CORP=P(corner1["pred_self"]),
             NLO=P(min(narrow), 0), NHI=P(max(narrow), 0), G1FULL=P(r1f["measured"]),
             R1F=PP(r1f["resid_self"]), R1I=PP(r1i["resid_self"]), R1O=PP(r1o["resid_self"]),
             R2F=PP(r2f["resid_self"]), R2I=PP(r2i["resid_self"]), R2O=PP(r2o["resid_self"]))


def CAVEATS(ti):
    return T(r"""<ul>
<li><b>The self-consistency prediction uses the posteriors being tested.</b> It answers "is the
measured pattern what these posteriors imply if they are right?", which is the question here. It
is not independent of the posteriors: a posterior that is wrong in a way that also shifts its own
predicted coverage could still pass. The data-conditioned checks and the posterior-mass check do not
have this weakness, and they show the small edge errors listed above.</li>
<li><b>Errors.</b> Residual errors are the binomial errors of the measured coverage. The truths are
stratified in \(g_k\) (the same count in every bin), so for the truth-binned checks the binomial
error is if anything conservative. The self-consistency residual has its own null error
(<code>null_se_self</code> in the JSON), which is nearly identical.</li>
<li><b>Toy bookkeeping.</b> For \(g_1\), the moment match to a truncated Gaussian converged for
$CONV of galaxies; $FB used the fixed-location width match and $PD (those with
\(\sigma_{\rm post} \ge \sigma_\pi\)) were treated as prior-dominated. The width match at fixed location
(variant in the JSON, <code>toy_alt</code>) does worse. The toy is one component at a time; the real
posterior is 2D and marginalized over seven nuisance parameters.</li>
<li><b>Gaussian formula for wide posteriors.</b> Clipping \(\sigma_{\rm post}\) at \(\sigma_\pi\) makes the
prediction a step function for $PD of \(g_1\) posteriors. Its global coverage for truths drawn from the box prior is
$GG, not 68%, because the truths follow the box, not \(N(0, \sigma_\pi^2)\).</li>
<li><b>Edge behaviour not diagnosed.</b> The excess posterior mass in the outermost 0.01 of the box
and the deficit just inside it could come from how the bounded flow handles its support boundary.
Would be good to check whether the paper model (1M training set) shows the same.</li>
<li>Equal galaxy weights only; the 100k-trained model and the Xu3 test set only.</li>
</ul>""", CONV=P(ti["g1"]["frac_moment_match_converged"]), FB=P(ti["g1"]["frac_fallback_width_match"]),
             PD=P(ti["g1"]["frac_prior_dominated"]), GG=P(ti["g1"]["gauss_global_68_truths_from_prior"]))


PAGES = [build()]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    for p in PAGES:
        (out / f"{p['slug']}.html").write_text(render(p))
        print("wrote", out / f"{p['slug']}.html")
