"""Explanatory pages for Sec. 5 (validating the NPE) of the paper-tests campaign.

Four pages: training convergence (R1), posterior calibration (R3), Tully-Fisher
importance-weight health (T6) and example posteriors / SNR dependence (R2a, R2b).

Numbers come from the campaign JSON files through ``J``. A few settings that are
not results (TF-prior constants, training settings, galaxy-selection targets,
SLURM job IDs) are read from their source: ``arch/tf_prior.py``, the canonical
model config, ``a_npe_stream.py`` and ``A_npe/STATUS.txt``.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

from test_pages_common import J, MODEL, ROOT, SCRIPTS, f, fig, page, pct, pm, table

SECTION = "Sec. 5: Validating the NPE"
ARCH = Path("/jet/home/xwang30/kl-nn/arch")
CONFIG = Path("/ocean/projects/phy250048p/shared/configs") / f"cfg_{MODEL}.json"


# --------------------------------------------------------------------- helpers
def S(template: str, **kw) -> str:
    """Fill ``<<name>>`` placeholders (avoids brace escaping in LaTeX strings)."""
    out = template
    for key, value in kw.items():
        out = out.replace(f"<<{key}>>", str(value))
    if "<<" in out and ">>" in out[out.index("<<"):]:
        left = re.search(r"<<(\w+)>>", out)
        if left:
            raise KeyError(f"unfilled placeholder {left.group(1)}")
    return out


@lru_cache(maxsize=None)
def _config() -> dict:
    return json.loads(CONFIG.read_text())


def C(*keys):
    """Look up a value in the canonical model config; raise on a missing key."""
    node = _config()
    for key in keys:
        node = node[key]
    return node


@lru_cache(maxsize=None)
def _tf():
    if str(ARCH) not in sys.path:
        sys.path.insert(0, str(ARCH))
    from tf_prior import TFPrior  # noqa: E402

    return TFPrior()


@lru_cache(maxsize=None)
def _stream_constants() -> dict:
    """Module-level constants of a_npe_stream.py, read without importing it."""
    tree = ast.parse((Path(SCRIPTS) / "a_npe_stream.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _job(status_rel: str, pattern: str) -> str:
    text = (ROOT / status_rel).read_text()
    m = re.search(pattern, text)
    if not m:
        raise KeyError(f"no match for {pattern} in {status_rel}")
    return m.group(1)


PNAME = {
    "g1": r"\(g_1\)", "g2": r"\(g_2\)", "theta_int": r"\(\theta_{\rm int}\)",
    "cosi": r"\(\cos i\)", "v0": r"\(v_0\)", "vcirc": r"\(v_{\rm circ}\)",
    "rscale": r"\(r_{\rm scale}\)", "hlr": r"\(r_{\rm hlr}\)",
    "halpha_flux_true": r"\(\log_{10} F_{\rm H\alpha}\)",
}
PDESC = {
    "g1": "shear component along the pixel axes",
    "g2": "shear component at 45° to the pixel axes",
    "theta_int": "intrinsic position angle of the disk (radians, periodic)",
    "cosi": "cosine of the disk inclination (1 = face-on, 0 = edge-on)",
    "v0": "systemic velocity offset (km/s)",
    "vcirc": "asymptotic circular (rotation) velocity (km/s)",
    "rscale": "turnover radius of the rotation curve (arcsec)",
    "hlr": "half-light radius (arcsec)",
    "halpha_flux_true": "log of the total Hα line flux (erg/s/cm²)",
}
NAMES9 = list(PNAME)


def _r3(weighting: str, name: str) -> dict:
    for row in J("npe", "R3", "rows", weighting):
        if row["parameter"] == name:
            return row
    raise KeyError(f"R3 row {weighting}/{name}")


def _implied_width_ratio(coverage68: float) -> float:
    """If a Gaussian ±1σ interval covers c, the true error is σ·z(c); return z(c)."""
    return NormalDist().inv_cdf(0.5 * (1.0 + coverage68))


def _gal(index: int) -> dict:
    return J("npe", "R2a", "galaxies", index)


# ============================================================ 1. training loss
def _page_training() -> dict:
    r = lambda k: J("npe", "R1", k)  # noqa: E731
    n_ep = r("n_epochs")
    gap = r("generalization_gap_valid_minus_train")
    patience = C("train", "early_stopping_patience")
    min_delta = C("train", "early_stopping_min_delta")
    n_train = C("data", "size")
    n_valid = C("test", "size")

    question = S(r"""<p>Did the network finish learning, and did it learn the general mapping from
data to posterior rather than memorizing the training galaxies? We look at the
training loss and the validation loss (the same loss evaluated on galaxies the network
never trains on) after each of the <<n>> training epochs.</p>""", n=n_ep)

    why = r"""<p>Sec. 5.1 of the paper shows the loss curve as the first piece of evidence that the
NPE is trustworthy. A reader needs to know (i) that training was not stopped while the
model was still improving quickly, since every later result would then describe an
unfinished model, and (ii) that the model is not overfitting, since an overfit model
gives posteriors that are too narrow for new galaxies. The loss curve cannot show that
the posteriors are <i>correct</i>; that is what the calibration test (next page) is for.</p>"""

    theory = S(r"""<p><b>What the loss is.</b> The NPE is a conditional normalizing flow that returns a
probability density \(q_\psi(\mathbf{x}\mid\mathbf{c})\) over the nine galaxy parameters
\(\mathbf{x} = (g_1, g_2, \theta_{\rm int}, \cos i, v_0, v_{\rm circ}, r_{\rm scale},
r_{\rm hlr}, \log_{10}F_{\rm H\alpha})\), given the feature vector \(\mathbf{c}\) that the
network extracts from one galaxy's image, spectra and metadata. Here \(\psi\) are the
network weights. The loss is the negative log-likelihood (NLL) of the true parameters,
averaged over galaxies:</p>
\[ \mathcal{L}(\psi) = -\frac{1}{N}\sum_{i=1}^{N} \log q_\psi(\mathbf{x}_i \mid \mathbf{c}_i). \]
<p><b>Why minimizing it gives the posterior.</b> The training galaxies are drawn as
\(\mathbf{x}\sim\pi_{\rm train}(\mathbf{x})\), then data \(\mathbf{c}\sim p(\mathbf{c}\mid\mathbf{x})\).
For a large training set the loss approaches its expectation over this process, which
splits into two terms:</p>
\[ \mathbb{E}\big[-\log q_\psi(\mathbf{x}\mid\mathbf{c})\big]
 = \mathbb{E}_{\mathbf{c}}\Big[ \mathrm{KL}\big(p(\mathbf{x}\mid\mathbf{c})\,\big\|\,q_\psi(\mathbf{x}\mid\mathbf{c})\big) \Big]
 + \mathbb{E}_{\mathbf{c}}\Big[ H\big(p(\mathbf{x}\mid\mathbf{c})\big) \Big]. \]
<p>Here \(p(\mathbf{x}\mid\mathbf{c})\) is the true posterior under the training prior,
\(\mathrm{KL}(p\|q) = \int p\log(p/q)\) is the Kullback–Leibler divergence (zero only when
\(q = p\), positive otherwise), and \(H(p) = -\int p\log p\) is the entropy of the true
posterior. The entropy term does not depend on the network. So a lower NLL means a
smaller average KL divergence from the true posterior, and the minimum is reached when
\(q_\psi\) equals the true posterior. This is why the NLL is the standard NPE loss. It
also means the NLL can never tell us how close we are to zero KL, because the entropy
term is unknown: we can compare NLL values, not judge one on its own.</p>
<p><b>Why the absolute value is odd-looking (e.g. strongly negative).</b> The network
works in rescaled coordinates: each parameter is mapped linearly onto a fixed interval
using its training-prior range (after taking \(\log_{10}\) of the Hα flux, and dividing
\(\theta_{\rm int}\) by \(\pi\)). A density in rescaled coordinates \(y\) and in physical
coordinates \(x\) differ by the Jacobian of this linear map, \(\log q_x = \log q_y +
\sum_k \log|{\rm d}y_k/{\rm d}x_k|\), which is the same constant for every galaxy and
every model with the same prior ranges. So the absolute NLL is only defined up to this
constant; differences are what matter. A negative NLL just means the density at the
truth is larger than 1 per unit rescaled volume, i.e. the posterior is much narrower than
the prior box. A difference of \(\Delta\) nats means the model puts \(e^{\Delta}\) times
more density on the truth, on average.</p>
<p><b>Train vs validation.</b> The validation set is a separate set of simulated galaxies
that the network never updates on. If the model memorizes the training galaxies, the
training NLL keeps falling while the validation NLL stalls and then rises. A small, stable
gap with a still-falling validation curve means the model generalizes. In this training
setup the noise added to the training images and spectra is drawn fresh every epoch
(and each galaxy is randomly shown as simulated or rotated by 90°), while the validation
noise is fixed. So the network cannot memorize the noise; any gap comes from the finite
number of distinct training galaxies.</p>""")

    procedure = S(r"""<ul>
<li>Read the per-epoch training and validation NLL from the loss file written during
training (<code><<csv>></code>). Nothing was retrained.</li>
<li>Training set: <<ntr>> simulated galaxies; validation set: <<nva>> separate galaxies
(from the model config). The CCL-pretrained feature extractor is kept frozen in this
training stage.</li>
<li>Training ran for up to <<n>> epochs with a warm-up then cosine-decaying learning rate,
and early stopping if the validation NLL did not improve by more than <<md>> for
<<pat>> epochs (config values). The epoch with the lowest validation NLL is kept.</li>
<li>Quantities recorded: the best epoch, NLL at that epoch, the gap (validation − train),
and how much the validation NLL still improved over the last 10 and last 50 epochs.</li>
</ul>""", csv=J("npe", "R1", "loss_csv"), ntr=f"{n_train:,}", nva=f"{n_valid:,}", n=n_ep,
              md=f(min_delta, 3), pat=patience)

    rows = [
        ["epochs trained", str(n_ep)],
        ["best (lowest validation NLL) epoch", str(r("best_epoch_1indexed"))],
        ["training NLL at best epoch", f(r("train_at_best"), 3)],
        ["validation NLL at best epoch", f(r("valid_at_best"), 3)],
        ["gap, validation − training", f(gap, 3, sign=True)
         + S(r" nats (density at truth \(e^{<<g>>}\approx <<x>>\times\) higher on training galaxies)",
             g=f(gap, 2), x=f(math.exp(gap), 2))],
        ["validation improvement over last 10 epochs", f(r("valid_improvement_last_10_epochs"), 4)],
        ["validation improvement over last 50 epochs", f(r("valid_improvement_last_50_epochs"), 3)],
        ["epoch-1 training / validation NLL", f(r("epoch1_train"), 3) + " / " + f(r("epoch1_valid"), 3)],
        ["early stopping triggered?", "yes" if r("early_stopping_triggered") else "no"],
    ]
    results = table(["quantity", "value"], rows) + fig(
        "A_npe/r1_loss_curve.png",
        "Training (blue) and validation (orange) NLL per galaxy vs epoch; the dot marks the best "
        "validation epoch. Inset: epochs 100–200. The epoch-1 training value lies above the "
        "plotted range.")

    interpretation = S(r"""<p><b>Converged.</b> The best validation epoch is the last one (<<best>> of <<n>>), and
over the final 10 epochs the validation NLL improved by only <<g10>> nats, about the size of the
early-stopping tolerance (<<md>>). Early stopping never fired because the validation NLL
kept edging down, reaching a new best at the final epoch, while the cosine schedule drove the learning rate to its minimum; the
curve is flat at the end, so more epochs at this learning rate would change little.</p>
<p><b>No harmful overfitting.</b> The validation NLL never turns up, and the gap of
<<gap>> nats per galaxy (spread over nine parameters) is modest. It says the model is
somewhat more confident on galaxies it has seen, which is expected with only
<<ntr>> training galaxies. Whether the validation posteriors are too narrow is answered
directly by the calibration test, which finds them close to correct.</p>
<p>At epoch 1 the validation NLL (<<v1>>) is lower than the training NLL (<<t1>>). This is
normal: the training value is averaged over the epoch while the model improves quickly,
whereas the validation value is measured at the end of the epoch.</p>""",
        best=r("best_epoch_1indexed"), n=n_ep, g10=f(r("valid_improvement_last_10_epochs"), 4),
        md=f(min_delta, 3), gap=f(gap, 2), ntr=f"{n_train:,}", v1=f(r("epoch1_valid"), 2),
        t1=f(r("epoch1_train"), 2))

    caveats = S(r"""<ul>
<li>This is the current model trained on <<ntr>> galaxies (validated on <<nva>>). The paper
model will be trained on a larger set; its loss curve and gap must be redone.</li>
<li>The NLL is in rescaled coordinates, so the absolute numbers (<<tb>>, <<vb>>) are not
physically meaningful and should not be compared with models that use different prior
ranges. Quote them only with that note, or show only the curve.</li>
<li>A low NLL does not guarantee calibrated posteriors: it is an average over all
parameters and galaxies and can hide, e.g., a slightly too-narrow shear posterior.
Calibration is tested separately.</li>
<li>The validation NLL measures the NPE before TF reweighting, i.e. the posterior under the
training prior (uniform \(v_{\rm circ}\)). The TF-weighted posteriors used for science are tested
on the calibration and TF-weight pages.</li>
</ul>""", ntr=f"{n_train:,}", nva=f"{n_valid:,}", tb=f(r("train_at_best"), 1), vb=f(r("valid_at_best"), 1))

    provenance = S(r"""<ul>
<li>Script: <code><<s>>/a_npe_report.py</code>, function <code>r1_loss</code> (reads the loss CSV
with <code>plot_losses.load_losses</code>); run on the login node, no SLURM job.</li>
<li>Loss file: <code><<csv>></code>. Training settings: <code><<cfg>></code>.</li>
<li>JSON: <code>A_npe/report.json</code> → <code>R1</code>. Figure: <code>A_npe/r1_loss_curve.png</code>
(paper copy <code>paper_figs/A_npe_loss_curve.*</code>).</li>
<li>Detailed group report: <a href='../A_npe/report.html'>A_npe/report.html</a>, section “R1”.</li>
</ul>""", s=SCRIPTS, csv=J("npe", "R1", "loss_csv"), cfg=CONFIG)

    answer = (f"Yes. The best validation loss is at the last epoch ({r('best_epoch_1indexed')} of {n_ep}), "
              f"it improved by only {f(r('valid_improvement_last_10_epochs'), 4)} nats over the final 10 epochs, "
              f"and the train–validation gap ({f(gap, 2)} nats) is modest with no upturn, so there is no "
              "sign of harmful overfitting.")

    return page(slug="training-convergence", title="Training convergence", code="R1", section=SECTION,
                answer=answer, question=question, why=why, theory=theory, procedure=procedure,
                results=results, interpretation=interpretation, caveats=caveats, provenance=provenance)


# ============================================================ 2. calibration
def _page_calibration() -> dict:
    n = J("npe", "R3", "n_galaxies")
    g1, g2 = _r3("equal", "g1"), _r3("equal", "g2")
    g1r, g2r = _r3("shape_noise_regularized", "g1"), _r3("shape_noise_regularized", "g2")
    th, thr = _r3("equal", "theta_int"), _r3("shape_noise_regularized", "theta_int")
    reg = lambda k: J("npe", "R3", "regularized_weight_diagnostics", k)  # noqa: E731
    val = lambda *k: J("npe", "validation_vs_fiber_gauge_03", *k)  # noqa: E731
    band = g1["ks_dkw95_band"]
    z_g1 = _implied_width_ratio(g1["cov68_pit"])
    z_th = _implied_width_ratio(th["cov68_summary"])
    others = [p for p in NAMES9 if p not in ("g1", "g2", "theta_int")]
    dev_others = max(abs(_r3("equal", p)["cov68_summary"] - 0.68) for p in others)

    question = r"""<p>Are the NPE posteriors <i>calibrated</i>? That is, when the posterior says “the
true value is in this interval with 68% probability”, is the truth actually inside it for
68% of galaxies, and likewise for every other probability level? We test this for all
nine parameters on the 100,000-galaxy Xu3 test set.</p>""".replace(
        "100,000", f"{n:,}")

    why = r"""<p>Sec. 5.2 of the paper. Everything in Sec. 6 uses the posterior: its mean as the
shear estimator, its width as the per-galaxy shape noise \(\sigma_{\epsilon,i}\), and (for
stacking) its full shape. If the posteriors were systematically too narrow or too wide,
the quoted uncertainties and the width-based weights and cuts would be wrong. Calibration
is the standard global check that an NPE has learned the right posterior and not just a
good point estimate.</p>"""

    theory = S(r"""<p><b>The probability-integral transform (PIT).</b> For galaxy \(i\), let
\(F_i(x) = P(X \le x \mid D_i)\) be the posterior cumulative distribution function (CDF) of
one parameter, given that galaxy's data \(D_i\). Define the PIT value
\(u_i = F_i(x_i^{\rm true})\): the posterior probability that the parameter is below its
true value, i.e. the rank of the truth among the posterior samples, as a fraction.</p>
<p><b>Claim:</b> if the truths are drawn from the prior \(\pi\), the data from the
likelihood, and the posterior is exact, then \(u\) is uniform on \([0,1]\).
<i>Why:</i> given the data \(D\), the truth is distributed exactly as the posterior, so
\(F_D(x^{\rm true})\) is a CDF evaluated at a draw from that same distribution, which is
uniform (this is how inverse-CDF sampling works). Averaging a uniform distribution over
all possible \(D\) still gives a uniform distribution. The converse is the useful part: a
non-uniform PIT distribution means the posterior is wrong in a specific way.</p>
<ul>
<li>Too many PITs near 0 and 1 (a U-shaped histogram): the truth often lies in the tails,
so the posterior is <b>too narrow</b> (overconfident).</li>
<li>Too many PITs near 0.5 (a hump): the posterior is <b>too wide</b> (underconfident).</li>
<li>More PITs near 0 than near 1 (or vice versa): the posterior is <b>biased</b> high (or low).</li>
</ul>
<p><b>The P–P plot</b> shows the empirical CDF of the PITs,
\(\hat F(u) = \frac{1}{N}\#\{i: u_i \le u\}\), against \(u\). For a calibrated posterior it
follows the diagonal. At \(N = <<n>>\) galaxies deviations are tiny, so we also plot
\(\hat F(u) - u\) directly. If \(\hat F(u) - u\) is positive at small \(u\) and negative at
large \(u\), there are extra PITs at both ends, i.e. the posterior is too narrow.</p>
<p><b>Coverage.</b> The central credible interval at level \(L\) runs from the
\((1-L)/2\) to the \((1+L)/2\) posterior quantile. The truth is inside it exactly when
\((1-L)/2 \le u_i \le (1+L)/2\), so coverage is just the fraction of PITs in that range and
should equal \(L\). Its statistical error is binomial,
\(\sqrt{L(1-L)/N}\) (e.g. <<se68>> percentage points for \(L = 68\%\) here). Coverage at
several \(L\) reads off the P–P curve at a few points.</p>
<p><b>How big a deviation is significant? The KS distance and the DKW band.</b> The
Kolmogorov–Smirnov (KS) distance is the largest vertical gap,
\(D = \max_u|\hat F(u) - u|\). The Dvoretzky–Kiefer–Wolfowitz (DKW) inequality bounds how
large \(D\) can get by chance alone when the PITs really are uniform:
\(P(D > \varepsilon) \le 2e^{-2N\varepsilon^2}\). Setting the right side to 0.05 gives the
95% band \(\varepsilon_{95} = \sqrt{\ln(2/0.05)/(2N)}\), which for \(N = <<n>>\) is
<<band>>. (This matches the usual asymptotic KS critical value \(1.36/\sqrt{N}\).) A curve
that leaves the band is miscalibrated at 95% confidence. Note how small this is: with
\(10^5\) galaxies we can detect a shift of half a percentage point in any quantile, which
is far below what matters for most uses.</p>
<p><b>Why the posterior must be TF-weighted here.</b> The claim above requires that the
truths come from the same prior the posterior assumes. The Xu3 test set draws
\(v_{\rm circ}\) from the Tully–Fisher (TF) relation, so the matching posterior is the
TF-reweighted one (see the TF-weight page). Every PIT and interval below uses the TF
importance weights.</p>
<p><b>What global calibration cannot tell you.</b></p>
<ul>
<li><i>It is not a test of usefulness.</i> A “posterior” that simply returns the prior for
every galaxy is perfectly calibrated (the truths are prior draws), but carries no
information. Calibration must be read together with the posterior widths (Sec. 6 shows
these).</li>
<li><i>It is an average over the whole prior.</i> Coverage at fixed true parameters need
not be 68% even for an exact posterior. Example: the shear prior is bounded,
\(|g_k| \le 0.1\). For a galaxy whose true \(g_1\) is near the edge, the posterior is pulled
toward the centre (shrinkage), so the truth falls in the outer tail more often than 16% of
the time: coverage is below 68% at large \(|g|\) and above 68% near \(g = 0\), and the two
average to 68%. An earlier campaign report found exactly this pattern (coverage higher for
small \(|g|\) than for large \(|g|\)); it is not re-measured here. So a correct global
coverage does not mean every subset is covered correctly.</li>
</ul>""", n=f"{n:,}", band=f(band, 4), se68=f(100 * g1["cov68_pit_se"], 2))

    procedure = S(r"""<ul>
<li><b>Data.</b> The existing Xu3 posterior cache: <<n>> test galaxies, each with
<<nd>> posterior draws (half drawn from the galaxy as simulated, half from the galaxy
rotated by 90° and rotated back) and the TF log-importance weight of every draw. No new
inference was run.</li>
<li><b>PITs for \(g_1\), \(g_2\).</b> For each galaxy and component, with normalized TF
weights \(w_j\) of draws \(g^{(j)}\):
\(u_i = \sum_j w_j\,[g^{(j)} < g^{\rm true}] + \tfrac12\sum_j w_j\,[g^{(j)} = g^{\rm true}]\)
(ties split in half). Coverage at \(L\) = 50, 68, 90, 95, 99% is the fraction of galaxies with
\(u_i\) inside \([(1-L)/2, (1+L)/2]\); the KS distance is computed from the same PITs.</li>
<li><b>The other seven parameters.</b> The compact cache stores draws for \((g_1, g_2)\)
only. For the other parameters it stores just the TF-weighted 16th percentile, mean and 84th
percentile per galaxy. So for them we can only count how often the truth is below the 16th
or above the 84th percentile: two points on the P–P curve (\(u = 0.16\), \(0.84\)) and the
68% coverage. No 95% coverage or KS distance is possible without new inference.</li>
<li><b>\(\theta_{\rm int}\) is an angle</b> (period \(2\pi\)), so “below” has no meaning
on its own. The stored interval is centred on the circular mean, and all differences are
wrapped into \([-\pi, \pi)\) with \({\rm wrap}(a) = ((a+\pi) \bmod 2\pi) - \pi\). The truth
counts as below the interval if \({\rm wrap}(\theta^{\rm true} - \bar\theta) <
{\rm wrap}(\theta_{16} - \bar\theta)\), where \(\bar\theta\) is the circular mean, and
similarly above. This avoids declaring a truth of \(+3.1\) rad “far” from a posterior at
\(-3.1\) rad.</li>
<li><b>Two galaxy weightings.</b> (i) Equal weights: every galaxy counts once (the paper
table). (ii) The shape-noise-regularized weights used later for shear estimation,
\(W_i \propto 1/(\sigma_i^2 + \sigma_{\rm sh}^2)\) with
\(\sigma_i = \sqrt{({\rm Var}\,g_1 + {\rm Var}\,g_2)/2}\) and \(\sigma_{\rm sh} = <<sh>>\) the
mean of \(\sigma_i\) over galaxies. These favour narrow posteriors; the weighted galaxy
count (Kish effective sample size, see the TF-weight page) is <<ess>>, and the binomial
errors and DKW band use it in place of \(N\).</li>
<li><b>Cross-check.</b> With the regularized weights and the stored 16–84% interval, the
\(g_1\)/\(g_2\) 68% coverage should reproduce the earlier fiber-gauge report
(<<r1>>/<<r2>>).</li>
</ul>""", n=f"{n:,}", nd=f"{J('npe', 'T6', 'draws_per_galaxy'):,}", sh=f(reg("shape_noise"), 4),
              ess=f"{reg('ess'):,.0f}", r1=pct(val("g1", "reference"), 2), r2=pct(val("g2", "reference"), 2))

    def cov(row, key):
        return f"{100 * row[key]:.2f} ± {100 * row[key + '_se']:.2f}"

    rows = []
    for p in NAMES9:
        e, rg = _r3("equal", p), _r3("shape_noise_regularized", p)
        rows.append([
            PNAME[p] + f"<br><small>{PDESC[p]}</small>",
            cov(e, "cov68_summary"),
            pct(e["frac_below_q16"], 2), pct(e["frac_above_q84"], 2),
            cov(e, "cov95_pit") if "cov95_pit" in e else "n/a (no draws)",
            (f"{f(e['ks_distance'], 4)} (band {f(e['ks_dkw95_band'], 4)})" if "ks_distance" in e else "n/a"),
            cov(rg, "cov68_summary"),
        ])
    main_table = table(["parameter", "68% coverage (%)", "truth below 16th pct (target 16%)",
                        "truth above 84th pct (target 16%)", "95% coverage (%)", "KS distance",
                        "68% coverage, regularized weights (%)"], rows)
    lvl_rows = []
    for L in (50, 68, 90, 95, 99):
        lvl_rows.append([f"{L}%", cov(g1, f"cov{L}_pit"), cov(g2, f"cov{L}_pit"),
                         cov(g1r, f"cov{L}_pit"), cov(g2r, f"cov{L}_pit")])
    lvl_table = table(["nominal level", "g<sub>1</sub> equal (%)", "g<sub>2</sub> equal (%)",
                       "g<sub>1</sub> regularized (%)", "g<sub>2</sub> regularized (%)"], lvl_rows)

    results = (
        "<p><b>All nine parameters</b> (equal galaxy weights unless noted; ± is the binomial error). "
        "68% coverage for g<sub>1</sub>, g<sub>2</sub> is from the stored 16–84% interval; the PIT-based "
        f"values are essentially identical ({pct(g1['cov68_pit'], 3)} vs {pct(g1['cov68_summary'], 3)} for "
        "g<sub>1</sub>).</p>" + main_table
        + "<p><b>Shear coverage at several levels</b> (PIT-based):</p>" + lvl_table
        + S("<p><b>Cross-check with the earlier report:</b> regularized-weight 68% coverage "
            "<<a>>/<<b>> vs reference <<c>>/<<d>> → <<ok>>.</p>",
            a=pct(val("g1", "reproduced_regularized"), 2), b=pct(val("g2", "reproduced_regularized"), 2),
            c=pct(val("g1", "reference"), 2), d=pct(val("g2", "reference"), 2),
            ok="reproduced" if val("pass") else "NOT reproduced")
        + fig("A_npe/r3_pp_all9.png",
              "P–P plots for all nine parameters (equal weights). Dashed: the diagonal expected for a "
              "calibrated posterior. For g<sub>1</sub> and g<sub>2</sub> the full curve is shown; for the "
              "other seven only the two points at u = 0.16 and 0.84 are known (the dotted line just "
              "connects them).")
        + fig("A_npe/r3_pp_deviation_g1g2.png",
              "The same g<sub>1</sub>, g<sub>2</sub> P–P curves shown as the deviation ECDF(u) − u. Grey: the "
              "DKW 95% band; a calibrated posterior stays inside it 95% of the time. g<sub>1</sub> is "
              "positive at small u and negative at large u (extra PITs at both ends) and just touches the "
              "band edge.", width="70%")
        + fig("A_npe/r3_pit_hist_g1g2.png",
              "PIT histograms for g<sub>1</sub> and g<sub>2</sub> (50 bins). Dashed: the flat density of a "
              "calibrated posterior.", width="80%")
    )

    interpretation = S(r"""<p><b>The shear posteriors are well calibrated.</b> Over <<n>> galaxies, the 68% and 95%
intervals cover the truth <<c68a>>/<<c68b>> and <<c95a>>/<<c95b>> of the time for
\(g_1\)/\(g_2\), and the tails are balanced (truth below the 16th percentile
<<ba>> of the time, above the 84th <<aa>> for \(g_1\)), so there is no sign of a posterior
biased to one side. \(g_2\)'s P–P curve stays inside the DKW band (KS <<ks2>> vs band
<<band>>).</p>
<p><b>\(g_1\) is very slightly too narrow.</b> Its KS distance (<<ks1>>) is just outside the
95% band, its 50% coverage is <<c50>> (about <<sig50>> binomial errors low) and its 99%
coverage is <<c99>>. Both tails have a small excess, the signature of overconfidence. The
size is small: if the posterior were Gaussian, a ±1σ interval that covers
<<c68a>> instead of 68% means the true scatter is about <<zg1>>σ, i.e. the posterior widths
are about <<pg1>> too small. With the regularized weights (which favour narrow posteriors)
the \(g_1\) KS distance (<<ks1r>>) falls inside its band (<<bandr>>).</p>
<p><b>The nuisance parameters are fine except \(\theta_{\rm int}\).</b> Six of the seven
nuisances have 68% coverage within <<dev>> percentage points of 68%. \(\theta_{\rm int}\)
covers only <<cth>> (<<cthr>> with regularized weights), with <i>both</i> tails heavy
(<<thb>> below, <<tha>> above). This is under-coverage, not bias: the position-angle
intervals are too narrow, roughly <<pth>> too small if Gaussian, or the posterior has
heavier tails than its 16–84% interval suggests. The shear estimate does not use
\(\theta_{\rm int}\) directly, but it is correlated with the shear through the observed
shape, so it is worth mentioning in Sec. 5.2.</p>""",
        n=f"{n:,}", c68a=pct(g1["cov68_pit"], 2), c68b=pct(g2["cov68_pit"], 2),
        c95a=pct(g1["cov95_pit"], 2), c95b=pct(g2["cov95_pit"], 2),
        ba=pct(g1["frac_below_q16"], 2), aa=pct(g1["frac_above_q84"], 2),
        ks2=f(g2["ks_distance"], 4), band=f(band, 4), ks1=f(g1["ks_distance"], 4),
        c50=pct(g1["cov50_pit"], 2), sig50=f((0.5 - g1["cov50_pit"]) / g1["cov50_pit_se"], 1),
        c99=pct(g1["cov99_pit"], 2), zg1=f(z_g1, 3), pg1=pct(1 - z_g1, 1),
        ks1r=f(g1r["ks_distance"], 4), bandr=f(g1r["ks_dkw95_band"], 4),
        dev=f(100 * dev_others, 1), cth=pct(th["cov68_summary"], 2), cthr=pct(thr["cov68_summary"], 2),
        thb=pct(th["frac_below_q16"], 2), tha=pct(th["frac_above_q84"], 2), pth=pct(1 - z_th, 0))

    caveats = S(r"""<ul>
<li><b>\(\theta_{\rm int}\) under-coverage</b> (<<cth>> at 68%) is real at this sample size
(the binomial error is <<seth>> points). Its cause is not diagnosed. Candidates: a too
narrow circular flow for \(\theta_{\rm int}\); multi-modal or very broad angle posteriors
(e.g. nearly face-on disks, where the position angle is poorly defined) for which an
interval centred on the circular mean is a poor summary. Checking needs the full angle
draws.</li>
<li><b>\(g_1\) slight overconfidence</b> (KS <<ks1>> vs band <<band>>). \(g_2\) is fine, so the
two components are not exactly symmetric; it is a percent-level effect.</li>
<li><b>Only a partial test for the seven nuisances.</b> The cache stores \(g_1\), \(g_2\) draws
only, so the nuisances are tested at two quantiles and there is no 95% coverage or KS test.
A full nine-parameter P–P needs a resampling run that stores all nine columns.</li>
<li><b>Global, not conditional.</b> Coverage is averaged over the test population. It says
nothing about whether subsets (large \(|g|\), edge-on, faint, low TF-ESS galaxies) are
covered correctly; for bounded shear an exact posterior is expected to over-cover at small
\(|g|\) and under-cover at large \(|g|\). The low-TF-ESS subset is checked on the next page.</li>
<li><b>Calibration assumes the test prior matches the assumed prior.</b> The TF relation used
to weight the posteriors is the one used to simulate the test set, so this checks the NPE,
not our knowledge of the real TF relation.</li>
<li>Current model (trained on <<ntr>> galaxies); to be redone for the paper model.</li>
</ul>""", cth=pct(th["cov68_summary"], 2), seth=f(100 * th["cov68_summary_se"], 2),
              ks1=f(g1["ks_distance"], 4), band=f(band, 4), ntr=f"{C('data', 'size'):,}")

    provenance = S(r"""<ul>
<li>Streaming pass over the cache: <code><<s>>/a_npe_stream.py</code>, SLURM job <<job>>
(RM-shared). It reuses <code>shear_bias_report.load_case</code> and
<code>load_shear_posterior_diagnostics</code> (the TF-weighted PITs and variances) and writes
<code>A_npe/a_npe_stream.npz</code>.</li>
<li>Coverage, KS and figures: <code><<s>>/a_npe_report.py</code>, function
<code>r3_calibration</code> (uses <code>coverage_metrics</code>, <code>posterior_pp_curve</code>,
<code>compose_shape_noise_regularized_weights</code> from <code>shear_bias_report.py</code>).</li>
<li>Cache (read-only): the Xu3 test-set cache of model <code><<model>></code>, case
<code><<case>></code>.</li>
<li>JSON: <code>A_npe/report.json</code> → <code>R3</code> (<code>rows/equal</code>,
<code>rows/shape_noise_regularized</code>, <code>regularized_weight_diagnostics</code>) and
<code>validation_vs_fiber_gauge_03</code>.</li>
<li>Figures: <code>A_npe/r3_pp_all9.png</code>, <code>r3_pp_deviation_g1g2.png</code>,
<code>r3_pit_hist_g1g2.png</code> (paper copies <code>paper_figs/A_npe_pp_*</code>).</li>
<li>Detailed group report: <a href='../A_npe/report.html'>A_npe/report.html</a>, section “R3”.</li>
</ul>""", s=SCRIPTS, job=_job("A_npe/STATUS.txt", r"STREAM_JOB=(\d+)"), model=MODEL,
              case=J("npe", "meta", "case").split(":", 1)[1])

    answer = (f"Yes, overall: over {n:,} test galaxies the g1/g2 68% intervals cover the truth "
              f"{pct(g1['cov68_pit'], 1)}/{pct(g2['cov68_pit'], 1)} and the 95% intervals "
              f"{pct(g1['cov95_pit'], 1)}/{pct(g2['cov95_pit'], 1)} of the time; g1 is about "
              f"{pct(1 - z_g1, 0)} too narrow and θ_int under-covers ({pct(th['cov68_summary'], 1)} at 68%).")

    return page(slug="posterior-calibration", title="Posterior calibration (P–P and coverage)", code="R3",
                section=SECTION, answer=answer, question=question, why=why, theory=theory,
                procedure=procedure, results=results, interpretation=interpretation, caveats=caveats,
                provenance=provenance)


# ============================================================ 3. TF weights
def _page_tf_weights() -> dict:
    t = lambda *k: J("npe", "T6", *k)  # noqa: E731
    tf = _tf()
    nd = t("draws_per_galaxy")
    q = t("ess_fraction_quantiles")
    low = t("lowest_ess_galaxies", 0)
    mu_log = (low["rmag_true"] - tf.intercept) / tf.slope
    dev_sigma = (math.log10(low["vcirc_true"]) - mu_log) / tf.scatter_dex
    sub = t("subset_shear_coverage")

    question = r"""<p>The Tully–Fisher (TF) prior on the rotation velocity is applied after the network
is trained, by reweighting posterior samples. Does that reweighting work reliably? In
particular, are there galaxies where only a handful of the 10,000 samples receive almost
all the weight, so that their TF posterior is effectively built from a few points?</p>""".replace(
        "10,000", f"{nd:,}")

    why = r"""<p>The TF relation is what breaks the degeneracy between a galaxy's intrinsic shape
and the shear, so every shear result in the paper uses TF-weighted posteriors. The paper
(NPE-architecture section, subsection “Incorporating the TF Prior”) describes this reweighting; this test is the
evidence that it is numerically safe on the test set. If many galaxies had degenerate
weights, their posterior means and widths would be noisy and their error bars
untrustworthy.</p>"""

    theory = S(r"""<p><b>Swapping priors by importance sampling.</b> The network is trained with a uniform
prior on \(v_{\rm circ}\), \(\pi_{\rm train}(v_{\rm circ})\), so its posterior
\(q(\mathbf{x}\mid D)\) is the posterior under that prior. The posterior we want uses the
TF prior \(\pi_{\rm TF}(v_{\rm circ}\mid m_r)\) (\(m_r\): the galaxy's \(r\)-band magnitude,
treated as known). By Bayes' theorem, the likelihood is the same in both, so</p>
\[ p_{\rm TF}(\mathbf{x}\mid D) \;\propto\; q(\mathbf{x}\mid D)\,
   \frac{\pi_{\rm TF}(v_{\rm circ}\mid m_r)}{\pi_{\rm train}(v_{\rm circ})}. \]
<p>Given draws \(\mathbf{x}_j \sim q\), \(j = 1\ldots N_{\rm draw}\), we give each draw the
weight</p>
\[ w_j = \frac{r_j}{\sum_k r_k}, \qquad r_j = \frac{\pi_{\rm TF}(v_{{\rm circ},j}\mid m_r)}{\pi_{\rm train}(v_{{\rm circ},j})}, \]
<p>and compute every posterior summary as a weighted one, e.g. the mean
\(\sum_j w_j g_{1,j}\). Dividing by \(\sum_k r_k\) (“self-normalized” weights) means we
never need the normalizing constants of either prior or of the posterior. Because
\(\pi_{\rm train}\) is uniform, \(r_j\) is just proportional to the TF density. Here the TF
prior is a Gaussian in \(\log_{10}v_{\rm circ}\) with mean \((m_r - <<b>>)/(<<a>>)\) and width
<<sc>> dex, truncated to <<vmin>>–<<vmax>> km/s; it is written as a density in
\(v_{\rm circ}\) itself (including the \(1/(v\ln 10)\) Jacobian), so both priors are densities
in the same variable.</p>
<p><b>Why weights can degenerate.</b> The weighted estimate is only as good as the number
of draws that carry weight. If the NPE posterior on \(v_{\rm circ}\) is much broader than the
TF prior, or sits in the TF tail, only the few draws that land where \(\pi_{\rm TF}\) is
large get weight. The standard summary is the Kish effective sample size</p>
\[ {\rm ESS} = \frac{\left(\sum_j r_j\right)^2}{\sum_j r_j^2} = \frac{1}{\sum_j w_j^2}. \]
<p>It equals \(N_{\rm draw}\) when all weights are equal and 1 when a single draw has all the
weight. The Monte Carlo error of a weighted mean is roughly that of an unweighted mean of
ESS independent draws, \(\approx\sigma/\sqrt{\rm ESS}\), so ESS is “how many samples the TF
posterior is really made of”. A low ESS means a few samples dominate: the estimate jumps
around depending on which draws happened to land near the TF peak, and the tails of the
TF posterior are not sampled at all.</p>
<p><b>What sets the ESS fraction.</b> A simple Gaussian example helps: if the NPE posterior on
\(\log v_{\rm circ}\) has width \(s\), the TF prior has width \(t\), and they are centred at
the same place, then \({\rm ESS}/N_{\rm draw} = t\sqrt{t^2+2s^2}/(t^2+s^2)\). This is 1 when
the data pin down \(v_{\rm circ}\) (\(s \ll t\)) and falls as \(\sqrt2\,t/s\) when the data say
little (\(s \gg t\)). An offset between the two centres lowers it further. So we expect low
ESS for galaxies whose kinematics constrain \(v_{\rm circ}\) poorly (faint, low Hα SNR,
face-on, where the line-of-sight velocity \(\propto v_{\rm circ}\sin i\) is small) and for
galaxies whose true \(v_{\rm circ}\) is far out in the TF tail.</p>""",
        b=f(tf.intercept, 1), a=f(tf.slope, 2), sc=f(tf.scatter_dex, 2),
        vmin=f(tf.vcirc_min, 0), vmax=f(tf.vcirc_max, 0))

    procedure = S(r"""<ul>
<li>Data: the Xu3 posterior cache, <<n>> galaxies × <<nd>> draws, with the TF
log-weight \(\log r_j\) stored for every draw (computed at inference time by
<code>tf_prior.py</code> with each galaxy's true \(m_r\)).</li>
<li>One streaming pass recomputed, for every galaxy, the normalized weights (subtracting the
maximum log-weight before exponentiating, for numerical safety), the Kish ESS and the largest
single weight \(\max_j w_j\). The recomputed ESS matches the ESS stored in the cache to a
relative difference of at most <<rd>>.</li>
<li>Summaries: the distribution of the ESS fraction \({\rm ESS}/N_{\rm draw}\); counts of
galaxies below ESS = 50, 100, 500, 1000; the distribution of the maximum weight; Spearman
rank correlations of the ESS fraction with true \(\cos i\), \(m_r\), central Hα SNR
\(\rho_S\), image SNR \(\rho_I\) and true \(v_{\rm circ}\) (rank correlation measures whether
one tends to rise with the other, without assuming a straight line).</li>
<li>Does low ESS break the shear posteriors? For subsets selected by ESS, the \(g_1\), \(g_2\)
coverage at 68% and 95% was computed from the PITs (equal weights; see the calibration
page).</li>
</ul>""", n=f"{t('n_galaxies'):,}", nd=f"{nd:,}", rd=f(t("stored_vs_recomputed_max_rel_diff"), 1))

    summary = table(["quantity", "value"], [
        ["draws per galaxy", f"{nd:,}"],
        ["ESS fraction: mean / median", f(t("ess_fraction_mean"), 3) + " / " + f(q["median"], 3)],
        ["ESS fraction: 16% / 84% / 95% quantiles",
         " / ".join(f(q[k], 3) for k in ("q16%", "q84%", "q95%"))],
        ["ESS fraction: 5% / 1% / 0.1% quantiles",
         " / ".join(f(q[k], 3) for k in ("q5%", "q1%", "q0.1%"))],
        ["galaxies with ESS &lt; 50 / 100 / 500 / 1000",
         " / ".join(str(t(k)) for k in ("count_ess_lt_50", "count_ess_lt_100", "count_ess_lt_500",
                                        "count_ess_lt_1000"))],
        ["smallest ESS (galaxy)", f"{f(t('ess_min'), 1)} (galaxy {t('ess_min_index')})"],
        ["largest single weight: median / 99th pct / max",
         " / ".join(f(t("max_weight_quantiles", k), 4) for k in ("median", "q99%", "max"))],
        ["galaxies with a single weight &gt; 0.1", str(t("count_max_weight_gt_0.1"))],
    ])
    sp_names = {"cosi": r"true \(\cos i\)", "rmag_true": r"\(m_r\) (fainter = larger)",
                "central_halpha_snr": r"central Hα SNR \(\rho_S\)", "image_snr": r"image SNR \(\rho_I\)",
                "vcirc_true": r"true \(v_{\rm circ}\)"}
    spear = table(["quantity", "Spearman ρ with ESS fraction"],
                  [[sp_names[k], f(v, 3, sign=True)] for k, v in t("spearman").items()])
    sub_rows = []
    labels = {"all": "all galaxies", "ess<50": "ESS &lt; 50", "ess<500": "ESS &lt; 500",
              "ess_frac<0.1": "ESS fraction &lt; 0.1", "ess_frac>=0.5": "ESS fraction ≥ 0.5"}
    for key, entry in sub.items():
        nn = entry["n"]
        if "g1_cov68_pit" not in entry:
            sub_rows.append([labels[key], str(nn), "too few", "", "", "", ""])
            continue
        e68 = 100 * math.sqrt(0.68 * 0.32 / nn)
        e95 = 100 * math.sqrt(0.95 * 0.05 / nn)
        sub_rows.append([labels[key], f"{nn:,}",
                         f"{100 * entry['g1_cov68_pit']:.1f}", f"{100 * entry['g2_cov68_pit']:.1f}",
                         f"± {e68:.1f}" if nn < 1000 else f"± {e68:.2f}",
                         f"{100 * entry['g1_cov95_pit']:.1f} / {100 * entry['g2_cov95_pit']:.1f} (± {e95:.1f})",
                         f(entry["median_cosi"], 3)])
    sub_table = table(["subset", "N", "g<sub>1</sub> 68% cov. (%)", "g<sub>2</sub> 68% cov. (%)",
                       "expected scatter of 68% cov.", "g<sub>1</sub> / g<sub>2</sub> 95% cov. (%)",
                       "median true cos i"], sub_rows)
    low_table = table(["galaxy", "ESS", "max weight", "true cos i", "m<sub>r</sub>", "ρ<sub>S</sub>",
                       "ρ<sub>I</sub>", "true v<sub>circ</sub> (km/s)", "PIT g<sub>1</sub> / g<sub>2</sub>"],
                      [[str(r["index"]), f(r["ess"], 1), f(r["max_weight"], 3), f(r["cosi"], 3),
                        f(r["rmag_true"], 2), f(r["central_halpha_snr"], 1), f(r["image_snr"], 0),
                        f(r["vcirc_true"], 0), f(r["pit_g1"], 3) + " / " + f(r["pit_g2"], 3)]
                       for r in t("lowest_ess_galaxies")])

    results = (
        "<p><b>Summary of the TF weights over the test set.</b></p>" + summary
        + fig("A_npe/t6_ess_distribution.png",
              "Left: distribution of the ESS fraction (ESS / number of draws). Right: log<sub>10</sub> ESS "
              "(log counts); the red line marks ESS = 50.")
        + "<p><b>What the ESS fraction depends on.</b></p>" + spear
        + fig("A_npe/t6_ess_dependence.png",
              "ESS fraction vs true cos i, true m<sub>r</sub>, central Hα SNR, image SNR and true "
              "v<sub>circ</sub>. Orange: binned median (solid) and 16/84% (dashed); red dot: the one galaxy "
              "with ESS &lt; 50.")
        + "<p><b>Shear coverage of low-ESS subsets</b> (equal weights). The “expected scatter” column is the "
          "binomial error of a 68% coverage for that N, so a correct posterior would land within about "
          "twice this of 68%.</p>" + sub_table
        + "<p><b>Galaxies with ESS &lt; 50.</b></p>" + low_table
    )

    s_low = sub["ess<500"]
    e68_low = math.sqrt(0.68 * 0.32 / s_low["n"])
    interpretation = S(r"""<p><b>The TF reweighting is healthy for essentially the whole test set.</b> The median
galaxy keeps <<med>> of its draws' worth of information (ESS ≈ <<medn>> of <<nd>>); 99% of
galaxies keep more than <<q1>> (ESS above ≈ <<q1n>>), so for them the Monte Carlo error of
the posterior mean is below \(\sigma/\sqrt{<<q1n>>}\), i.e. a few percent of the posterior
width. Only <<n50>> galaxy has ESS below 50 and <<n500>> have ESS below 500.</p>
<p><b>The trends are the expected ones.</b> ESS is lower for faint galaxies (larger \(m_r\)),
low Hα SNR and nearly face-on disks (large \(\cos i\)), and higher for fast rotators. These
are the galaxies whose spectra constrain \(v_{\rm circ}\) least, so their \(v_{\rm circ}\)
posterior is broad compared with the <<sc>>-dex TF prior, exactly as the Gaussian example in
the theory section predicts.</p>
<p><b>The worst galaxy is a TF outlier.</b> Galaxy <<gid>> (ESS <<ess>>, one draw carries
<<mw>> of the weight) has true \(v_{\rm circ} = <<vc>>\) km/s, while the TF prior for its
magnitude \(m_r = <<mr>>\) is centred at \(10^{<<mu>>} \approx <<vmu>>\) km/s: its true
velocity is <<dsig>> TF widths below the mean. The data and the TF prior disagree, so only the
few NPE draws in the overlap survive. With <<n>> galaxies drawn from the TF relation, one or
two such outliers are expected. Its shear PITs (<<p1>>, <<p2>>) are unremarkable.</p>
<p><b>Low-ESS galaxies are somewhat under-covered.</b> The <<nl>> galaxies with ESS &lt; 500 have
68% coverage <<c1>> (\(g_1\)) and <<c2>> (\(g_2\)); the binomial scatter for N = <<nl>> is
±<<e>>, so \(g_2\) is about <<z2>> errors low and \(g_1\) is consistent with 68%. This subset is
only <<frac>> of the sample, so it has no visible effect on the global calibration, but it
suggests the posteriors of galaxies with badly sampled TF weights are too narrow, as expected
when the TF posterior's tails are represented by very few draws. Galaxies with ESS fraction ≥
0.5 are covered correctly. (This extreme tail is not dominated by face-on disks: its
median true \(\cos i\) is <<mc>>, vs <<mca>> overall, even though the ESS fraction falls
towards face-on on average.)</p>""",
        med=pct(q["median"], 0), medn=f"{q['median'] * nd:,.0f}", nd=f"{nd:,}", q1=pct(q["q1%"], 1),
        q1n=f"{q['q1%'] * nd:.0f}", n50=t("count_ess_lt_50"), n500=t("count_ess_lt_500"),
        sc=f(tf.scatter_dex, 1), gid=low["index"], ess=f(low["ess"], 1), mw=pct(low["max_weight"], 0),
        vc=f(low["vcirc_true"], 0), mr=f(low["rmag_true"], 2), mu=f(mu_log, 2), vmu=f(10 ** mu_log, 0),
        dsig=f(abs(dev_sigma), 1), n=f"{t('n_galaxies'):,}", p1=f(low["pit_g1"], 2), p2=f(low["pit_g2"], 2),
        nl=s_low["n"], c1=pct(s_low["g1_cov68_pit"], 1), c2=pct(s_low["g2_cov68_pit"], 1),
        e=pct(e68_low, 1), z2=f((0.68 - s_low["g2_cov68_pit"]) / e68_low, 1),
        frac=pct(s_low["n"] / t("n_galaxies"), 2),
        mc=f(s_low["median_cosi"], 2), mca=f(sub["all"]["median_cosi"], 2))

    caveats = S(r"""<ul>
<li><b>Low-ESS subset coverage.</b> The ESS &lt; 500 subset (<<nl>> galaxies) is under-covered in
\(g_2\) at about the <<z2>>σ level. The subset is small, so this is suggestive, not conclusive. If
the paper model or real data have more low-ESS galaxies, one could draw more samples for those
galaxies (the NPE is cheap), or exclude galaxies below an ESS threshold.</li>
<li><b>ESS measures sampling noise, not correctness.</b> A high ESS means the weighted estimate is
stable, not that the NPE's posterior is right. Importance weights also cannot recover posterior
mass in regions where the NPE put no draws at all; that failure would show up as bad coverage,
which we do not see globally.</li>
<li><b>The TF relation is assumed known exactly</b> and \(m_r\) is treated as noise-free (the
<<sc>>-dex width is meant to absorb some \(m_r\) uncertainty). The test set is simulated with
the same TF relation, so a mismatch between the assumed and the true TF relation is not
tested here.</li>
<li>The TF outlier explanation for galaxy <<gid>> is our reading of its true \(v_{\rm circ}\) and
\(m_r\); no extra inference was run on it.</li>
</ul>""", nl=s_low["n"], z2=f((0.68 - s_low["g2_cov68_pit"]) / e68_low, 1), sc=f(tf.scatter_dex, 1),
              gid=low["index"])

    provenance = S(r"""<ul>
<li>Weight pass: <code><<s>>/a_npe_stream.py</code> (<code>recompute_weight_health</code>), SLURM job
<<job>>; summaries and figures: <code><<s>>/a_npe_report.py</code> (<code>t6_ess</code>).</li>
<li>TF prior and weights: <code>kl-nn/arch/tf_prior.py</code> (<code>TFPrior</code>,
<code>truncated_tf_log_prob</code>, <code>posterior_importance_weights</code>); the TF constants quoted
above are read from its defaults. Paper text: <code>sections/nn-arch.tex</code>, subsection
“Incorporating the TF Prior”.</li>
<li>JSON: <code>A_npe/report.json</code> → <code>T6</code>. Figures: <code>A_npe/t6_ess_distribution.png</code>,
<code>A_npe/t6_ess_dependence.png</code>.</li>
<li>Detailed group report: <a href='../A_npe/report.html'>A_npe/report.html</a>, section “T6”.</li>
</ul>""", s=SCRIPTS, job=_job("A_npe/STATUS.txt", r"STREAM_JOB=(\d+)"))

    answer = (f"Yes. The median galaxy keeps {pct(q['median'], 0)} of its {nd:,} draws as effective samples, "
              f"only {t('count_ess_lt_50')} galaxy has ESS below 50 and {t('count_ess_lt_500')} below 500; "
              f"those few are somewhat under-covered "
              f"(g2 68% coverage {pct(s_low['g2_cov68_pit'], 0)}).")

    return page(slug="tf-weight-health", title="Tully–Fisher importance weights", code="T6", section=SECTION,
                answer=answer, question=question, why=why, theory=theory, procedure=procedure,
                results=results, interpretation=interpretation, caveats=caveats, provenance=provenance)


# ============================================================ 4. examples
def _page_examples() -> dict:
    gals = [_gal(i) for i in range(len(J("npe", "R2a", "galaxies")))]
    g0 = gals[0]
    gid = g0["index"]
    r2b = lambda *k: J("stencil", "R2b", *k)  # noqa: E731
    arms = r2b(str(gid), "rows")
    arms503 = r2b("503", "rows")
    rho_rows = r2b(f"{gid}_rhoS", "rows")
    rho_levels = {k: float(k.replace("hasnr", "")) for k in rho_rows if k.startswith("hasnr")}
    lo_key, hi_key = min(rho_levels, key=rho_levels.get), max(rho_levels, key=rho_levels.get)
    rho_cat = r2b(f"{gid}_rhoS", "central_halpha_snr")
    n_seeds = arms["snr50"]["n_seeds"]
    stream = _stream_constants()
    lo_pi, hi_pi = C("par_ranges", "g1")
    sigma_pi = (hi_pi - lo_pi) / math.sqrt(12)
    th_post = g0["tf_posterior"]["theta_int"]
    th_half = 0.5 * (th_post["q84"] - th_post["q16"])
    th_off = (th_post["mean"] - g0["truth"]["theta_int"]) / th_half
    g1_post = g0["tf_posterior"]["g1"]
    g1_off = (g1_post["mean"] - g0["truth"]["g1"]) / (0.5 * (g1_post["q84"] - g1_post["q16"]))
    s2t = math.sin(2 * g0["truth"]["theta_int"])
    c2t = math.cos(2 * g0["truth"]["theta_int"])
    c9 = lambda g, *k: J("stencil", "corner9d", str(g), *k)  # noqa: E731
    th9 = c9(gid, "posterior_mean")[2]
    n_out, n_pair, max_off = 0, 0, 0.0
    for g in gals:
        for p in NAMES9:
            tp, tv = g["tf_posterior"][p], g["truth"][p]
            lo, hi, mid = tp["q16"], tp["q84"], tp["mean"]
            if p == "theta_int":
                def wrap(a):
                    return (a + math.pi) % (2 * math.pi) - math.pi
                lo, hi, tv = wrap(lo - mid), wrap(hi - mid), wrap(tv - mid)
            half = 0.5 * (hi - lo)
            n_pair += 1
            if tv < lo or tv > hi:
                n_out += 1
                max_off = max(max_off, (lo - tv) / half if tv < lo else (tv - hi) / half)

    question = S(r"""<p>Do individual NPE posteriors look sensible, with the truth inside the bulk of the
posterior and the expected correlations between parameters? And which part of the data does
the shear information come from: if we change the image SNR, or the Hα (spectrum) SNR, of one
galaxy while keeping everything else fixed, how does its shear posterior change?</p>""")

    why = r"""<p>Sec. 5.1 shows one or two example posteriors so the reader can see what the NPE
returns. The SNR experiment supports the physical picture of kinematic lensing used
throughout the paper: the image measures the <i>observed</i> shape and orientation, the
spectra measure the rotation (and, via the TF relation, the <i>intrinsic</i> shape), and the
shear is the difference. If the spectra are the bottleneck, improving the image does little,
which matters for survey design and for interpreting which galaxies have narrow posteriors.
The original plan was a figure at two image SNRs; the result below changes that plan.</p>"""

    theory = S(r"""<p><b>Parameters.</b> <<plist>></p>
<p><b>What a corner plot shows.</b> A corner plot shows a many-dimensional posterior through
its one- and two-dimensional projections. The diagonal panels are the 1D marginal posterior of
each parameter (a histogram of the weighted draws); the off-diagonal panels are the 2D marginals
of each pair, drawn as contours. A tilted or curved (“banana”-shaped) 2D contour means the two
parameters are correlated in the posterior: the data constrain a combination of them better than
each one alone. The truth is marked by lines. The contours here enclose 39.3% and 86.5% of the
posterior mass. These odd-looking numbers are the probability inside the 1σ and 2σ ellipses of a
2D Gaussian, \(1 - e^{-1/2}\) and \(1 - e^{-2}\), so for a Gaussian the contours are the familiar
1σ and 2σ ellipses. (The single-number “68%” applies only to 1D.)</p>
<p><b>Which data carry the shear information.</b> A sheared disk looks like a different
ellipse. From the image alone, a round disk seen at an angle and a sheared disk can look
identical, because the observed axis ratio mixes inclination and shear. The spectra break this:
the velocity field gives the inclination through the rotation amplitude (\(\propto v_{\rm circ}\sin i\),
with \(v_{\rm circ}\) tied to the magnitude by the TF relation), and the kinematic axis gives the
intrinsic orientation. Roughly:</p>
<ul>
<li>the shear component that changes the apparent axis ratio (along the galaxy's own axes) is
degenerate with inclination and is constrained only as well as the kinematics plus TF prior
pin down the inclination;</li>
<li>the component at 45° to the galaxy axes shows up as a misalignment between the photometric
and kinematic axes, which the velocity field measures directly.</li>
</ul>
<p>For a typical galaxy the image SNR is already high, so the observed ellipse is known well;
the uncertainty is dominated by the intrinsic shape, i.e. by the spectra. So we expect the
image SNR to matter little and the Hα SNR to matter a lot. We also expect one shear component to
stay close to the prior width \(\sigma_\pi = <<wid>>/\sqrt{12} = <<spi>>\), the standard deviation of
the uniform shear prior on \([<<lo>>, <<hi>>]\), when the TF prior is too broad to fix the
inclination.</p>
<p><b>Width convention.</b> This page uses the per-galaxy shape noise in the Xu et al. (2024)
form, \(\sigma_{\epsilon,i} = \sqrt{\sigma_{g_1}^2 + \sigma_{g_2}^2}\), with \(\sigma_{g_k}\) the
TF-weighted posterior standard deviation. (The galaxy selection below instead uses
\(\sigma_i = \sqrt{(\sigma_{g_1}^2+\sigma_{g_2}^2)/2}\), smaller by \(\sqrt2\).)</p>""",
        plist="; ".join(f"{PNAME[p]}: {PDESC[p]}" for p in NAMES9)
              + r". \(\rho_I\): image signal-to-noise ratio (SNR); \(\rho_S\): central Hα SNR of the "
                r"spectra; \(m_r\): \(r\)-band magnitude.",
        spi=f(sigma_pi, 4), lo=f(lo_pi, 1), hi=f(hi_pi, 1), wid=f(hi_pi - lo_pi, 1))

    procedure = S(r"""<p><b>(a) Example posteriors from the cache.</b> Three Xu3 test galaxies were chosen to be
<i>typical</i>, not best cases. All are moderately inclined (true \(\cos i\) inside
[<<c0>>, <<c1>>] or a narrower window), have image and Hα SNR closest to \(\rho_I = <<ti>>\),
\(\rho_S = <<ts>>\) (the ranking score is \(|\ln(\rho_I/<<ti>>)| + |\ln(\rho_S/<<ts>>)|\)), and have a TF
ESS fraction above the median. Galaxy <<g0>> (“typical width”) was additionally required to have
a small true shear and a posterior width \(\sigma_i\) within 10% of the test-set median
(<<med>>); galaxies <<g1>> and <<g2>> came from the first selection, the second differing in
inclination from the first. The width condition was added because the first pick (<<g1>>) has a
\(g_2\) posterior piled against the \(+0.1\) prior edge, which is not typical. From the cache we
have the \((g_1, g_2)\) draws and TF weights, and the 16th/mean/84th percentiles of the other
parameters.</p>
<p><b>(b) Full nine-parameter corner plots.</b> Because the cache stores draws of \(g_1, g_2\)
only, the same three galaxies were re-sampled with the NPE (<<nd>> draws, half from the
90°-rotated galaxy, TF weights as in production), storing all nine parameters. This is a
<i>new noise realization</i> (noise seed 0), so these posteriors are not the cached ones.</p>
<p><b>(c) SNR experiment.</b> For galaxy <<g0>>, the image SNR stored with the galaxy was
overwritten with \(\rho_I = <<r50>>\), its catalog value <<rcat>>, and <<r200>>, keeping the Hα
SNR at its catalog value \(\rho_S = <<scat>>\). Both the added pixel noise and the SNR value the
network receives as metadata follow the override. In a second experiment the Hα SNR was set to
\(\rho_S = <<slo>>\) and <<shi>> with the image SNR at its catalog value. Each setting was run
with <<ns>> noise seeds; the same seeds are used for every SNR level, so within a seed the noise
has the same pattern and only its amplitude changes (this isolates the effect of SNR from the
luck of a particular noise draw). Per seed we compute the TF-weighted means and standard
deviations of \(g_1\), \(g_2\) and \(\sigma_{\epsilon,i}\); tables give the average over seeds
and the scatter between seeds. A second galaxy (503) was run at the three image SNRs as a
cross-check (the report does not record why it was chosen).</p>""",
        c0=f(stream["COSI_RANGE"][0], 2), c1=f(stream["COSI_RANGE"][1], 2),
        ti=f(stream["TARGET_IMAGE_SNR"], 0), ts=f(stream["TARGET_HALPHA_SNR"], 0),
        g0=gid, g1=gals[1]["index"], g2=gals[2]["index"], med=f(J("npe", "R2a", "median_sigma_i"), 4),
        nd=f"{J('npe', 'T6', 'draws_per_galaxy'):,}",
        r50=f(arms["snr50"]["image_snr"], 0), rcat=f(arms["snrcatalog"]["image_snr"], 0),
        r200=f(arms["snr200"]["image_snr"], 0), scat=f(r2b(str(gid), "central_halpha_snr"), 1),
        slo=f(rho_levels[lo_key], 0), shi=f(rho_levels[hi_key], 0), ns=n_seeds)

    # --- tables
    sel_table = table(
        ["galaxy", "selection", "ρ<sub>I</sub>", "ρ<sub>S</sub>", "m<sub>r</sub>", "TF ESS fraction",
         "σ<sub>i</sub> (percentile)"],
        [[str(g["index"]), g["selection"].replace("_", " "), f(g["image_snr"], 0), f(g["central_halpha_snr"], 1),
          f(g["rmag_true"], 2), f(g["tf_ess_fraction"], 3),
          f"{f(g['sigma_i'], 4)} ({g['sigma_i_percentile']:.0f}%)"] for g in gals])

    post_rows = []
    for p in NAMES9:
        tp = g0["tf_posterior"][p]
        post_rows.append([PNAME[p], f(g0["truth"][p], 4), f(tp["mean"], 4),
                          f"[{f(tp['q16'], 4)}, {f(tp['q84'], 4)}]"])
    post_table = table(["parameter", "truth", "TF-weighted mean", "16–84% interval"], post_rows)

    def c9_rows(g):
        truth, mean = c9(g, "truth"), c9(g, "posterior_mean")
        out = []
        for k, p in enumerate(NAMES9):
            tv, mv = truth[k], mean[k]
            if p == "halpha_flux_true":
                tv, mv = math.log10(tv), math.log10(mv)
            out.append(f(tv, 3) + " / " + f(mv, 3))
        return out
    c9_gals = list(J("stencil", "corner9d"))
    c9_table = table(["parameter"] + [f"galaxy {g}: truth / mean" for g in c9_gals],
                     [[PNAME[p]] + [c9_rows(g)[k] for g in c9_gals] for k, p in enumerate(NAMES9)]
                     + [["TF ESS fraction"] + [f(c9(g, "tf_ess_fraction"), 3) for g in c9_gals]])

    def arm_table(rows, keys, labels, snr_col):
        out = []
        for k, lab in zip(keys, labels):
            r = rows[k]
            out.append([lab, snr_col(k, r), pm(r["sd_g1"], r["sd_g1_seed_std"], 4),
                        pm(r["sd_g2"], r["sd_g2_seed_std"], 4), pm(r["sigma_eps"], r["sigma_eps_seed_std"], 4),
                        f"{f(r['mean_g1'], 4, True)} / {f(r['mean_g2'], 4, True)}",
                        f"{f(r['mean_g1_seed_std'], 4)} / {f(r['mean_g2_seed_std'], 4)}",
                        f(r["ess_fraction"], 2)])
        return table(["setting", "ρ", "σ<sub>g1</sub>", "σ<sub>g2</sub>", "σ<sub>ε,i</sub>",
                      "mean ĝ<sub>1</sub> / ĝ<sub>2</sub>", "seed scatter of ĝ<sub>1</sub> / ĝ<sub>2</sub>",
                      "TF ESS fraction"], out)

    img_keys, img_labels = ("snr50", "snrcatalog", "snr200"), ("ρ_I low", "ρ_I catalog", "ρ_I high")
    img_col = lambda k, r: "ρ<sub>I</sub> = " + f(r["image_snr"], 0)  # noqa: E731
    rho_keys = (lo_key, "snrcatalog", hi_key)
    rho_col = lambda k, r: "ρ<sub>S</sub> = " + (f(rho_levels[k], 0) if k in rho_levels else f(rho_cat, 1))  # noqa: E731
    t_img = arm_table(arms, img_keys, img_labels, img_col)
    t_503 = arm_table(arms503, img_keys, img_labels, img_col)
    t_rho = arm_table(rho_rows, rho_keys, ("ρ_S low", "ρ_S catalog", "ρ_S high"), rho_col)

    g1s = f"galaxy {gals[1]['index']}"
    g2s = f"galaxy {gals[2]['index']}"
    results = (
        "<h3>(a) Cached posteriors of three typical galaxies</h3>" + sel_table
        + fig(f"A_npe/r2a_corner_gal{gid}.png",
              f"Galaxy {gid} from the cache. Left: g<sub>1</sub>–g<sub>2</sub> corner (TF-weighted; 39.3% and "
              "86.5% contours; orange: truth). Right: for every parameter, the TF-weighted mean (dot) and "
              "16–84% interval (bar) minus the truth, in units of the interval half-width σ<sub>68</sub>; a "
              "well-behaved posterior has most bars crossing 0.")
        + f"<p>Galaxy {gid}, cached posterior summaries:</p>" + post_table
        + fig(f"A_npe/r2a_corner_gal{gals[1]['index']}.png",
              f"Same for {g1s} (the first SNR-window pick; its g<sub>2</sub> posterior is piled against the "
              "+0.1 prior edge).", width="85%")
        + fig(f"A_npe/r2a_corner_gal{gals[2]['index']}.png", f"Same for {g2s}.", width="85%")
        + "<h3>(b) Full nine-parameter posteriors (new noise realization)</h3>" + c9_table
        + "".join(fig(f"B_inference/corner9d_gal{g}.png",
                      f"Galaxy {g}: nine-parameter TF-weighted NPE posterior (catalog SNRs "
                      f"ρ<sub>I</sub> = {f(c9(g, 'image_snr'), 0)}, ρ<sub>S</sub> = "
                      f"{f(c9(g, 'central_halpha_snr'), 1)}), 39.3% and 86.5% contours, truth in orange. "
                      "Axes for g<sub>1</sub>, g<sub>2</sub> span the full prior ±0.1 and cos i spans [0, 1].")
                  for g in c9_gals)
        + f"<h3>(c) Changing the image SNR (galaxy {gid})</h3>"
        + f"<p>Values are averages over {n_seeds} noise seeds ± the scatter between seeds.</p>" + t_img
        + fig(f"B_inference/r2b_g1g2_contours_gal{gid}.png",
              f"Galaxy {gid}, noise seed 0: g<sub>1</sub>–g<sub>2</sub> posterior at the low (blue) and high "
              "(orange) image SNR. Thick/thin contours enclose 68%/95%; dots are posterior means, the star is "
              "the truth. The legend's σ values are for this seed only.", width="60%")
        + "<p>Cross-check, galaxy 503:</p>" + t_503
        + fig("B_inference/r2b_g1g2_contours_gal503.png", "Galaxy 503, noise seed 0, low vs high image SNR.",
              width="60%")
        + f"<h3>(d) Changing the Hα SNR (galaxy {gid}, image SNR at its catalog value)</h3>" + t_rho
        + fig(f"B_inference/r2b_g1g2_contours_rhoS_gal{gid}.png",
              f"Galaxy {gid}, noise seed 0: ρ<sub>S</sub> = {f(rho_levels[lo_key], 0)} (blue) vs "
              f"{f(rho_levels[hi_key], 0)} (orange). At the low Hα SNR this particular seed piles up against "
              "the g<sub>1</sub> = +0.1 prior edge; the other seeds do not (see the large seed scatter of "
              "ĝ<sub>1</sub> in the table).", width="60%")
    )

    a50, a200 = arms["snr50"], arms["snr200"]
    rl, rh = rho_rows[lo_key], rho_rows[hi_key]
    interpretation = S(r"""<p><b>The example posteriors behave sensibly.</b> In the cached posteriors of the three
galaxies, the truth lies outside the 16–84% interval for <<nout>> of the <<npair>>
galaxy–parameter pairs (about <<nexp>> are expected for correct 68% intervals), and never by
more than <<maxoff>> interval half-widths. The 2D contours show the correlations the physics
predicts. In the nine-parameter corner of galaxy <<gid>>, the poorly constrained shear component is strongly
correlated with \(\cos i\), \(v_{\rm circ}\) and the half-light radius: changing the inclination
changes the apparent axis ratio, which the model can trade against shear, and \(v_{\rm circ}\)
must change with \(\cos i\) to keep the observed rotation \(\propto v_{\rm circ}\sin i\) fixed. This is
the shape–shear degeneracy that the TF prior is there to break.</p>
<p><b>For galaxy <<gid>> one shear component is nearly prior-limited.</b> Its true
\(\theta_{\rm int} = <<th>>\) rad gives \(\sin 2\theta_{\rm int} = <<s2t>>\) and
\(\cos 2\theta_{\rm int} = <<c2t>>\), so \(g_2\) lies almost along the galaxy's own axes (the
axis-ratio component) and \(g_1\) is mostly the 45° component. As expected from the theory,
\(\sigma_{g_2} \approx <<sg2>>\) at every SNR, close to the prior width \(\sigma_\pi = <<spi>>\),
while \(g_1\) is constrained (\(\sigma_{g_1}\approx <<sg1>>\)).</p>
<p><b>Image SNR barely matters.</b> Going from \(\rho_I = <<r50>>\) to <<r200>> (a factor of
<<fac>> in SNR) changes \(\sigma_{\epsilon,i}\) of galaxy <<gid>> from <<se50>> to <<se200>>,
within the seed-to-seed scatter (≈<<sss>>); for galaxy 503 it shrinks by only <<p503>>. So at the
SNRs typical of the test set, the image is not what limits the shear measurement.</p>
<p><b>Hα SNR matters a lot.</b> At fixed image SNR, going from \(\rho_S = <<slo>>\) to <<shi>>
shrinks \(\sigma_{g_1}\) from <<sg1lo>> to <<sg1hi>> (a factor <<rat>>), while \(\sigma_{g_2}\)
stays at the prior-limited ≈<<sg2hi>>. The TF weights also become healthier (ESS fraction
<<esslo>> → <<esshi>>), because better spectra pin down \(v_{\rm circ}\). This confirms that the
shear information comes through the spectra. <b>For the paper</b>, the planned “two image SNRs”
figure would show two almost identical contours; the Hα-SNR figure shows the physics better
(or both can be shown, with the image-SNR one as the null result).</p>
<p><b>Noise moves the posterior by about its own width.</b> Across the <<ns>> noise seeds at fixed
SNR, the posterior mean of \(g_1\) for galaxy <<gid>> scatters by <<msd>>, comparable to
\(\sigma_{g_1}\approx <<sg1>>\). This is consistent with the posterior width reflecting real,
noise-driven uncertainty (when the data dominate over the prior, the two should be similar), and it
is why a single example galaxy's posterior mean can sit a full width away from the truth.</p>""",
        gid=gid, th=f(g0["truth"]["theta_int"], 2), s2t=f(s2t, 2), c2t=f(c2t, 2),
        sg2=f(arms["snrcatalog"]["sd_g2"], 3), spi=f(sigma_pi, 4), sg1=f(arms["snrcatalog"]["sd_g1"], 3),
        r50=f(a50["image_snr"], 0), r200=f(a200["image_snr"], 0), fac=f(a200["image_snr"] / a50["image_snr"], 0),
        se50=f(a50["sigma_eps"], 4), se200=f(a200["sigma_eps"], 4),
        sss=f(max(a50["sigma_eps_seed_std"], a200["sigma_eps_seed_std"]), 3),
        p503=pct(1 - arms503["snr200"]["sigma_eps"] / arms503["snr50"]["sigma_eps"], 0),
        slo=f(rho_levels[lo_key], 0), shi=f(rho_levels[hi_key], 0), sg1lo=f(rl["sd_g1"], 3),
        sg1hi=f(rh["sd_g1"], 3), rat=f(rl["sd_g1"] / rh["sd_g1"], 1), sg2hi=f(rh["sd_g2"], 3),
        esslo=f(rl["ess_fraction"], 2), esshi=f(rh["ess_fraction"], 2), ns=n_seeds,
        msd=f(arms["snrcatalog"]["mean_g1_seed_std"], 3), nout=n_out, npair=n_pair,
        nexp=f(0.32 * n_pair, 0), maxoff=f(max_off, 1))

    caveats = S(r"""<ul>
<li><b>The cache stores only \(g_1, g_2\) draws</b>, so the nine-parameter corners had to come from a
new inference with a new noise realization. They are therefore <i>not</i> the same posteriors as
the cached ones used for calibration and for Sec. 6, and their numbers differ (e.g. galaxy <<gid>>'s
\(g_1\) mean is <<m9>> in the new run vs <<mc>> in the cache).</li>
<li><b>Galaxy <<gid>>'s cached posterior is not a clean textbook example.</b> Its \(g_1\)–\(g_2\)
posterior is a curved, tilted band (a “banana”) with \(g_2\) piled against the \(+0.1\) prior
edge; the \(g_1\) mean is <<g1off>> half-widths from the truth, and the \(\theta_{\rm int}\) mean
(<<thm>>) is offset from the truth (<<tht>>) by <<thoff>> interval half-widths, consistent with
the \(\theta_{\rm int}\) under-coverage found on the calibration page. In the new run
\(\theta_{\rm int}\) is offset the other way (mean <<th9>>). The shape of the \(g_1\)–\(g_2\)
posterior also differs between the cached and new noise realizations (tilted band vs nearly
vertical band). We believe this is noise (the seed scatter of \(\hat g_1\) is as large as its
width), but only the five new seeds were compared; plotting all seeds would confirm it.</li>
<li><b>Why galaxy 503 was used</b> as the second SNR galaxy is not recorded.</li>
<li>At \(\rho_S = <<slo>>\) the seed-0 contour in the figure piles up at the \(g_1 = +0.1\) edge; the
table averages over seeds are the more reliable summary. The seed-0 legend widths differ from the
seed averages for the same reason.</li>
<li>Only one galaxy (plus one cross-check) was used for the SNR experiment. The conclusion that
image SNR barely matters holds for typical inclinations and SNRs; it may not hold at very low image
SNR (below the range tested) or for nearly face-on disks.</li>
<li>These are simulated galaxies with the same simulator used for training; a comparison with MCMC
posteriors for a few galaxies is still pending.</li>
</ul>""", gid=gid, m9=f(c9(gid, "posterior_mean")[0], 3), mc=f(g1_post["mean"], 3),
              g1off=f(abs(g1_off), 1), thm=f(th_post["mean"], 3), tht=f(g0["truth"]["theta_int"], 3),
              thoff=f(th_off, 1), th9=f(th9, 3), slo=f(rho_levels[lo_key], 0))

    provenance = S(r"""<ul>
<li>(a) Cached posteriors: <code><<s>>/a_npe_stream.py</code> (<code>pick_corner_galaxies</code>, SLURM job
<<job>>) and <code><<s>>/a_npe_report.py</code> (<code>pick_typical_width</code>, <code>r2a_corner</code>).
JSON: <code>A_npe/report.json</code> → <code>R2a</code>. Figures <code>A_npe/r2a_corner_gal*.png</code>.</li>
<li>(b)–(d) New inference: <code><<s>>/b_r2b_sample.py</code> (calls <code>train.sample_density</code> with an
SNR override, TF weights from <code>tf_prior.posterior_importance_weights</code>), run on the login node
on CPU (no SLURM job; the GPU jobs were cancelled, see <code>B_inference/STATUS.txt</code>). Samples in
<code>$S/cache/paper_tests_B_inference/r2b/</code>. Tables and figures: <code><<s>>/b_make_report.py</code>
(<code>r2b_table</code>, <code>r2b_figure</code>, <code>rhos_figure</code>, <code>corner_figure</code>).</li>
<li>JSON: <code>B_inference/report.json</code> → <code>R2b</code> (keys <code><<gid>></code>, <code>503</code>,
<code><<gid>>_rhoS</code>) and <code>corner9d</code>.</li>
<li>Detailed group reports: <a href='../A_npe/report.html'>A_npe/report.html</a> (“R2a”) and
<a href='../B_inference/report.html'>B_inference/report.html</a> (sections 1–2).</li>
</ul>""", s=SCRIPTS, job=_job("A_npe/STATUS.txt", r"STREAM_JOB=(\d+)"), gid=gid)

    answer = (f"The posteriors look sensible, and the shear information comes from the spectra: raising the image "
              f"SNR of galaxy {gid} from {f(a50['image_snr'], 0)} to {f(a200['image_snr'], 0)} leaves its shape "
              f"noise unchanged ({f(a50['sigma_eps'], 3)} vs {f(a200['sigma_eps'], 3)}), while raising the Hα SNR "
              f"from {f(rho_levels[lo_key], 0)} to {f(rho_levels[hi_key], 0)} shrinks σ_g1 from "
              f"{f(rl['sd_g1'], 3)} to {f(rh['sd_g1'], 3)}.")

    return page(slug="example-posteriors", title="Example posteriors and their dependence on SNR",
                code="R2a, R2b", section=SECTION, answer=answer, question=question, why=why, theory=theory,
                procedure=procedure, results=results, interpretation=interpretation, caveats=caveats,
                provenance=provenance)


PAGES = [_page_training(), _page_calibration(), _page_tf_weights(), _page_examples()]
