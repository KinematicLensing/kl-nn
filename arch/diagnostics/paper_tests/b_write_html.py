#!/usr/bin/env python3
"""Write the self-contained Group B report.html from report.json and PNGs."""
from __future__ import annotations

import base64
import json
from pathlib import Path

S = Path("/ocean/projects/phy250048p/shared")
OUT = S / "reports/paper-tests/B_inference"
rep = json.loads((OUT / "report.json").read_text())
T = rep["T2b"]; X = T["xu3like"]; F = T["fibrepair"]; L = T["legacy45255702"]
CONS = rep["T2b_xu3_consistency"]


def img(name, cap, width="100%"):
    data = base64.b64encode((OUT / f"{name}.png").read_bytes()).decode()
    return (f'<figure><img style="width:{width}" src="data:image/png;base64,{data}" alt="{name}">'
            f"<figcaption>{cap}</figcaption></figure>")


def pm(v, e, p=3):
    return f"{v:.{p}f} ± {e:.{p}f}"


def mat(block):
    R, E = block["R"], block["R_se"]
    return (f'<table class="matrix"><tr><td>{pm(R[0][0], E[0][0])}</td><td>{pm(R[0][1], E[0][1])}</td></tr>'
            f'<tr><td>{pm(R[1][0], E[1][0])}</td><td>{pm(R[1][1], E[1][1])}</td></tr></table>')


CSS = """
:root{--ink:#1b1f24;--muted:#5c6770;--line:#d9dee3;--paper:#fbfbf9;--navy:#1f4e79;--orange:#b85c38;--go:#1f6b4a;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#e8eaed;--muted:#a9b1b8;--line:#3a4046;--paper:#16181b;--navy:#8fb8e0;--orange:#e39a7a;--go:#7cc7a2;--card:#1f2226}}
:root[data-theme="dark"]{--ink:#e8eaed;--muted:#a9b1b8;--line:#3a4046;--paper:#16181b;--navy:#8fb8e0;--orange:#e39a7a;--go:#7cc7a2;--card:#1f2226}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:var(--paper);font:16px/1.55 "Iowan Old Style","Palatino Linotype",Palatino,"Times New Roman",serif}
header{background:#1f4e79;color:#f4f7fa;padding:2.2rem 1.25rem 1.8rem}
header p{color:#d5e2ee;max-width:46rem}
header .in{max-width:920px;margin:0 auto}
main{max-width:920px;margin:0 auto;padding:1.5rem 1rem 4rem}
h1{font-size:2rem;margin:0 0 .4rem;font-weight:600}
h2{font-size:1.35rem;margin:2.4rem 0 .7rem;color:var(--navy)}
h3{font-size:1.08rem;margin:1.4rem 0 .4rem}
p,li{max-width:46rem}
.meta{font:13px/1.4 ui-sans-serif,system-ui,sans-serif;letter-spacing:.02em;text-transform:uppercase;color:#b9c9d8}
.verdict{border:2px solid var(--go);padding:1rem 1.1rem;margin:1.4rem 0 1.8rem;background:var(--card)}
.verdict strong{color:var(--go)}
.callout{border-left:4px solid var(--navy);padding:.2rem 0 .2rem 1rem;margin:1.1rem 0;color:var(--muted)}
.warn{border-left-color:var(--orange)}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:14px/1.4 ui-sans-serif,system-ui,sans-serif;margin:.8rem 0 1.2rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.4rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.matrix{width:auto}.matrix td{border:1px solid var(--line);text-align:center;padding:.4rem .6rem}
figure{margin:1.2rem 0 1.6rem}figure img{max-width:100%;height:auto;display:block;background:#fff}
figcaption{font:13px/1.4 ui-sans-serif,system-ui,sans-serif;color:var(--muted);margin-top:.45rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}
code{font-size:.85em;word-break:break-all}
footer{font:13px/1.45 ui-sans-serif,system-ui,sans-serif;color:var(--muted);margin-top:2.5rem}
@media (max-width:700px){.grid{grid-template-columns:1fr}}
"""

# ---------------- R2b ----------------
g = rep["R2b"]["6447"]; g5 = rep["R2b"]["503"]; gs = rep["R2b"]["6447_rhoS"]


def r2b_rows(block, order, names):
    out = []
    for k, nm in zip(order, names):
        r = block["rows"][k]
        out.append(f"<tr><td>{nm}</td><td>{r['image_snr']:.0f}</td><td>{pm(r['sd_g1'], r['sd_g1_seed_std'], 4)}</td>"
                   f"<td>{pm(r['sd_g2'], r['sd_g2_seed_std'], 4)}</td><td>{pm(r['sigma_eps'], r['sigma_eps_seed_std'], 4)}</td>"
                   f"<td>{r['seed0_sigma_eps']:.4f}</td><td>{r['mean_g1']:+.4f} / {r['mean_g2']:+.4f}</td>"
                   f"<td>{r['mean_g1_seed_std']:.4f} / {r['mean_g2_seed_std']:.4f}</td><td>{r['ess_fraction']:.2f}</td></tr>")
    return "".join(out)


R2B_HEAD = ("<tr><th>arm</th><th>ρ<sub>I</sub></th><th>σ<sub>g1</sub></th><th>σ<sub>g2</sub></th><th>σ<sub>ε,i</sub></th>"
            "<th>σ<sub>ε,i</sub> seed 0</th><th>mean ĝ1 / ĝ2</th><th>seed std of ĝ1 / ĝ2</th><th>TF ESS frac</th></tr>")

# ---------------- T2b tables ----------------
def overall_row(name, blk, extra=""):
    o = blk["overall"]
    return (f"<tr><td>{name}</td><td>{pm(o['R'][0][0], o['R_se'][0][0])}</td><td>{pm(o['R'][1][1], o['R_se'][1][1])}</td>"
            f"<td>{pm(o['R'][0][1], o['R_se'][0][1])}</td><td>{pm(o['R'][1][0], o['R_se'][1][0])}</td>"
            f"<td>{o['c'][0]*1e3:+.2f} ± {o['c_se'][0]*1e3:.2f}</td><td>{o['c'][1]*1e3:+.2f} ± {o['c_se'][1]*1e3:.2f}</td>"
            f"<td>{o['weight_ess']:.0f}</td><td>{extra}</td></tr>")


def bins_rows(blk, key, fmt_edge):
    out = []
    for b in blk[key]:
        if "R" not in b:
            out.append(f"<tr><td>{fmt_edge(b)}</td><td>{b['n']}</td><td colspan=6>too few</td></tr>"); continue
        pred = 1 - b["mean_sigma_comp_sq"] / (0.2 / 12 ** 0.5) ** 2
        out.append(f"<tr><td>{fmt_edge(b)}</td><td>{b['n']}</td><td>{pm(b['R'][0][0], b['R_se'][0][0])}</td>"
                   f"<td>{pm(b['R'][1][1], b['R_se'][1][1])}</td><td>{b['R'][0][1]:+.3f} / {b['R'][1][0]:+.3f}</td>"
                   f"<td>{b['mean_sigma_eps']:.4f}</td><td>{pred:.3f}</td></tr>")
    return "".join(out)


def fit_rows():
    out = []
    for name, blk in (("Xu3-like stencil, TF-weighted mean (primary)", X["tf_equal"]),
                      ("Xu3-like stencil, unweighted mean", X["proposal"]),
                      ("proposal stencil, unweighted mean", F["proposal"]),
                      ("proposal stencil, TF target (pop. ESS 208)", F["tf_target"]),
                      ("proposal stencil, TF mean + equal bases (misspecified)", F["tf_equal"]),
                      ("legacy 45255702, proposal stencil, unweighted", L["proposal"])):
        f = blk["fit_R_sigma"]; pc = f["per_component"]; tr = f["trace_vs_sigma_eps_sq"]
        out.append(f"<tr><td>{name}</td><td>{pm(pc['a'], pc['a_se'])}</td><td>{pc['b']:.0f} ± {pc['b_se']:.0f}</td>"
                   f"<td>{pc['b_fixed_a1']:.0f}</td><td>{pm(tr['a'], tr['a_se'])}</td><td>{tr['b']:.0f} ± {tr['b_se']:.0f}</td>"
                   f"<td>{tr['residual_rms']:.2f}</td></tr>")
    return "".join(out)


def sel_rows(name, blk):
    out = []
    for s in blk["selection"]:
        v, e = s["value"], s["se"]
        rsel = v["R_sel"]; rse = e["R_sel"]
        sig = s["R_sel_significance_sigma"]
        out.append(f"<tr><td>{name}</td><td>{s['threshold']:.3f}</td><td>{v['frac_by_state'][0]:.3f}</td>"
                   f"<td>{v['dfrac_dg'][0]:+.2f} / {v['dfrac_dg'][1]:+.2f}</td>"
                   f"<td>{s['R_gamma_diag']:.3f}</td><td>{pm(rsel[0][0], rse[0][0])}</td><td>{pm(rsel[1][1], rse[1][1])}</td>"
                   f"<td>{rsel[0][1]:+.3f} / {rsel[1][0]:+.3f}</td>"
                   f"<td>{pm(s['R_sel_diag'], s['R_sel_diag_se'])}</td>"
                   f"<td>{(f'{sig:.1f}' if sig is not None else '—')}</td>"
                   f"<td>{(100 * s['R_sel_over_R_gamma']):+.1f}%</td><td>{s['n_flips']['g1_plus_vs_g1_minus']} / {s['n_flips']['g2_plus_vs_g2_minus']}</td></tr>")
    return "".join(out)


xs = X["tf_equal"]["selection"]
s05, s075 = xs[1], xs[2]
xfit = X["tf_equal"]["fit_R_sigma"]["per_component"]
c1 = CONS["fit_from_xu3like_tf_equal"]
rw = CONS["resp_weighted_xu3like"]; rwp = CONS["resp_weighted_proposal_stencil"]
xo = X["tf_equal"]["overall"]
regen = rep["regen_probe"]
import math
z1 = abs(xo['R'][0][0] - (1 + c1['g1']['raw_low_m'])) / math.hypot(xo['R_se'][0][0], c1['g1']['raw_low_m_se'])
z2 = abs(xo['R'][1][1] - (1 + c1['g2']['raw_low_m'])) / math.hypot(xo['R_se'][1][1], c1['g2']['raw_low_m_se'])
fp = {round(q['threshold'], 3): q for q in F['proposal']['selection']}

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper tests B: inference</title><style>{CSS}</style></head><body>
<header><div class="in"><div class="meta">KL-NN paper tests · Group B · fibrepair NPE · 29 Sep 2026</div>
<h1>R2b posterior vs S/N, 9D corners, and T2b shear and selection response</h1>
<p>Model <code>CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair</code> (best checkpoint). Estimator: the TF-weighted posterior mean of (g1, g2) from 10<sup>4</sup> identity+R90 draws. Width: σ<sub>ε,i</sub> = √(σ<sub>g1</sub>² + σ<sub>g2</sub>²), the Xu+24 quadrature of the TF-weighted 1D marginal standard deviations. This is not the √((Var g1+Var g2)/2) used in <code>shear_bias_report.py</code>; the two differ by a factor √2.</p></div></header>
<main>

<div class="verdict"><div class="kicker">Answers</div>
<p><strong>R2b.</strong> For the chosen galaxy (Xu3 #6447, cos i = 0.61, ρ<sub>S</sub> = 23.9), going from ρ<sub>I</sub> = 50 to 200 barely changes the shear posterior. σ<sub>ε,i</sub> is {g['rows']['snr50']['sigma_eps']:.4f} at ρ<sub>I</sub> = 50 and {g['rows']['snr200']['sigma_eps']:.4f} at ρ<sub>I</sub> = 200 (mean of 5 noise draws; the scatter between draws is ≈0.001). g2 stays at the prior width. The Hα S/N is what moves the posterior: at fixed ρ<sub>I</sub>, σ<sub>g1</sub> goes from {gs['rows']['hasnr8']['sd_g1']:.3f} at ρ<sub>S</sub> = 8 to {gs['rows']['hasnr60']['sd_g1']:.3f} at ρ<sub>S</sub> = 60. The planned "two image S/N" figure will therefore show two almost identical contours; a ρ<sub>S</sub> panel is provided as an alternative.</p>
<p><strong>T2b.</strong> On a new Xu3-like five-shear stencil (1000 bases, shared noise), the response is R = diag({xo['R'][0][0]:.3f}, {xo['R'][1][1]:.3f}) ± 0.014, close to the Xu3 raw 1+m = ({1+c1['g1']['raw_low_m']:.3f}, {1+c1['g2']['raw_low_m']:.3f}). Per component, R follows a + bσ² with b = {xfit['b']:.0f} ± {xfit['b_se']:.0f} (Gaussian shrinkage predicts −300) and a = {xfit['a']:.2f} ± {xfit['a_se']:.2f}. Calibrating the Xu3 ensemble with this R(σ) leaves m = ({rw['g1']['m_after_resp_weighted_calibration']:+.3f} ± {rw['g1']['se']:.3f}, {rw['g2']['m_after_resp_weighted_calibration']:+.3f} ± {rw['g2']['se']:.3f}).
<strong>The σ-cut does induce a significant selection response.</strong> At t = 0.05, R<sub>sel</sub> = {pm(s05['R_sel_diag'], s05['R_sel_diag_se'])}, which is {100*s05['R_sel_over_R_gamma']:+.0f}% of R<sub>γ</sub> ({s05['R_sel_significance_sigma']:.1f}σ). At t = 0.075 it is {pm(s075['R_sel_diag'], s075['R_sel_diag_se'])} ({100*s075['R_sel_over_R_gamma']:+.1f}%, {s075['R_sel_significance_sigma']:.1f}σ). Because R<sub>sel</sub> > 0, a σ-cut makes ĝ over-respond relative to the unselected response. t = 0.1 is not a cut: σ<sub>ε,i</sub> never exceeds ≈0.085, and the prior-limited ceiling is √2·σ<sub>π</sub> = 0.082.</p></div>

<h2>1. R2b: g1–g2 posterior at two image S/N</h2>
<p><b>Galaxy choice.</b> Xu3 row 6447, the galaxy the NPE-validation agent proposed. Truth: g = ({g['truth'][0]:+.4f}, {g['truth'][1]:+.4f}), θ<sub>int</sub> = {g['truth'][2]:.2f}, cos i = {g['truth'][3]:.3f}, catalog ρ<sub>I</sub> = 166, ρ<sub>S</sub> = {g['central_halpha_snr']:.1f}. Xu3 medians are ρ<sub>I</sub> = 174 and ρ<sub>S</sub> = 25.5.</p>
<p><b>Method.</b> The <code>image_snr</code> field of the LMDB record is overridden. Both the pixel-noise level and the network's observation context follow it; ρ<sub>S</sub> stays at the catalog value. Five noise seeds (0–4) are shared across the S/N arms, so each seed has the same noise pattern at a different amplitude. These draws are fresh, not the Xu3 cache's. Sampling uses <code>train.sample_density</code>, 10<sup>4</sup> identity+R90 draws, and TF weights as in <code>cache_posteriors.py</code>. The figure shows seed 0; the table averages the 5 seeds.</p>
{img('r2b_g1g2_contours_gal6447', 'Paper candidate (paper_figs/B_r2b_g1g2_contours_gal6447.png/.pdf). Contours enclose 68% (thick) and 95% (thin) of the TF-weighted posterior mass (80×80 histogram, Gaussian smoothing 1.8 bins). Dots mark posterior means and the star marks the truth. The axes are the training-prior support ±0.1.', '70%')}
<div class="tw"><table><thead>{R2B_HEAD}</thead><tbody>{r2b_rows(g, ['snr50','snrcatalog','snr200'], ['ρ_I = 50','catalog','ρ_I = 200'])}</tbody></table></div>
<p>Second galaxy, Xu3 503 (cos i = 0.50, ρ<sub>S</sub> = 27.6, catalog ρ<sub>I</sub> = 147). σ<sub>ε,i</sub> is {g5['rows']['snr50']['sigma_eps']:.4f} at ρ<sub>I</sub> = 50 and {g5['rows']['snr200']['sigma_eps']:.4f} at ρ<sub>I</sub> = 200, only about 5% narrower at the higher S/N.</p>
<div class="tw"><table><thead>{R2B_HEAD}</thead><tbody>{r2b_rows(g5, ['snr50','snrcatalog','snr200'], ['ρ_I = 50','catalog','ρ_I = 200'])}</tbody></table></div>
<h3>Alternative: vary ρ<sub>S</sub> at fixed ρ<sub>I</sub> = 166</h3>
{img('r2b_g1g2_contours_rhoS_gal6447', 'Paper alternative (paper_figs/B_r2b_g1g2_contours_rhoS_gal6447.*). ρ<sub>S</sub> = 8 vs 60, noise seed 0. At ρ<sub>S</sub> = 8 this seed piles up against the g1 = +0.1 prior edge; the other seeds do not (see the seed std of ĝ1).', '70%')}
<div class="tw"><table><thead>{R2B_HEAD}</thead><tbody>{r2b_rows(gs, ['hasnr8','snrcatalog','hasnr60'], ['ρ_S = 8','ρ_S = 23.9 (catalog)','ρ_S = 60'])}</tbody></table></div>
<div class="callout warn">Galaxy 6447's σ<sub>g2</sub> ≈ 0.055 at every S/N, compared with the prior σ<sub>π</sub> = 0.0577. One shear component is prior-limited even for a typical galaxy, so σ<sub>ε,i</sub> (the quadrature of both components) stays ≳ σ<sub>π</sub> whenever one component is unconstrained.</div>

<h2>2. 9D corner plots (catalog S/N)</h2>
<p>These are full 9-parameter TF-weighted posteriors in physical units (log<sub>10</sub> for the Hα flux), 10<sup>4</sup> identity+R90 draws each. They were resampled here because the Xu3 cache keeps only (g1, g2). The noise realizations are new (seed 0), so these posteriors are not the cached ones. Contours enclose 39% and 86%, the 1σ and 2σ levels of a 2D Gaussian.</p>
""" + "".join(img(f"corner9d_gal{k}", f"Xu3 galaxy {k}: ρ<sub>I</sub> = {v['image_snr']:.0f}, ρ<sub>S</sub> = {v['central_halpha_snr']:.1f}, true cos i = {v['truth'][3]:.3f}, TF ESS fraction {v['tf_ess_fraction']:.2f}. Paper copy: paper_figs/B_corner9d_gal{k}.png/.pdf.") for k, v in rep["corner9d"].items()) + f"""

<h2>3. T2b: the stencils</h2>
<p><b>Can the legacy stencil be reused?</b> Yes, and it does not need the fiber repair. The stencil <code>shear_response_simv3_cosi_5k</code> (1000 bases from <code>train_1m_simv3_cosi.csv</code> × 5 shears, Δg = 0.01, noiseless) was generated on 7 Sep with right-handed fibers. A dry run of <code>repair_minor_axis_fiber_sign.py</code> classifies all 5000 LMDB and all 5000 FITS rows as <code>match</code>, with 0 <code>swap_minor</code>. (The CSV-based "legacy SVD" prediction says <code>swap_minor</code> for all 5000; that prediction is wrong for this catalog, as it was for g02/g002.) Eight rows re-rendered with the current simulator are <b>bitwise identical</b> to the stored FITS in the image, all 5 spectra and the fiber offsets: bitwise_identical = {regen['bitwise_identical']}, file <code>regen_probe.json</code>. The LMDB was copied to <code>datasets/paper_tests_B_shear_response_simv3_cosi_5k_fibrepair</code> (md5-identical; the repair would be a no-op). The original was not touched.</p>
<div class="callout warn"><b>Side finding.</b> The legacy model 45255702 was trained on the pre-repair (swap_minor) catalogs but its stencil is right-handed. The legacy shear-response report (R ≈ 0.49–0.58) therefore evaluated 45255702 in a different fiber gauge from the one it was trained on. Those numbers should not be quoted as the legacy model's in-gauge response.</div>
<p><b>Why a second stencil.</b> The legacy bases come from the uniform training proposal: ρ<sub>I</sub> ~ U[10, 1000] and v<sub>circ</sub> not on the TF relation. Posterior TF weights are degenerate for many of these bases (TF ESS fraction < 0.05 for 28% of rows, versus 0.3% on the Xu3-like stencil), so the production estimator (TF-weighted mean) is noisy and misspecified there. The TF-population ESS is only 208 of 1000. I therefore also built a <b>Xu3-like stencil</b>. Its 1000 bases were drawn (seed 2718) from <code>test_100k_simv3_xu3_tf.csv</code>, the TF-conformed, catalog-S/N Xu3 draw that is <i>not</i> the paper test set, after removing the 284 rows with ρ<sub>S</sub> > 150 (outside model support). It was rendered with the current simulator (fibers <code>match</code> 5000/5000). That catalog shares g, θ<sub>int</sub>, sin i, v<sub>0</sub> and r<sub>scale</sub> column values with the paper Xu3 set row for row (same LHS draw), but rmag, hlr, v<sub>circ</sub> and both S/N differ. The runbook's rule against using the paper Xu3 as generator input is respected.</p>
<p>Inference ran on CPU (RM-shared; GPU-shared was queued about 8 h) with <code>cache_posteriors.py</code> unchanged: <code>--matched-group-size 5</code>, shared image and spectral noise within each 5-shear group, 10<sup>4</sup> identity+R90 draws, seed 42, 100 partitions × 50 rows. Errors are bootstrap over base galaxies, 1000–2000 resamples.</p>
{img('t2b_sigma_hist', 'σ<sub>ε,i</sub> at zero shear. The Xu3-like stencil tracks Xu3 except for a deficit below 0.03 (3.1% vs 8.8% of Xu3). The Xu3 test set spans |g| ≤ 0.1, while the stencil has |g| ≤ 0.01, which suggests σ<sub>ε,i</sub> falls with |g| (T1). Dotted lines mark the cut thresholds.', '75%')}

<h2>4. T2b: shear response R</h2>
<div class="tw"><table><thead><tr><th>stencil / estimator</th><th>R11</th><th>R22</th><th>R12</th><th>R21</th><th>c1 ×10³</th><th>c2 ×10³</th><th>base ESS</th><th>note</th></tr></thead><tbody>
{overall_row('Xu3-like, TF-weighted mean (primary)', X['tf_equal'], 'production analogue')}
{overall_row('Xu3-like, unweighted mean', X['proposal'])}
{overall_row('proposal stencil, unweighted mean', F['proposal'])}
{overall_row('proposal stencil, TF target', F['tf_target'], 'legacy-report convention')}
{overall_row('proposal stencil, TF mean, equal bases', F['tf_equal'], 'TF weights degenerate')}
{overall_row('legacy 45255702, proposal stencil, unweighted', L['proposal'], 'gauge-mismatched')}
{overall_row('legacy 45255702, proposal stencil, TF target', L['tf_target'], 'gauge-mismatched')}
</tbody></table></div>
<p>Xu3 test set, raw low-|g| fit, same cache estimator: 1+m = ({1+c1['g1']['raw_low_m']:.3f} ± {c1['g1']['raw_low_m_se']:.3f}, {1+c1['g2']['raw_low_m']:.3f} ± {c1['g2']['raw_low_m_se']:.3f}). The Xu3-like stencil gives R = ({xo['R'][0][0]:.3f}, {xo['R'][1][1]:.3f}), differing by {z1:.1f}σ (g1) and {z2:.1f}σ (g2). The off-diagonals (−0.014, −0.013 ± 0.007) are marginal at 2σ.</p>
<h3>Binned by σ<sub>ε,i</sub> (zero-shear state), Xu3-like stencil, TF-weighted mean</h3>
<div class="tw"><table><thead><tr><th>σ<sub>ε,i</sub> bin</th><th>n</th><th>R11</th><th>R22</th><th>R12 / R21</th><th>mean σ<sub>ε,i</sub></th><th>Gaussian 1−⟨σ<sub>k</sub>²⟩/σ<sub>π</sub>²</th></tr></thead>
<tbody>{bins_rows(X['tf_equal'], 'by_sigma', lambda b: f"[{b['lo']:.3f}, {b['hi']:.3f})")}</tbody></table></div>
<h3>Same, proposal stencil, unweighted mean</h3>
<div class="tw"><table><thead><tr><th>σ<sub>ε,i</sub> bin</th><th>n</th><th>R11</th><th>R22</th><th>R12 / R21</th><th>mean σ<sub>ε,i</sub></th><th>Gaussian</th></tr></thead>
<tbody>{bins_rows(F['proposal'], 'by_sigma', lambda b: f"[{b['lo']:.3f}, {b['hi']:.3f})")}</tbody></table></div>
<div class="callout warn"><b>Suspicious: R > 1 for the narrowest Xu3-like posteriors.</b> The 31 bases with σ<sub>ε,i</sub> < 0.03 have R ≈ 1.4 ± 0.1 (median of the per-base trace response 1.14; 90th percentile 2.3). All but one have cos i > 0.9. A likely cause is that the fiber positions fed to the network are computed from the <i>noiseless true</i> sheared galaxy axes (<code>compute_fiber_offsets(g1, g2, θ<sub>int</sub>, sin i)</code>). For a nearly face-on disk the observed axis direction is set mostly by the shear, so the fiber layout leaks the shear orientation without noise. This is a hypothesis, not tested here. It matters for the paper's claims about narrow-posterior galaxies and for any width cut that keeps mainly face-on systems.</div>
<h3>Binned by true cos i</h3>
<div class="tw"><table><thead><tr><th>cos i bin</th><th>n</th><th>R11</th><th>R22</th><th>R12 / R21</th><th>mean σ<sub>ε,i</sub></th><th>Gaussian</th></tr></thead>
<tbody>{bins_rows(X['tf_equal'], 'by_cosi', lambda b: f"Xu3-like [{b['lo']:.1f}, {b['hi']:.1f})")}{bins_rows(F['proposal'], 'by_cosi', lambda b: f"proposal [{b['lo']:.1f}, {b['hi']:.1f})")}</tbody></table></div>
{img('t2b_R_vs_cosi', 'Diagonal response vs true cos i. The steep rise toward face-on is the shrinkage pattern: edge-on disks have wider shear posteriors.', '75%')}

<h2>5. T2b: the R(σ) = a + bσ² calibration (Sec. 6.4)</h2>
<p>For a Gaussian prior of width σ<sub>π</sub> and a Gaussian likelihood, the posterior-mean response is exactly R = 1 − σ<sub>post</sub>²/σ<sub>π</sub>², with σ<sub>post</sub> the <i>per-component</i> posterior width. So b = −1/σ<sub>π</sub>² = −300 applies to R<sub>kk</sub> vs σ<sub>g_k</sub>². Written against σ<sub>ε,i</sub>² = σ<sub>g1</sub>² + σ<sub>g2</sub>², the same model gives b = −1/(2σ<sub>π</sub>²) = −150 for the trace. Both fits use σ from the zero-shear state and OLS over bases, with bootstrap errors.</p>
<div class="tw"><table><thead><tr><th>sample</th><th>a (per comp.)</th><th>b (per comp.; pred −300)</th><th>b with a≡1</th><th>a (trace vs σ<sub>ε</sub>²)</th><th>b (trace; pred −150)</th><th>per-base resid. rms</th></tr></thead><tbody>{fit_rows()}</tbody></table></div>
{img('t2b_R_vs_sigma', 'Paper candidate (paper_figs/B_t2b_R_vs_sigma.*). Pooled R<sub>11</sub> and R<sub>22</sub> of the Xu3-like stencil (TF-weighted mean) in deciles of the per-component posterior variance, with the linear fit, the proposal-stencil fit and the Gaussian-shrinkage line. The slope is close to −1/σ<sub>π</sub>²; the intercept exceeds 1 because of the R > 1 face-on points and the non-Gaussian uniform prior.')}
<p><b>Applying it to Xu3.</b> Dividing each galaxy by R(σ<sub>i</sub>) is unstable (R → 0 near the prior width) and leaves m ≈ ({c1['g1']['calibrated_low_m']:+.2f}, {c1['g2']['calibrated_low_m']:+.2f}). The stable ensemble version regresses ĝ<sub>k,i</sub> on R(σ<sub>k,i</sub>)·g<sub>k,i</sub> (|g| < 0.02, equivalent to weights ∝ R<sub>i</sub>). With a nested bootstrap over stencil bases and Xu3 galaxies, the residual m is <b>({rw['g1']['m_after_resp_weighted_calibration']:+.3f} ± {rw['g1']['se']:.3f}, {rw['g2']['m_after_resp_weighted_calibration']:+.3f} ± {rw['g2']['se']:.3f})</b> using the Xu3-like fit and ({rwp['g1']['m_after_resp_weighted_calibration']:+.3f} ± {rwp['g1']['se']:.3f}, {rwp['g2']['m_after_resp_weighted_calibration']:+.3f} ± {rwp['g2']['se']:.3f}) using the proposal-stencil fit. The mean of R(σ<sub>i</sub>) over Xu3 low-|g| galaxies is ({c1['g1']['predicted_1pm_from_stencil_fit']:.3f}, {c1['g2']['predicted_1pm_from_stencil_fit']:.3f}), above the measured 1+m, because Xu3 has more σ < 0.03 galaxies than the stencil and the linear model over-predicts R there.</p>

<h2>6. T2b: selection response of the width cut σ<sub>ε,i</sub> < t</h2>
<p>Metacal-style, using the matched noise: <i>sel<sub>s</sub></i> is the cut evaluated on state s (its own σ<sub>ε,i</sub>, measured on the sheared image with the same noise). R<sub>γ</sub> is the mean per-base R over bases selected at zero shear. R<sub>sel,ij</sub> = [⟨ĝ<sub>i</sub>(0)⟩<sub>sel(+g_j)</sub> − ⟨ĝ<sub>i</sub>(0)⟩<sub>sel(−g_j)</sub>]/2Δg. The total response of the cut sample, R<sub>tot</sub> (own-state selection), equals R<sub>γ</sub> + R<sub>sel</sub> to within noise. Also listed is the change in the selected fraction per unit shear, df/dg = [f(+) − f(−)]/2Δg. Flips counts bases whose selection differs between the + and − states. Part of this is MC noise in σ between independently sampled states; that adds variance but no bias.</p>
<div class="tw"><table><thead><tr><th>sample</th><th>t</th><th>f(g=0)</th><th>df/dg1 / dg2</th><th>R<sub>γ</sub> diag</th><th>R<sub>sel,11</sub></th><th>R<sub>sel,22</sub></th><th>R<sub>sel,12</sub> / R<sub>sel,21</sub></th><th>R<sub>sel</sub> diag</th><th>signif. (σ)</th><th>R<sub>sel</sub>/R<sub>γ</sub></th><th>flips g1 / g2</th></tr></thead><tbody>
{sel_rows('Xu3-like, TF mean', X['tf_equal'])}{sel_rows('proposal, unweighted', F['proposal'])}{sel_rows('proposal, TF target', F['tf_target'])}</tbody></table></div>
{img('t2b_selection_response', 'Left: selection response vs threshold. Middle: R<sub>γ</sub> (solid) and R<sub>tot</sub> (dashed). Right: selected fraction.')}
<ul>
<li><b>Significant.</b> For the Xu3-like stencil, R<sub>sel</sub> is {pm(s05['R_sel_diag'], s05['R_sel_diag_se'])} at t = 0.05 ({s05['R_sel_significance_sigma']:.1f}σ, {100*s05['R_sel_over_R_gamma']:+.0f}% of R<sub>γ</sub>) and {pm(s075['R_sel_diag'], s075['R_sel_diag_se'])} at t = 0.075 ({s075['R_sel_significance_sigma']:.1f}σ, {100*s075['R_sel_over_R_gamma']:+.1f}%). The proposal stencil agrees: {pm(fp[0.05]['R_sel_diag'], fp[0.05]['R_sel_diag_se'])} at 0.05 and {pm(fp[0.03]['R_sel_diag'], fp[0.03]['R_sel_diag_se'])} at 0.03. At t = 0.03 the Xu3-like stencil keeps only 31 bases, so R<sub>sel</sub> is not measurable there.</li>
<li>The selected <i>fraction</i> does not move with g: every df/dg is within 1σ of 0 (bootstrap SE 0.1–0.3 per unit g). The effect is a shear-orientation-dependent reshuffling: galaxies whose ĝ lines up with the applied shear become narrower and enter the cut. It is not a change in the number selected.</li>
<li><b>Implication for Sec. 6.3–6.4.</b> A width cut changes m by +R<sub>sel</sub>/R<sub>γ</sub> on top of the change in R<sub>γ</sub>. Calibrating a cut sample with R(σ) alone would leave ≈ +{100*s05['R_sel_over_R_gamma']:.0f}% (t = 0.05) or +{100*s075['R_sel_over_R_gamma']:.0f}% (t = 0.075) of residual m. The correction needs the metacal-style R<sub>sel</sub> as well.</li>
</ul>

<h2>7. Caveats</h2>
<ul>
<li>The stencil |g| ≤ 0.01 measures the response at zero shear; the Xu3 m is fitted over |g| < 0.02.</li>
<li>The Xu3-like stencil comes from the non-cosi Xu3 draw; its σ<sub>ε,i</sub> distribution lacks Xu3's σ < 0.03 tail (see Fig. in §3).</li>
<li>The flow samples are drawn independently for each state; only the image and spectral noise is shared. MC noise in ĝ is ≈7×10<sup>−4</sup> per row, adding ≈0.05 of scatter to per-base R and none to the mean.</li>
<li>R2b uses one noise realization per figure; the widths are averaged over 5 seeds. The image S/N override also changes the network's S/N context, as it would for real data.</li>
<li>The Xu3 consistency check uses equal weights over all 100k (low-|g| fit, per component) with the same TF-weighted means as the cache, recomputed from the cached draws (max deviation from the cached means 4×10<sup>−9</sup>).</li>
</ul>

<h2>8. Bookkeeping</h2>
<div class="tw"><table><thead><tr><th>Item</th><th>Value</th></tr></thead><tbody>
<tr><td>Scripts</td><td><code>kl-nn/arch/diagnostics/paper_tests/b_*.py, b_*.slurm</code></td></tr>
<tr><td>Proposal stencil (copy)</td><td><code>$S/datasets/paper_tests_B_shear_response_simv3_cosi_5k_fibrepair</code>; manifest <code>$S/samples/shear_response_simv3_cosi_5k_manifest.csv</code></td></tr>
<tr><td>Xu3-like stencil</td><td><code>$S/paper_tests_B_inference/samples/</code>, <code>$S/fits/paper_tests_B_shear_response_xu3like_5k</code>, <code>$S/datasets/paper_tests_B_shear_response_xu3like_5k</code></td></tr>
<tr><td>Caches</td><td><code>$S/cache/paper_tests_B_inference/</code> (stencil caches, r2b/, xu3_sigma_summary.npz, summaries)</td></tr>
<tr><td>Jobs</td><td>see STATUS.txt</td></tr>
</tbody></table></div>
<footer>Machine-readable: report.json, t2b_xu3like.json, t2b_fibrepair.json, t2b_legacy45255702.json, regen_probe.json. Generated by b_make_report.py and b_write_html.py.</footer>
</main></body></html>
"""
(OUT / "report.html").write_text(html)
print("wrote", OUT / "report.html", len(html) // 1024, "kB")
