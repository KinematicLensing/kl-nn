"""Write paper-tests/index.html summarizing the pre-writing test campaign.

Numbers are transcribed from each group's report.json / final summary.
Re-run after any group changes: python write_index.py
"""
import html
import os
from pathlib import Path

ROOT = Path("/ocean/projects/phy250048p/shared/reports/paper-tests")

GROUPS = [
    ("A_npe/report.html", "A_npe: NPE validation (R1, R2a, R3, T6)"),
    ("A_shear/report.html", "A_shear: shear estimators (R-raw, T1, T3, T4, T5, R6, T2a, T7)"),
    ("B_inference/report.html", "B_inference: SNR contours, 9D corners, matched-shear stencil (R2b, T2b)"),
    ("C_stacking/report.html", "C_stacking: NPE likelihood stacking at constant shear (T8)"),
]

# (test, question, answer, implication for the paper)
ROWS = [
    ("R1", "Did training converge?",
     "Best validation NLL at the last epoch (200): train −18.34, valid −17.80. Validation improved by 0.0013 over the last 10 epochs.",
     "Converged, no overfitting. Loss is in normalized target coordinates."),
    ("R3", "Are posteriors calibrated?",
     "g1/g2 68% coverage 67.25/67.80%, 95% coverage 94.87/95.13% (equal weights, N = 10^5). g1 KS 0.0049 is just outside the 95% band (slightly too narrow). Nuisances within 0.8 pp except θ_int 64.5% (heavy tails).",
     "Sec. 5.2: well calibrated overall. The θ_int under-coverage should be mentioned."),
    ("T6", "Are TF importance weights healthy?",
     "Median ESS fraction 0.598; one galaxy with ESS < 50; 73 with ESS < 500, whose g2 coverage is 54.8%.",
     "Healthy; the TF reweighting is safe."),
    ("R2a/R2b", "Do example posteriors behave sensibly?",
     "9D corners for galaxies 6447, 54792 and 5327. Galaxy 6447: σ_ε,i ≈ 0.060 at ρ_I = 50 and 200 (image SNR barely matters) but σ_g1 = 0.037 → 0.016 for ρ_S = 8 → 60.",
     "Sec. 5.1: show the ρ_S panel (or both). Image SNR has little effect on the shear posterior; Hα SNR drives it."),
    ("R-raw", "How biased is the raw per-galaxy posterior mean?",
     "Equal weights, full Xu3 sample: m1 = −0.616 ± 0.015, m2 = −0.592 ± 0.014; 10^4 c ≈ −0.7/+0.8 ± 1.6; m+ = −0.648, m× = −0.576 (≈4σ apart); σ_ε mean 0.061 (Xu24 definition).",
     "Sec. 6.1: c ≈ 0 but m ≈ −0.6. The shrinkage is anisotropic in the galaxy frame."),
    ("T3", "Does shrinkage explain m?",
     "m = −⟨σ_post²⟩/σ_π² (σ_π = 0.0577) fits the binned data (χ² ≈ 16–18 for 12 dof) and predicts −0.607 vs −0.616 measured. The earlier −σ²/(σ²+σ_π²) form is wrong for posterior widths. The galaxy-frame prediction fits poorly.",
     "Sec. 6.2: use m = −σ_post²/σ_π². Use the selection-free D1 figure."),
    ("T1", "Does the posterior width depend on the true shear?",
     "Yes: d lnσ/d|g| = −1.53 ± 0.02 (−0.81 ± 0.06 away from the prior edge); the quadratic term is stronger along g+ than g×.",
     "σ is not a shear-independent selection variable."),
    ("T2a / T2b", "Do posterior-width cuts cause selection bias?",
     "Yes. Selection excess in m at matched kept fraction: +0.10 ± 0.02 at 20% kept, 0.00 at 70%. Stencil selection response R_sel/R_γ: +12% (4.5σ) at σ_ε,i < 0.05 and +3% (3.5σ) at < 0.075. Inverse-variance weighting adds no bias (+0.002 ± 0.002).",
     "Sec. 6.3: present the cut as a trade-off curve with a selection-bias caveat, or correct it with R_sel. Weighting is safe."),
    ("R6", "How does m change with a width cut?",
     "Raw m1/m2 = −0.017/+0.027 at σ_ε,i < 0.045 (20.8% kept); −0.097/−0.030 at < 0.05 (26.3%); overshoots to ≈ +0.1 at 0.03–0.035.",
     "Part of the approach to m ≈ 0 is the selection bias above."),
    ("T2b (calibration)", "Does response calibration work?",
     "Xu3-like stencil: R11/R22 = 0.423/0.414; R(σ) = a + bσ² with b = −350 ± 9 (Gaussian prediction −300). Ensemble-calibrated Xu3 m = −0.049 ± 0.031 / −0.000 ± 0.030. Per-galaxy division is unstable.",
     "Sec. 6.4: the ensemble R(σ) calibration brings m to within ~5%."),
    ("T2b (leak?)", "Is anything off in the stencil?",
     "The narrowest posteriors (σ_ε < 0.03, nearly all cos i > 0.9) have R ≈ 1.4 > 1. Hypothesis: fibers are placed from the noiseless true sheared axes, which leaks the shear direction for face-on disks.",
     "Test before claiming face-on performance; mention in limitations."),
    ("T4 / T5 / T7", "Are the fit conventions robust?",
     "The low-|g| fit is fine (m moves ≤ 0.05 across fit windows). Bootstrap/fit error ratio is 0.95–1.07. Std vs 16–84% half-width changes thresholds by 1–17% but not the conclusions.",
     "Quote fit errors. Use the std (Xu24) and state it."),
    ("T8", "Does stacking NPE likelihoods at constant shear remove the bias?",
     "Yes. On the fixed-shear benchmark (25 cells × 32 galaxies × 16 noise realizations; KDE of q(g|D_i) on a 0.0005 grid, no snapping), the stacked posterior mean at N = 32, R = 16 gives m1 = −0.013 ± 0.006, m2 = −0.000 ± 0.005 (c ≈ 1e-4), vs −0.407/−0.371 for the per-galaxy mean. Already ≈ 0 at R = 1. Insensitive to bandwidth ×0.5/×2 and to the grid. The moment-Gaussian approximation is biased by box truncation (−0.027).",
     "Sec. 6.5: stacking removes the shrinkage to ~1% (residual m1 ≈ 2σ). Caveats: repeated noise of one galaxy over-counts information (posterior too narrow at R = 16; fine at R = 1); the benchmark uses training-prior nuisances, not TF-conformed ones, so the TF-weighted case needs a new benchmark; needs N×R ≳ 32; the benchmark is brighter than Xu3."),
]


def figs():
    folder = ROOT / "paper_figs"
    names = sorted(p.name for p in folder.glob("*.png"))
    return "".join(
        f"<figure style='display:inline-block;width:30%;margin:.3rem;vertical-align:top'>"
        f"<a href='paper_figs/{n}'><img src='paper_figs/{n}' style='width:100%'></a>"
        f"<figcaption style='font-size:12px'>{html.escape(n)}</figcaption></figure>"
        for n in names
    )


def main():
    links = "".join(
        f"<li><a href='{p}'>{html.escape(t)}</a>"
        f"{'' if (ROOT / p).exists() else ' (pending)'}</li>"
        for p, t in GROUPS
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
        for r in ROWS
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>paper-tests campaign</title>
<style>
body {{ font: 16px/1.5 Palatino, serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; font: 14px/1.4 ui-sans-serif, sans-serif; }}
th, td {{ border-bottom: 1px solid #ccc; padding: .35rem .45rem; text-align: left; vertical-align: top; }}
</style></head><body>
<h1>paper-tests campaign</h1>
<p>Pre-writing tests for Sec. 5 (NPE validation) and Sec. 6 (shear estimation)
of the KL-NN paper. Model: <code>CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair</code>
(100k-trained, CCL-pretrained frozen encoder). Test set: Xu3 only. Estimator:
TF-weighted posterior mean. Shape noise follows Xu+24,
σ<sub>ε,i</sub> = √(σ<sub>g1</sub>² + σ<sub>g2</sub>²), from posterior
standard deviations. m and c come from a linear fit over |g| &lt; 0.02. Plan:
<code>kl-nn/repo_reports/PAPER_TESTS_PLAN.md</code>. Scripts:
<code>kl-nn/arch/diagnostics/paper_tests/</code>.</p>
<ul>{links}</ul>
<h2>Summary</h2>
<table><thead><tr><th>Test</th><th>Question</th><th>Answer</th><th>Implication for the paper</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Paper-candidate figures</h2>
{figs()}
</body></html>"""
    (ROOT / "index.html").write_text(page)
    print("wrote", ROOT / "index.html")


if __name__ == "__main__":
    main()
