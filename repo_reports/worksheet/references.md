# Citation audit for the cluster worksheet

Every citation in the worksheet's Given boxes (`citations.tex`), with the line of the source it points to. Each was
read in the source text on 2026-10-26: `pdftotext -layout` of the arXiv or author PDF, or rendered page images where
the text layer was missing or garbled (KS93, Zaroubi 1995, Madsen et al. 2004, Kleijn & van der Vaart 2012).
Downloaded copies: `/tmp/claude-99649/…/scratchpad/refs/<paper>/` (temporary).

## Lensing

| Macro | Used in | Source, section, equation | Quoted line (≤20 words) |
|---|---|---|---|
| `\cBSpsi` | C1 | B&S01 §3.1.3 Magnification and Distortion, Eq. (3.12); Kilbinger15 §3.3, Eq. (19) | B&S: "where we have introduced the components of the shear γ ≡ γ1 + iγ2 = \|γ\|e^{2iϕ}"; Kilbinger: "This defines the convergence and shear as second derivatives of the potential" |
| `\cBSpoisson` | C4, D1 | B&S01 §3.1.2 The Lens Equation, unnumbered, after Eq. (3.9); S96 §2.1, Eqs. (3a)–(3b) | "two-dimensional analogue of the Newtonian gravitational potential and satisfies the Poisson equation ∇²ψ(θ) = 2κ(θ)" |
| `\cBSreduced` | C3 | B&S01 §4.2 Galaxy Shapes and Sizes, and their Transformation, Eq. (4.7); Kilbinger15 §3.3, Eq. (20) | "where the asterisk denotes complex conjugation, and g is the reduced shear" |
| `\cBSmsd` | C3, D3 | B&S01 §5.2.2, Eq. (5.9) and the text after it | "Hence, the reduced shear g = γ(1 − κ)⁻¹ is invariant under the transformation (5.9)" |
| `\cBSks` | D2, D3 | B&S01 §3.1.3, Eq. (3.15); §5.2.1 Linear Inversion of Shear Maps, Eqs. (5.1)–(5.4); KS93 §2.1, Eqs. (2.1.12), (2.1.15) | "The constant κ0 in (5.4) appears because a constant surface mass density does not cause any shear" |
| `\cJeffreyKS` | D2 | J18 §2.2.1 Theory, Eqs. (12)–(14) | "The inverse Fourier transform then returns the convergence reconstruction in configuration space (Kaiser & Squires 1993)." |
| `\cBStan` | B5, C2 | B&S01 §5.3.1 Aperture Mass Measures, Eq. (5.25); Kilbinger15 §3.8, Eq. (31) | B&S: "where we have defined the tangential component γt of the shear relative to the point θ0"; Kilbinger: "The minus sign, by convention, results in a positive value of γt for the tangential alignment around a mass overdensity." |
| `\cBSgtmean` | C2 | Kilbinger15 §3.12 Galaxy-galaxy lensing, Eq. (44); B&S01 §3.1.5, footnote 5 | Kilbinger: "This result is very general as it makes no assumption about the actual mass distribution."; B&S: "the magnitude of the shear can be calculated from \|γ\|(θ) = κ̄(θ) − κ(θ)" |
| `\cBSsis` | C1 | B&S01 §3.1.5 An Illustrative Example: Isothermal Spheres, Eqs. (3.17)–(3.18) | "and the deflection potential is ψ = θE \|θ\|. From that, the shear is obtained using (3.12)" |
| `\cWBnfw` | C5 | WB00 §2 Convergence and Shear of an NFW Object, Eqs. (11)–(16) | "Since the NFW density profile is spherically symmetric, the radial dependence of the shear can be written as" |
| `\cJeffreyWF` | E2 | J18 §2.3.1 Theory, Eqs. (17)–(18), (19)–(21) | "The Wiener filter is the linear minimum-variance solution…"; "so the maximum a posteriori (MAP) solution is that of the Wiener reconstruction." |
| `\cSchneiderMap` | G4 | S96 §2.2, Eqs. (9), (14)–(16); §3, Eq. (24); B&S01 §5.3.1, Eqs. (5.20)–(5.30) | S96: "Thus, we can define the signal-to-noise ratio S at position x0 as" |

## Statistics and inference

| Macro | Used in | Source, section, equation | Quoted line (≤20 words) |
|---|---|---|---|
| `\cHBLwls` | A1–A3, E1 | HBL10 §1 Standard practice, Eqs. (5), (7); the covariance is stated in the text after (7) | "the matrix [AᵀC⁻¹A]⁻¹ that appears in equation (5) is just the covariance matrix … for the parameters in X" |
| `\cHoggBayes` | A4 | H12 §1 Generalities, Eqs. (6), (9)–(11); §3 Posterior probabilities, Eqs. (37)–(38) | "These two factorizations taken together lead to what is sometimes called 'Bayes's theorem'" |
| `\cInterim` | A4, B3 | Sch15 §2.2.2 Hierarchical inference via importance sampling, Eqs. (13)–(15); HMB10 §2 Method, Eqs. (5)–(9) | Sch15: "We refer to Pr(ωn\|I0) … as 'interim priors'."; HMB10 eq. (9): "an importance-sampling approximation" |
| `\cMisspec` | B4 | KvdV12 §1 Introduction and §2.2, Eq. (2.7) | "θ∗ is the parameter value θ minimizing the Kullback-Leibler divergence θ ↦ P0 log(p0/pθ)" |
| `\cZaroubiWF` | E2 | Z95 §2b Wiener Filter, Eqs. (2–11a,b); §2c Conditional Probability | "it is easily shown that the MAP estimator coincides with the WF and conditional mean field i.e., s_MV = s_mean = s_MAP" |
| `\cZaroubiMSE` | E3 | Z95 §2b, Eqs. (2–3)–(2–5); §3b, Eq. (3–9) | "suppression the shot noise roughly by the ratio of prior/(prior + noise) (if S and N are diagonal)" |
| `\cMacKayLaplace` | F1 | ITILA03 Ch. 27 Laplace's Method, Eqs. (27.6)–(27.8) | "The idea behind the Laplace approximation is simple." |
| `\cMacKayGamma` | F4 | M92t Ch. 2 §2.4, Eq. (2.23); M92 §4, Eq. (23) (preprint) | "γ is thus a measure of the effective number of parameters which are well determined by the data." |
| `\cGaussNewton` | F2 | MNT04 Ch. 3, Eq. (3.4b); §3.1 The Gauss–Newton Method, Eq. (3.8) | "It is easily seen that the gradient and the Hessian of L are L′(h) = Jᵀf + JᵀJh, L″(h) = JᵀJ." |
| `\cBAresponse`, `\cBAconst`, `\cBAvarying`, `\cBAbias`, `\cBAthird` | B2, B3, G1, G2 | BA14 §1 Eq. (1); §2.1 Eqs. (2)–(7); §2.2 Eqs. (8)–(10); §2.3 Eqs. (11)–(15); Appendix A | Eq. (1): "ĝ = (1 + m)g + c"; §2.2: "the posterior for any shear model can be calculated as long as the model predicts shears in the regime where (4) holds" |

## Caveats found while checking

- **B&S01 never defines γ_×**, and writes κ = ½∇²ψ only as ∇²ψ = 2κ, so the worksheet adds Kilbinger15 for both.
- **B&S01 gives the κ̄ − κ relation only for \|γ\|** (footnote 5). The general, azimuthally averaged form is Kilbinger15
  Eq. (44), which the C2 Given box now quotes.
- **Kernel convention:** B&S01 and J18 write D̂ = π(k₁² − k₂² + 2ik₁k₂)/k² with a π⁻¹ in front; the worksheet absorbs
  the π. KS93 uses Σ = 2κ, a potential of opposite sign, and calls the wavevector κ.
- **J18 Eq. (21)** prints the posterior precision as (S⁻¹ + A N⁻¹ A†); the standard form is S⁻¹ + A†N⁻¹A. The worksheet
  cites J18 only for the statements, not for that formula.
- **J18 says "MAP" and "minimum variance"**, not "posterior mean"; Z95 states s_MV = s_mean = s_MAP.
- **ITILA03 does not contain** γ = Σλ/(λ+α). It comes from MacKay's 1992 thesis and paper. The thesis number (2.23) is
  certain; the journal paper's (23) was read from the preprint.
- **MNT04 treat unweighted least squares** (with F = ½‖f‖²). The weighted form JᵀFJ is marked as derived in the
  worksheet.
- **KvdV12 Corollary 4.2** appears to contain a typo ("→ 1"), so the worksheet cites §1 and Eq. (2.7) instead.
- **Not checked:** White (1982), Econometrica 50, 1. It is not openly available, and KvdV12 is cited in its place.
- **Derived here (no literature source):** A5 (truncation formula), B1 (pull distribution), B2 (Gaussian shrinkage
  m), D4 (cell cancellation), E4 (chain prior ↔ power spectrum), E5, F3 (noise vs posterior covariance), G3 (difference
  map), G4 (empirical-null z), G6 (look-elsewhere count). G5 uses our own results.
