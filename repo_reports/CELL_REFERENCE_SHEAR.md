# The reference shear of a stacked cell: derivation

Written 2026-10-23 for the cluster substructure test (`h_cluster_analysis.py: analyse_cells`). It derives what
the stacked cell estimate converges to when the true shear varies within a cell, and why we compare it with the
precision-matrix-weighted mean of the true shears. Equations are LaTeX (`$…$`, `$$…$$`); see the end of
`KAPPA_FITTING_EXPLAINED.md` for how to view them rendered.

## 1. Setup and notation

- A cell holds galaxies $i = 1, \dots, N$ at positions $\mathbf{x}_i$.
- Each galaxy has a true reduced shear $\mathbf{g}_i = (g_1, g_2)_i$ in the image frame. It comes from GalSim at
  $\mathbf{x}_i$, with $\gamma$ and $\kappa$ summed over the halos before forming $g = \gamma/(1-\kappa)$.
- Each galaxy's NPE gives a shear likelihood $L_i(\mathbf{g}) \propto q_i(\mathbf{g}) / \pi_{\rm train}(\mathbf{g})$,
  stored on a grid.
- **Cell frame.** Polar cells are stacked in tangential/cross components about the main-halo centre. Galaxy $i$ sits
  at position angle $\phi_i$, and its $(g_t, g_\times)$ are obtained with

$$
R_i = -\begin{pmatrix} \cos 2\phi_i & \sin 2\phi_i \\ -\sin 2\phi_i & \cos 2\phi_i \end{pmatrix},
\qquad (g_t, g_\times)_i^{\mathsf T} = R_i\, \mathbf{g}_i ,
$$

  i.e. $g_t = -\mathrm{Re}(g\, e^{-2i\phi})$ and $g_\times = -\mathrm{Im}(g\, e^{-2i\phi})$. $R_i$ is orthogonal
  ($R_i R_i^{\mathsf T} = I$, $\det R_i = 1$). For Cartesian cells, $R_i = I$.
- **True shears in the cell frame:** $\mathbf{t}_i \equiv R_i\, \mathbf{g}_i$. These differ from galaxy to galaxy,
  because the shear varies inside the cell.

## 2. What the stack computes

The stack assumes, falsely, that all galaxies in the cell share one cell-frame shear $\mathbf{G}$. Galaxy $i$ then
has image-frame shear $R_i^{\mathsf T}\mathbf{G}$, and the stacked posterior is

$$
p(\mathbf{G} \mid \text{cell}) \;\propto\; \pi(\mathbf{G}) \prod_i L_i\!\left(R_i^{\mathsf T}\mathbf{G}\right),
\qquad \pi \text{ uniform on } |\mathbf{G}| < 0.1 .
$$

We report its mean, $\hat{\mathbf{G}}$, and its covariance, $\Sigma_{\hat G}$ (the error bars).

## 3. Gaussian-likelihood approximation

Suppose each likelihood is Gaussian in the image frame,

$$
L_i(\mathbf{g}) \;\propto\; \exp\!\left[-\tfrac12 (\mathbf{g}-\hat{\mathbf{g}}_i)^{\mathsf T} F_i\, (\mathbf{g}-\hat{\mathbf{g}}_i)\right],
$$

with peak $\hat{\mathbf{g}}_i$ (the galaxy's noisy shear measurement) and precision $F_i = C_i^{-1}$ (inverse
covariance). Two assumptions go into this:

- **(A1) Unbiased, linear response:** $\mathbb{E}[\hat{\mathbf{g}}_i] = \mathbf{g}_i$, averaging over the noise.
- **(A2) Fixed width:** $F_i$ does not depend on $\mathbf{g}$. The likelihood shifts with the shear but keeps its
  shape (a location family).

**Rotate to the cell frame.** Write $\mathbf{d}_i \equiv R_i\hat{\mathbf{g}}_i$ for galaxy $i$'s measurement in the
cell frame. Because $R_i$ is orthogonal, $R_i^{\mathsf T}\mathbf{G} - \hat{\mathbf{g}}_i = R_i^{\mathsf T}(\mathbf{G}-\mathbf{d}_i)$.
The exponent becomes

$$
(R_i^{\mathsf T}\mathbf{G}-\hat{\mathbf{g}}_i)^{\mathsf T} F_i\,(R_i^{\mathsf T}\mathbf{G}-\hat{\mathbf{g}}_i)
= (\mathbf{G}-\mathbf{d}_i)^{\mathsf T}\,(R_i F_i R_i^{\mathsf T})\,(\mathbf{G}-\mathbf{d}_i)
\equiv (\mathbf{G}-\mathbf{d}_i)^{\mathsf T} F_i'\,(\mathbf{G}-\mathbf{d}_i),
$$

so in the cell frame galaxy $i$'s precision is $F_i' = R_i F_i R_i^{\mathsf T}$.

**Multiply the likelihoods.** Ignoring the prior bound for now (assumption A3 below), the log of the product is a sum
of quadratics in $\mathbf{G}$:

$$
\ln p(\mathbf{G}) = -\tfrac12 \sum_i (\mathbf{G}-\mathbf{d}_i)^{\mathsf T} F_i'\,(\mathbf{G}-\mathbf{d}_i) + \text{const}.
$$

Set the gradient to zero,

$$
\sum_i F_i'\,(\mathbf{G}-\mathbf{d}_i) = 0,
$$

to get the stacked estimate and its covariance:

$$
\hat{\mathbf{G}} = F_{\rm tot}^{-1} \sum_i F_i'\,\mathbf{d}_i,
\qquad F_{\rm tot} \equiv \sum_i F_i',
\qquad \Sigma_{\hat G} = F_{\rm tot}^{-1}.
$$

For a Gaussian posterior the mean equals the peak, so this is also what the grid mean returns.

## 4. What $\hat{\mathbf{G}}$ converges to: the reference shear

Average over noise realizations. With (A1),
$\mathbb{E}[\mathbf{d}_i] = R_i\,\mathbb{E}[\hat{\mathbf{g}}_i] = R_i\mathbf{g}_i = \mathbf{t}_i$. With (A2), the
weights $F_i'$ don't depend on the noise, so the average passes through them:

$$
\mathbb{E}[\hat{\mathbf{G}}] = F_{\rm tot}^{-1} \sum_i F_i'\,\mathbf{t}_i \;\equiv\; \mathbf{g}_{\rm ref}.
$$

This is the **reference shear**: the precision-matrix-weighted mean of the galaxies' true cell-frame shears.

**Its scatter.** Assume the noises of different galaxies are independent, with
$\mathrm{Cov}(\hat{\mathbf{g}}_i) = C_i = F_i^{-1}$, so $\mathrm{Cov}(\mathbf{d}_i) = R_i C_i R_i^{\mathsf T} = F_i'^{-1}$. Then

$$
\mathrm{Cov}(\hat{\mathbf{G}})
= F_{\rm tot}^{-1}\Big(\sum_i F_i' F_i'^{-1} F_i'\Big) F_{\rm tot}^{-1}
= F_{\rm tot}^{-1}\Big(\sum_i F_i'\Big) F_{\rm tot}^{-1}
= F_{\rm tot}^{-1} = \Sigma_{\hat G}.
$$

So, under (A1)–(A3),

$$
\hat{\mathbf{G}} - \mathbf{g}_{\rm ref} \sim \mathcal{N}(0, \Sigma_{\hat G}),
$$

and the per-component pull $(\hat G_k - g_{{\rm ref},k}) / \sqrt{(\Sigma_{\hat G})_{kk}}$ is a standard normal. That is
the property the synthetic validation tests: pull mean 0, spread 1.

**Two remarks.**

1. **The spread of the true shears doesn't enter the error bar.** $\Sigma_{\hat G} = F_{\rm tot}^{-1}$ is the same
   however much the $\mathbf{t}_i$ differ, because the stack never "sees" that variation. It is still the correct
   error about $\mathbf{g}_{\rm ref}$, because $\mathbf{g}_{\rm ref}$ is exactly what $\hat{\mathbf{G}}$ estimates. It
   would *not* be the right error about some other choice of cell truth, such as the shear at the cell centre.
2. **This is the general statement for a misspecified model.** When you fit a model (here: constant shear in the
   cell) that is wrong, the fit converges to the parameter that best approximates the truth. For Gaussian errors
   that is the weighted least-squares projection above (White 1982). For non-Gaussian likelihoods the pseudo-true
   value is the $\mathbf{G}$ that maximises $\sum_i \mathbb{E}\big[\ln L_i(R_i^{\mathsf T}\mathbf{G})\big]$, which
   reduces to $\mathbf{g}_{\rm ref}$ in the Gaussian case.

## 5. Why not a simpler reference?

- **Plain mean, $\frac1N \sum_i \mathbf{t}_i$.** Correct only if every $F_i'$ is the same. KL galaxies differ a lot in
  precision, so this would mislabel a correct stack as biased. We store it as `plain_mean` for comparison.
- **Scalar weights, $\sum_i w_i\mathbf{t}_i / \sum_i w_i$.** Correct only if each $F_i' \propto I$. KL likelihoods are
  strongly anisotropic: the kinematics pin one shear direction much better than the other, so $F_i$ has very unequal
  eigenvalues and a galaxy-dependent orientation. With matrix weights the two components also mix:
  $g_{{\rm ref},t}$ depends on the galaxies' true $g_\times$ through the off-diagonal terms of $F_i'$, and vice versa.
  A scalar weight can't capture that.
- **Shear at the cell centre (or the area average).** That is what a map "means", but the stack doesn't estimate
  it. The difference $\mathbf{g}_{\rm ref} - \mathbf{g}(\text{centre})$ is a deterministic binning effect, largest
  where the shear changes quickly (near the halo centre) and in big cells.

## 6. How the code computes it

`analyse_cells` in `h_cluster_analysis.py`:

1. $C_i$ = the covariance of galaxy $i$'s normalised likelihood on the $401 \times 401$ grid (`grid_moments`).
2. $F_i' = (R_i C_i R_i^{\mathsf T})^{-1}$ and $\mathbf{t}_i = R_i\mathbf{g}_i$.
3. $\mathbf{g}_{\rm ref} = \big(\sum_i F_i'\big)^{-1} \sum_i F_i'\,\mathbf{t}_i$.
4. $\hat{\mathbf{G}}$ and $\Sigma_{\hat G}$ are the mean and covariance of the stacked posterior on a $201 \times 201$
   grid over $|\mathbf{G}| < 0.1$. This is computed from the full product of the KDE likelihoods, not from the Gaussian
   approximation.

Doubling both grids (likelihoods $401^2 \to 801^2$, stacks $201^2 \to 401^2$) changes $\hat{\mathbf{G}}$ by
$\sim 10^{-6}$ on average and at most $4.5\times10^{-5}$ per cell (tested on s3_sub1, 2026-10-25).

## 7. Where the approximations can fail

- **(A1) nonlinear or biased response.** If a galaxy's likelihood doesn't shift one-for-one with the true shear
  ($\mathbb{E}[\hat{\mathbf{g}}_i] \neq \mathbf{g}_i$, e.g. the cubic term at large $|g|$), then
  $\mathbb{E}[\hat{\mathbf{G}}] \neq \mathbf{g}_{\rm ref}$. That is exactly the bias the NPE run is meant to measure, so
  it's a feature of the test, not a flaw.
- **(A2) shape changing with $g$.** If the likelihood's width or shape depends on the shear, the weights pick up the
  noise and $\mathbb{E}[\hat{\mathbf{G}}]$ shifts slightly. With strongly non-Gaussian NPE likelihoods the Gaussian
  formula for $\mathbf{g}_{\rm ref}$ is an approximation. The grid-based $C_i$ already includes each likelihood's
  actual width, including the truncation at the box edge.
- **(A3) the prior bound.** The stack has a uniform prior on $|\mathbf{G}| < 0.1$. When $F_{\rm tot}$ is small (1–2
  broad likelihoods), the posterior is cut off by that bound and its mean is pulled toward zero. This is the shrinkage
  seen in the validation. Round 1, whose synthetic likelihoods were too narrow, gave about 0.005 for 1–2 galaxy cells.
  Round 2 (s3_sub1, realistic widths) gives $-0.010$ for 2-galaxy polar cells and $-0.002$ for 3. The 50-realization
  NPE ensemble gives $-0.011$ and $-0.005$ for 1 and 2 galaxies, and a precision-weighted $+0.0004 \pm 0.0003$ for
  cells with $n \ge 3$.
- **Covariance from a truncated likelihood.** For a galaxy whose likelihood is cut off by the box edge, $C_i$ is
  smaller than the width of the untruncated Gaussian, which over-weights that galaxy in $\mathbf{g}_{\rm ref}$. In
  round 2, a few cells sit 3–4σ off $\mathbf{g}_{\rm ref}$ when averaged over 30 realizations (offsets $\le 0.012$, in
  2–3 galaxy cells). Recomputing $\mathbf{g}_{\rm ref}$ with the exact, untruncated synthetic precisions did **not**
  remove them, so they come from the stack itself (the prior bound, or the KDE's reflection at the box edge), not from
  this weighting. Across cells with at least 3 galaxies they amount to about 0.15 of a cell's error bar. Their cause
  is still open; Sec. 8.4 has a possible reading in BA14's terms.

## 8. Connection to Bernstein & Armstrong (2014)

BA14 (MNRAS 438, 1880; arXiv:1304.1843), Sec. 2, sets out the Bayesian shear inference our stacking follows. Their
§2.2 extends it to non-constant shear. This section maps their notation onto ours. Equation numbers are BA14's.

### 8.1 Constant shear: BA14 Sec. 2.1

They start from the posterior of one shear $\mathbf{g}$ shared by $N$ galaxies with independent data $D_i$ (their
eq. 2):

$$
P(\mathbf{g} \mid D) \;\propto\; P(\mathbf{g}) \prod_i P(D_i \mid \mathbf{g}).
$$

**Our stack is this equation.** Our likelihood $L_i(\mathbf{g}) = q_i(\mathbf{g}\mid D_i)/\pi_{\rm train}(\mathbf{g})$
is $P(D_i \mid \mathbf{g})$ up to a factor that doesn't depend on $\mathbf{g}$. The NPE was trained under
$\pi_{\rm train}$, so dividing its posterior by $\pi_{\rm train}$ leaves the likelihood. The only differences are that
our cell stack works in each galaxy's rotated frame (Sec. 1) and its prior is uniform on $|\mathbf{G}| < 0.1$.

BA14 then assume that **the posterior, not each galaxy's likelihood**, is close to Gaussian, and Taylor-expand each
likelihood about $\mathbf{g} = 0$ to second order (their eqs. 3–4):

$$
P_i = P(D_i \mid \mathbf{0}),
\qquad \mathbf{Q}_i = \nabla_{\mathbf{g}} P(D_i \mid \mathbf{g})\big|_{\mathbf{0}},
\qquad \mathbf{R}_i = \nabla_{\mathbf{g}}\nabla_{\mathbf{g}} P(D_i \mid \mathbf{g})\big|_{\mathbf{0}},
$$

$$
P(D_i \mid \mathbf{g}) \approx P_i + \mathbf{g}\cdot\mathbf{Q}_i + \tfrac12\, \mathbf{g}\cdot\mathbf{R}_i\cdot\mathbf{g}.
$$

Summing the logs and dropping the prior gives a Gaussian posterior with (their eqs. 6–7)

$$
C_g^{-1} = \sum_i \left( \frac{\mathbf{Q}_i\mathbf{Q}_i^{\mathsf T}}{P_i^2} - \frac{\mathbf{R}_i}{P_i} \right),
\qquad \bar{\mathbf{g}} = C_g \sum_i \frac{\mathbf{Q}_i}{P_i}.
$$

**Translation.** The two combinations that appear are derivatives of $\ln P(D_i \mid \mathbf{g})$ at $\mathbf{g}=0$:

$$
\frac{\mathbf{Q}_i}{P_i} = \nabla \ln P_i\big|_{\mathbf{0}},
\qquad \frac{\mathbf{Q}_i\mathbf{Q}_i^{\mathsf T}}{P_i^2} - \frac{\mathbf{R}_i}{P_i} = -\nabla\nabla \ln P_i\big|_{\mathbf{0}} \equiv \mathbf{A}_i .
$$

For our Gaussian approximation $\ln L_i = -\tfrac12(\mathbf{g}-\hat{\mathbf{g}}_i)^{\mathsf T}F_i(\mathbf{g}-\hat{\mathbf{g}}_i)$ (Sec. 3):

$$
\mathbf{A}_i = F_i \qquad\text{and}\qquad \nabla \ln L_i\big|_{\mathbf{0}} = F_i\,\hat{\mathbf{g}}_i ,
$$

so BA14's eqs. 6–7 become

$$
C_g^{-1} = \sum_i F_i, \qquad \bar{\mathbf{g}} = \Big(\sum_i F_i\Big)^{-1} \sum_i F_i\,\hat{\mathbf{g}}_i ,
$$

which is our $\hat{\mathbf{G}}$ and $\Sigma_{\hat G}$ (Sec. 3) with $R_i = I$. **BA14's $\mathbf{A}_i$ is our precision
matrix $F_i$.** In the cell frame both transform the same way: $\mathbf{Q}_i \to R_i\mathbf{Q}_i$ and
$\mathbf{A}_i \to R_i\mathbf{A}_iR_i^{\mathsf T} = F_i'$.

**What we do differently:**
- **We don't Taylor-expand.** BA14 need the quadratic form so that each galaxy can be summarised by six numbers
  ($P$, $\mathbf{Q}$, $\mathbf{R}$). We keep each galaxy's whole likelihood on a $401^2$ grid and multiply the grids
  directly. This matches BA14 when their quadratic approximation holds, and automatically includes the higher-order
  terms they add as perturbations in their Appendix A, which they say are needed at the $10^{-3}$ level for
  $g \approx 0.03$.
- **We expand nowhere in particular.** BA14 expand about $\mathbf{g}=0$, the weak-shear limit. Our grid product is
  exact at any $|g| < 0.1$. Only the *reference* $\mathbf{g}_{\rm ref}$ uses the Gaussian form, through $F_i$.
- **$P(D_i\mid\mathbf{g})$ comes from a network.** BA14 compute it from a prior on the unlensed galaxy population,
  using either model fitting or Fourier-domain moments. Ours comes from the NPE (posterior ÷ training prior), so the
  galaxy-population prior, here the TF relation and the training distribution, is built in through training.

### 8.2 Non-constant shear: BA14 Sec. 2.2, in their words and in ours

**What BA14 say.** Once $P_i$, $\mathbf{Q}_i$ and $\mathbf{R}_i$ are computed for each galaxy, the posterior of *any*
shear-field model follows, as long as the model's shears stay in the regime where the quadratic expansion holds. For a
model with parameters $\mathbf{a}$ that predicts a shear $\mathbf{g}_i(\mathbf{a})$ at each galaxy (their eq. 8,
written per galaxy):

$$
-\ln P(\mathbf{a} \mid D) \approx \text{const} - \ln P(\mathbf{a})
- \sum_i \mathbf{g}_i(\mathbf{a}) \cdot \frac{\mathbf{Q}_i}{P_i}
+ \frac12 \sum_i \mathbf{g}_i(\mathbf{a}) \cdot \mathbf{A}_i \cdot \mathbf{g}_i(\mathbf{a}).
$$

As printed, their eq. 8 also has a $\sum_i$ inside the bracket of the quadratic term. The per-galaxy form above is what
you get by applying their eq. 5 galaxy by galaxy, which is how the text describes it.

They make three further points:

1. **A linear model with a Gaussian prior has a closed-form solution.** If $\mathbf{g}_i$ is linear in $\mathbf{a}$ and
   the prior is quadratic, the most probable $\mathbf{a}$ solves one matrix equation.
2. **A Fourier example (their eqs. 9–10).** Write the field as a sum of Fourier modes,
   $\mathbf{g}_i = \sum_\mu \mathrm{Re}\big(\mathbf{a}_\mu e^{i\mathbf{k}_\mu\cdot\mathbf{x}_i}\big)$. If the sources are
   spread uniformly enough that $\sum_i \mathbf{A}_i e^{i\mathbf{k}\cdot\mathbf{x}_i} \approx 0$ for $\mathbf{k}\neq 0$,
   the posterior separates mode by mode. Each Fourier coefficient then has an independent $2\times2$ Gaussian
   posterior, with covariance set by the same $C_g$ as the constant-shear case. The same holds for any decomposition
   into orthogonal functions.
3. **Correlation functions** of the shear field should be similarly straightforward, at least when shape and
   measurement noise dominate the sample variance. They leave this for future work.

**What it means.** A galaxy's information about the shear is a local quadratic form centred on its own shear, the same
whatever produced that shear. To fit a shear field, you don't need a "shear estimate per galaxy". You evaluate each
galaxy's likelihood at the shear the model predicts *for that galaxy* and add up the logs. A constant shear is the
special case $\mathbf{g}_i(\mathbf{a}) = \mathbf{a}$ for every galaxy.

### 8.3 How our analyses fit into BA14 Sec. 2.2

All three of our cluster analyses are their eq. 8 with a different model $\mathbf{g}_i(\mathbf{a})$, and with the full
$\ln L_i$ in place of its quadratic expansion:

| Our analysis | Parameters $\mathbf{a}$ | Model $\mathbf{g}_i(\mathbf{a})$ | Prior $P(\mathbf{a})$ |
|---|---|---|---|
| Cell stacking (this note) | one shear $\mathbf{G}_c$ per cell | $R_i^{\mathsf T}\mathbf{G}_{c(i)}$: constant within each cell | uniform, $\lVert\mathbf{G}\rVert < 0.1$ |
| κ map (`KAPPA_FITTING_EXPLAINED.md`) | κ in 144 pixels | KS kernel, then $g = \gamma/(1-\kappa)$ at $\mathbf{x}_i$ | Gaussian: amplitude + smoothness |
| Halo fit (built, not used) | $M_{\rm main}$ (+ $M_{\rm sub}$) | NFW reduced shear at $\mathbf{x}_i$ | log-uniform masses |

- **Cells.** A piecewise-constant model makes eq. 8 split into independent cell stacks, Sec. 8.1 applied per cell. The
  true field is not piecewise constant, so the cell model is *misspecified* (Sec. 4). In BA14's terms, assume each
  galaxy's expected score responds linearly to its own true cell-frame shear $\mathbf{t}_i$:
  $\mathbb{E}[\mathbf{Q}_i/P_i] \approx \mathbf{A}_i\mathbf{t}_i$. This is the per-galaxy version of the condition in
  their Sec. 2.3, where, for a population at one shear $\mathbf{g}$, the large-$N$ posterior is centred on
  $\mathbf{A}^{-1}\mathbf{b}$ with $\mathbf{b} = \langle\mathbf{Q}_i/P_i\rangle$, and must equal $\mathbf{g}$. It is our
  assumption (A1). Then the stack converges to

$$
\mathbb{E}[\hat{\mathbf{G}}] \to \Big(\sum_i \mathbf{A}_i\Big)^{-1} \sum_i \mathbf{A}_i\,\mathbf{t}_i = \mathbf{g}_{\rm ref},
$$

  the same reference shear as Sec. 4, now in BA14 notation.
- **κ map.** This is eq. 8 with $\mathbf{a}$ = the κ pixels, a nearly linear model (KS kernel plus the $1/(1-\kappa)$
  correction), and a quadratic prior. By BA14's point 1, if the model were exactly linear and each likelihood exactly
  quadratic, the MAP map would have a closed form, and that closed form is the Wiener filter. Their Fourier example is
  the same idea with Fourier modes as the basis and no prior. We solve numerically because the full $\ln L_i$ and
  $g = \gamma/(1-\kappa)$ are not quite quadratic and linear.
- **Halo fit.** This is eq. 8 with an NFW model, the most direct use of their §2.2.

### 8.4 BA14's finite-N bias (their Sec. 2.3) and our sparse cells

BA14 show that the posterior mean $\bar{\mathbf{g}}$ carries a small bias at finite $N$ (their eq. 15). It comes from
the scatter of $\mathbf{A}_i$ and $\mathbf{Q}_i/P_i$ from galaxy to galaxy, and from their correlation. To leading
order the bias is $O(1/N)$, smaller than the statistical error $\sigma_g/\sqrt N$ by about $\sqrt N$. That bias concerns
large $N$; for a cell with 2–3 galaxies the $1/N$ expansion doesn't hold. Still, the same mechanism, together with the
$|\mathbf{G}| < 0.1$ prior bound (A3), is a natural candidate for the sparse-cell offsets in Sec. 7. They also shrink
quickly with $N$ ($-0.010$ at $n=2$, $-0.002$ at $n=3$). One test would tell them apart: rerun the validation without
the prior bound and without the KDE edge reflection.

## References

- Bernstein, G. M. & Armstrong, R. 2014, *Bayesian lensing shear measurement*, MNRAS 438, 1880. arXiv:1304.1843
  (`BA14` in main.bib). Sec. 2.1 (eqs. 2–7), Sec. 2.2 (non-constant shear, eqs. 8–10), Sec. 2.3 (bias, eqs. 11–15),
  Appendix A (third-order terms).
- White, H. 1982, *Maximum likelihood estimation of misspecified models*, Econometrica 50, 1.
