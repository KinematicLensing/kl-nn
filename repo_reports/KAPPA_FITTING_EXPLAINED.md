# How the likelihood-based κ map works, from the ground up

Written 2026-10-24 for the cluster substructure test. Code: `arch/diagnostics/paper_tests/h_cluster_analysis.py`
(`likelihood_kappa`, `kaiser_squires`, `model_kappa`); tuning in `h_cluster_tune.py`. Equations are LaTeX; see
Sec. 13 for how to view them rendered.

Related notes:
- `MASS_MAP_BASICS.md`: cells versus pixels, and why shear is non-local. A gentler start than this note.
- `KAPPA_MAP_METHODS.md`: Kaiser–Squires versus our method, and earlier work using similar methods. Its settings and
  results sections predate round 2.
- `CELL_REFERENCE_SHEAR.md`: what a stacked cell estimates, and the connection to BA14.

This note goes through every step of the κ fit, explains the statistics it relies on (Wiener filtering,
regularisation, the Laplace approximation), and says what the resulting map can and can't be trusted for.

---

## 0. The short version

1. **Unknowns:** $\kappa$, the projected mass in units of the critical density, on a grid of $12\times12$ pixels, each
   0.5 Mpc across.
2. **Forward model:** for any trial κ map, compute the reduced shear $\mathbf{g}$ it predicts at each galaxy's exact
   position.
3. **Data term:** score the trial map with each galaxy's own NPE shear likelihood, evaluated at that predicted shear,
   and add up the log-likelihoods over all galaxies.
4. **Prior:** add a Gaussian prior that prefers small, smooth κ. This is what a Wiener filter does.
5. **Fit:** the map that maximises data term + prior is the MAP map. We find it with an optimiser (L-BFGS).
6. **Uncertainty:** a quadratic approximation around the MAP gives each pixel's noise σ, and hence the S/N maps.
7. **What to trust:** the prior shrinks and blurs the map, so pixel values are biased low. The map's *structure*,
   and above all the *difference* between two maps made the same way, is reliable.

---

## 1. What we start with

- **About 150 background galaxies**, each at a known position $\mathbf{x}_i$ in the field (radius 3 Mpc around the
  cluster).
- **One shear likelihood per galaxy**, $L_i(g_1, g_2)$. The NPE gives a posterior $q_i(\mathbf{g}\mid\text{data})$; we
  divide by the training prior $\pi_{\rm train}$, which is uniform on the box $|g_1|, |g_2| < 0.1$:

$$
L_i(\mathbf{g}) \;\propto\; \frac{q_i(\mathbf{g} \mid \text{data}_i)}{\pi_{\rm train}(\mathbf{g})}.
$$

  We turn the posterior samples into a smooth density with a kernel density estimate (KDE) and store $\log L_i$ on a
  $401\times401$ grid in $(g_1, g_2)$. Reading off $\log L_i$ at any shear is then a bilinear interpolation. (Doubling
  the grid changes the results by $\sim10^{-6}$; see `CELL_REFERENCE_SHEAR.md`, Sec. 6.)
- **What $L_i$ says:** how compatible each shear value is with galaxy $i$'s image and spectrum. A typical KL
  likelihood is about 0.05 wide per component and elongated. The kinematics pin one shear direction much better than
  the other.

The same $L_i$ goes into the cell stacking. The κ map is a different way of combining them.

---

## 2. Step 1: from a κ map to the shear at each galaxy (the forward model)

### 2.1 Why κ fixes the shear

Both κ and the shear γ are second derivatives of one function, the lensing potential $\psi$:

$$
\kappa = \tfrac12(\psi_{11} + \psi_{22}), \qquad
\gamma_1 = \tfrac12(\psi_{11} - \psi_{22}), \qquad
\gamma_2 = \psi_{12}.
$$

So once you know κ everywhere, γ is fixed. In Fourier space, where each derivative becomes a factor of the
wavevector $\mathbf{k}$, this becomes one multiplication per mode:

$$
\hat\gamma(\mathbf{k}) = D(\mathbf{k})\,\hat\kappa(\mathbf{k}),
\qquad D(\mathbf{k}) = \frac{k_1^2 - k_2^2 + 2i\,k_1k_2}{|\mathbf{k}|^2}
$$

(Kaiser & Squires 1993). $|D| = 1$. It doesn't change the size of a mode, only how the shear pattern is oriented. In
real space the same relation says that a lump of mass shears the galaxies *around* it, tangentially, with a strength
falling off as $1/r^2$. That is the non-local link from `MASS_MAP_BASICS.md`.

### 2.2 What the code does

1. **Pad.** Copy the $12\times12$ κ map into a $24\times24$ array of zeros. This assumes $\kappa = 0$ outside the
   field, and it keeps the FFT's periodic wrap-around from making the far edge of the field act on the near edge.
2. **FFT, multiply by $D$, inverse FFT.** This gives $\gamma_1$ and $\gamma_2$ on the grid.
3. **Interpolate to each galaxy.** Bilinear interpolation of $\gamma_1$, $\gamma_2$ and $\kappa$ to the galaxy's exact
   position. No binning happens anywhere.
4. **Reduced shear.** Galaxies respond to $g = \gamma/(1-\kappa)$, not to γ. We form
   $\mathbf{g}_i = \boldsymbol{\gamma}_i/(1-\kappa_i)$ exactly. This matters near the cluster centre, where κ reaches
   0.2.

The result is a function $\kappa \mapsto \{\mathbf{g}_i(\kappa)\}$ at all galaxies. It is *almost* linear in κ:
everything is linear except the $1/(1-\kappa)$ factor, which is a small correction where κ is small.

---

## 3. Step 2: scoring a trial map with the galaxies (the data term)

$$
\log \mathcal{L}(\kappa) = \sum_i \log L_i\big(\mathbf{g}_i(\kappa)\big).
$$

Each galaxy says how plausible its predicted shear is. Adding the log-likelihoods multiplies the likelihoods, which
is the correct way to combine independent galaxies. This is what the stacking does too, but here each galaxy gets its
*own* predicted shear instead of one common shear per cell.

It does three things for free:

- **Weighting.** A galaxy with a tight likelihood changes $\log L_i$ a lot when its predicted shear moves, so it pulls
  hard on the map; a broad one barely matters. No weights have to be chosen.
- **Direction.** Because $L_i$ is elongated, the galaxy constrains the shear component it measures well and is
  lenient about the other.
- **Exact positions.** Each galaxy is compared with the shear at its own position, not with a cell average.

### 3.1 Why per-galaxy NPE shrinkage doesn't enter

A single galaxy's *posterior mean* shear is shrunk toward zero (raw $m \approx -0.6$). That is because the likelihood
is wide compared with the training box, and cutting it off at the box edge drags its mean inward. The κ fit never uses
a posterior mean. It evaluates the likelihood *function* at small shears ($|g| \lesssim 0.05$, well inside the box),
where its shape is correct. That's the same reason the stacking is unbiased (stacked $m \approx -0.01$ against raw
$-0.6$). The cluster runs agree: the NPE subhalo signal in the difference maps matches the forecast from exact
Gaussian likelihoods.

---

## 4. Why the data alone are not enough (an ill-posed problem)

Finding κ from noisy shears is an **inverse problem**: we know how to go from cause (mass) to effect (shear), and want
to go back. Ours is badly ill-posed for three reasons.

1. **Too few numbers.** There are 144 unknown pixels and about 150 galaxies, each giving a noisy 2-component
   measurement.
2. **Noise far above the signal.** Each galaxy's shear is uncertain by about 0.05, while the subhalo changes the shear
   by about 0.01. A single galaxy can hardly tell whether the subhalo is there.
3. **Small scales are weakly constrained.** A pattern of κ that alternates from pixel to pixel produces shears that
   largely cancel at the galaxies. The data barely constrain such patterns, so an unregularised fit would fill them
   with noise.

If we maximised the data term alone, the result would be a map that fits the noise: a speckle of large positive and
negative pixels. This is the general problem of inverting a smoothing operator, and every mass-mapping method has to
deal with it. KS deals with it by smoothing afterwards; we do it with a prior.

An analogy: drawing a curve through scattered, noisy points. A curve flexible enough to go through every point wiggles
wildly, and the wiggles are noise. You get a sensible curve by telling the fit to prefer smooth curves. How strongly
you prefer smoothness sets how much detail survives.

---

## 5. Step 3: the prior

We add

$$
\log p(\kappa) = -\frac{1}{2\sigma_\kappa^2}\left[\sum_{\text{pixels } a} \kappa_a^2
+ \ell^2 \sum_{\text{neighbours } (a,b)} (\kappa_a - \kappa_b)^2\right].
$$

- **The amplitude term**, with scale $\sigma_\kappa$, says pixel values are typically no bigger than about
  $\sigma_\kappa$. Smaller $\sigma_\kappa$ means stronger pulling toward zero.
- **The smoothness term**, with length $\ell$ in pixels, penalises differences between neighbouring pixels. Larger
  $\ell$ means smoother maps.
- **Settings in use:** $\sigma_\kappa = 0.03$ and $\ell = 1$ pixel, with 0.5 Mpc pixels. These were chosen on synthetic
  data (Sec. 10).

### 5.1 What this prior means

This is a Gaussian random field prior. Both terms are quadratic in κ, so the prior says "κ is a Gaussian random field
with a particular covariance". In Fourier space the neighbour-difference term is (approximately) $\ell^2|\mathbf{k}|^2$
times the amplitude term, so the prior's power spectrum is roughly

$$
P_{\rm prior}(k) \;\propto\; \frac{\sigma_\kappa^2}{1 + \ell^2 k^2}.
$$

It is flat on large scales and falls off on scales smaller than about $\ell$. In words: the prior expects κ
fluctuations up to about $\sigma_\kappa$ in size, and expects very little structure on scales below the smoothing
length. This is exactly the kind of prior a Wiener filter uses.

---

## 6. What a Wiener filter is

### 6.1 The basic idea

Suppose you measure a signal with noise: data = signal + noise. Both the signal and the noise are random, and you know
roughly how big each is at every scale. In Fourier language, you know the signal power $S(k)$ and the noise power
$N(k)$ for each mode $k$. What is the best guess of the signal?

The **Wiener filter** (Wiener 1949) answers this for Gaussian signal and noise. It multiplies each Fourier mode of the
data by

$$
W(k) = \frac{S(k)}{S(k) + N(k)}.
$$

- **Where the signal dominates** ($S \gg N$, usually large scales), $W \approx 1$: keep the data as measured.
- **Where the noise dominates** ($S \ll N$, usually small scales), $W \approx 0$: throw it away and guess zero.
- **In between**, keep the fraction of the data you believe is signal.

So it is a smoothing filter whose shape is *set by the signal-to-noise at each scale* rather than chosen by hand. Of
all linear estimates, it has the smallest average squared error.

### 6.2 The same thing as a Bayesian estimate

Put a Gaussian prior on the signal (covariance $\mathsf S$) and a Gaussian likelihood on the data (noise covariance
$\mathsf N$). The posterior is then also Gaussian, and its mean and its peak (the MAP) are both the Wiener-filter
estimate. In matrix form, for data $\mathbf{d} = \mathsf{A}\mathbf{s} + \mathbf{n}$, where $\mathsf A$ is the response
of the data to the signal:

$$
\hat{\mathbf{s}} = \left(\mathsf S^{-1} + \mathsf A^{\mathsf T}\mathsf N^{-1}\mathsf A\right)^{-1}\mathsf A^{\mathsf T}\mathsf N^{-1}\mathbf{d}.
$$

Read it as two parts:

- **The data's information:** $\mathsf A^{\mathsf T}\mathsf N^{-1}\mathsf A$.
- **The prior's information:** $\mathsf S^{-1}$.

The estimate weighs the data against the prior, mode by mode. For a single Fourier mode with $\mathsf A = 1$ this
reduces to $W = S/(S+N)$ above. Wiener filtering is widely used in cosmology, for example for reconstructing
large-scale structure (Zaroubi et al. 1995) and for weak-lensing mass maps (Jeffrey et al. 2018).

### 6.3 The price: shrinkage

The Wiener filter is **biased toward zero** wherever the noise matters. A real feature on a scale where $W = 0.4$ comes
out at 40% of its true height. That is the trade it makes on purpose: a little bias in exchange for much less noise,
which lowers the total error. It is the same shrinkage, in a different place, as the per-galaxy posterior-mean
shrinkage in Sec. 6 of the paper.

### 6.4 Ours is a Wiener filter with three changes

1. **The data term isn't Gaussian.** Each galaxy enters through its NPE likelihood: non-Gaussian, elongated, bounded by
   the training box.
2. **The forward model isn't quite linear,** because of $g = \gamma/(1-\kappa)$.
3. **So there's no closed form.** We find the MAP numerically instead of applying a formula.

Close to the solution, the data term is nearly quadratic and the model nearly linear. The map then behaves like a
Wiener filter, and the formulas in Sec. 8 apply.

---

## 7. Step 4: finding the most probable map

We maximise

$$
\log p(\kappa \mid \text{data}) = \sum_i \log L_i\big(\mathbf{g}_i(\kappa)\big) + \log p(\kappa) + \text{const}
$$

over the 144 pixel values. Equivalently, we minimise its negative.

- **Optimiser:** **L-BFGS** (in PyTorch). It is a standard method for smooth problems with many unknowns. It follows
  the gradient, and builds up an approximation to the curvature from the last few steps to take larger, better-aimed
  steps.
- **Gradient:** PyTorch computes it exactly by automatic differentiation through the FFTs, the interpolation and the
  likelihood grids.
- **Starting point:** $\kappa = 0$.
- **Convergence:** we use 400 iterations. Rerunning with 3000 gives the same map to all printed digits, for both the
  NPE and synthetic data. A fit takes a few seconds.

The result is the **MAP map**.

---

## 8. Step 5: how uncertain is each pixel? (S/N maps)

### 8.1 The Laplace approximation

Near its peak, the log-posterior is approximately a downward parabola, a quadratic in κ. Its curvature matrix
$\mathsf H$ (the Hessian, $144\times144$) says how sharply the posterior falls off in each direction. Treating the
posterior as a Gaussian with covariance $\mathsf H^{-1}$ is the **Laplace approximation**.

### 8.2 Why we don't use the exact Hessian

$\log L_i$ is read from its grid by bilinear interpolation. Inside each grid cell that interpolation is flat along
$g_1$ and along $g_2$, so it has **zero curvature** in those directions. Its curvature lives only in the kinks between
cells. The exact Hessian of our objective therefore misses most of what the data say, and can even have negative
directions. In our first try, 12 pixels came out with $\sigma = 0$.

The fix is the standard **Gauss–Newton** form. It uses each galaxy's *known* likelihood width instead of the
interpolant's curvature:

$$
\mathsf H_d = \sum_i \mathsf J_i^{\mathsf T} F_i\, \mathsf J_i .
$$

- $\mathsf J_i$ ($2\times144$): how galaxy $i$'s predicted shear changes when each pixel changes,
  $\mathsf J_i = \partial\mathbf{g}_i/\partial\kappa$. This is the forward model, linearised; PyTorch computes it.
- $F_i$ ($2\times2$): galaxy $i$'s precision, the inverse of the covariance of its likelihood grid. It is the same
  matrix that weights the reference shear in the cell analysis.

$\mathsf H_d$ is the data's information about the map, the $\mathsf A^{\mathsf T}\mathsf N^{-1}\mathsf A$ of Sec. 6.2.
The prior adds its exact Hessian $\mathsf H_p$ (the prior is quadratic). Then $\mathsf H = \mathsf H_d + \mathsf H_p$.

### 8.3 Two different "uncertainties"

In the linear-Gaussian case of Sec. 6.2 we can work out exactly how the MAP map behaves. Let the true map be
$\kappa_{\rm true}$, and the data $\mathbf{d} = \mathsf A\kappa_{\rm true} + \mathbf{n}$ with noise covariance $\mathsf N$:

$$
\hat\kappa = \mathsf H^{-1}\mathsf A^{\mathsf T}\mathsf N^{-1}\mathbf{d},
\qquad \mathsf H = \mathsf H_d + \mathsf H_p,
\qquad \mathsf H_d = \mathsf A^{\mathsf T}\mathsf N^{-1}\mathsf A.
$$

- **Average over noise realizations:**
  $\mathbb{E}[\hat\kappa] = \mathsf H^{-1}\mathsf H_d\,\kappa_{\rm true} \equiv \mathsf R\,\kappa_{\rm true}$, where
  $\mathsf R$ is the *resolution matrix* (Sec. 9).
- **Scatter over noise realizations:**

$$
\mathrm{Cov}_{\rm noise}(\hat\kappa) = \mathsf H^{-1}\mathsf A^{\mathsf T}\mathsf N^{-1}\,\mathsf N\,\mathsf N^{-1}\mathsf A\,\mathsf H^{-1} = \mathsf H^{-1}\mathsf H_d\,\mathsf H^{-1}.
$$

- **Posterior covariance:** $\mathsf H^{-1}$.

The two covariances answer different questions:

- $\mathsf H^{-1}$ (posterior): "given these data, how unsure am I about the true κ?" It includes the uncertainty the
  prior itself allows: where the data say little, the true κ could be anything the prior permits.
- $\mathsf H^{-1}\mathsf H_d\mathsf H^{-1}$ (noise): "if I repeated the observation with new noise, how much would
  *this map* change?" It is always smaller, because the prior also damps how much noise gets into the map.

For "is this blob real or a noise fluctuation?", the noise covariance is the right one. The S/N maps use

$$
\text{S/N per pixel} = \frac{\hat\kappa}{\sigma_{\rm noise}},
\qquad \sigma_{\rm noise} = \sqrt{\mathrm{diag}\!\left(\mathsf H^{-1}\mathsf H_d\mathsf H^{-1}\right)}.
$$

**Check against brute force:** the scatter of the subhalo-aperture value over 20 synthetic noise realizations is
0.0029. The noise covariance predicts 0.0034 (12% conservative); the posterior covariance would say 0.0067, about 2×
too large. Per pixel, $\sigma_{\rm noise} \approx 0.005$.

---

## 9. What the map gets wrong on purpose: bias and resolution

### 9.1 The resolution matrix

From Sec. 8.3, on average the MAP map is not the truth but $\mathsf R\,\kappa_{\rm true}$, with

$$
\mathsf R = \mathsf H^{-1}\mathsf H_d = \left(\mathsf H_d + \mathsf H_p\right)^{-1}\mathsf H_d .
$$

Each row of $\mathsf R$ says how one output pixel is built from the true κ around it:

- **Where the data are strong** compared with the prior (many precise galaxies), $\mathsf R \approx I$, the identity:
  the pixel is close to the truth.
- **Where the data are weak**, the row of $\mathsf R$ is a broad, low bump: the pixel is a *blurred, shrunk* average of
  the true κ nearby.

The width of those bumps is the map's **effective resolution**. The height is the shrinkage. Both are set jointly by
the galaxies (their number, positions and precision) and by the prior ($\sigma_\kappa$, $\ell$). The pixel size only
sets the finest scale the map *could* show.

### 9.2 "About one galaxy per pixel": what that means

Pixels are not estimated from the galaxies inside them. Every galaxy's shear depends on every pixel through the
non-local kernel, so the whole map is fitted jointly. A pixel with no galaxy is still constrained by galaxies around
it, only weakly. The answer to "does it resolve statistically?":

- Noise *does* average down: many galaxies each contribute a little to every pixel.
- The prior's bias does *not*: $\mathbb{E}[\hat\kappa] = \mathsf R\,\kappa_{\rm true} \neq \kappa_{\rm true}$ however
  many noise realizations you average. Individual pixel values are biased low by design, by the most where data are
  weakest.

### 9.3 How big the bias is here

- The reconstructed main halo near the subhalo position sits on average 0.0086 below the true κ (20 synthetic
  realizations, 0.5 Mpc aperture at 1 Mpc from the centre).
- In the NPE difference map of the detailed run, the subhalo's peak is 0.012, against a true pixel value of 0.030:
  about 40%.
- The main halo's central pixel is likewise far below its true value.

### 9.4 Why the difference map is the right thing to look at

In the linear picture, $\mathsf R$ is the *same* for both scenarios: same galaxies, same precisions, same prior. So

$$
\mathbb{E}\left[\hat\kappa_{\rm main+sub} - \hat\kappa_{\rm main}\right]
= \mathsf R\left(\kappa_{\rm true,\,main+sub} - \kappa_{\rm true,\,main}\right) = \mathsf R\,\kappa_{\rm subhalo}.
$$

The difference map is the subhalo's κ seen through the same resolution, and the main halo's shrinkage cancels. Because
the two scenarios also share their noise draws, most of the noise cancels too. What's left shows how visible the
subhalo makes itself in a map at this source density: where it appears, how big and how sharp.

To get unbiased *amplitudes*, for example the subhalo mass, you would fit a parametric model to the galaxies (the halo
fit, kept in the code) instead of reading values off the map.

---

## 10. How the settings were chosen

The pixel size, $\sigma_\kappa$ and $\ell$ set the balance between noise and blur (Sec. 9.1), so we chose them on
synthetic data before looking at the NPE run.

- **Synthetic data:** 20 noise realizations of one galaxy set (141 galaxies). Each galaxy's likelihood is a Gaussian
  with its NPE width (matched to within 6%), centred on a noisy copy of its true shear.
- **Grid:**
  - likelihood map: pixel 0.5, 0.75 or 1 Mpc; $\sigma_\kappa$ 0.03, 0.05 or 0.1; $\ell$ 0.5, 1 or 2 pixels;
  - KS: Gaussian smoothing σ of 0.5, 0.7, 1.0 or 1.5 cells.
- **Scores**, with aperture value $a$ = the reconstructed κ averaged over a 0.5 Mpc disk:
  - *contrast*: $\big(\overline{a_{\rm main+sub}} - \overline{a_{\rm main}}\big) / \mathrm{sd}(a_{\rm main})$ at the
    subhalo position, where the bar and sd are over realizations;
  - *false-peak rate*: the fraction of (realization, null aperture) pairs in the main-only maps whose value, relative to
    that aperture's mean, exceeds the contrast's numerator. There are 10 null apertures, on rings at 1 and 2 Mpc from
    the centre and at least 1.2 Mpc from the subhalo;
  - *RMS error*: of the reconstructed minus true κ (main + sub) over pixels inside the field.
- **Rule (fixed in advance):** the highest contrast among settings with a false-peak rate $\le 0.10$.
- **Result:**
  - likelihood map: 0.5 Mpc, $\sigma_\kappa = 0.03$, $\ell = 1$ (contrast 3.0, false-peak rate 0.00);
  - KS: 0.7 cells (contrast 1.25, false-peak rate 0.07).

The choice sits at the grid's edge (smallest pixel, smallest $\sigma_\kappa$), so a slightly stronger prior or finer
pixels might do a little better.

**Why a strong prior wins here:** $\sigma_\kappa = 0.03$ shrinks the subhalo but suppresses the noise even more. With
contrast as the target, the Wiener-filter trade (Sec. 6.3) tips toward smoothing at a source density this low.

---

## 11. For comparison: Kaiser–Squires

KS goes the other way, from shear to κ: $\hat\kappa(\mathbf{k}) = D^*(\mathbf{k})\,\hat\gamma(\mathbf{k})$. It puts one
stacked shear per Cartesian cell (0.98 Mpc) on a grid, applies the inverse kernel $D^*$ in Fourier space, and smooths
the result with a Gaussian of σ = 0.7 cells.

- **Strengths:** it is linear, with no fit, and its smoothing makes its maps quiet.
- **Weaknesses:**
  - it sees only cell averages;
  - every cell counts the same, whatever its galaxy count or precision;
  - it assumes $g \approx \gamma$;
  - it can't resolve anything smaller than a cell.

On the same data its tuned contrast is 1.25, against 3.0 for the likelihood map. Over 50 NPE realizations it detects
the subhalo above 2σ in 22% of them, against 60% for the likelihood map.

---

## 12. Glossary

- **κ (convergence):** projected mass density / critical density. It is what a mass map shows.
- **γ, g (shear, reduced shear):** the stretching of background galaxies; $g = \gamma/(1-\kappa)$ is what they actually
  respond to.
- **KS kernel $D$:** the Fourier-space factor turning κ into γ.
- **Likelihood $L_i$:** how compatible a shear value is with galaxy $i$'s data (posterior / training prior).
- **Prior / regularisation:** the extra term that makes an ill-posed fit well defined by preferring small, smooth
  maps.
- **Wiener filter:** the best linear estimate for Gaussian signal and noise; keeps each mode in proportion to its
  signal fraction $S/(S+N)$; equal to the Bayesian posterior mean under Gaussian assumptions.
- **MAP:** maximum a posteriori; the single most probable map.
- **L-BFGS:** the gradient-based optimiser used to find the MAP.
- **Hessian $\mathsf H$:** the curvature of the log-posterior at the peak; its inverse is the Laplace covariance.
- **Gauss–Newton:** approximating the data curvature by $\mathsf J^{\mathsf T}F\mathsf J$, from the model's derivatives
  and each galaxy's known precision.
- **Noise vs posterior covariance:** scatter of the map over noise realizations ($\mathsf H^{-1}\mathsf H_d\mathsf H^{-1}$)
  vs uncertainty about the true κ ($\mathsf H^{-1}$).
- **Resolution matrix $\mathsf R$:** what the map shows on average, $\mathsf R\,\kappa_{\rm true}$. It sets the effective
  resolution and the shrinkage.
- **Mass-sheet degeneracy:** shear can't fix the overall level of κ; only its shape (Bartelmann & Schneider 2001).

---

## 13. Viewing the equations

The math is written for KaTeX: inline as `$x$` (no space just inside the dollar signs) and display as `$$` blocks.

- **Cursor / VS Code preview:** open the file and press Ctrl+Shift+V (Cmd+Shift+V on Mac), or "Open Preview to the
  Side". The built-in preview renders `$…$` math when the setting **Markdown › Math: Enabled**
  (`markdown.math.enabled`) is on, which is the default. If it is off, turn it on.
- **If the built-in preview still fails:** install the *Markdown Preview Enhanced* extension (shd101wyy), whose preview
  renders KaTeX/MathJax.
- **Standalone HTML:** `python repo_reports/md2html.py repo_reports/KAPPA_FITTING_EXPLAINED.md` writes a `.html`
  next to the file, rendered with KaTeX. Open it in any browser.

## References

- Bartelmann, M. & Schneider, P. 2001, *Weak gravitational lensing*, Phys. Rep. 340, 291. arXiv:astro-ph/9912508.
- Jeffrey, N. et al. 2018, *Improving weak lensing mass map reconstructions using Gaussian and sparsity priors:
  application to DES SV*, MNRAS 479, 2871. arXiv:1801.08945.
- Kaiser, N. & Squires, G. 1993, *Mapping the dark matter with weak gravitational lensing*, ApJ 404, 441.
- Marshall, P. J., Hobson, M. P., Gull, S. F. & Bridle, S. L. 2002, *Maximum-entropy weak lens reconstruction:
  improved methods and application to data*, MNRAS 335, 1037. arXiv:astro-ph/0112396.
- Nocedal, J. & Wright, S. J. 2006, *Numerical Optimization*, 2nd ed., Springer. Covers L-BFGS (Ch. 7) and
  Gauss–Newton (Ch. 10).
- Wiener, N. 1949, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*, MIT Press.
- Zaroubi, S., Hoffman, Y., Fisher, K. B. & Lahav, O. 1995, *Wiener reconstruction of the large-scale structure*,
  ApJ 449, 446. arXiv:astro-ph/9410080.
