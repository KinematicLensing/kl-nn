# Mass maps from scratch: what the cells do, what the pixels do

Written 2026-10-23 for the cluster substructure test. Companion to `KAPPA_MAP_METHODS.md`, which compares the
two κ methods in more depth; this note starts from zero.

## 1. Two different products

In the cluster test we make two things from the same galaxies. They are easy to mix up.

1. **Shear cells.** We group about 5 galaxies per cell (by expected count, ≈ 1 Mpc² each) and stack their
   likelihoods to get **one shear per cell**. These are measurements of the shear. We use them for the bias
   analysis (pulls, m and c), and they are the input to the Kaiser–Squires (KS) map.
2. **κ maps.** A picture of the projected **mass**, κ, over the field. We make it two ways:
   - **KS:** computed from the cell shears, so its pixels *are* the cells (~1 Mpc).
   - **Likelihood-based:** fitted directly to the individual galaxies. Its pixels are a separate grid
     (0.5 Mpc now, maybe 0.75 Mpc after tuning) and have nothing to do with the cells.

So the cell size limits the shear measurements and the KS map. The likelihood map has its own pixel grid. The
question is what a pixel finer than a cell can actually mean (Sec. 4).

## 2. The key fact: shear is not local

κ at a point is the mass *there*. The shear at a point is caused by mass *everywhere*. A clump of mass
stretches background galaxies all around it, tangentially, with a strength that falls off with distance like
1/r² from a compact clump. In formula form:

  γ(**θ**) = (1/π) ∫ D(**θ** − **θ**′) κ(**θ**′) d²**θ**′,  with D(**θ**) = −1/(θ₁ − iθ₂)²

(Kaiser & Squires 1993; Bartelmann & Schneider 2001). The kernel D falls off as 1/|**θ**|². So:

- **Every galaxy's shear depends on the whole κ map**, mostly on the mass near it, but not only that.
- **Every patch of mass affects the shear of many galaxies** around it, not just the galaxies sitting on top
  of it.

This is what makes mass mapping possible, and it is also why "how many galaxies are in this pixel" is not the
right way to think about what a map pixel knows.

## 3. Method A: Kaiser–Squires (shear → mass)

The integral above can be undone exactly. In Fourier space it becomes a multiplication by a kernel of
magnitude 1, so we can divide it back out (Kaiser & Squires 1993). The steps:

1. **Measure the shear on a regular grid.** In our case that's the stacked shear of each Cartesian cell.
2. **Fourier transform, apply the inverse kernel, transform back.** That gives κ on the same grid.
3. **Smooth**, because the noise of each cell's shear goes straight into κ.

The resolution is set by the grid you measured the shear on: one κ value per cell. KS cannot produce anything
finer than its input cells, because the inversion needs a shear value at every grid point. Making cells smaller
than ~5 galaxies would give cells with 0–1 galaxies, many of them empty or extremely noisy.

## 4. Method B: fitting a mass map to the galaxies (mass → shear)

Here we never compute a cell shear. Instead:

1. **Guess a κ map** on a pixel grid, e.g. 12 × 12 pixels of 0.5 Mpc.
2. **Predict the shear at every galaxy's exact position** from that guess, using the integral in Sec. 2 (in
   practice with an FFT on the grid, then interpolating to the galaxy).
3. **Score the guess** with each galaxy's own likelihood: how probable is the predicted shear, given that
   galaxy's data? Add up the log-likelihoods of all 141 galaxies.
4. **Add a prior** that prefers maps with small, smooth κ (more on why below).
5. **Adjust the map to make score + prior as high as possible.** The result is the most probable map.

**Why pixels can be finer than cells here.** Because of Sec. 2, each galaxy informs many pixels at once: its
predicted shear depends on the κ in all pixels, weighted by distance. A pixel with no galaxy in it is still
constrained by the galaxies around it. So there is no need to collect ~5 galaxies into a pixel before it means
anything. The data only need to constrain the map *as a whole*.

**But finer pixels are not the same as finer resolution.** What the data can actually distinguish is set by how
much information the galaxies carry, not by the pixel size:

- The mean spacing between our galaxies is about 3.1′, or 0.45 Mpc: one galaxy per 9.7 arcmin².
- Each galaxy measures shear with noise σ_g ≈ 0.04 per component, while the subhalo adds only ~0.01–0.03 of
  shear.
- So a single galaxy can't tell you much about the mass right next to it. A feature in the map only becomes
  real once enough galaxies around it agree, which in practice means structure on scales of ~1 Mpc or more.

When the pixels are smaller than what the data can resolve (0.5 Mpc pixels with 141 galaxies: 144 unknowns,
about one galaxy each), the data alone can't pin down each pixel. Many different fine-grained maps fit the
galaxies equally well. **The prior decides between them**: it picks the smoothest, smallest map that fits.
This is regularisation, the same idea as a Wiener filter (Jeffrey et al. 2018).

Two good ways to think about the pixel size:

- **Fine pixels + prior** ≈ drawing a smooth curve through scattered noisy points. The curve is drawn at fine
  resolution, but the wiggles you can trust are set by the spacing and noise of the points, and by how stiff
  you make the curve (the prior).
- **Coarse pixels** build the smoothing in by force: a 1 Mpc pixel simply cannot represent anything smaller.

So the pixel size and the prior strength together set the *effective* resolution of the map. That is why the
next step is to tune them (e.g. 0.75 Mpc pixels, a smaller σ_κ, a stronger smoothness term) on synthetic data,
where we know the true map. Too fine and too weak a prior gives a speckled map of noise; too coarse or too
strong a prior blurs the subhalo away.

## 5. Our numbers

| | Shear cells | KS map | Likelihood map (now) | Likelihood map (candidate) |
|---|---|---|---|---|
| Size | ~1 Mpc² per cell (polar ≈ equal area; Cartesian 0.98 Mpc squares) | = Cartesian cells, 0.98 Mpc | 0.5 Mpc pixels, 12 × 12 | 0.75 Mpc pixels, 8 × 8 |
| Galaxies per element | ~5 expected (2–9 actual) | ~5 | ~1 | ~2.2 |
| What sets the resolution | cell size | cell size + smoothing | prior + galaxy information | pixel size + prior |
| Used for | bias analysis, KS input | mass map | mass map, residual | mass map, residual |

For reference, the subhalo's extra shear is concentrated within ~0.5–1 Mpc of it, so pixels much bigger than
~1 Mpc would wash it out.

## 6. Reading the maps

- **Residual map:** the with-subhalo κ map minus the *true* main-halo κ (decided 2026-10-23: for the paper we
  subtract the true main halo as if it were known, rather than a fitted one). A subhalo should show up as a
  positive blob at its position; elsewhere the residual is noise.
- **S/N map (planned):** κ divided by its uncertainty in each pixel, from the curvature of the fit. A blob only
  means something relative to the noise around it, so this makes "visible by eye" honest.
- **Peaks are shrunk:** the prior pulls κ toward zero, so the subhalo's peak comes out lower than the truth.
  Always compare with the truth run through the same pipeline, as in the synthetic tests.
- **Mass-sheet degeneracy:** shear can't fix the overall level of κ (a constant can be added without changing
  the shear), so only the *shape* of the map is meaningful, not its zero point (Bartelmann & Schneider 2001).

## 7. In one paragraph

Cells measure the shear: you need several galaxies in each to beat the noise, which is why they're ~1 Mpc. A
mass map is a different object: you guess a mass distribution, predict every galaxy's shear from it, and keep
the guess that fits best. Since mass affects the shear of galaxies all around it, every galaxy helps constrain
many pixels, so the map's pixels don't each need their own galaxies. But the detail you can trust is still
limited by the number of galaxies and their noise, so a map with fine pixels relies on its prior to stay smooth.
Tuning the pixel size and prior on synthetic data is how we choose a map that shows real structure, like the
subhalo, without inventing structure from noise.

## References

- Bartelmann, M. & Schneider, P. 2001, Phys. Rep. 340, 291. arXiv:astro-ph/9912508.
- Jeffrey, N. et al. 2018, MNRAS 479, 2871. arXiv:1801.08945.
- Kaiser, N. & Squires, G. 1993, ApJ 404, 441. doi:10.1086/172297.
- See `KAPPA_MAP_METHODS.md` for the full comparison and earlier applications (Bridle et al. 1998; Marshall
  et al. 2002; Schneider et al. 2017).
