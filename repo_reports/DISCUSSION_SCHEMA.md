# Sec. 7 Discussion: outline (2026-11-02)

Every point rests on a result already in the paper or an item in `kl-paper-gaps.md`; nothing here needs a new run.
The framing stays exploratory: what the NPE enables and what it still needs, not a finished pipeline.

## 7.1 What fast, amortized KL inference enables

1. **Survey-scale cost.** One comparison to make the speed concrete: 10^7 DESI-KL galaxies at 0.3 s each is about
   35 GPU-days, versus about 380 GPU-years for MCMC at 20 min per galaxy (Sec. 5.1, intro). Could also stay
   qualitative.
2. **Exploring survey design and sample selection** (Eason, 2026-10-28). The shape-noise maps of Sec. 5.3 were cheap
   because inference is fast. The same approach can test target-selection cuts for a KL sample (SNR, size,
   inclination, magnitude) before spending fiber time, and check them later against real measurements.
3. **Reusing the likelihoods.** Each galaxy's shear likelihood is computed once and can then be reused without
   rerunning inference:
   - the TF prior can be swapped afterwards (Sec. 4.4);
   - the likelihood plugs into any shear-field model (Eq. `halo-post`), so comparing models is cheap.

   Compare models with the likelihood ratio, or with the evidence when they aren't nested, not by each model's
   detection σ. Examples that fit our ~0.2–0.5 Mpc resolution:
   - tidally truncated vs untruncated subhalos;
   - offsets between dark matter and galaxies or gas (the Bullet-cluster SIDM test, via the free-position fit);
   - merging clusters.

   Cored vs cuspy profiles only as motivation, since they are below our resolution. Caveat: telling profiles apart
   needs far more signal than detecting a subhalo (about 4σ here), i.e. stacking many clusters, which the same
   likelihood product handles.

## 7.2 Limitations of the current NPE

4. **No per-galaxy shear.** Single posteriors are prior-dominated (Appendix B), so shear only comes from combining
   galaxies, through cells or a model fit. KL-II probes substructure galaxy by galaxy, which we have not shown.
   - The halo fit doesn't need galaxies across the whole cluster: the subhalo information comes from within about
     1 Mpc of it.
   - The ±0.1 training prior caps |g|, so the NPE can't be used near massive cluster cores; a wider prior needs a
     retrained model.
5. **Simulations vs real data.** Our forward-model simplifications:
   - a single source redshift;
   - Hα only (Xu et al. fit four lines, so adding [O II] and [O III] should lower the shape noise);
   - a thin disk with an arctan rotation curve;
   - true-valued metadata (m_r, ρ_I, ρ_S);
   - a TFR identical to the TF prior;
   - isolated galaxies.

   Population shift relative to the training prior is inherent to NPE; don't frame it as a flaw. Next steps: the
   MCMC comparison (Lauren), then real DESI data.
6. **Shear-bias requirements.**
   - Stacking at constant shear gives m ≈ −1% ± 0.5% (Sec. 6.1); the cluster test constrains m only to ±0.05.
   - Both fall short of Stage-IV cosmic-shear requirements of a few ×10^-3 (DESC SRD; **exact number to verify**).
   - Cosmic shear also has no groups of galaxies sharing one shear, so it would need a forward model of the shear
     field, as in field-level inference (Zhou+24, already cited).

   This shows where the method stands without claiming a pipeline.

## 7.3 Future work (one short paragraph, or merged into 7.2)

7. - more emission lines;
   - several source redshifts;
   - noisy metadata and m_r uncertainty in the TF prior;
   - the Xu5 sample;
   - the MCMC comparison, then a pilot on DESI peculiar-velocity data;
   - optionally, neural ratio estimation, which learns the likelihood directly instead of dividing the posterior by
     the training prior.

## Open questions for Eason

- Three subsections, or 4–6 plain paragraphs?
- Quote the GPU-time comparison, or keep it qualitative?
- Mention NRE as future work, or keep it out?
- How far to speculate on cosmic shear (suggested: two sentences on what's missing, no promises)?
