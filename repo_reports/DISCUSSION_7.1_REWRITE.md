# Discussion 7.1: suggested rewrite (2026-11-04)

Status: suggestion only. Nothing here has been applied to `sections/discussion.tex`. Eason is editing ¶1–3 himself.

## Change from the 11-03 version
In ¶2, the MCMC side is now "fewer than ~100 galaxies". This rests on the forward-model timing below; Eason
rounds it up generously because CPU time is cheaper than GPU time.

## Timing behind ¶2 (same simulator as our training data)
- Script: `arch/diagnostics/paper_tests/time_forward_model.py`. It runs the unmodified `data_generate/generate_fits.py` for Xu3 galaxy 6447, then exits before writing anything.
- Measured on 1 RM-shared core over 10 calls: 0.115–0.122 s per call, mean 0.117 s. One call renders the image and 5 fiber spectra and computes chi^2.
- One emcee fit with KL-I's settings (100 walkers x 30,000 steps = 3e6 calls) costs 3.5e5 s, about 100 core-hours (4 core-days) per galaxy.
- The training set costs a few CPU-seconds per galaxy x 1.1e6, about 1e3 core-hours, which is about the same as 10 emcee fits.
- KL-I, KL-II and DESI-KL publish no chain run times.
- Caveats:
  - Timed on one galaxy only.
  - The chain length is KL-I's.
  - KL-I used slit spectra, not fibers.
  - Lauren's NUTS/GPU pipeline is different.

## Suggested rewrite

> ¶1. As we mentioned in Sec.~\ref{sec:intro}, the cost of inference is one of the main challenges in applying KL to a full survey, and ongoing work continues to improve the speed of forward-model fitting (Everett et al., in prep.; Willett et al., in prep.). NPEs offer a complementary route. Once trained, our NPE measures one galaxy in $\sim$0.3\,s on a single GPU, so a survey of order $10^7$ galaxies could be measured in about a month of single-GPU time, or a few days when parallelized.
>
> ¶2. More immediately, our NPE can be used to quickly explore our data and parameter space in simulations. For example, it can help us understand how the photometric and spectroscopic data constrain the morphological and kinematic parameters, which can inform modeling choices such as which nuisance parameters to include, the priors, and details of the simulator. It can also inform survey decisions, such as selection cuts, targeting, or exposure times. As demonstrated in this work, an NPE trained on $10^6$ simulated galaxies already covers the parameter space it was trained on well, and with parallelization it takes about one day to generate the training data and another day to train. With the same compute and the same simulator, MCMC fits would measure fewer than $\sim$100 galaxies, far from enough to explore the parameter space in depth, while our trained NPE measures 100 galaxies in under a minute on one GPU. A faster forward model would also speed up the generation of our training data, so the NPE would keep a large advantage.
>
> ¶3. Fast inference also makes it practical to explore the science cases enabled by KL, such as distinguishing between dark matter models. Galaxy clusters are good laboratories for dark matter self-interactions at high velocities (Bechtol et al.~\citep{Bechtol22}). Self-interacting dark matter (SIDM) not only forms a core at the center of a halo, but also changes the density profile out to the virial radius, as well as the population of subhalos, on the $\sim$0.2--2\,Mpc scales probed by weak lensing~\citep{Banerjee20, Bechtol22}. Cluster substructure is particularly informative. For a velocity-dependent cross section, scattering within a subhalo happens at galactic velocities, while scattering between the subhalo and its host happens at cluster velocities and evaporates the subhalo, so subhalos probe the cross section over a wide range of velocities~\citep{Bechtol22}. Bhattacharyya et al.~\citep{Bhattacharyya22} found that subhalos in the outskirts of clusters, where tidal disruption is weaker, are the most useful for this, which is the regime of the subhalo we detected in Sec.~\ref{ssec:substructure}. Leaving the subhalo position free, as we did there, also measures the offset between a subhalo's dark matter and its galaxies, which depends on the self-interaction cross section~\citep{Kahlhoefer14, Fischer22}.
>
> In this work, we assumed a high-quality selection of DESI/LS galaxies, with a density of $\sim$0.14\,arcmin$^{-2}$. A KL sample from the Roman Space Telescope is expected to reach $\sim$4\,arcmin$^{-2}$ with a shape noise of 0.035~\citep{KLRoman}, about an order of magnitude less shear noise per unit area than the weak lensing sample in the same forecast. At that density, the subhalo of Sec.~\ref{ssec:substructure} would be detected at much higher significance, and less massive subhalos would become detectable in individual clusters.

## Notes
- The five new citations (Bechtol22, Banerjee20, Bhattacharyya22, Kahlhoefer14, Fischer22) are already in `main.bib`.
- "About an order of magnitude" comes from my own calculation using KL-Roman's numbers (σ²/n gives 9–18× against their Roman WL sample). KL-Roman does not state this comparison itself.
- "A month of single-GPU time" is 10^7 x 0.3 s, about 35 days.
