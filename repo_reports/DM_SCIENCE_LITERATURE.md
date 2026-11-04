# Dark-matter science reachable with KL + NPE: literature notes (2026-11-03, updated for Roman)

For Discussion 7.1. Summaries are mostly from abstracts. The Snowmass halo white paper Sec. 3 and the KL-Roman
number-density section were read in full; the other papers were not. Check numbers in the papers before quoting them.

## What KL can reach: DESI-like (this paper) vs Roman-like (forward-looking)

| | DESI-KL (our Sec. 6) | KL-Roman (Xu et al. 2023, arXiv:2201.00739) |
|---|---|---|
| Source density | ~0.11 arcmin^-2 behind z = 0.13 | 4 arcmin^-2 adopted; 8.13 before halving "to account for observational inefficiencies" |
| Shape noise (total, both components) | σ_eff = 0.041 (our Sec. 5.3) | 0.035 |
| Mean source separation | ~3 arcmin | 0.35–0.5 arcmin |
| Area | DESI footprint | ~2000 deg² (HLIS ∩ HLSS) |
| Sources | z = 0.3 in our simulations | Hα/[O III] emitters, z ~ 1–2 |

Shear noise per unit area, σ²/n (total σ):

| Survey | σ²/n per arcmin² | Relative to KL-Roman |
|---|---|---|
| DESI-KL | 1.6×10^-2 | ~50–100× higher |
| KL-Roman | 1.5–3.1×10^-4 | 1 |
| Roman WL (0.37 at 51 arcmin^-2, Xu+23) | 2.7×10^-3 | 9–18× higher |
| LSST-like WL (0.37 at ~27 arcmin^-2) | 5.1×10^-3 | 17–34× higher |

Even weighting by area (LSST ~18,000 deg² vs ~2000 deg²), KL-Roman's total statistical power is comparable to LSST's.
This ignores cosmic variance and source-redshift differences, so it's a rough comparison.

**DESI-like KL (this paper).** Resolution ~0.2–0.5 Mpc for z_l = 0.13. A 3×10^14 M_sun subhalo is detected at a
median of ~4σ in a single cluster. KL's strength there is per-galaxy precision, so individual systems need only ~100
sources.

**Roman-like KL.**
- A 1 arcmin² cell holds 4–8 sources, giving a per-component shear noise of ~0.01 per arcmin². That is mass mapping at
  arcmin to sub-arcmin resolution (~0.1–0.3 Mpc for lenses at z ~ 0.2–0.5) with very low noise.
- Naively scaling our 6.3 result (√n and σ_ε only, same lens and subhalo) gives ~7–10× higher significance, i.e.
  ~25–40σ instead of 3.9σ. Equivalently, much lower subhalo masses become detectable in single clusters.
- This scaling is not a computed result. Roman sources are at higher z (better lensing efficiency for z < 1 lenses),
  and Roman KL uses slitless spectroscopy, not fibers, so a Roman NPE needs its own forward model and training.

**Which DM physics this reaches.** DM physics that changes halos or subhalos of ≳10^11–10^12 M_sun on scales of
≳100 kpc.
- **SIDM at cluster velocities (v ~ 1000–3000 km/s)** is the main candidate in both regimes.
- **Roman-like KL** also reaches galaxy-scale subhalos (satellite lensing, halo shapes) system by system or in small
  stacks. It makes **small-scale cosmic shear** (WDM, fuzzy DM, decaying DM) a realistic, baryon-limited target rather
  than a weak case.
- **WDM and fuzzy DM through substructure remain out of reach in either regime**: their subhalo effects are at
  ≲10^10 M_sun and kpc scales.

## A. Overviews (start here)

1. **Bechtol et al. 2022, "Snowmass2021 Cosmic Frontier White Paper: Dark Matter Physics from Halo Measurements",
   arXiv:2203.07354, Sec. 3 "Cluster Scale" (pp. 17–21).** Read in full. Clusters are "high-energy DM particle
   colliders". The four cluster probes:
   - **Outer density profiles (3.2.1).** SIDM rearranges the profile out to the virial radius, so weak lensing at
     ~0.2–2 h^-1 Mpc probes σ/m at v ~ 1000 km/s. DES Y1 already gives σ/m ≲ 2 cm²/g; LSST forecasts < 0.5 cm²/g.
     Fig. 4 shows the ΔΣ residuals.
   - **Merging clusters (3.2.2).** DM–gas–galaxy offsets. The Bullet Cluster gives σ/m < 5 cm²/g from the optical
     depth. Ensembles of minor mergers proved fragile, and the paper says better lensing data alone is not enough
     without theory advances.
   - **BCG oscillations in SIDM cores (3.2.3).** Cores are ~100 kpc for σ/m ~ 1 cm²/g, and ~10 kpc BCG offsets imply
     σ/m ~ 0.1 cm²/g. Too central for DESI-like KL. Roman-like KL resolves ~100 kpc, but cluster cores also exceed
     our ±0.1 shear prior, so this needs a wider-prior NPE.
   - **Substructure in massive clusters (3.2.4).** For velocity-dependent SIDM, subhalo-internal scattering
     (v ~ 100–300 km/s) and host–subhalo scattering (cluster velocities, causing evaporation) probe different
     velocities. Lensing of cluster substructure can therefore probe the cross section over a broad velocity range.
     **This is the paragraph that motivates our Sec. 6.3 most directly.**
2. **Rose et al. 2024, "Introducing the DREAMS Project", arXiv:2405.00766, Sec. IV** (Tim's suggestion).
   - WDM suites suppress halos of ~10^10 M_sun by ~30% in the warmest models, and the paper states "WDM is not
     expected to impact halo abundance at these scales" (groups and clusters).
   - **WDM is not reachable through substructure**, even with Roman resolution. The WDM route is the small-scale
     cosmic-shear power spectrum (E).
   - Cyr-Racine is a co-author; DREAMS plans SIDM and cluster suites, so it is worth watching.

## B. Cluster substructure and SIDM (closest to Sec. 6.3)

3. **Bhattacharyya et al. 2022, ApJ 932, 30, arXiv:2106.08292.**
   - Velocity-dependent SIDM cluster zoom-ins: fewer surviving subhalos and different inner subhalo profiles, which
     they propose to measure with weak lensing.
   - **Subhalos in cluster outskirts, where disruption matters less, are the most useful.**
   - Caveat: degenerate with subhalo disruption in CDM.
   - Forecast: LSST reaches σ/m ~ 1 cm²/g at v ~ 2000 km/s. Roman-like KL, with 17–34× lower noise per area than
     LSST, could weigh outskirt subhalos individually instead of only statistically.
4. **Banerjee et al. 2020, JCAP 02, 024, arXiv:1906.12026.** SIDM changes the cluster beyond the core: the subhalo
   distribution and the density out to the splashback radius. Current weak lensing gives σ/m ≲ 2 cm²/g.
5. **Ragagnin et al. 2024, A&A 687, A270, arXiv:2404.01383, "Dianoga SIDM".** Rare vs frequent scattering, with
   baryons. SIDM suppresses subhalos (less with baryons; more for frequent scattering), and produces **a tail of
   diffuse subhalos in cluster outskirts** and more compact ones in cores.
6. **Zeng, Peter, ..., Cyr-Racine, Vogelsberger 2022, MNRAS 513, 4845, arXiv:2110.00259** (Peter and Cyr-Racine).
   Core collapse vs host evaporation vs tidal heating. It targets ~10^8 M_sun subhalos (the strong-lensing regime), so
   cite it only for the physics of evaporation.
7. **Okabe et al. 2014, ApJ 784, 90, arXiv:1304.2399, weak-lensing subhalos in Coma (Subaru).** **The observational
   precedent.**
   - Weak-lensing masses of 32 individual subhalos down to ~10^-3 of the cluster mass.
   - Truncation radii and mass-to-light ratios decrease toward the center.
   - Mass function consistent with CDM; truncated NFW preferred.
   - It needed the nearest rich cluster. KL-Roman's noise per area could do this for many more distant clusters.

## C. Mergers and DM–galaxy offsets (our free-position fit)

8. **Harvey et al. 2015, Science 347, 1462, arXiv:1503.07675** (offsets in many minor mergers) and **Wittman et al.
   2018, ApJ 869, 104, arXiv:1701.05877** (offsets change with better data; revised limits). The cautionary pair:
   offsets need precise mass localization. Our localization of ~0.2 Mpc at DESI density should improve several-fold
   at Roman density.
9. **Kahlhoefer et al. 2014, MNRAS 437, 2865, arXiv:1308.3419**: SIDM shifts the halo centroid relative to the
   galaxies (deceleration, evaporation), depending on the angular dependence of the scattering. **Fischer et al. 2022,
   arXiv:2109.10035**: frequent vs rare scattering in unequal-mass mergers (our "subhalo falling into a cluster"
   configuration); offsets are largest for frequent scattering, and the subhalo evaporates.

## D. Galaxy–galaxy lensing

10. **Kumar & More 2025, MNRAS, arXiv:2409.05795, HSC Y3 satellite lensing in redMaPPer clusters.** Subhalo mass vs
    cluster-centric distance: stripped within ~0.3 h^-1 Mpc, field-like outside. Today this needs stacks of thousands
    of satellites. With KL-Roman (~4–8 sources per arcmin² at σ 0.035), stacks can be far smaller, enough to split by
    host, radius and luminosity simultaneously.
11. **Halo shapes.** SIDM makes halos rounder, most strongly in the center.
    - **Peter et al. 2013, MNRAS 430, 105, arXiv:1208.3026** (Annika Peter): σ/m ≥ 0.1 cm²/g allowed by halo shapes.
    - **Clampitt & Jain 2016, arXiv:1506.03536**: weak-lensing ellipticity of LRG halos.
    - **Gonzalez et al. 2023, arXiv:2308.13461**: stacked lensing of cluster-halo shapes in matched CDM/SIDM.
    - The anisotropic shear signal is small and shape-noise limited, which is the regime where KL's low shape noise
      matters most. A strong forward-looking case for Roman-like KL.

## E. Cosmic shear (WDM, fuzzy DM and decaying DM all suppress small-scale power)

12. **Dentler et al. 2022, MNRAS 515, 5646, arXiv:2111.01199, fuzzy DM with DES Y1.** m_a > 10^-23 eV with Planck.
    The authors say the constraint could improve by up to three orders of magnitude with cosmic shear at ~0.1 arcmin
    resolution, *if* baryonic feedback and fuzzy DM are modelled precisely in the halo model.
    - KL-Roman's low noise per area (0.35–0.5 arcmin separation) is exactly what the small-scale signal needs.
    - The limiting factor becomes baryonic feedback, not shape noise.
    - At DESI-like density this is not a KL case.
13. **KiDS-1000 + CMB decaying-DM limits (arXiv:2211.14334)** and **KiDS-1000 "deviations from a purely CDM power
    spectrum" (arXiv:2502.04449).** Found by title only; authors and journals still to check. Same logic as 12.

## Suggested framing for 7.1

- **What this paper demonstrates.** At DESI-like density, the NPE likelihoods detect a massive subhalo in a single
  cluster and locate it to ~0.2 Mpc, with no calibration.
- **What the same machinery enables at Roman-like density** (4–8 arcmin^-2, σ 0.035; 9–34× lower noise per area than
  WL surveys):
  - weighing individual outskirt subhalos and their truncation, i.e. SIDM at cluster velocities (Snowmass 3.2.4;
    Bhattacharyya+22; Banerjee+20; Ragagnin+24; precedent Okabe+14);
  - DM–galaxy offsets of infalling groups (Kahlhoefer+14, Fischer+22, with the Wittman+18 caution);
  - satellite stripping and halo shapes from galaxy–galaxy lensing in small stacks;
  - small-scale cosmic shear for WDM, fuzzy DM and decaying DM, limited by baryons.
- **Why amortized inference matters for Roman.** The sample is much larger, so per-galaxy MCMC is even less feasible.
  The NPE needs retraining for Roman's slitless spectroscopy, but only once.
- **Be explicit** that WDM and fuzzy DM *substructure* (≲10^10 M_sun, kpc) stays out of reach in either regime
  (DREAMS Sec. IV).
- **Cyr-Racine.** I found no cluster-scale weak-lensing paper of his; his substructure work (ETHOS; the substructure
  power spectrum) is in the strong-lensing regime. ETHOS is still useful for mapping particle models onto structure
  formation.
