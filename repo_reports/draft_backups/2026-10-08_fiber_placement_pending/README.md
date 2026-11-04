# Pending fiber-placement changes (saved 2026-10-08)

Eason is replacing the paper folder with his Overleaf edit. These changes were made on the
cluster copy (2026-10-06) and are NOT yet in Overleaf. Re-insert them into the new draft.
Full copies of the changed files are in this folder (data-generate.tex, nn-arch.tex, main.bib),
with unified diffs against the pre-change backups (data-generate.diff, main.bib.diff).

## 1. sections/data-generate.tex, Sec. 3.1 (ssec:kl-model)
Replaces the old true-axis text: everything after the sentence ending "...each offset fiber has
an exposure time of 600 s." (the T = GRP matrix, eqns fiber-transform/fiber-matrices, the SVD
and its two sign conditions) up to \subsection{Training and Validation Data}. New text:

```latex
In a real survey, the fibers are placed along axes measured from the photometry of the galaxy. DESI-PV, for example, places its offset fibers along the major axis given by the position angle in the Siena Galaxy Atlas (SGA) \citep{DESIPVTargetSel2023}, which is measured from the second moments of the galaxy's light distribution in the Legacy Survey images \citep{SGA2020}. If we instead placed our fibers on the true observed axes of the galaxy, the fiber positions would give the model a noise-free measurement of the galaxy's orientation, which for nearly face-on disks is set almost entirely by the shear. We therefore emulate the measurement used for targeting.

For each galaxy, we add a single realization of sky and read noise to its noiseless $r$-band image, using the Legacy Survey $r$-band observing configuration of \citep{KLDESI} (Table~\ref{tab:obs-conf}). We refer to this as the targeting image. Its SNR, defined as in Eq.~(\ref{eqn:snr-image}), is not chosen by us, but follows from the brightness and size of each galaxy under realistic Legacy Survey imaging. For the galaxies in our Xu3 test set (Sec.~\ref{ssec:test-set}), the SNR of the targeting image agrees with the image SNR measured in the Legacy Survey catalog to within a factor of two for 90\% of the galaxies, with a median ratio of 1.0. The targeting noise realization is fixed for each galaxy, and is independent of the image noise added during training and inference (Sec.~\ref{ssec:datasets}).

We then measure the position angle $\beta$ of the major axis from the adaptive second moments of the targeting image \citep{Hirata2003, Mandelbaum2005}, as implemented in \textsc{GalSim}. Both these and the SGA moments estimate the second moments of the light distribution; we use adaptive moments since they do not require choosing a surface brightness threshold. The measurement fails to converge for 0.3\% of our galaxies, all of them large, nearly edge-on disks, since the iteration starts from a small, round weight function that is far from their elongated shape. For these galaxies, we draw $\beta$ uniformly at random. The measured major axis is then the unit vector $(\cos\beta, \sin\beta)$ on the image, and the minor axis is the major axis rotated by $90\degree$ counterclockwise. We place the offset fibers along these two axes. Since an image only determines its axes up to a rotation of $180\degree$, we restrict $\beta$ to $(-90\degree, 90\degree]$, which fixes the order of the fibers. For galaxies with $\cos i < 0.5$, the measured position angle is typically within $0.2\degree$ of the true observed major axis, while for nearly face-on galaxies with $\cos i > 0.8$, whose images are close to round, the typical error grows to about $2\degree$. The largest errors therefore occur for nearly face-on galaxies with low SNR, where noise can dominate the small ellipticity of the image.
```

## 2. Table tab:obs-conf (same file): two new rows after "Exposure time"
```latex
        Sky level & 44.54 & - & e$^{-}$ s$^{-1}$ pixel$^{-1}$ \\
        Read noise & 2.6 & - & e$^{-}$ \\
```
and appended to the caption (after "...central/offset fibers respectively."):
```latex
The image sky level and read noise are only used for the targeting image that sets the fiber positions (Sec.~\ref{ssec:kl-model}).}
```

## 3. sections/nn-arch.tex, R90 paragraph (Sec. 4.3)
"own axes" -> "measured axes":
```latex
Its spectra stay the same, since our fibers are placed relative to the galaxy's measured axes (Sec.~\ref{ssec:kl-model}).
```

## 4. main.bib: two new entries
```bibtex
@article{SGA2020,
    author = "Moustakas, John and Lang, Dustin and Dey, Arjun and Juneau, St{\'e}phanie and Meisner, Aaron and others",
    title = "{Siena Galaxy Atlas 2020}",
    eprint = "2307.04888",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.GA",
    doi = "10.3847/1538-4365/acfaa2",
    journal = "Astrophys. J. Suppl.",
    volume = "269",
    number = "1",
    pages = "3",
    year = "2023"
}

@article{Mandelbaum2005,
    author = "Mandelbaum, Rachel and Hirata, Christopher M. and Seljak, Uros and Guzik, Jacek and Padmanabhan, Nikhil and Blake, Cullen and Blanton, Michael R. and Lupton, Robert and Brinkmann, Jonathan",
    title = "{Systematic errors in weak lensing: Application to SDSS galaxy-galaxy weak lensing}",
    eprint = "astro-ph/0501201",
    archivePrefix = "arXiv",
    doi = "10.1111/j.1365-2966.2005.09282.x",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "361",
    pages = "1287--1322",
    year = "2005"
}
```

## Pending decisions (discuss with Eason, per his request, after the fix)
- Phrasing of the faint-galaxy fallback (unweighted second moments when adaptive moments do not
  converge: ~5% of training galaxies, none in the test sets). The failure sentence above ("The
  measurement fails to converge for 0.3% ... we draw beta uniformly at random") is OUT OF DATE
  after the 2026-10-06 targeting fix: every test-set galaxy now has a converged HSM measurement,
  and Eason said no error mention is needed if all measurements are good.
- The accuracy (0.2 deg / 2 deg) and SNR-ratio numbers were computed before the fix; they only
  change for the 0.3% of Xu3 galaxies that were fixed. Re-check when re-inserting.
