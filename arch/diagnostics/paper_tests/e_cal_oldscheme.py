"""Sec. 6.4 (2026-10-01): the earlier (codex/cursor, legacy-model) R(sigma) calibration scheme
re-run on the current model, for comparison with the R^2-weighted ensemble calibration.
Old scheme: sigma = sqrt((Var g1 + Var g2)/2); fit R = a + b sigma^2 on the stencil (R_11, R_22
pooled... old code used the scalar trace response) iterating the keep rule sigma/R < 0.2 with R
clipped to [0.25, 1]; on Xu3 keep sigma/R < 0.2, divide each galaxy by R, use shape-noise-regularized
weights on the kept galaxies, fit m on |g| < 0.02."""
import sys
import numpy as np
sys.path.insert(0, '.')
import make_paper_figures_sec6 as M
z = np.load(M.PT / 'B_inference/t2b_xu3like.tf_equal.cubes.npz')
Rst = z['R']; sd0 = z['sd'][:, 0, :]
s_st = np.sqrt((sd0 ** 2).mean(1)); Rtr = 0.5 * (Rst[:, 0, 0] + Rst[:, 1, 1])
keep = np.ones(len(s_st), bool)
for it in range(20):
    A = np.column_stack([np.ones(keep.sum()), s_st[keep] ** 2])
    a, b = np.linalg.lstsq(A, Rtr[keep], rcond=None)[0]
    Rp = np.clip(a + b * s_st ** 2, 0.25, 1.0)
    new = s_st / Rp < 0.2
    if np.array_equal(new, keep):
        break
    keep = new
print(f"stencil fit (keep rule): a={a:.3f} b={b:.1f} kept {keep.sum()}/{len(keep)} iters {it}")
D = M.load_xu3()
s = np.sqrt(D['var'].mean(1))
for lab, (aa, bb) in (("old-rule fit", (a, b)), ("current fit a=1.19 b=-350", (1.1926, -350.29))):
    R = np.clip(aa + bb * s ** 2, 0.25, 1.0)
    mask = s / R < 0.2
    w = M.weights(D, 'weighted', mask)
    out = []
    for k in range(2):
        e = D['e'][:, k] / R
        f = M.fit(D['g'][:, k], e, w)
        out.append(f"m{k+1}={f['low_m']:+.3f}+-{f['low_m_se']:.3f}")
    print(lab, f"kept {mask.mean():.3f} Neff/N {M.kish(w[w > 0]) / len(w):.3f}", *out)
