"""Sec. 6.4 (2026-10-01): selection response of the ensemble response calibration
(ghat_i / R(sigma_k,i) with weights R^2, R = a + b sigma_k^2 clipped at 0) on the Xu3-like stencil.
Total response of the calibrated mean = R_gamma_cal + R_sel_cal (ideal: 1)."""
import numpy as np
z = np.load('/ocean/projects/phy250048p/shared/reports/paper-tests/B_inference/t2b_xu3like.tf_equal.cubes.npz')
g, sd, d = z['ghat'], z['sd'], float(z['delta'])
a, b = 1.1926492704684541, -350.2905502397149
Rc = np.clip(a + b * sd ** 2, 0, None)            # (nb, 5 states, 2 comps)
nb = len(g)

def cal_mean(gi, Ri):                              # weighted mean of ghat/R with weights R^2
    return (Ri * gi).sum() / (Ri ** 2).sum()

def stat(idx):
    out = []
    for k, (p, q) in enumerate(((1, 2), (3, 4))):
        Rg = (cal_mean(g[idx, p, k], Rc[idx, 0, k]) - cal_mean(g[idx, q, k], Rc[idx, 0, k])) / (2 * d)
        Rs = (cal_mean(g[idx, 0, k], Rc[idx, p, k]) - cal_mean(g[idx, 0, k], Rc[idx, q, k])) / (2 * d)
        Rt = (cal_mean(g[idx, p, k], Rc[idx, p, k]) - cal_mean(g[idx, q, k], Rc[idx, q, k])) / (2 * d)
        out += [Rg, Rs, Rt]
    return np.array(out)

v = stat(np.arange(nb))
rng = np.random.default_rng(3)
reps = np.array([stat(rng.integers(0, nb, nb)) for _ in range(2000)])
se = reps.std(0)
for k in range(2):
    print(f"g{k+1}: R_gamma_cal {v[3*k]:.3f}+-{se[3*k]:.3f}  R_sel_cal {v[3*k+1]:.4f}+-{se[3*k+1]:.4f}  R_tot {v[3*k+2]:.3f}+-{se[3*k+2]:.3f}")
rs = (reps[:, 1] + reps[:, 4]) / 2
print(f"mean R_sel_cal {(v[1]+v[4])/2:.4f}+-{rs.std():.4f}")
