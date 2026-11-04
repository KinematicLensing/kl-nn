"""Sec. 6.4 (2026-10-01): response-calibration variants on Xu3 (100k).
R(sigma_k) = a + b sigma_k^2 from the Xu3-like stencil fit, or the Gaussian prediction a=1, b=-1/sigma_pi^2.
(1) metacal-like ratio  <ghat>/<R>  with equal weights  (no width-dependent weights -> no selection term)
(2) regression of ghat on R*g with weights R^2  (= Sigma R ghat / Sigma R^2, the Tab. 2 estimator)"""
import sys
import numpy as np
sys.path.insert(0, '.')
import make_paper_figures_sec6 as M
D = M.load_xu3()
SP2 = 0.2 ** 2 / 12
for lab, ab in (("stencil", (1.1926492704684541, -350.2905502397149)), ("gauss", (1.0, -1 / SP2))):
    for k in range(2):
        t, e = D["g"][:, k], D["e"][:, k]
        R = np.clip(ab[0] + ab[1] * D["sd"][:, k] ** 2, 0, None)
        low = np.abs(t) < M.LOW_G
        raw = M.fit(t, e, np.full(len(t), 1 / len(t)))
        m_ratio = (1 + raw["low_m"]) / R[low].mean() - 1
        tt, ee, ww = M.calibrated_component(D, ab, k)
        f = M.fit(tt, ee, ww / ww.sum())
        neff = M.kish(ww[ww > 0])
        print(f"{lab} g{k+1}: <R>_low={R[low].mean():.3f}  ratio m={m_ratio:+.3f}  R2-weighted m={f['low_m']:+.3f}+-{f['low_m_se']:.3f}  Neff={neff:.0f}")
