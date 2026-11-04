#!/usr/bin/env python3
"""Time the kl-tools forward model as an MCMC fit would call it (Discussion: cost of one emcee fit).

Runs data_generate/generate_fits.py for one galaxy (Xu3 galaxy 6447, the paper's example galaxy) with the
FiberLikelihood built exactly as in data generation. Right after the likelihood is constructed, it calls the
log-likelihood N times at parameters jittered around the fiducial values (one call = render the image and the
five fiber spectra and compute chi^2, i.e. one walker step), prints the timings, and exits before anything is
written. generate_fits.py itself is not modified.

Usage: python time_forward_model.py [N_CALLS]
"""
import os
import sys
import time

N_CALLS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SRC = "/jet/home/xwang30/kl-nn/data_generate/generate_fits.py"

ANCHOR = "        calibration_getter = getattr(\n"
TIMING = '''        import numpy as _np, time as _time
        _rng = _np.random.default_rng(0)
        _dv = get_GlobalDataVector(0)
        _theta0 = _np.array(sampled_pars_value, dtype=float)
        # the noiseless datavector has no noise arrays; a real fit has them, and chi^2 divides by their std
        _dv.noise = [_np.ones_like(_np.asarray(_dv.get_data(_i), dtype=float)) for _i in range(likelihood.Nobs)]
        _times = []
        for _k in range(N_CALLS):
            _theta = _theta0 + _rng.normal(0, 1, _theta0.size) * sampled_pars_std * 1000 * 0.1
            _t0 = _time.perf_counter()
            likelihood(list(_theta), _dv)
            _times.append(_time.perf_counter() - _t0)
        _times = _np.array(_times)
        print("TIMING sampled_pars", list(sampled_pars))
        print("TIMING per-call seconds", _np.round(_times, 4).tolist())
        print("TIMING mean %.4f s  median %.4f s  (first call %.4f s)" % (_times.mean(), _np.median(_times), _times[0]))
        print("TIMING mean excluding first call %.4f s" % _times[1:].mean())
        sys.exit(0)
'''

src = open(SRC).read()
assert src.count(ANCHOR) == 1, "anchor not found once in generate_fits.py"
src = src.replace(ANCHOR, TIMING + ANCHOR)

# Xu3 galaxy 6447 (reports/paper-tests-1m/mcmc/package/truth.csv); galaxy_axis layout = a single rendering pass
sys.argv = [SRC, "-ID", "6447", "-d", "timing_only",
            "-g1", "0.0145920703396579", "-g2", "-0.0042680728448773",
            "-theta_int", "2.4496671825750944", "-sini", "0.7938892115120704",
            "-v0", "8.183714816991596", "-vcirc", "235.5402290867125",
            "-rscale", "0.306141543771561", "-hlr", "1.60064435005188",
            "--rmag-true", "19.533092682901334", "--halpha-flux-true", "2.3490540907397686e-17",
            "--image-snr", "166.21834269554338", "--central-halpha-snr", "23.886021859123108"]
sys.path.insert(0, os.path.dirname(SRC))
g = {"__name__": "__main__", "__file__": SRC, "N_CALLS": N_CALLS, "sys": sys}
t_start = time.perf_counter()
try:
    exec(compile(src, SRC, "exec"), g)
except SystemExit:
    pass
print("TIMING total wall incl. setup %.1f s" % (time.perf_counter() - t_start))
