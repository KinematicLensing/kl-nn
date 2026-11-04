#!/usr/bin/env python3
"""T2b gate: re-render a few legacy stencil rows with the current simulator and
subtract the stored noiseless FITS (images, 5 spectra, fiber headers).

Writes new FITS only to $S/fits/paper_tests_B_regen_probe/ and a JSON summary to
the B_inference report folder. Never touches the legacy stencil.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits

DG = Path('/jet/home/xwang30/kl-nn/data_generate')
sys.path.insert(0, str(DG))
from generate_fits_wrapper import build_generate_command  # noqa: E402
from observation_schema import classify_fiber_offset_sign  # noqa: E402

S = Path('/ocean/projects/phy250048p/shared')
CSV = S / 'samples/samples_shear_response_simv3_cosi_5k.csv'
STORED = S / 'fits/shear_response_simv3_cosi_5k'
REGEN = 'paper_tests_B_regen_probe'
OUT = S / 'reports/paper-tests/B_inference/regen_probe.json'
PART = 50

def load(path):
    with fits.open(path, memmap=False) as h:
        spec = [np.asarray(h[k].data, float) for k in range(1, 6)]
        pos = np.asarray([(h[k].header['FIBERDX'], h[k].header['FIBERDY']) for k in range(1, 6)])
        img = np.asarray(h[6].data, float)
    return img, spec, pos

def main():
    tab = pd.read_csv(CSV, float_precision='round_trip')
    # 3 bases spanning theta_int, all 5 states of the first, g1+/g2- of others
    bases = tab.iloc[::5].copy()
    bases['b'] = np.arange(len(bases))
    pick = bases.iloc[np.argsort(bases.theta_int.values)[[5, 500, 995]]].b.values
    ids = [5 * pick[0] + s for s in range(5)] + [5 * pick[1] + 1, 5 * pick[1] + 4, 5 * pick[2] + 3]
    rows = []
    for sid in ids:
        row = tab.iloc[sid]
        part = sid // PART + 1
        cmd = build_generate_command(row, part=part, dataset=REGEN)
        (S / 'fits' / REGEN / f'part_{part}').mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, cwd=str(DG), stdout=subprocess.DEVNULL)
        a = load(STORED / f'part_{part}' / f'gal_{sid}.fits')
        b = load(S / 'fits' / REGEN / f'part_{part}' / f'gal_{sid}.fits')
        kw = dict(g1=float(row.g1), g2=float(row.g2), theta_int=float(row.theta_int), sini=float(row.sini))
        rows.append(dict(
            ID=int(sid), g1=float(row.g1), g2=float(row.g2), theta_int=float(row.theta_int),
            stored_fiber_label=classify_fiber_offset_sign(a[2], **kw),
            regen_fiber_label=classify_fiber_offset_sign(b[2], **kw),
            image_max_abs_diff=float(np.max(np.abs(a[0] - b[0]))),
            image_max_abs=float(np.max(np.abs(a[0]))),
            spec_max_abs_diff=[float(np.max(np.abs(x - y))) for x, y in zip(a[1], b[1])],
            spec_max_abs=[float(np.max(np.abs(x))) for x in a[1]],
            fiber_xy_max_abs_diff=float(np.max(np.abs(a[2] - b[2]))),
        ))
        print(rows[-1], flush=True)
    ok = all(r['image_max_abs_diff'] == 0 and max(r['spec_max_abs_diff']) == 0 and r['fiber_xy_max_abs_diff'] == 0 for r in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(bitwise_identical=ok, rows=rows), indent=1))
    print('bitwise identical:', ok)

if __name__ == '__main__':
    main()
