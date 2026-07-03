import os
from os.path import join
from argparse import ArgumentParser
import numpy as np
import pandas as pd
SCR_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse input arguments
parser = ArgumentParser()
parser.add_argument('-i', type=int, default=0, help='start index')
parser.add_argument('-j', type=int, default=1, help='stop index')
parser.add_argument('-n', type=int, default=1, help='array id')
parser.add_argument('-s', type=str, default='samples_small.csv', help='sample file')
parser.add_argument('-d', type=str, default='small', help='dataset name')
parser.add_argument('--low_psf', action='store_true', help='whether to use low psf')
args = parser.parse_args()
i = args.i
j = args.j
n = args.n
s = args.s
d = args.d
low_psf = args.low_psf
low_psf_str = '--low_psf' if low_psf else ''

SAMP_FILE = f'/ocean/projects/phy250048p/shared/samples/{s}'

os.system(f"mkdir /ocean/projects/phy250048p/shared/fits/{d}/part_{n}/")

df = pd.read_csv(SAMP_FILE)
nsamps = len(df)
chunk = np.array(df.iloc[i:j]) if j < nsamps else np.array(df.iloc[i:])

for row in chunk:
    ID, g1, g2, theta_int, sini, v0, vcirc, rscale, hlr, dx_disk, dy_disk, dx_spec, dy_spec = row[:13]
    # dx_disk, dy_disk, dx_spec, dy_spec = 0., 0., 0., 0.
    ID = int(ID)
    os.system(f"python {join(SCR_DIR, 'generate_fits.py')} -n={n} -d={d} -ID={ID} -g1={g1} -g2={g2} -theta_int={theta_int} -sini={sini} -v0={v0} -vcirc={vcirc} -rscale={rscale} -hlr={hlr} -dx_disk={dx_disk} -dy_disk={dy_disk} -dx_spec={dx_spec} -dy_spec={dy_spec} {low_psf_str}")