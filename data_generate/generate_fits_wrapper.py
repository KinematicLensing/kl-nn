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
args = parser.parse_args()
i = args.i
j = args.j
n = args.n
s = args.s
d = args.d

SAMP_FILE = f'/ocean/projects/phy250048p/shared/samples/{s}'

os.system(f"mkdir /ocean/projects/phy250048p/shared/fits/{d}/part_{n}/")

df = pd.read_csv(SAMP_FILE)
nsamps = len(df)
chunk = np.array(df.iloc[i:j]) if j < nsamps else np.array(df.iloc[i:])

for row in chunk:
    ID, g1, g2, theta_int, sini, v0, vcirc, rscale, hlr = row
    ID = int(ID)
    os.system(f"python {join(SCR_DIR, 'generate_fits.py')} -n={n} -d={d} -ID={ID} -g1={g1} -g2={g2} -theta_int={theta_int} -sini={sini} -v0={v0} -vcirc={vcirc} -rscale={rscale} -hlr={hlr}")