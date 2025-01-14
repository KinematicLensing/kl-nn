# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
_USER_RUNNER_CLASS_ = False
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from argparse import ArgumentParser
import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-i', type=int, default=0, help='start index')
parser.add_argument('-j', type=int, default=1, help='stop index')
group = parser.add_mutually_exclusive_group()
args = parser.parse_args()
i = args.i
j = args.j

SAMP_FILE = '/xdisk/timeifler/wxs0703/desi_nn_mcmc/samples/samples.csv'

df = pd.read_csv(SAMP_FILE)
nsamps = len(df)
chunk = np.array(df.iloc[i:j]) if j < nsamps else np.array(df.iloc[i:])

for row in chunk:
    ID, g1, g2, theta_int, sini, v0, vcirc, rscale, hlr = row
    ID = int(ID)
    os.system(f"python generate_training_set.py -ID={ID} -g1={g1} -g2={g2} -theta_int={theta_int} -sini={sini} -v0={v0} -vcirc={vcirc} -rscale={rscale} -hlr={hlr}")