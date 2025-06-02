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
parser.add_argument('-n', type=int, default=1, help='array id')
group = parser.add_mutually_exclusive_group()
args = parser.parse_args()
i = args.i
j = args.j
n = args.n

SAMP_FILE = '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_test_low_g_1m.csv'

os.system(f"mkdir /xdisk/timeifler/wxs0703/kl_nn/test_data/temp_{n}")

df = pd.read_csv(SAMP_FILE)
nsamps = len(df)
chunk = np.array(df.iloc[i:j]) if j < nsamps else np.array(df.iloc[i:])

for row in chunk:
    ID, g1, g2, theta_int, sini, v0, vcirc, rscale, hlr = row
    ID = int(ID)
    os.system(f"python generate_training_set.py -n={n} -isTrain={0} -ID={ID} -g1={g1} -g2={g2} -theta_int={theta_int} -sini={sini} -v0={v0} -vcirc={vcirc} -rscale={rscale} -hlr={hlr}")