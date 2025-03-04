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

SAMP_FILE = '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_massive.csv'

df = pd.read_csv(SAMP_FILE)
no_file = np.load('no_file.npy')

for i in no_file[:2]:
    ID, g1, g2, theta_int, sini, v0, vcirc, rscale, hlr = df.iloc[i]
    ID = int(ID)
    os.system(f"python generate_training_set.py -ID={ID} -g1={g1} -g2={g2} -theta_int={theta_int} -sini={sini} -v0={v0} -vcirc={vcirc} -rscale={rscale} -hlr={hlr}")