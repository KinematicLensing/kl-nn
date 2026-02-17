from __future__ import print_function
from os.path import join
import time

import numpy as np
import pandas as pd
import pyxis as px
from astropy.io import fits
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-s', type=str, default='small', help='sample name')
parser.add_argument('-d', type=str, default='small', help='dataset name')
parser.add_argument('-n', type=int, default=4000, help='number of samples per db entry')
parser.add_argument('-N', type=int, default=250, help='number of db entries')
parser.add_argument('--nspec', type=int, default=5, help='number of spectra per sample')
args = parser.parse_args()
n = args.n
N = args.N
nspec = args.nspec
sample_name = args.s
dataset_name = args.d

def normalize(form, data, pars=None):
    '''
    Normalizes data into one of three forms:
    '01': normalize between 0 and 1, pars = (min, max)
    '-11': normalize between -1 and 1, pars = (min, max)
    'std': standardize to center around 0 with std dev of 1, pars = (mean, std)
    '''
    if form == 'std': 
        mean, std = pars if pars is not None else (data.mean(), data.std())
        new_data = (data-mean)/std
        
        return new_data
    
    else:
        min_val, max_val = pars if pars is not None else (np.min(data), np.max(data))
        
        if form == '01':
            new_data = (data - min_val)/(max_val-min_val)
            
            return new_data
        
        elif form == '-11':
            new_data = (2*data - (max_val+min_val))/(max_val-min_val)
            
            return new_data
        
        else:
            raise ValueError("Invalid form, must be '01', '-11', or 'std'.")

par_ranges = \
{
    'g1': [-0.1, 0.1],
    'g2': [-0.1, 0.1],
    'theta_int': [-np.pi, np.pi],
    'sini': [0, 1],
    'v0': [-30, 30],
    'vcirc': [60, 540],
    'rscale': [0.1, 10],
    'hlr': [0.1, 1],
}

def main():
    data_dir = f'/ocean/projects/phy250048p/shared/fits/{dataset_name}/'
    samp_dir = f'/ocean/projects/phy250048p/shared/samples/samples_{sample_name}.csv'
    save_dir = f'/ocean/projects/phy250048p/shared/datasets/{dataset_name}'
    samples = pd.read_csv(samp_dir)
    for i, values in enumerate(par_ranges.values()):
        samples.iloc[:, i+1] = normalize('-11', samples.iloc[:, i+1], values)
    samples.to_csv(f'/ocean/projects/phy250048p/shared/samples/normalized/samples_{dataset_name}_normalized.csv', index=False)
        
    with px.Writer(dirpath=save_dir, map_size_limit=200000, ram_gb_limit=2) as db:
        
        for index in range(N):
            start = time.time()
            folder = index+1
            img_stack = np.full((n, 1, 48, 48), 0.)
            spec_stack = np.full((n, 1, nspec, 64), 0.)
            start_id = index*n
            file_id = index*n
            ids = np.arange(start_id, start_id+n, dtype=np.uint64)
            fids = np.array(samples.iloc[ids])[:, 1:]

            for i in range(n):
                
                ID = file_id + i

                with fits.open(join(data_dir, f'part_{folder}/gal_{ID}.fits')) as hdu:
                    img_stack[i, 0] = hdu[nspec+1].data
                    
                    specs = np.full((nspec, 64), 0.)
                    for k in range(nspec):
                        spec = hdu[k+1].data
                        specs[k, :spec.shape[0]] = spec
                    spec_stack[i, 0] = specs
            
            # img_stack = normalize('std', img_stack, (img_mean, img_std))
            # spec_stack = normalize('std', spec_stack, (spec_mean, spec_std))
            
            db.put_samples({'img': img_stack,
                            'spec': spec_stack,
                            'fid_pars': fids,
                            'id': ids})
            t = round(time.time() - start, 2)
            
            print(f'folder {folder} complete, {t} seconds')

if __name__ == '__main__':
    main()