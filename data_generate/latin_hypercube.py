from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

FIGDIR = '/ocean/projects/phy250048p/shared/figures/'
SAMPDIR = '/ocean/projects/phy250048p/shared/samples/'

def main():
    # Define sample limits
    # Do this for all 8 parameters even if only fitting for shear, necessary for simulation
    param_list = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr']
    sample_limits = [[-0.1, 0.1],     # g1
                     [-0.1, 0.1],     # g2
                     [-np.pi, np.pi], # theta_int
                     [0, 1],          # sini
                     [-30, 30],       # v0
                     [60, 540],       # vcirc
                     [0.1, 2],       # rscale
                     [0.1, 3]]        # hlr
    ndim = len(sample_limits)
    nsamples = int(1e4)

    sample_centers = []
    sample_scale = []
    for limit in sample_limits:
        sample_centers.append((limit[-1] + limit[0])/2)
        sample_scale.append(limit[-1] - limit[0])

    # Initialize Latin Hypercube Sampler
    LHS = LatinHypercube(8, scramble=True)
    samples = LHS.random(nsamples) - 0.5
    df = pd.DataFrame(samples, columns=param_list)

    for i, param in enumerate(param_list):
        df[param] = df[param]*sample_scale[i] + sample_centers[i]
    
    # df['sini'] = np.sqrt(1-df['sini']**2)
    
    # Save parameter samples
    df.to_csv(join(SAMPDIR, 'samples_small_1m.csv'))


if __name__ == '__main__':
    main()