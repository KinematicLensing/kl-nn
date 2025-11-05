from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import os

FIGDIR = '/ocean/projects/phy250048p/shared/figures/'
SAMPDIR = '/ocean/projects/phy250048p/shared/samples/'

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(SAMPDIR, exist_ok=True)

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
                     [0.1, 10],       # rscale
                     [0.1, 5]]        # hlr
    ndim = len(sample_limits)
    nsamples = int(1e5)

    sample_centers = []
    sample_scale = []
    for limit in sample_limits:
        sample_centers.append((limit[-1] + limit[0])/2)
        sample_scale.append(limit[-1] - limit[0])

    # Initialize Latin Hypercube Sampler
    LHS = LatinHypercube(8, scramble=False)
    samples = LHS.random(nsamples) - 0.5
    df = pd.DataFrame(samples, columns=param_list)

    for i, param in enumerate(param_list):
        df[param] = df[param]*sample_scale[i] + sample_centers[i]

    # # Code for sampling a uniform circular distribution of shear values, but probably retired
    # df['g1+g2'] = np.sqrt(df['g1+g2'])
    # param_list[:2] = ['g1', 'g2']

    # g1 = df['g1+g2']*np.cos(df['phi'])
    # g2 = df['g1+g2']*np.sin(df['phi'])

    # df = df.drop(columns=['g1+g2', 'phi'])
    # df.insert(0, 'g2', g2)
    # df.insert(0, 'g1', g1)

    # # Plot distribution of samples in parameter space (to check if distribution is sampled correctly)
    # plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})
    # fig, axs = plt.subplots(2, 2)

    # for i, ax in enumerate(axs.reshape(-1)):
    #     p1 = param_list[i*2]
    #     p2 = param_list[i*2+1]
    #     ax.scatter(df[p1], df[p2], marker='.', color='blue', s=0.1)
    #     ax.set_xlabel(p1)
    #     ax.set_ylabel(p2)
    #     ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
    # plt.tight_layout()
    # #plt.savefig(join(FIGDIR, 'sample_dist_low_g.jpg'), dpi=300)
    # plt.close(fig)
    
    # Save parameter samples
    df.to_csv(join(SAMPDIR, 'samples_small.csv'))


if __name__ == '__main__':
    main()