from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGDIR = '/ocean/projects/phy250048p/shared/figures/'
SAMPDIR = '/ocean/projects/phy250048p/shared/samples/'

def main():
    # Define sample limits
    # Do this for all 8 parameters even if only fitting for shear, necessary for simulation
    defaults = {'g1': 0.,
                'g2': 0.,
                'theta_int': 0.,
                'sini': 0.7,
                'v0': 0.,
                'vcirc': 300.,
                'rscale': 1.5,
                'hlr': 2.0,
                'n_s': 1.0}
    param_list = list(defaults.keys())
    sampled_params = ['theta_int', 'sini']
    sample_size = [40, 25]
    sample_limits = [[-0.1, 0.1],     # g1
                     [-0.1, 0.1],     # g2
                     [-np.pi, np.pi], # theta_int
                     [0, 1],          # sini
                     [-30, 30],       # v0
                     [60, 540],       # vcirc
                     [0.1, 10],       # rscale
                     [0.1, 1],        # hlr
                     [0.5, 5]]        # n_s
    ndim = len(sample_limits)
    nsamples = np.prod(sample_size)

    sample_centers = []
    sample_scale = []
    for limit in sample_limits:
        sample_centers.append((limit[-1] + limit[0])/2)
        sample_scale.append(limit[-1] - limit[0])

    # create meshgrid for sampled parameters
    steps = [1/(s-1) for s in sample_size]
    sample_grid = np.mgrid[0:1.0001:steps[0], 0:1.0001:steps[1]]
    # print(sample_grid[:10])
    for i in range(len(sampled_params)):
        param = sampled_params[i]
        param_idx = param_list.index(param)
        sample_grid[i] = (sample_grid[i] - 0.5)*sample_scale[param_idx] + sample_centers[param_idx]
        
    samples = np.zeros((nsamples, ndim))
    for i in range(ndim):
        if param_list[i] in sampled_params:
            param_idx = sampled_params.index(param_list[i])
            samples[:, i] = sample_grid[param_idx].flatten()
        else:
            samples[:, i] = defaults[param_list[i]]
    
    df = pd.DataFrame(samples, columns=param_list)

    # Save parameter samples
    df.to_csv(join(SAMPDIR, 'samples_sini_theta_var.csv'))


if __name__ == '__main__':
    main()