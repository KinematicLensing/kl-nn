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
    param_list = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr', 'dx_disk', 'dy_disk', 'dx_spec', 'dy_spec']
    sample_limits = [[-0.1, 0.1],     # g1
                     [-0.1, 0.1],     # g2
                     [-np.pi, np.pi], # theta_int
                     [0, 1],          # sini
                     [-30, 30],       # v0
                     [60, 540],       # vcirc
                     [0.1, 2],       # rscale
                     [0.1, 3],        # hlr
                     [-0.5, 0.5],         # dx_disk
                     [-0.5, 0.5],      # dy_disk
                     [-0.5, 0.5],      # dx_spec
                     [-0.5, 0.5]]        # dy_spec
    ndim = len(sample_limits)
    defaults = {
                # 'g1': 0.,
                # 'g2': 0.,
                # 'theta_int': np.arange(-np.pi, np.pi, np.pi/16),
                # 'sini': 0.0,
                # 'v0': 0.,
                # 'vcirc': 300.,
                # 'rscale': 1.5,
                # 'hlr': 2.0,
                'dx_disk': 0.,
                'dy_disk': 0.,
                'dx_spec': 0.,
                'dy_spec': 0.
                }
    nsamples = int(5e6)

    sample_centers = []
    sample_scale = []
    for limit in sample_limits:
        sample_centers.append((limit[-1] + limit[0])/2)
        sample_scale.append(limit[-1] - limit[0])

    # Initialize Latin Hypercube Sampler
    LHS = LatinHypercube(ndim, scramble=True)
    samples = LHS.random(nsamples) - 0.5
    df = pd.DataFrame(samples, columns=param_list)

    for i, param in enumerate(param_list):
        df[param] = df[param]*sample_scale[i] + sample_centers[i]
    
    # # Optional set some parameters to default values, comment out if not needed
    # for param, _ in df.items():
    #     if param in defaults.keys():
    #         if type(defaults[param]) != np.ndarray:
    #             df[param] = defaults[param]
    #         else:
    #             values = defaults[param]
    #             df_list = [df.copy() for _ in range(len(values))]
    #             for i, value in enumerate(values):
    #                 # print(df_list[i][param][0])
    #                 df_list[i][param] = value
    #             df = pd.concat(df_list, ignore_index=True)

    # df['sini'] = np.sqrt(1-df['sini']**2)

    # df_rot90 = df.copy()
    # df_rot90['theta_int'] = df_rot90['theta_int'] - np.pi/2
    # theta_mask = df_rot90['theta_int'] < -np.pi
    # df_rot90.loc[theta_mask, 'theta_int'] += 2*np.pi
    # # df_rot90['g1'] = -df_rot90['g1']
    # # df_rot90['g2'] = -df_rot90['g2']
    # df = pd.concat([df, df_rot90], ignore_index=True)
    
    # Save parameter samples
    df.to_csv(join(SAMPDIR, 'samples_pretrain_5m.csv'))


if __name__ == '__main__':
    main()