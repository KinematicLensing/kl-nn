from os.path import join
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from tng_rotation_fit import fit_galaxy_rotation_params

FIGDIR = '/ocean/projects/phy250048p/shared/figures/'
SAMPDIR = '/ocean/projects/phy250048p/shared/samples/'

def main():
    # Define sample limits
    # Sampling 4 parameters for TNG
    param_list = ['g1', 'g2', 'i', 'theta_int']
    sample_limits = [[-0.1, 0.1],     # g1
                     [-0.1, 0.1],     # g2
                     [0, np.pi],      # i (inclination angle in radians)
                     [-np.pi, np.pi]] # theta_int
    ndim = len(sample_limits)
    nsamples = int(1e4)
    ngals = 10

    sample_centers = []
    sample_scale = []
    for limit in sample_limits:
        sample_centers.append((limit[-1] + limit[0])/2)
        sample_scale.append(limit[-1] - limit[0])

    for n in range(50, 50+ngals):
        fit = fit_galaxy_rotation_params(n)
        print(
            f'galaxy {n}: fitted v0={fit.v0:.3f}, vcirc={fit.vcirc:.3f}, '
            f'rscale={fit.rscale:.3f}, rmse={fit.rmse:.3f}, bins={fit.n_profile_bins}'
        )

        # Initialize Latin Hypercube Sampler
        LHS = LatinHypercube(4, scramble=True, seed=n)
        samples = LHS.random(nsamples) - 0.5
        df = pd.DataFrame(samples, columns=param_list)

        for i, param in enumerate(param_list):
            df[param] = df[param]*sample_scale[i] + sample_centers[i]

        df['v0'] = fit.v0
        df['vcirc'] = fit.vcirc
        df['rscale'] = fit.rscale
        df['rmse'] = fit.rmse
        df = df[['g1', 'g2', 'theta_int', 'i', 'v0', 'vcirc', 'rscale', 'rmse']]
        df.insert(0, 'row_id', np.arange(nsamples, dtype=np.int64))

        # Save parameter samples
        df.to_csv(join(SAMPDIR, f'samples_test_tng_10k_{n}.csv'), index=False)


if __name__ == '__main__':
    main()
