# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
_USER_RUNNER_CLASS_ = False
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, copy, os
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle

# From KL-tools
import priors, likelihood
from cube import FiberModelCube
from parameters import Pars, MetaPars
from likelihood import LogPosterior, GrismLikelihood, get_GlobalDataVector, FiberLikelihood
from velocity import VelocityMap
from datavector import FiberDataVector
from emission import LINE_LAMBDAS

import ipdb
import time

########################### Parsing arguments ##################################
parser = ArgumentParser()
parser.add_argument('-n', type=int, default=1, help='folder id')
parser.add_argument('-isTrain', type=int, default=1, help='folder id')
parser.add_argument('-ID', type=int, default=0, help='sample id')
parser.add_argument('-g1', type=float, default=0, help='shear component 1')
parser.add_argument('-g2', type=float, default=0, help='shear component 2')
parser.add_argument('-theta_int', type=float, default=0, help='position angle')
parser.add_argument('-sini', type=float, default=0.5, help='inclination angle')
parser.add_argument('-v0', type=float, default=0, help='systemic velocity')
parser.add_argument('-vcirc', type=float, default=300, help='max rotation velocity')
parser.add_argument('-rscale', type=float, default=1, help='velocity scale radius')
parser.add_argument('-hlr', type=float, default=1, help='half-light radius')
group = parser.add_mutually_exclusive_group()
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing` sequencial pool).')
args = parser.parse_args()
isTrain = bool(args.isTrain)
n = args.n
ID = args.ID
g1 = args.g1
g2 = args.g2
theta_int = args.theta_int
sini = args.sini
v0 = args.v0
vcirc = args.vcirc
rscale = args.rscale
hlr = args.hlr

fiber_blur = 3.4 # pixels
atm_psf_fwhm = 1.0 # arcsec
fiber_rad = 0.75 # arcsec
fiber_offset = 1.5 # arcsec
exptime_offset = 600 # seconds
exptime_photo = -1
ADD_NOISE = False

TRAIN_DIR = f'/xdisk/timeifler/wxs0703/kl_nn/train_data/temp_{n}/'
TEST_DIR = f'/xdisk/timeifler/wxs0703/kl_nn/test_data/temp_{n}/'

##################### Setting up observation configurations ####################

default_photo_conf = {'INSTNAME': "CTIO/DECam", 'OBSTYPE': 0, 'NAXIS': 2,
    'NAXIS1': 48, 'NAXIS2': 48, 'PIXSCALE': 0.2637, 'PSFTYPE': "airy_fwhm",
    'PSFFWHM': 1.0, 'DIAMETER': 378.2856, 'GAIN': 4.0,
    'NOISETYP': 'ccd', 'RDNOISE': 2.6, 'ADDNOISE': ADD_NOISE
}

default_fiber_conf = {'INSTNAME': "DESI", 'OBSTYPE': 1,
    'SKYMODEL': "../data/Skyspectra/spec-sky.dat", 'PSFTYPE': "airy_fwhm", 
    'PSFFWHM': 1.0, 'DIAMETER': 332.42, 'EXPTIME': 180, 'GAIN': 1.0,
    'NOISETYP': 'ccd', 'ADDNOISE': ADD_NOISE, 'FIBERRAD': 0.75, 'FIBRBLUR': 3.4
}

default_obs_conf, _index_ = [], 0

### Fiber observations
emlines = ['O2', 'Hb', 'O3_1', 'O3_2', 'Ha']
channels = ['b', 'r', 'r', 'r', 'z']
rdnoise = [3.41, 2.6, 2.6, 2.6, 2.6]
wavelength_range = np.array([
                        [482.3, 487.11], 
                        [629.7, 634.51],
                        [642.4, 647.21], 
                        [648.7, 653.51],
                        [851, 855.81],
                    ])
SPEC_MASK = 1
spec_mask_str = ("{0:0%db}"%len(emlines)).format(SPEC_MASK)
spec_mask = np.array([int(bit) for bit in spec_mask_str])
Nspec_used = np.sum(spec_mask)
blockids = [int(np.sum(spec_mask[:i])*spec_mask[i]) for i in range(len(spec_mask))]
    
### Choose fiber configurations
cosi = np.sqrt(1-sini**2)
A = np.array([[1+g1, g2],
              [g2, 1-g1]])
R = np.array([[np.cos(theta_int), -np.sin(theta_int)],
              [np.sin(theta_int), np.cos(theta_int)]])
P = np.array([[1, 0],
              [0, cosi]])
T = np.matmul(A, np.matmul(R, P))
U, S, Vh = np.linalg.svd(T)
offsets = [(fiber_offset*np.cos(0),         fiber_offset*np.sin(0)),
           (fiber_offset*np.cos(np.pi),   fiber_offset*np.sin(np.pi)),
           (0,0)]
offsets = np.matmul(offsets, U)
OFFSETX = 1

for i in range(len(emlines)):
    if spec_mask[i]==1:
        eml, bid, chn, rdn = emlines[i], blockids[i], channels[i], rdnoise[i]
        _bp = "../data/Bandpass/DESI/%s.dat"%(chn)
        for (dx, dy) in offsets:
            _conf = copy.deepcopy(default_fiber_conf)
            _conf.update({'OBSINDEX': _index_, 'SEDBLKID': bid, 'BANDPASS': _bp,
                'RDNOISE': rdn, 'FIBERDX': dx, 'FIBERDY': dy})
            if np.abs(dx)>1e-3 or np.abs(dy)>1e-3:
                _conf.update({'EXPTIME': exptime_offset*OFFSETX})
            default_obs_conf.append(_conf)
            _index_+=1
            

### Photometry observations
photometry_band = ['r', 'g', 'z']
sky_levels = [44.54, 19.02, 168.66]
LS_DR9_exptime = [60, 100, 80]
PHOT_MASK = 7
phot_mask = np.array([int(bit) for bit in ("{0:0%db}"%len(photometry_band)).format(PHOT_MASK)])
Nphot_used = np.sum(phot_mask)

for i in range(len(photometry_band)):
    if phot_mask[i]==1:
        _bp = "../data/Bandpass/CTIO/DECam.%s.dat"%photometry_band[i]
        _conf = copy.deepcopy(default_photo_conf)
        _conf.update({"OBSINDEX": _index_, 'BANDPASS': _bp, "SKYLEVEL": sky_levels[i],
            "EXPTIME": exptime_photo if exptime_photo>0 else LS_DR9_exptime[i]})
        default_obs_conf.append(_conf)
        _index_+=1

def main():
    ########################### Initialization #################################
    redshift = 0.3
    ################## Params Sampled & Fiducial Values ########################
    sampled_pars = [
        "g1",
        "g2",
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        'hlr'
        ]
    sampled_pars_value_dict = {
        "g1": g1,
        "g2": g2,
        "theta_int": theta_int,
        "sini": sini,
        "v0": v0,
        "vcirc": vcirc,
        "rscale": rscale,
        "hlr": hlr
    }
    ########################### Supporting #################################
    sampled_pars_label = {
        "g1": r'g_1', 
        "g2": r'g_2', 
        "theta_int": r'{\theta}_{\mathrm{int}}',
        "sini": r'\mathrm{sin}(i)',
        "v0": r'v_0', 
        "vcirc": r'v_\mathrm{circ}', 
        "rscale": r'r_\mathrm{scale}',
        "hlr": r'\mathrm{hlr}'
    }
    param_limit = {
        "g1": [-1.0, 1.0],
        "g2": [-1.0, 1.0],
        "theta_int": [-np.pi, np.pi],
        "sini": [0, 1.0],
        "v0": [-30, 30],
        "vcirc": [60, 540],
        "rscale": [0, 10],
        "hlr": [0, 5]
    }
    sampled_pars_std_dict = {
        "g1": 0.01,
        "g2": 0.01,
        "eint1": 0.01,
        "eint2": 0.01,
        "theta_int": 0.01,
        "sini": 0.01,
        "v0": 1,
        "vcirc": 1,
        "rscale": 0.01,
        "hlr": 0.01
    }
    sampled_pars_value = [sampled_pars_value_dict[k] for k in sampled_pars]
    sampled_pars_std=np.array([sampled_pars_std_dict[k] for k in sampled_pars])
    sampled_pars_std /= 1000

    meta_pars = {
        ### priors
        'priors': {
            'g1': priors.UniformPrior(-0.99, 0.99,),
            'g2': priors.UniformPrior(-0.99, 0.99),
            'theta_int': priors.UniformPrior(-np.pi/2., np.pi/2.),
            'sini': priors.UniformPrior(-1., 1.),
            'v0': priors.GaussPrior(0, 10),
            'vcirc': priors.LognormalPrior(300, 0.06, clip_sigmas=3),
            'rscale': priors.UniformPrior(0.1, 5),
            'hlr': priors.UniformPrior(0.1, 5),
        },
        ### velocity model
        'velocity': {
            'model': 'default',
            'v0': 'param',
            'vcirc': 'param',
            'rscale': 'param',
        },
        ### intensity model
        'intensity': {
            ### Inclined Exp profile
            'type': 'inclined_exp',
            'flux': 1.0, # counts
            'hlr': 1.0,
        },
        ### misc
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('arcsec')
        },
        'run_options': {
            'run_mode': 'ETC',
            #'remove_continuum': True,
            'use_numba': False,
            'alignment_params': 'sini_pa', # eint | inc_pa | sini_pa | eint_eigen
        },
        ### 3D underlying model dimension
        'model_dimension':{
            'Nx': 64,
            'Ny': 64,
            'lblue': 300,
            'lred': 1200,
            'resolution': 500000,
            'scale': 0.11, # arcsec
            'lambda_range': wavelength_range[np.where(spec_mask==1)[0]],
            'lambda_res': 0.08, # nm
            'super_sampling': 4,
            'lambda_unit': 'nm',
        },
        ### SED model
        # typical values: cont 4e-16, emline 1e-16-1e-15 erg/s/cm2/nm
        'sed':{
            'z': redshift,
            'continuum_type': 'temp',
            'restframe_temp': '../data/Simulation/GSB2.spec',
            'temp_wave_type': 'Ang',
            'temp_flux_type': 'flambda',
            'cont_norm_method': 'flux',
            'obs_cont_norm_wave': 850,
            'obs_cont_norm_flam': 3.0e-17,
            'em_Ha_flux': 1.2e-16,
            'em_Ha_sigma': 0.065,
            'em_O2_flux': 8.8e-17,
            'em_O2_sigma': [0.065, 0.065],
            'em_O2_share': [0.45, 0.55],
            'em_O3_1_flux': 2.4e-17,
            'em_O3_1_sigma': 0.065,
            'em_O3_2_flux': 2.8e-17,
            'em_O3_2_sigma': 0.065,
            'em_Hb_flux': 1.2e-17,
            'em_Hb_sigma': 0.065,
        },
        ### observation configurations
        'obs_conf': default_obs_conf,
    }
    pars = Pars(sampled_pars, meta_pars)
     
    fiberlike = FiberLikelihood(pars, None, sampled_theta_fid=sampled_pars_value)
    datavector = get_GlobalDataVector(0)
    if isTrain:
        print(f'Train #{ID} generated')
        datavector.to_fits(os.path.join(TRAIN_DIR, f'training_{ID}.fits'), overwrite=True)
    else:
        print(f'Train #{ID} generated')
        datavector.to_fits(os.path.join(TEST_DIR, f'testing_{ID}.fits'), overwrite=True)
    
    return 0

if __name__ == '__main__':

    rc = main()

    if rc != 0:
        print(f'Tests failed with return code of {rc}')
