import os
from os.path import join
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from astropy.io import fits
import jax

from kl_pipe.parameters import ImagePars
from kl_pipe.spectral import (
    CubePars,
    FiberPars,
    desi_z_R,
    halpha_vac_line,
    make_spectral_config,
)
from kl_pipe.tng import FiberDataVector, TNG50Galaxy, TNGRenderConfig


TEMPLATE_DIR = '/jet/home/xwang30/kl-tools/data'
OUTPUT_ROOT = '/ocean/projects/phy250048p/shared/fits'

jax.config.update('jax_enable_x64', True)


def log(msg: str) -> None:
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {msg}', flush=True)


def compute_fiber_offsets(
    g1: float,
    g2: float,
    cosi: float,
    theta_int: float,
    fiber_offset: float = 1.5,
) -> np.ndarray:
    """Compute 5 transformed fiber locations from shear, inclination, and PA."""
    A = np.array([[1.0 + g1, g2], [g2, 1.0 - g1]])
    R = np.array(
        [[np.cos(theta_int), -np.sin(theta_int)], [np.sin(theta_int), np.cos(theta_int)]]
    )
    P = np.array([[1.0, 0.0], [0.0, cosi]])

    T = A @ (R @ P)
    U, _, _ = np.linalg.svd(T)

    # Keep orientation consistent with the transformed x-axis reference.
    v_ref = T @ np.array([1.0, 0.0])
    if np.dot(U[:, 0], v_ref) < 0:
        U *= -1.0

    offsets = np.array(
        [
            (fiber_offset * np.cos(0.0), fiber_offset * np.sin(0.0)),
            (fiber_offset * np.cos(np.pi), fiber_offset * np.sin(np.pi)),
            (0.0, 0.0),
            (fiber_offset * np.cos(np.pi / 2.0), fiber_offset * np.sin(np.pi / 2.0)),
            (fiber_offset * np.cos(3.0 * np.pi / 2.0), fiber_offset * np.sin(3.0 * np.pi / 2.0)),
        ]
    )
    return offsets @ U


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-s', type=int, required=True, help='subhalo id (TNG galaxy index)')
    parser.add_argument('-g1', type=float, required=True, help='shear component 1')
    parser.add_argument('-g2', type=float, required=True, help='shear component 2')
    parser.add_argument('-cosi', type=float, required=True, help='cos(inclination angle)')
    parser.add_argument('-theta_int', type=float, required=True, help='intrinsic position angle [rad]')
    parser.add_argument('-ID', type=int, default=0, help='global sample id for output naming')
    parser.add_argument('-d', type=str, default='test_tng_10k', help='dataset directory name')
    parser.add_argument('-z', type=float, default=0.3, help='target redshift')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = join(OUTPUT_ROOT, args.d, f'galaxy_{args.s}')
    os.makedirs(out_dir, exist_ok=True)
    out_file = join(out_dir, f'gal_{args.ID}.fits')

    log(
        f'Start sample: galaxy={args.s}, ID={args.ID}, '
        f'g1={args.g1:.6f}, g2={args.g2:.6f}, cosi={args.cosi:.6f}, theta_int={args.theta_int:.6f}'
    )

    if not (-1.0 <= args.cosi <= 1.0):
        raise ValueError(f'cosi must be in [-1, 1], got {args.cosi}')

    log('Loading TNG galaxy data')
    tng = TNG50Galaxy(index=args.s)
    galaxy = tng.get_galaxy()
    log('Galaxy loaded')

    params = {
        'theta_int': args.theta_int,
        'cosi': args.cosi,
        'x0': 0.0,
        'y0': 0.0,
        'g1': args.g1,
        'g2': args.g2,
    }

    image_pars = ImagePars(shape=(48, 48), pixel_scale=0.2637, indexing='ij')

    # Use a clean render configuration; PSF is applied by observation configs.
    render_cfg = TNGRenderConfig(
        image_pars=image_pars,
        band='r',
        use_dusted=True,
        center_on_peak=True,
        use_native_orientation=False,
        pars=params,
        use_cic_gridding=True,
        target_redshift=args.z,
        preserve_gas_stellar_offset=True,
    )

    spec_cfg = make_spectral_config(lines=[halpha_vac_line()], R_func=desi_z_R)
    cube_pars_phot = CubePars.from_range(image_pars, 565.5, 717.0, 5.0)
    cube_pars_spec = CubePars.from_range(image_pars, 851.0, 855.81, 0.08)
    line_fluxes = {'Ha': 1e-16}
    line_continua = {'Ha': 1e-16}

    phot_obs_cfg = {
        'INSTNAME': 'CTIO/DECam',
        'OBSTYPE': 0,
        'NAXIS': 2,
        'NAXIS1': 48,
        'NAXIS2': 48,
        'PIXSCALE': 0.2637,
        'PSFTYPE': 'airy_fwhm',
        'PSFFWHM': 1.0,
        'DIAMETER': 378.2856,
        'GAIN': 4.0,
        'NOISETYP': 'ccd',
        'RDNOISE': 2.6,
        'ADDNOISE': False,
        'BANDPASS': join(TEMPLATE_DIR, 'Bandpass/CTIO/DECam.r.dat'),
        'SKYLEVEL': 44.54,
        'EXPTIME': 60,
    }

    base_obs_cfg = {
        'INSTNAME': 'DESI',
        'OBSTYPE': 1,
        'SKYMODEL': join(TEMPLATE_DIR, 'Skyspectra/spec-sky.dat'),
        'PSFTYPE': 'airy_fwhm',
        'PSFFWHM': 1.0,
        'DIAMETER': 332.42,
        'EXPTIME': 180,
        'GAIN': 1.0,
        'NOISETYP': 'ccd',
        'ADDNOISE': False,
        'FIBERRAD': 0.75,
        'FIBRBLUR': 3.4,
        'BANDPASS': join(TEMPLATE_DIR, 'Bandpass/DESI/z.dat'),
        'RDNOISE': 2.6,
    }

    log('Generating photometric image')
    phot_pars = FiberPars.from_cube_pars(cube_pars_phot, phot_obs_cfg)
    phot_gen = FiberDataVector(galaxy, phot_pars)
    cube_phot = phot_gen.generate_cube(
        render_cfg,
        cube_pars_phot,
        spec_cfg,
        line_fluxes=line_fluxes,
        line_continua=line_continua,
    )
    image = np.asarray(phot_gen.fiber_observe_cube(cube_phot), dtype=np.float32)

    log('Computing transformed 5-fiber layout')
    offsets = compute_fiber_offsets(args.g1, args.g2, args.cosi, args.theta_int, fiber_offset=1.5)

    log('Generating spectral cube')
    center_cfg = dict(base_obs_cfg)
    center_cfg.update({'FIBERDX': float(offsets[2, 0]), 'FIBERDY': float(offsets[2, 1])})
    center_pars = FiberPars.from_cube_pars(cube_pars_spec, center_cfg)
    center_gen = FiberDataVector(galaxy, center_pars)
    cube = center_gen.generate_cube(
        render_cfg,
        cube_pars_spec,
        spec_cfg,
        line_fluxes=line_fluxes,
        line_continua=line_continua,
    )

    log('Observing cube with 5 fibers')
    spectra = []
    for obs_idx, (dx, dy) in enumerate(offsets):
        obs_cfg = dict(base_obs_cfg)
        obs_cfg.update(
            {
                'OBSINDEX': obs_idx,
                'FIBERDX': float(dx),
                'FIBERDY': float(dy),
            }
        )
        if np.abs(dx) > 1e-3 or np.abs(dy) > 1e-3:
            obs_cfg['EXPTIME'] = 600

        fiber_pars = FiberPars.from_cube_pars(cube_pars_spec, obs_cfg)
        fiber_gen = FiberDataVector(galaxy, fiber_pars)
        spec, _ = fiber_gen.fiber_observe_cube(cube)
        spectra.append(np.asarray(spec, dtype=np.float32))

    flux = np.asarray(spectra, dtype=np.float32)

    # Save data-only FITS with photometric image, spectra, and fiber geometry metadata.
    primary = fits.PrimaryHDU()
    primary.header['GALID'] = int(args.s)
    primary.header['SAMPLEID'] = int(args.ID)
    primary.header['G1'] = float(args.g1)
    primary.header['G2'] = float(args.g2)
    primary.header['COSI'] = float(args.cosi)
    primary.header['THETAINT'] = float(args.theta_int)
    primary.header['REDSHIFT'] = float(args.z)
    primary.header['NFIBERS'] = 5

    image_hdu = fits.ImageHDU(data=image, name='IMAGE')
    flux_hdu = fits.ImageHDU(data=flux, name='FLUX')
    fiber_tbl = fits.BinTableHDU.from_columns(
        [
            fits.Column(name='fiber_id', format='J', array=np.arange(5, dtype=np.int32)),
            fits.Column(name='dx_arcsec', format='E', array=offsets[:, 0].astype(np.float32)),
            fits.Column(name='dy_arcsec', format='E', array=offsets[:, 1].astype(np.float32)),
        ],
        name='FIBERS',
    )

    hdul = fits.HDUList([primary, image_hdu, flux_hdu, fiber_tbl])
    hdul.writeto(out_file, overwrite=True)
    log(f'Saved FITS: {out_file}')
    return 0


if __name__ == '__main__':
    rc = main()
    if rc != 0:
        print(f'Failed with return code {rc}', flush=True)
