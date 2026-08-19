from argparse import ArgumentParser
from os.path import join
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

try:
    from .observation_schema import (
        CURRENT_OBSERVATION_MODEL_VERSION,
        DEFAULT_HALPHA_FLUX_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_AXIS_FIBER_LAYOUT,
        LEGACY_OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import (
        CURRENT_OBSERVATION_MODEL_VERSION,
        DEFAULT_HALPHA_FLUX_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_AXIS_FIBER_LAYOUT,
        LEGACY_OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )

FIGDIR = '/ocean/projects/phy250048p/shared/figures/'
SAMPDIR = '/ocean/projects/phy250048p/shared/samples/'

PARAMETER_LIMITS = {
    'g1': (-0.1, 0.1),
    'g2': (-0.1, 0.1),
    'theta_int': (-np.pi, np.pi),
    'sini': (0.0, 1.0),
    'v0': (-30.0, 30.0),
    'vcirc': (60.0, 540.0),
    'rscale': (0.1, 2.0),
    'hlr': (0.1, 3.0),
}
LEGACY_SUBPIXEL_PARAMETER_LIMITS = {
    'dx_disk': (-0.5, 0.5),
    'dy_disk': (-0.5, 0.5),
    'dx_spec': (-0.5, 0.5),
    'dy_spec': (-0.5, 0.5),
}
DEFAULT_RMAG_RANGE = (15.0, 23.4)


def generate_samples(
    nsamples: int,
    *,
    seed: int | None = None,
    observation_model_version: int = CURRENT_OBSERVATION_MODEL_VERSION,
    rmag_range: tuple[float, float] = DEFAULT_RMAG_RANGE,
    halpha_flux_range: tuple[float, float] = DEFAULT_HALPHA_FLUX_RANGE,
    fiber_layout: str | None = None,
) -> pd.DataFrame:
    """Generate proposals without imposing a Tully--Fisher relation.

    The eight inference parameters, ``rmag_true``, and
    ``halpha_flux_true`` use separate randomized Latin hypercubes. Magnitude
    and integrated H-alpha flux are independent of each other and of every
    inference parameter. Legacy-v1 samples retain the historical four
    sub-pixel-offset columns; v2 intentionally does not sample them.
    """

    if nsamples <= 0:
        raise ValueError('nsamples must be positive')
    if observation_model_version not in (
        LEGACY_OBSERVATION_MODEL_VERSION,
        CURRENT_OBSERVATION_MODEL_VERSION,
    ):
        raise ValueError('observation_model_version must be 1 or 2')
    rmag_min, rmag_max = (float(value) for value in rmag_range)
    if not np.isfinite([rmag_min, rmag_max]).all() or rmag_min >= rmag_max:
        raise ValueError('rmag_range must contain two finite increasing values')
    halpha_min, halpha_max = (float(value) for value in halpha_flux_range)
    if (
        not np.isfinite([halpha_min, halpha_max]).all()
        or halpha_min <= 0.0
        or halpha_min >= halpha_max
    ):
        raise ValueError(
            'halpha_flux_range must contain two positive finite increasing values'
        )

    parameter_limits = dict(PARAMETER_LIMITS)
    if observation_model_version == LEGACY_OBSERVATION_MODEL_VERSION:
        parameter_limits.update(LEGACY_SUBPIXEL_PARAMETER_LIMITS)
    parameter_names = list(parameter_limits)
    seed_sequence = np.random.SeedSequence(seed)
    parameter_seed, magnitude_seed, halpha_seed = seed_sequence.spawn(3)
    parameter_lhs = LatinHypercube(
        len(parameter_names),
        scramble=True,
        seed=np.random.default_rng(parameter_seed),
    )
    unit_samples = parameter_lhs.random(nsamples)
    df = pd.DataFrame(unit_samples, columns=parameter_names)

    for name, (lower, upper) in parameter_limits.items():
        df[name] = lower + df[name] * (upper - lower)

    if observation_model_version == CURRENT_OBSERVATION_MODEL_VERSION:
        magnitude_lhs = LatinHypercube(
            1,
            scramble=True,
            seed=np.random.default_rng(magnitude_seed),
        )
        unit_magnitudes = magnitude_lhs.random(nsamples)[:, 0]
        df[RMAG_TRUE_COLUMN] = rmag_min + unit_magnitudes * (rmag_max - rmag_min)
        halpha_lhs = LatinHypercube(
            1,
            scramble=True,
            seed=np.random.default_rng(halpha_seed),
        )
        unit_halpha_fluxes = halpha_lhs.random(nsamples)[:, 0]
        df[HALPHA_FLUX_TRUE_COLUMN] = (
            halpha_min + unit_halpha_fluxes * (halpha_max - halpha_min)
        )

    if fiber_layout is None:
        fiber_layout = (
            GALAXY_AXIS_FIBER_LAYOUT
            if observation_model_version == CURRENT_OBSERVATION_MODEL_VERSION
            else IMAGE_AXIS_FIBER_LAYOUT
        )
    if fiber_layout not in (IMAGE_AXIS_FIBER_LAYOUT, GALAXY_AXIS_FIBER_LAYOUT):
        raise ValueError('fiber_layout must be image_axis or galaxy_axis')
    df[FIBER_LAYOUT_COLUMN] = fiber_layout
    df[OBSERVATION_MODEL_VERSION_COLUMN] = np.int16(observation_model_version)
    return df


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--nsamples', type=int, default=int(5e6))
    parser.add_argument('--seed', type=int)
    parser.add_argument(
        '--observation-model-version',
        type=int,
        choices=(
            LEGACY_OBSERVATION_MODEL_VERSION,
            CURRENT_OBSERVATION_MODEL_VERSION,
        ),
        # A no-argument invocation historically generated the legacy schema.
        # Simulator-v2 jobs must opt in explicitly so an old output path can
        # never be silently overwritten with a different observation model.
        default=LEGACY_OBSERVATION_MODEL_VERSION,
    )
    parser.add_argument('--rmag-min', type=float, default=DEFAULT_RMAG_RANGE[0])
    parser.add_argument('--rmag-max', type=float, default=DEFAULT_RMAG_RANGE[1])
    parser.add_argument(
        '--halpha-flux-min', type=float, default=DEFAULT_HALPHA_FLUX_RANGE[0]
    )
    parser.add_argument(
        '--halpha-flux-max', type=float, default=DEFAULT_HALPHA_FLUX_RANGE[1]
    )
    parser.add_argument('--fiber-layout', choices=('image_axis', 'galaxy_axis'))
    parser.add_argument(
        '--output',
        default=join(SAMPDIR, 'samples_pretrain_5m.csv'),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    df = generate_samples(
        args.nsamples,
        seed=args.seed,
        observation_model_version=args.observation_model_version,
        rmag_range=(args.rmag_min, args.rmag_max),
        halpha_flux_range=(args.halpha_flux_min, args.halpha_flux_max),
        fiber_layout=args.fiber_layout,
    )
    
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
    
    # Preserve the historical first-column sample ID while naming it
    # explicitly for robust downstream column lookup.
    df.to_csv(args.output, index_label='ID')


if __name__ == '__main__':
    main()
