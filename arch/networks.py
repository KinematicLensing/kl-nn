import logging
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
import math
import normflows as nf
from nflows.flows.base import Flow
from nflows.distributions.base import Distribution
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import Permutation, ReversePermutation
from nflows.utils import torchutils

from circular_spline import CircularAutoregressiveRationalQuadraticSpline

import config
from utils import resolve_feature_index
from data import (
    D4_ELEMENTS,
    D4_INVERSES,
    TFCalculator,
    apply_d4_to_datavector,
    transform_d4_feature_blocks,
    transform_d4_fiber_mask,
    transform_d4_parameters,
)

FLOW_TYPES = ("affine", "circular_rqs")


class ConditionalNormalWithCircularTheta(Distribution):
    """Conditional Gaussian base on R^(D-1) times Uniform(S1).

    The circular coordinate is the final feature and is represented on the
    canonical normalized interval [-1, 1). Its density is exactly 1/2 with
    respect to that coordinate. This is a proper compact latent base, unlike
    drawing theta from a Gaussian and wrapping the result afterward.
    """

    def __init__(self, features, context_encoder):
        super().__init__()
        if type(features) is not int or features < 2:
            raise ValueError("features must be an integer of at least two")
        self.features = features
        self.linear_features = features - 1
        self.context_encoder = context_encoder
        self.register_buffer(
            "_normal_log_z",
            torch.tensor(
                0.5 * self.linear_features * math.log(2.0 * math.pi),
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def _compute_params(self, context):
        if context is None:
            raise ValueError("Context cannot be None for the conditional base")
        params = self.context_encoder(context)
        expected = 2 * self.linear_features
        if params.shape != (context.shape[0], expected):
            raise RuntimeError(
                "Circular base context encoder must return shape "
                f"({context.shape[0]}, {expected}); got {tuple(params.shape)}"
            )
        return params.chunk(2, dim=-1)

    def _log_prob(self, inputs, context):
        if inputs.ndim != 2 or inputs.shape[1] != self.features:
            raise ValueError(
                f"Expected inputs with shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )
        means, log_stds = self._compute_params(context)
        linear = inputs[:, :-1]
        normalized = (linear - means) * torch.exp(-log_stds)
        log_prob = -0.5 * normalized.square().sum(dim=-1)
        log_prob -= log_stds.sum(dim=-1)
        log_prob -= self._normal_log_z.to(dtype=inputs.dtype)
        log_prob -= math.log(2.0)
        theta = inputs[:, -1]
        on_circle = torch.isfinite(theta) & (theta >= -1.0) & (theta < 1.0)
        return torch.where(
            on_circle,
            log_prob,
            torch.full_like(log_prob, -torch.inf),
        )

    def _sample(self, num_samples, context):
        means, log_stds = self._compute_params(context)
        context_size = context.shape[0]
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(torch.exp(log_stds), num_samples)
        linear = means + stds * torch.randn_like(means)
        theta = 2.0 * torch.rand(
            context_size * num_samples,
            1,
            device=context.device,
            dtype=linear.dtype,
        ) - 1.0
        samples = torch.cat((linear, theta), dim=-1)
        return torchutils.split_leading_dim(
            samples, [context_size, num_samples]
        )

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return torch.cat((means, means.new_zeros(means.shape[0], 1)), dim=-1)


class PeriodicThetaFlow(Flow):
    """Flow whose public theta coordinate is canonicalized on the circle."""

    def __init__(self, transform, distribution, theta_index):
        super().__init__(transform, distribution)
        self.theta_index = int(theta_index)

    def _canonicalize_theta(self, inputs):
        canonical = inputs.clone()
        theta = canonical[..., self.theta_index]
        outside = torch.isfinite(theta) & ((theta < -1.0) | (theta >= 1.0))
        wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
        canonical[..., self.theta_index] = torch.where(
            outside, wrapped, theta
        )
        return canonical

    def _sample(self, num_samples, context):
        return self._canonicalize_theta(super()._sample(num_samples, context))

    def _log_prob(self, inputs, context):
        return super()._log_prob(self._canonicalize_theta(inputs), context)

    def transform_to_noise(self, inputs, context=None):
        return super().transform_to_noise(
            self._canonicalize_theta(inputs), context=context
        )

### Main Network ###
class KLNPE(nn.Module):
    '''
    Main network consisting of feature extraction branches for images and spectra,
    followed by either point estimate or density estimate layers.
    '''
    def __init__(self, 
                 feature_extractor=None,
                 mode=config.train['mode'],    # 0 = point estimate, 1 = density estimate, 2 = density estimate with TF prior
                 batch_size=config.train['batch_size'],
                 nfeatures=config.train['feature_number'],
                 nspec=config.data['nspec'],
                 # Lognormal TF prior parameters (only used when mode == 2)
                 vcirc_dex=config.tf['scatter'],   # scatter in dex; fixed, represents TF relation scatter
                 vcirc_min=config.par_ranges.get('vcirc', [60.0, 540.0])[0],
                 vcirc_max=config.par_ranges.get('vcirc', [60.0, 540.0])[1],
                 vcirc_idx=None,
                 backbone_type=None,
                 posterior_symmetry=None):

        self.bs = batch_size
        self.nfeatures = nfeatures
        self.nspecs = nspec
        self.feature_names = tuple(
            config.train.get(
                'feature_names',
                ["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            )
        )
        if len(self.feature_names) != self.nfeatures:
            raise ValueError(
                "feature_names length must equal nfeatures; "
                f"got {len(self.feature_names)} and {self.nfeatures}"
            )
        if posterior_symmetry is None:
            posterior_symmetry = config.train.get("posterior_symmetry", "none")
        self.posterior_symmetry = str(posterior_symmetry).lower()
        if self.posterior_symmetry not in ("none", "d4"):
            raise ValueError("posterior_symmetry must be 'none' or 'd4'")
        if mode in (0, 1, 2):
            self.mode = mode
        else:
            raise ValueError('Mode must be 0 (point estimate), 1 (density estimate), or 2 (density estimate with TF prior)!')

        # Initialize apparent magnitude TF calculator with standard values
        self.tf_calc = TFCalculator(slope=config.tf['slope'], intercept=config.tf['intercept'])

        # Lognormal TF prior settings (only used when mode == 2)
        # dex is fixed (TF scatter); mu is supplied per-galaxy at runtime from magnitude measurements
        self.vcirc_dex = float(vcirc_dex)
        self.vcirc_log_scale = vcirc_dex * torch.log(torch.tensor(10.)).item()  # convert dex -> natural-log std
        self.vcirc_min = float(vcirc_min)
        self.vcirc_max = float(vcirc_max)
        self.vcirc_jac = 0.5 * (self.vcirc_max - self.vcirc_min)  # |dv/dx| for x in [-1, 1]
        if vcirc_idx is None:
            vcirc_idx = resolve_feature_index(
                self.feature_names, 'vcirc', aliases=('v_circ',)
            )
        self.vcirc_idx = int(vcirc_idx)
        if self.vcirc_idx < 0 or self.vcirc_idx >= self.nfeatures:
            raise ValueError(
                f'vcirc_idx={self.vcirc_idx} is out of bounds for nfeatures={self.nfeatures}'
            )

        super(KLNPE, self).__init__()

        if backbone_type is None:
            backbone_type = config.train.get("backbone_type", "legacy")
        self.feature_extractor = (
            build_feature_extractor(backbone_type, nspec=self.nspecs)
            if feature_extractor is None
            else feature_extractor
        )
        if self.posterior_symmetry == "d4":
            if self.mode == 0:
                raise ValueError("D4 posterior symmetry requires density mode 1 or 2")
            if not callable(getattr(self.feature_extractor, "transform_features", None)):
                raise ValueError(
                    "D4 posterior symmetry requires an equivariant feature extractor "
                    "with transform_features(features, element)"
                )

        # Define point estimate or density estimate layers
        if self.mode == 0:
            ### Fully-connected layers
            self.fully_connected_layer = MLP([1024, 512, 256, self.nfeatures])
            self.loss = nn.MSELoss()
        elif self.mode >= 1:
            # Normalizing flow for density estimation
            self.layer_norm = nn.LayerNorm(1024)
            self.setup_flows()
            if self.flow_type == "circular_rqs":
                self.flow = PeriodicThetaFlow(
                    self.transform, self.base, theta_index=self.theta_idx
                )
            else:
                self.flow = Flow(self.transform, self.base)

    
    def forward(self, x, y, true, fp, mag=None, snr=None):
        '''
        x: image tensor
        y: spectrum tensor
        true: target tensor of shape (batch, nfeatures)
        fp: fiber position tensor of shape (batch, nspecs, 2)
        '''
        raw_features = self.feature_extractor(x, y, fp)

        if self.mode == 0:
            prediction = self.fully_connected_layer(raw_features)
            return self.loss(prediction, true)

        if self.posterior_symmetry == "d4":
            branch_log_prob = self._d4_branch_log_prob_from_features(
                raw_features, true
            )
            per_galaxy_log_prob = branch_log_prob.mean(dim=1)
        else:
            context = self.layer_norm(raw_features)
            per_galaxy_log_prob = self.flow.log_prob(true, context=context)

        if self.mode == 2:
            # TF quantities are D4 scalars. Compute one weight per galaxy and
            # apply it only after averaging that galaxy's eight branch scores.
            weights = self._compute_tf_weights(true, mag, snr)
            return -(weights * per_galaxy_log_prob).mean()
        return -per_galaxy_log_prob.mean()

    def _d4_contexts_from_features(self, raw_features):
        """Build group-major flow contexts as LN(rho_g(raw_features))."""
        if self.posterior_symmetry != "d4":
            raise RuntimeError("D4 contexts require posterior_symmetry='d4'")
        if raw_features.ndim != 2 or raw_features.shape[-1] != 1024:
            raise ValueError("raw feature tensor must have shape (batch, 1024)")
        return torch.stack(
            tuple(
                self.layer_norm(
                    self.feature_extractor.transform_features(raw_features, element)
                )
                for element in D4_ELEMENTS
            ),
            dim=0,
        )

    def _d4_branch_log_prob_from_features(self, raw_features, parameters):
        """Return the eight branch log densities with shape (batch, group)."""
        if parameters.ndim != 2 or parameters.shape[0] != raw_features.shape[0]:
            raise ValueError("parameters must have shape (batch, nfeatures)")
        contexts = self._d4_contexts_from_features(raw_features)
        transformed_parameters = torch.stack(
            tuple(
                transform_d4_parameters(
                    parameters,
                    element,
                    feature_names=self.feature_names,
                )
                for element in D4_ELEMENTS
            ),
            dim=0,
        )
        group_count, batch_size = transformed_parameters.shape[:2]
        log_prob = self.flow.log_prob(
            transformed_parameters.reshape(group_count * batch_size, self.nfeatures),
            context=contexts.reshape(group_count * batch_size, -1),
        )
        return log_prob.reshape(group_count, batch_size).transpose(0, 1)

    def _d4_mixture_log_prob_from_features(self, raw_features, parameters):
        branch_log_prob = self._d4_branch_log_prob_from_features(
            raw_features, parameters
        )
        return torch.logsumexp(branch_log_prob, dim=1) - math.log(len(D4_ELEMENTS))

    def posterior_log_prob(self, x, y, parameters, fp):
        """Evaluate the configured conditional posterior at normalized parameters."""
        if self.mode == 0:
            raise RuntimeError("posterior_log_prob requires density mode 1 or 2")
        raw_features = self.feature_extractor(x, y, fp)
        if self.posterior_symmetry == "d4":
            return self._d4_mixture_log_prob_from_features(raw_features, parameters)
        context = self.layer_norm(raw_features)
        return self.flow.log_prob(parameters, context=context)
    
    def posterior_mean(self, samples):
        """Return an equivariant posterior mean from normalized sample clouds.

        Scalars and shear use arithmetic means. ``theta_int`` uses its directed
        circular mean on the normalized ``[-1, 1)`` coordinate.
        """
        if samples.ndim < 2 or samples.shape[-1] != self.nfeatures:
            raise ValueError("samples must have shape (..., samples, nfeatures)")
        mean = samples.mean(dim=-2)
        theta_idx = resolve_feature_index(self.feature_names, "theta_int")
        theta = samples[..., theta_idx]
        theta_mean = torch.atan2(
            torch.sin(math.pi * theta).mean(dim=-1),
            torch.cos(math.pi * theta).mean(dim=-1),
        ) / math.pi
        mean[..., theta_idx] = torch.remainder(theta_mean + 1.0, 2.0) - 1.0
        return mean

    def point_estimate(self, x, y, fp):
        '''
        Run through feature extraction and return point estimate of parameters
        '''
        z = self.feature_extractor(x, y, fp)
        z = self.fully_connected_layer(z)
        return z
    
    def extract_latent(self, x, y, true, fp):
        '''
        Run through feature extraction but map from true parameters to latent space in flow
        '''
        z = self.feature_extractor(x, y, fp)
        z = self.layer_norm(z)
        latent = self.flow.transform_to_noise(true, context=z)

        return latent

    def _norm_to_vcirc(self, v_norm):
        v_norm = v_norm.clamp(min=-1.0, max=1.0)
        v_circ = self.vcirc_min + 0.5 * (v_norm + 1.0) * (self.vcirc_max - self.vcirc_min)
        return v_circ.clamp(min=1e-8)

    def _tf_log_prob_from_vnorm(self, v_norm, vcirc_mu):
        """
        v_norm: normalized vcirc in [-1, 1], shape (...)
        vcirc_mu: TF prior center in km/s, broadcastable to v_norm
        """
        min_val = 1
        v_circ = self._norm_to_vcirc(v_norm)
        v_circ = torch.where(
            torch.isfinite(v_circ) & (v_circ > 0),
            v_circ,
            torch.full_like(v_circ, min_val),
        )
        mu = vcirc_mu.to(device=v_circ.device, dtype=v_circ.dtype)
        mu = torch.where(
            torch.isfinite(mu) & (mu > 0),
            mu,
            torch.full_like(mu, min_val),
        )
        prior = torch.distributions.LogNormal(
            loc=torch.log(mu),
            scale=torch.full_like(v_circ, self.vcirc_log_scale)
        )
        # return prior.log_prob(v_circ)
        return prior.log_prob(v_circ) + torch.log(torch.full_like(v_circ, self.vcirc_jac))
    
    def _get_tf_prior_params(self, mag, snr):
        """
        Computes the LogNormal prior parameters for vcirc based on magnitude.
        Uses 0.1 dex base width, modified with a magnitude-dependent observational error.
        """
        # Calculate the expected vcirc center (mu)
        vcirc_mu = self.tf_calc.mag_to_vcirc(mag)
        
        if snr is None:
            # Calculate magnitude-dependent observational uncertainty
            # SNR = 5 * 10**(-0.4 * (mag - 23.4))
            snr = 5.0 * torch.pow(10.0, -0.4 * (mag - 23.4))

        sigma_m = 1.086 / snr
        
        # Propagate error into the TF space (sigma_total_dex = sqrt(0.1^2 + (sigma_m / slope)^2))
        slope = self.tf_calc.slope
        sigma_total_dex = torch.sqrt(self.vcirc_dex**2 + (sigma_m / slope)**2)
        
        # Convert dex back to natural log space for LogNormal prior evaluation
        sigma_total_ln = sigma_total_dex * math.log(10.0)
        
        return vcirc_mu, sigma_total_ln

    def _compute_tf_weights(self, true, mag, snr):
        """
        Computes normalized importance weights based on the TF prior for the batch.
        """
        if mag is None:
            return torch.ones(true.size(0), device=true.device, dtype=true.dtype)
            
        v_norm = true[:, self.vcirc_idx]
        v_circ = self._norm_to_vcirc(v_norm)
        
        # Regularize non-finite elements
        min_val = 1.0
        v_circ = torch.where(
            torch.isfinite(v_circ) & (v_circ > 0),
            v_circ,
            torch.full_like(v_circ, min_val),
        )
        
        # Compute prior parameters
        vcirc_mu, sigma_total_ln = self._get_tf_prior_params(mag, snr)
        
        # Match devices and dtypes
        vcirc_mu = vcirc_mu.to(device=v_circ.device, dtype=v_circ.dtype)
        sigma_total_ln = sigma_total_ln.to(device=v_circ.device, dtype=v_circ.dtype)
        
        # Setup the physical LogNormal distribution
        prior = torch.distributions.LogNormal(
            loc=torch.log(vcirc_mu),
            scale=sigma_total_ln
        )
        
        # Calculate the physical log prob and apply change of variables back to normalized [-1, 1] space
        log_prob_physical = prior.log_prob(v_circ)
        log_jacobian = math.log(self.vcirc_jac)
        log_prob_tf = log_prob_physical + log_jacobian
        
        # Convert to importance weights and normalize across the batch elements
        weights = torch.exp(log_prob_tf).detach()
        weights = weights / (weights.mean() + 1e-8)
        return weights

    def _kde_log_density_1d(self, values):
        """
        Gaussian KDE log-density estimate at sample locations.
        values: (N,)
        returns: (N,)
        """
        n = values.shape[0]
        if n < 2:
            return torch.zeros_like(values)

        std = values.std(unbiased=False).clamp(min=1e-6)
        bandwidth = (1.06 * std * (float(n) ** (-1.0 / 5.0))).clamp(min=1e-6)

        diffs = (values[:, None] - values[None, :]) / bandwidth
        log_norm = torch.log(torch.tensor(math.sqrt(2.0 * math.pi), device=values.device, dtype=values.dtype) * bandwidth)
        log_kernel = -0.5 * diffs.pow(2) - log_norm
        return torch.logsumexp(log_kernel, dim=1) - torch.log(torch.tensor(float(n), device=values.device, dtype=values.dtype))

    def setup_flows(self):
        """Set up the selected conditional posterior flow.

        The circular construction moves ``theta_int`` to the final
        autoregressive coordinate once. Every later permutation leaves it
        there, so Euclidean coordinates never condition on an unwrapped angle,
        while theta can still condition on all seven Euclidean parameters.
        """
        num_layers = int(config.flow['num_layers'])
        if num_layers <= 0:
            raise ValueError("flow num_layers must be positive")
        hidden_units = 256
        num_blocks = 2
        context_size = 1024

        self.flow_type = str(config.flow.get('flow_type', 'affine')).lower()
        if self.flow_type not in FLOW_TYPES:
            raise ValueError(
                f"flow_type must be one of {FLOW_TYPES}; got {self.flow_type!r}"
            )
        self.theta_idx = resolve_feature_index(self.feature_names, "theta_int")

        if self.flow_type == "affine":
            # Preserve the historical architecture and state-dict layout.
            self.base = ConditionalDiagonalNormal(
                shape=[self.nfeatures],
                context_encoder=MLP(
                    [context_size, 128, 64, self.nfeatures * 2]
                ),
            )
            transforms = []
            for _ in range(num_layers):
                transforms.append(ReversePermutation(features=self.nfeatures))
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=self.nfeatures,
                        hidden_features=hidden_units,
                        num_blocks=num_blocks,
                        context_features=context_size,
                    )
                )
            self.transform = CompositeTransform(transforms)
            return

        if self.nfeatures < 2:
            raise ValueError("circular_rqs requires at least two parameters")
        num_bins = config.flow.get('num_bins', 8)
        if type(num_bins) is not int or num_bins < 2:
            raise ValueError("flow num_bins must be an integer of at least two")

        non_theta = [
            index for index in range(self.nfeatures) if index != self.theta_idx
        ]
        canonical_order = non_theta + [self.theta_idx]
        internal_theta_idx = self.nfeatures - 1
        scalar_reverse = list(reversed(range(internal_theta_idx))) + [
            internal_theta_idx
        ]
        self.circular_internal_theta_idx = internal_theta_idx
        self.circular_boundary_permutation = tuple(canonical_order)
        self.circular_layer_permutation = tuple(scalar_reverse)

        transforms = [
            Permutation(torch.tensor(canonical_order, dtype=torch.long))
        ]
        for _ in range(num_layers):
            # Unlike the old implementation, this permutation cannot move
            # theta away from the index declared circular in the spline.
            transforms.append(
                Permutation(torch.tensor(scalar_reverse, dtype=torch.long))
            )
            transforms.append(
                CircularAutoregressiveRationalQuadraticSpline(
                    num_input_channels=self.nfeatures,
                    num_blocks=num_blocks,
                    num_hidden_channels=hidden_units,
                    ind_circ=[internal_theta_idx],
                    num_context_channels=context_size,
                    num_bins=num_bins,
                    tail_bound=1.0,
                    identity_init=True,
                )
            )

        self.base = ConditionalNormalWithCircularTheta(
            features=self.nfeatures,
            context_encoder=MLP(
                [context_size, 128, 64, 2 * (self.nfeatures - 1)]
            ),
        )
        self.transform = CompositeTransform(transforms)

    def _draw_flow_samples(
        self,
        num_samples,
        context,
        *,
        sample_id=None,
        canonical_theta=False,
    ):
        """Draw finite samples for every context row, with bounded-coordinate safety."""
        max_tries = 5
        bad_samples_tolerance = 0.75
        theta_idx = getattr(
            self,
            "theta_idx",
            resolve_feature_index(self.feature_names, "theta_int"),
        )
        last_bad = None
        for attempt in range(max_tries):
            samples = self.flow.sample(num_samples, context=context)
            is_circular = (
                getattr(self, "flow_type", "affine") == "circular_rqs"
            )
            if is_circular:
                # A circular sample is an equivalence class modulo two in the
                # theta/pi coordinate. Always return its canonical [-1, 1)
                # representative; unlike clipping, this preserves the angle.
                samples = samples.clone()
                theta = samples[..., theta_idx]
                outside = torch.isfinite(theta) & (
                    (theta < -1.0) | (theta >= 1.0)
                )
                wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
                samples[..., theta_idx] = torch.where(outside, wrapped, theta)
            if canonical_theta:
                samples = samples.clone()
                nonperiodic = [index for index in range(self.nfeatures) if index != theta_idx]
                samples[..., nonperiodic] = samples[..., nonperiodic].clamp(-1.5, 1.5)
                if not is_circular:
                    samples[..., theta_idx] = torch.remainder(
                        samples[..., theta_idx] + 1.0, 2.0
                    ) - 1.0
            else:
                samples = samples.clone()
                nonperiodic = [index for index in range(self.nfeatures) if index != theta_idx]
                samples[..., nonperiodic] = samples[..., nonperiodic].clamp(-1.5, 1.5)
                if not is_circular:
                    samples[..., theta_idx] = samples[..., theta_idx].clamp(-1.5, 1.5)

            finite = torch.isfinite(samples).all(dim=-1)
            bad_count = int((~finite).sum().item())
            total_count = finite.numel()
            last_bad = (bad_count, total_count)
            if bad_count / total_count > bad_samples_tolerance:
                logging.warning(
                    "Sampling for %s produced %d/%d non-finite samples; "
                    "retrying (remaining=%d)",
                    sample_id,
                    bad_count,
                    total_count,
                    max_tries - attempt - 1,
                )
                continue
            if bad_count:
                repaired = samples.clone()
                repair_failed = False
                for context_index in range(samples.shape[0]):
                    valid_indices = torch.nonzero(
                        finite[context_index], as_tuple=False
                    ).flatten()
                    invalid_indices = torch.nonzero(
                        ~finite[context_index], as_tuple=False
                    ).flatten()
                    if invalid_indices.numel() and not valid_indices.numel():
                        repair_failed = True
                        break
                    if invalid_indices.numel():
                        choices = valid_indices[
                            torch.randint(
                                valid_indices.numel(),
                                (invalid_indices.numel(),),
                                device=samples.device,
                            )
                        ]
                        repaired[context_index, invalid_indices] = samples[
                            context_index, choices
                        ]
                if repair_failed:
                    continue
                samples = repaired
            return samples
        raise RuntimeError(
            f"Sampling for {sample_id} failed after {max_tries} attempts; "
            f"last non-finite count was {last_bad}"
        )

    def _apply_tf_resampling(self, samples, mag, snr):
        """Apply the existing per-galaxy TF replacement to a candidate bank."""
        if samples.shape[0] != 1:
            raise ValueError("TF inference resampling currently requires batch size 1")
        if mag is None or snr is None:
            raise ValueError("mode 2 sampling requires both mag and snr")
        candidates = samples[0]
        v_circ = self._norm_to_vcirc(candidates[:, self.vcirc_idx])
        mag_tensor = torch.as_tensor(
            mag, device=candidates.device, dtype=candidates.dtype
        ).reshape(-1)
        snr_tensor = torch.as_tensor(
            snr, device=candidates.device, dtype=candidates.dtype
        ).reshape(-1)
        vcirc_mu, sigma_total_ln = self._get_tf_prior_params(mag_tensor, snr_tensor)
        prior = torch.distributions.LogNormal(
            loc=torch.log(vcirc_mu[0]),
            scale=torch.full_like(v_circ, sigma_total_ln[0]),
        )
        tf_log_p_v = prior.log_prob(v_circ)
        flow_log_p_v = self._kde_log_density_1d(v_circ)
        log_w = tf_log_p_v - flow_log_p_v
        safe_log_w = log_w.float()
        finite = torch.isfinite(safe_log_w)
        fallback = not bool(finite.any())
        if not fallback:
            safe_log_w = torch.where(
                finite,
                safe_log_w,
                torch.full_like(safe_log_w, -torch.inf),
            )
            maximum = safe_log_w.max()
            fallback = not bool(torch.isfinite(maximum))
        if not fallback:
            weights = torch.softmax(safe_log_w - maximum, dim=0)
            fallback = not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0)
        if fallback:
            logging.warning(
                "Mode 2 sampling: invalid log-weights; falling back to uniform resampling."
            )
            weights = torch.full(
                (candidates.shape[0],),
                1.0 / candidates.shape[0],
                device=candidates.device,
                dtype=torch.float32,
            )
            log_w = torch.zeros_like(log_w)
        else:
            log_w = torch.where(
                torch.isfinite(log_w),
                log_w,
                torch.full_like(log_w, -torch.inf),
            )
        indices = torch.multinomial(
            weights, num_samples=candidates.shape[0], replacement=True
        )
        return candidates[indices].unsqueeze(0), log_w[indices]

    def _d4_sample_from_features(self, raw_features, num_samples, sample_id=None):
        if raw_features.shape[0] != 1:
            raise ValueError("D4 posterior sampling currently requires one galaxy")
        if num_samples <= 0 or num_samples % len(D4_ELEMENTS):
            raise ValueError("num_samples must be positive and divisible by 8")
        contexts = self._d4_contexts_from_features(raw_features)[:, 0]
        per_component = num_samples // len(D4_ELEMENTS)
        branch_samples = self._draw_flow_samples(
            per_component,
            contexts,
            sample_id=sample_id,
            canonical_theta=True,
        )
        aligned = tuple(
            transform_d4_parameters(
                branch_samples[index],
                D4_INVERSES[element],
                feature_names=self.feature_names,
            )
            for index, element in enumerate(D4_ELEMENTS)
        )
        return torch.cat(aligned, dim=0).unsqueeze(0)

    def _d4_sample_log_prob(self, raw_features, samples, chunk_size=256):
        if raw_features.shape[0] != 1 or samples.shape[0] != 1:
            raise ValueError("D4 sample scoring currently requires one galaxy")
        scores = []
        for start in range(0, samples.shape[1], chunk_size):
            candidates = samples[0, start : start + chunk_size]
            candidate_features = raw_features.expand(candidates.shape[0], -1)
            scores.append(
                self._d4_mixture_log_prob_from_features(
                    candidate_features, candidates
                )
            )
        return torch.cat(scores, dim=0)

    def sample(
        self,
        x,
        y,
        num_samples,
        fp,
        mag=None,
        snr=None,
        return_log_prob=False,
        log_context=None,
        sample_id=None,
    ):
        """Sample one galaxy's configured conditional posterior.

        The D4 posterior uses a balanced eight-component mixture. Samples drawn
        in each transformed frame are mapped back to the input frame before TF
        replacement and before returning. ``log_context`` is retained only for
        backward call compatibility.
        """
        del log_context
        if self.mode == 0:
            raise RuntimeError("sample requires density mode 1 or 2")
        if x.shape[0] != 1:
            raise ValueError("sample currently requires a single-galaxy batch")
        raw_features = self.feature_extractor(x, y, fp)

        if self.posterior_symmetry == "d4":
            samples = self._d4_sample_from_features(
                raw_features, num_samples, sample_id=sample_id
            )
        else:
            context = self.layer_norm(raw_features)
            samples = self._draw_flow_samples(
                num_samples,
                context,
                sample_id=sample_id,
                canonical_theta=False,
            )

        tf_log_correction = None
        if self.mode == 2:
            samples, tf_log_correction = self._apply_tf_resampling(
                samples, mag, snr
            )

        if not return_log_prob:
            return samples.reshape(1, num_samples, self.nfeatures)

        if self.posterior_symmetry == "d4":
            flow_log_prob = self._d4_sample_log_prob(raw_features, samples)
        else:
            context = self.layer_norm(raw_features).expand(num_samples, -1)
            flow_log_prob = self.flow.log_prob(
                samples.reshape(num_samples, self.nfeatures),
                context=context,
            )
        if tf_log_correction is not None:
            flow_log_prob = flow_log_prob + tf_log_correction
        return samples.reshape(1, num_samples, self.nfeatures), flow_log_prob

    def evaluate_conditional_2d(
        self,
        x,
        y,
        true_params,
        idx1,
        idx2,
        fp=None,
        grid_bins=200,
        bounds=(-1, 1)
    ):
        '''
        Diagnostic: Sample 2 parameters conditional on all other parameters being fixed.
        Evaluates the flow log_prob over a 2D grid and samples from the resulting PDF.
        
        true_params: Tensor of shape (1, nfeatures) containing the fixed parameter values.
        idx1, idx2: Integers representing the parameter indices for g1 and g2.
        '''
        # 1. Extract context 'z' identically to your sample() function
        z = self.feature_extractor(x, y, fp)
        z = self.layer_norm(z)
        
        # 2. Create a 2D grid for the two parameters
        g1_vals = torch.linspace(bounds[0], bounds[1], grid_bins, device=z.device)
        g2_vals = torch.linspace(bounds[0], bounds[1], grid_bins, device=z.device)
        G1, G2 = torch.meshgrid(g1_vals, g2_vals, indexing='ij')
        
        flat_g1 = G1.flatten()
        flat_g2 = G2.flatten()
        num_grid_points = flat_g1.size(0)
        
        # 3. Prepare the massive batch of parameter vectors
        # Clone the true_params vector for every point on the grid
        theta_grid = true_params.repeat(num_grid_points, 1)
        
        # Overwrite the g1 and g2 columns with the grid points
        theta_grid[:, idx1] = flat_g1
        theta_grid[:, idx2] = flat_g2
        
        # Repeat the context vector to match the grid size
        z_rep = z.repeat(num_grid_points, 1)
        
        # 4. Evaluate log probabilities for the entire grid in one forward pass
        log_probs = self.flow.log_prob(theta_grid, context=z_rep)
        
        # 5. Convert to normalized probabilities 
        # (Subtract max for numerical stability before exp to avoid overflow)
        probs = torch.exp(log_probs - log_probs.max())
        probs = probs / probs.sum()
        
        return probs.view(grid_bins, grid_bins), g1_vals, g2_vals

class FeatureExtractor(nn.Module):
    def __init__(self, nspec=config.data['nspec']):
        super().__init__()
        self.nspecs = nspec
        
        # Vision Transformer for images
        # self.img_net = VisionTransformer(
        #     in_channels=1, embed_dim=512, img_size=48, patch_size=6, 
        #     num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1
        # )
        self.img_net = ImgCNN()

        # CNN for spectra
        self.spec_net = SpecCNN(self.nspecs)

    def forward(self, x, y, fp):
        '''Extracts raw features independently for both modalities'''
        x = nn.functional.normalize(x, dim=[2, 3])
        y = nn.functional.normalize(y, dim=[2, 3])
        # fp = nn.functional.normalize(fp, dim=[1, 2])
        fp = fp / 1.5
        fp = fp.view(fp.size(0), -1)
        
        img_feats = self.img_net(x).view(x.size(0), -1)     # Shape: (B, 512)
        spec_feats = self.spec_net(y).view(y.size(0), -1)   # Shape: (B, 512 - 2*nspecs)
        spec_feats = torch.cat((spec_feats, fp), dim=-1)     # Shape: (B, 512)
        
        z = torch.cat((img_feats, spec_feats), -1)

        return z
    

BACKBONE_TYPES = ("legacy", "stage3", "stage4_d4")


class DivisibleMeanPool1d(nn.Module):
    """Deterministic adaptive-mean equivalent for evenly divisible bins."""

    def __init__(self, output_size):
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = int(output_size)

    def forward(self, inputs):
        input_size = inputs.shape[-1]
        if input_size < self.output_size or input_size % self.output_size:
            raise ValueError(
                "spectral feature length must be a positive multiple of "
                f"{self.output_size}; got {input_size}"
            )
        bin_size = input_size // self.output_size
        return inputs.reshape(
            *inputs.shape[:-1], self.output_size, bin_size
        ).mean(dim=-1)


class SharedSpecCNN(nn.Module):
    """Encode every fiber spectrum with the same wavelength-aware Conv1d network."""

    def __init__(self, embedding_dim=128, pooled_length=8):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if pooled_length <= 0:
            raise ValueError("pooled_length must be positive")
        self.embedding_dim = int(embedding_dim)
        self.pooled_length = int(pooled_length)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # For the 64-bin spectra this maps 16 -> 8 exactly like adaptive
            # average pooling, but its CUDA backward is deterministic.
            DivisibleMeanPool1d(self.pooled_length),
        )
        # Flattening the reduced wavelength axis, rather than globally averaging it,
        # preserves absolute line-position information needed for radial velocity.
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * self.pooled_length, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, spectra):
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, fibers, wavelength)")
        batch_size, _, fiber_count, wavelength_count = spectra.shape
        if wavelength_count < 4:
            raise ValueError("spectra must contain at least four wavelength samples")
        shared_input = spectra[:, 0].reshape(
            batch_size * fiber_count, 1, wavelength_count
        )
        encoded = self.projection(self.encoder(shared_input))
        return encoded.reshape(batch_size, fiber_count, self.embedding_dim)


class FiberSetAttention(nn.Module):
    """Position-free self-attention over a set of physical fiber tokens."""

    def __init__(self, token_dim=128, num_heads=4, feedforward_dim=256):
        super().__init__()
        if token_dim <= 0 or token_dim % num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        self.self_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.feedforward_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens, observed_mask, key_padding_mask=None):
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, fibers, token_dim)")
        if observed_mask.shape != tokens.shape[:2]:
            raise ValueError("observed_mask shape must match the token batch and fibers")
        if observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must be a bool tensor")
        if key_padding_mask is not None:
            if key_padding_mask.shape != observed_mask.shape:
                raise ValueError("key_padding_mask shape must match observed_mask")
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be a bool tensor")

        attended, _ = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        observed = observed_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        tokens = self.attention_norm(tokens + attended) * observed
        tokens = self.feedforward_norm(tokens + self.feedforward(tokens)) * observed
        return tokens


class Stage3FeatureExtractor(nn.Module):
    """Permutation-aware image/spectra fusion for the Stage 3 CCL experiment."""

    output_dim = 1024

    def __init__(
        self,
        nspec=config.data['nspec'],
        spectral_embedding_dim=128,
        token_dim=128,
        num_heads=4,
        fiber_position_scale=1.5,
        img_net=None,
    ):
        super().__init__()
        if nspec <= 0:
            raise ValueError("nspec must be positive")
        if fiber_position_scale <= 0 or not math.isfinite(fiber_position_scale):
            raise ValueError("fiber_position_scale must be positive and finite")
        self.nspecs = int(nspec)
        self.fiber_position_scale = float(fiber_position_scale)
        self.img_net = ImgCNN() if img_net is None else img_net
        self.spec_net = SharedSpecCNN(embedding_dim=spectral_embedding_dim)

        # Coordinates contain spin-1 (x,y), scalar (r^2), and spin-2
        # (x^2-y^2,2xy) components. There is deliberately no storage-index
        # embedding: physical positions, spectra, and masks define each token.
        coordinate_dim = 5
        spectral_strength_dim = 1
        observation_dim = 1
        self.token_projection = nn.Sequential(
            nn.Linear(
                spectral_embedding_dim
                + coordinate_dim
                + spectral_strength_dim
                + observation_dim,
                token_dim,
            ),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.fiber_set_encoder = FiberSetAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            feedforward_dim=2 * token_dim,
        )
        self.image_query = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, token_dim),
        )
        self.image_fiber_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attended_norm = nn.LayerNorm(token_dim)
        self.fiber_projection = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.fusion_norm = nn.LayerNorm(self.output_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    @staticmethod
    def _validate_inputs(image, spectra, fiber_positions, fiber_mask):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must have shape (batch, 1, height, width)")
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, fibers, wavelength)")
        if fiber_positions.ndim != 3 or fiber_positions.shape[-1] != 2:
            raise ValueError("fiber_positions must have shape (batch, fibers, 2)")
        if image.shape[0] != spectra.shape[0] or image.shape[0] != fiber_positions.shape[0]:
            raise ValueError("image, spectra, and fiber_positions batch sizes must match")
        if spectra.shape[2] != fiber_positions.shape[1]:
            raise ValueError("spectra and fiber_positions must have the same fiber count")
        if fiber_mask is not None:
            if fiber_mask.dtype != torch.bool:
                raise TypeError("fiber_mask must be a bool tensor")
            if fiber_mask.shape != spectra.shape[:1] + spectra.shape[2:3]:
                raise ValueError("fiber_mask must have shape (batch, fibers)")

    @staticmethod
    def _coordinate_features(fiber_positions):
        x_coord, y_coord = fiber_positions.unbind(dim=-1)
        return torch.stack(
            (
                x_coord,
                y_coord,
                x_coord.square() + y_coord.square(),
                x_coord.square() - y_coord.square(),
                2.0 * x_coord * y_coord,
            ),
            dim=-1,
        )

    @staticmethod
    def _relative_spectral_strength(spectra):
        """Return per-fiber L2 norms relative to the full spectral datavector."""
        fiber_norms = torch.linalg.vector_norm(spectra, dim=-1, keepdim=True)
        total_norm = torch.linalg.vector_norm(fiber_norms, dim=2, keepdim=True)
        return fiber_norms / total_norm.clamp_min(torch.finfo(spectra.dtype).tiny)

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        self._validate_inputs(image, spectra, fiber_positions, fiber_mask)
        batch_size, _, fiber_count, _ = spectra.shape
        explicit_fiber_mask = fiber_mask is not None
        if fiber_mask is None:
            fiber_mask = torch.ones(
                (batch_size, fiber_count),
                dtype=torch.bool,
                device=spectra.device,
            )
        else:
            fiber_mask = fiber_mask.to(device=spectra.device)
            # Explicit variable-fiber batches require at least one real token.
            # The all-observed training path skips this host-visible check.
            if torch.any(~fiber_mask.any(dim=1)):
                raise ValueError("every sample must contain at least one observed fiber")
        key_padding_mask = ~fiber_mask if explicit_fiber_mask else None

        spectral_mask = fiber_mask[:, None, :, None]
        position_mask = fiber_mask.unsqueeze(-1)
        spectra = torch.where(spectral_mask, spectra, torch.zeros_like(spectra))
        fiber_positions = torch.where(
            position_mask,
            fiber_positions,
            torch.zeros_like(fiber_positions),
        )

        image = F.normalize(image, dim=(-2, -1))
        # Per-fiber normalization isolates line shape, while the relative norm
        # retains all amplitude information present under the legacy global
        # spectral normalization.
        relative_spectral_strength = self._relative_spectral_strength(spectra)
        spectra = F.normalize(spectra, dim=-1)
        normalized_positions = fiber_positions / self.fiber_position_scale

        image_features = self.img_net(image).reshape(batch_size, -1)
        if image_features.shape[1] != 512:
            raise ValueError(
                f"image encoder must return 512 features; got {image_features.shape[1]}"
            )
        spectral_features = self.spec_net(spectra)
        coordinate_features = self._coordinate_features(normalized_positions)
        spectral_strength_feature = relative_spectral_strength[:, 0].to(
            spectral_features.dtype
        )
        observation_feature = fiber_mask.unsqueeze(-1).to(spectral_features.dtype)
        fiber_tokens = self.token_projection(
            torch.cat(
                (
                    spectral_features,
                    coordinate_features,
                    spectral_strength_feature,
                    observation_feature,
                ),
                dim=-1,
            )
        )
        fiber_tokens = fiber_tokens * observation_feature
        fiber_tokens = self.fiber_set_encoder(
            fiber_tokens,
            fiber_mask,
            key_padding_mask=key_padding_mask,
        )

        image_query = self.image_query(image_features).unsqueeze(1)
        attended_fibers, _ = self.image_fiber_attention(
            image_query,
            fiber_tokens,
            fiber_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_fibers = self.attended_norm(attended_fibers.squeeze(1))
        fiber_features = self.fiber_projection(attended_fibers)

        joint_features = torch.cat((image_features, fiber_features), dim=-1)
        fused_features = joint_features + self.fusion_mlp(
            self.fusion_norm(joint_features)
        )
        return self.output_norm(fused_features)


class D4OrbitFeatureExtractor(nn.Module):
    """Exactly D4-equivariant multimodal features from a shared orbit backbone.

    The raw 1024 channels are interpreted as scalar, directed spin-1, and
    spin-2 blocks. Every forward pass evaluates all eight complete datavector
    views in one shared-backbone batch, maps each output back to the input
    frame, and averages the aligned orbit.
    """

    output_dim = 1024
    default_scalar_channels = 512
    default_spin1_channels = 256
    default_spin2_channels = 256

    def __init__(
        self,
        nspec=config.data['nspec'],
        base_backbone=None,
        scalar_channels=default_scalar_channels,
        spin1_channels=default_spin1_channels,
        spin2_channels=default_spin2_channels,
    ):
        super().__init__()
        channel_counts = (scalar_channels, spin1_channels, spin2_channels)
        if any(type(value) is not int or value < 0 for value in channel_counts):
            raise ValueError("D4 feature channel counts must be non-negative integers")
        if spin1_channels % 2 or spin2_channels % 2:
            raise ValueError("spin-1 and spin-2 channel counts must be even")
        if sum(channel_counts) != self.output_dim:
            raise ValueError(
                f"D4 feature channel counts must sum to {self.output_dim}"
            )
        self.nspecs = int(nspec)
        self.scalar_channels = scalar_channels
        self.spin1_channels = spin1_channels
        self.spin2_channels = spin2_channels
        self.base_backbone = (
            Stage3FeatureExtractor(nspec=self.nspecs)
            if base_backbone is None
            else base_backbone
        )

    def transform_features(self, features, element):
        """Express equivariant features in the frame transformed by ``element``."""
        return transform_d4_feature_blocks(
            features,
            element,
            scalar_channels=self.scalar_channels,
            spin1_channels=self.spin1_channels,
            spin2_channels=self.spin2_channels,
        )

    def _build_orbit_batch(self, image, spectra, fiber_positions, fiber_mask=None):
        image_views = []
        spectrum_views = []
        position_views = []
        mask_views = []
        for element in D4_ELEMENTS:
            view_image, view_spectra, _, view_positions = apply_d4_to_datavector(
                image,
                spectra,
                fp=fiber_positions,
                element=element,
            )
            image_views.append(view_image)
            spectrum_views.append(view_spectra)
            position_views.append(view_positions)
            if fiber_mask is not None:
                mask_views.append(transform_d4_fiber_mask(fiber_mask, element))

        orbit_image = torch.cat(image_views, dim=0)
        orbit_spectra = torch.cat(spectrum_views, dim=0)
        orbit_positions = torch.cat(position_views, dim=0)
        orbit_mask = torch.cat(mask_views, dim=0) if mask_views else None
        if image.is_contiguous(memory_format=torch.channels_last):
            orbit_image = orbit_image.contiguous(memory_format=torch.channels_last)
        if spectra.is_contiguous(memory_format=torch.channels_last):
            orbit_spectra = orbit_spectra.contiguous(memory_format=torch.channels_last)
        return orbit_image, orbit_spectra, orbit_positions, orbit_mask

    def aligned_orbit_features(
        self,
        image,
        spectra,
        fiber_positions,
        fiber_mask=None,
    ):
        """Return all eight raw orbit features aligned to the input frame."""
        if fiber_positions is None:
            raise ValueError("fiber_positions are required for the D4 orbit backbone")
        batch_size = image.shape[0]
        orbit = self._build_orbit_batch(
            image,
            spectra,
            fiber_positions,
            fiber_mask=fiber_mask,
        )
        if orbit[3] is None:
            raw_features = self.base_backbone(orbit[0], orbit[1], orbit[2])
        else:
            raw_features = self.base_backbone(*orbit)
        expected_shape = (len(D4_ELEMENTS) * batch_size, self.output_dim)
        if raw_features.shape != expected_shape:
            raise ValueError(
                "D4 base backbone must return shape "
                f"{expected_shape}; got {tuple(raw_features.shape)}"
            )
        raw_features = raw_features.reshape(
            len(D4_ELEMENTS), batch_size, self.output_dim
        )
        return torch.stack(
            tuple(
                self.transform_features(raw_features[index], D4_INVERSES[element])
                for index, element in enumerate(D4_ELEMENTS)
            ),
            dim=0,
        )

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        return self.aligned_orbit_features(
            image,
            spectra,
            fiber_positions,
            fiber_mask=fiber_mask,
        ).mean(dim=0)



class _D4PairLinear(nn.Module):
    """Bias-free multiplicity mixing that preserves a two-component irrep."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        if input_channels % 2 or output_channels % 2:
            raise ValueError("D4 pair-linear channel counts must be even")
        self.input_pairs = input_channels // 2
        self.output_pairs = output_channels // 2
        self.weight = nn.Parameter(torch.empty(self.output_pairs, self.input_pairs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features):
        pairs = features.reshape(*features.shape[:-1], self.input_pairs, 2)
        projected = torch.einsum("...ic,oi->...oc", pairs, self.weight)
        return projected.reshape(*features.shape[:-1], 2 * self.output_pairs)


class D4EquivariantCCLProjector(nn.Module):
    """Projection head whose scalar/spin blocks obey the D4 feature action."""

    def __init__(
        self,
        scalar_channels=512,
        spin1_channels=256,
        spin2_channels=256,
        output_dim=128,
    ):
        super().__init__()
        if output_dim <= 0 or output_dim % 8:
            raise ValueError("D4 CCL projector output_dim must be divisible by 8")
        self.scalar_channels = int(scalar_channels)
        self.spin1_channels = int(spin1_channels)
        self.spin2_channels = int(spin2_channels)
        self.output_scalar_channels = output_dim // 2
        self.output_spin1_channels = output_dim // 4
        self.output_spin2_channels = output_dim // 4
        hidden_channels = max(128, 4 * self.output_scalar_channels)
        self.scalar_projector = nn.Sequential(
            nn.Linear(self.scalar_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.output_scalar_channels),
        )
        self.spin1_projector = _D4PairLinear(
            self.spin1_channels, self.output_spin1_channels
        )
        self.spin2_projector = _D4PairLinear(
            self.spin2_channels, self.output_spin2_channels
        )

    def forward(self, features):
        expected = self.scalar_channels + self.spin1_channels + self.spin2_channels
        if features.shape[-1] != expected:
            raise ValueError(
                f"D4 CCL projector expected {expected} features; "
                f"got {features.shape[-1]}"
            )
        scalar_end = self.scalar_channels
        spin1_end = scalar_end + self.spin1_channels
        return torch.cat(
            (
                self.scalar_projector(features[..., :scalar_end]),
                self.spin1_projector(features[..., scalar_end:spin1_end]),
                self.spin2_projector(features[..., spin1_end:]),
            ),
            dim=-1,
        )


def build_feature_extractor(backbone_type, nspec=config.data['nspec']):
    if backbone_type == "legacy":
        return FeatureExtractor(nspec=nspec)
    if backbone_type == "stage3":
        return Stage3FeatureExtractor(nspec=nspec)
    if backbone_type == "stage4_d4":
        return D4OrbitFeatureExtractor(nspec=nspec)
    raise ValueError(
        f"backbone_type must be one of {BACKBONE_TYPES}; got {backbone_type!r}"
    )

class VICRegPretrain(nn.Module):
    def __init__(self, backbone=None, projector_dim=128):
        super().__init__()

        self.backbone = (
            build_feature_extractor(
                config.pretrain.get("backbone_type", "legacy"),
                nspec=config.data["nspec"],
            )
            if backbone is None
            else backbone
        )
        
        # Get feature dimensions dynamically
        dim_in = 1024
        
        # Projector networks mapping onto a common high-dimensional space
        self.projector = MLP([dim_in, 2048, 512, projector_dim], use_batchnorm=True, use_dropout=True)
        
        self.vicreg_loss = VICRegLoss(lam=20.0, mu=20.0, nu=5.0, gamma=1.0)

    def forward(self, x1, y1, fp1, x2, y2, fp2, return_components=False):
        # 1. Extract features from different views
        z1 = self.backbone.forward(x1, y1, fp1)
        z2 = self.backbone.forward(x2, y2, fp2)
        
        # 2. Project to high-dimensional space
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if not dist.is_initialized():
            # 3. Compute loss
            loss = self.vicreg_loss(z1, z2, return_components=return_components)
            return loss
        else:
            # 3. Gather features across distributed processes
            z1_gathered = torch.cat(all_gather(z1), dim=0)
            z2_gathered = torch.cat(all_gather(z2), dim=0)

            # 4. Compute loss on gathered features
            loss = self.vicreg_loss(z1_gathered, z2_gathered, return_components=return_components)
            return loss
        
    def extract_features(self, x, y, fp):
        z = self.backbone.forward(x, y, fp)
        # z = self.projector(z)
        return z

class VICRegLoss(nn.Module):
    def __init__(self, lam=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.eps = eps

    def forward(self, z1, z2, return_components=False):
        # 1. Invariance Loss
        sim_loss = F.mse_loss(z1, z2)

        # Center the representations
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # 2. Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2))

        # 3. Covariance Loss
        batch_size = z1.size(0)
        num_features = z1.size(1)
        
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        # Mask out diagonal elements
        diag_mask = ~torch.eye(num_features, device=z1.device).bool()
        cov_loss = (cov_z1[diag_mask].pow(2).sum() / num_features) + (cov_z2[diag_mask].pow(2).sum() / num_features)

        # Compute effective dimensions
        eig_z1 = torch.linalg.eigvalsh(cov_z1)
        eig_z2 = torch.linalg.eigvalsh(cov_z2)
        sum_ev_z1 = eig_z1.sum()
        sum_sq_ev_z1 = (eig_z1 ** 2).sum()
        sum_ev_z2 = eig_z2.sum()
        sum_sq_ev_z2 = (eig_z2 ** 2).sum()
        eff_dim_z1 = (sum_ev_z1 ** 2) / (sum_sq_ev_z1)
        eff_dim_z2 = (sum_ev_z2 ** 2) / (sum_sq_ev_z2)

        sim_component = self.lam * sim_loss
        var_component = self.mu * var_loss
        cov_component = self.nu * cov_loss
        total_loss = sim_component + var_component + cov_component

        # Total Loss
        if return_components:
            return total_loss, sim_component, var_component, cov_component, eff_dim_z1, eff_dim_z2
        else:
            return total_loss

class CCLPretrain(nn.Module):
    def __init__(self, backbone=None, projector_dim=128):
        super().__init__()

        self.backbone = (
            build_feature_extractor(
                config.pretrain.get("backbone_type", "legacy"),
                nspec=config.data["nspec"],
            )
            if backbone is None
            else backbone
        )

        # Get feature dimensions dynamically
        dim_in = 1024

        # Stage 4 keeps the contrastive head equivariant too. Its spin blocks
        # use bias-free multiplicity mixing, while scalar channels may use an
        # unrestricted scalar MLP. Legacy/Stage 3 checkpoints retain the
        # original projection head exactly.
        if isinstance(self.backbone, D4OrbitFeatureExtractor):
            self.projector = D4EquivariantCCLProjector(
                scalar_channels=self.backbone.scalar_channels,
                spin1_channels=self.backbone.spin1_channels,
                spin2_channels=self.backbone.spin2_channels,
                output_dim=projector_dim,
            )
        else:
            self.projector = MLP(
                [dim_in, 2048, 512, projector_dim],
                use_batchnorm=True,
                use_dropout=False,
            )

        feature_names = list(config.train["feature_names"])
        configured_scales = config.pretrain.get("ccl_label_scales", {})
        label_scales = [
            float(configured_scales.get(name, 1.0)) for name in feature_names
        ]
        theta_idx = resolve_feature_index(feature_names, "theta_int")
        self.ccl_loss = ContinuousContrastiveLoss(
            temperature=float(config.pretrain.get("ccl_temperature", 0.1)),
            sigma_label=float(config.pretrain.get("ccl_sigma_label", 0.15)),
            d_cutoff=float(config.pretrain.get("ccl_d_cutoff", 0.40)),
            label_scales=label_scales,
            theta_idx=theta_idx,
            distance_reduction=config.pretrain.get(
                "ccl_distance_reduction", "mean"
            ),
        )

    def _compute_ccl_loss(self, projected_features, labels, return_diagnostics):
        if dist.is_initialized():
            projected_features = torch.cat(all_gather(projected_features), dim=0)
            labels = torch.cat(all_gather(labels), dim=0)
        return self.ccl_loss(
            projected_features,
            labels,
            return_diagnostics=return_diagnostics,
        )

    def forward(self, x, y, fp, labels, return_diagnostics=False):
        backbone_features = self.backbone.forward(x, y, fp)
        projected_features = self.projector(backbone_features)
        return self._compute_ccl_loss(
            projected_features,
            labels,
            return_diagnostics,
        )

    def extract_features(self, x, y, fp):
        return self.backbone.forward(x, y, fp)


class ContinuousContrastiveLoss(nn.Module):
    """Continuous contrastive loss with fixed normalized parameter geometry."""

    def __init__(
        self,
        temperature=0.1,
        sigma_label=0.15,
        d_cutoff=0.40,
        label_scales=None,
        theta_idx=None,
        distance_reduction="mean",
    ):
        super().__init__()
        for name, value in (
            ("temperature", temperature),
            ("sigma_label", sigma_label),
            ("d_cutoff", d_cutoff),
        ):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if theta_idx is not None and type(theta_idx) is not int:
            raise TypeError("theta_idx must be an integer or None")
        if distance_reduction not in ("mean", "sum"):
            raise ValueError("distance_reduction must be 'mean' or 'sum'")

        scales = torch.as_tensor(
            [1.0] if label_scales is None else label_scales,
            dtype=torch.float32,
        )
        if (
            scales.ndim != 1
            or torch.any(~torch.isfinite(scales))
            or torch.any(scales <= 0)
        ):
            raise ValueError(
                "label_scales must be a 1D sequence of positive finite values"
            )

        self.temperature = float(temperature)
        self.sigma_label = float(sigma_label)
        self.theta_idx = theta_idx
        self.distance_reduction = distance_reduction
        self.register_buffer("label_scales", scales)
        self.register_buffer(
            "delta_bg",
            torch.exp(
                torch.tensor(
                    -(d_cutoff ** 2) / (2 * (sigma_label ** 2)),
                    dtype=torch.float32,
                )
            ),
        )

    def pairwise_label_distance_sq(self, labels):
        """Return fixed-scale pairwise distances for normalized KL labels."""
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, nfeatures)")
        if self.label_scales.numel() not in (1, labels.shape[1]):
            raise ValueError(
                "label_scales must contain one value or one value per label feature"
            )
        if self.theta_idx is not None and not 0 <= self.theta_idx < labels.shape[1]:
            raise ValueError("theta_idx is outside the label feature dimension")

        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        if self.theta_idx is not None:
            theta_delta = label_diff[..., self.theta_idx]
            theta_delta = torch.atan2(
                torch.sin(math.pi * theta_delta),
                torch.cos(math.pi * theta_delta),
            ) / math.pi
            label_diff = label_diff.clone()
            label_diff[..., self.theta_idx] = theta_delta

        scaled_diff_sq = (label_diff / self.label_scales) ** 2
        if self.distance_reduction == "mean":
            return torch.mean(scaled_diff_sq, dim=-1)
        return torch.sum(scaled_diff_sq, dim=-1)

    def _target_distribution(self, labels):
        batch_size = labels.shape[0]
        if batch_size < 2:
            raise ValueError("continuous contrastive loss requires at least two rows")
        label_dist_sq = self.pairwise_label_distance_sq(labels).float()
        weights = torch.exp(-label_dist_sq / (2 * (self.sigma_label ** 2)))

        mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        weights_masked = weights.masked_fill(mask, 0.0)
        row_sum = torch.sum(weights_masked, dim=1, keepdim=True)
        delta_bg = self.delta_bg.to(device=labels.device, dtype=weights.dtype)
        target_mass = row_sum / (row_sum + delta_bg)
        positive_probs = weights_masked / row_sum.clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        weights_norm = positive_probs * target_mass
        return mask, weights_norm, positive_probs, target_mass.squeeze(1)

    @staticmethod
    def _target_statistics(positive_probs, target_mass):
        row_entropy = -torch.sum(
            torch.special.xlogy(positive_probs, positive_probs), dim=1
        )
        concentration = torch.sum(positive_probs.square(), dim=1)
        effective_positives = concentration.clamp_min(
            torch.finfo(concentration.dtype).tiny
        ).reciprocal()
        effective_positives = torch.where(
            target_mass > 0, effective_positives, torch.zeros_like(concentration)
        )
        uniform_baseline = target_mass * math.log(positive_probs.shape[1] - 1)
        return {
            "target_entropy": torch.mean(target_mass * row_entropy).detach(),
            "uniform_baseline": torch.mean(uniform_baseline).detach(),
            "effective_positives": torch.mean(effective_positives).detach(),
            "target_mass": torch.mean(target_mass).detach(),
        }

    def target_statistics(self, labels):
        """Return batch-level diagnostics for the soft-positive target only."""
        _, _, positive_probs, target_mass = self._target_distribution(labels)
        return self._target_statistics(positive_probs, target_mass)

    def forward(self, z, labels, return_diagnostics=False):
        z = F.normalize(z, dim=1)
        batch_size = z.shape[0]
        if batch_size < 2:
            raise ValueError("continuous contrastive loss requires at least two rows")
        if labels.shape[0] != batch_size:
            raise ValueError("z and labels must have the same batch dimension")

        mask, weights_norm, positive_probs, target_mass = (
            self._target_distribution(labels)
        )

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        log_prob_sim = F.log_softmax(
            sim_matrix.masked_fill(mask, -torch.inf),
            dim=1,
        )
        # Avoid the undefined 0 * -inf diagonal contribution.
        log_prob_sim = log_prob_sim.masked_fill(mask, 0.0)

        loss = -torch.sum(weights_norm * log_prob_sim, dim=1).mean()
        if not return_diagnostics:
            return loss

        diagnostics = self._target_statistics(positive_probs, target_mass)
        diagnostics["excess_loss"] = (
            loss.detach() - diagnostics["target_entropy"]
        )
        return loss, diagnostics



class MLP(nn.Module):
    '''
    A simple MLP with Linear and ReLU
    '''
    
    def __init__(self, layers, use_batchnorm=False, use_dropout=False):
        
        super(MLP,self).__init__()

        modules = nn.ModuleList([])
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2:
                modules.append(nn.ReLU(True))
                if use_batchnorm:
                    modules.append(nn.BatchNorm1d(layers[i+1], affine=False))
                if use_dropout:
                    modules.append(nn.Dropout(0.1))

        self.mlp = nn.Sequential(*modules)

    def forward(self,x):

        x = self.mlp(x)
        return x

class ResidualBlock(nn.Module):
    '''
    A residual block object that skips layers until stride > 1, i.e. the size of data shrinks
    '''
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
            
    def forward(self,x):
        
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

### ViT classes ###
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=6, img_size=48, embed_dim=512, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)  
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, img_size=48, patch_size=6, num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_embed = PatchEmbedding(in_channels, patch_size, img_size, embed_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token

### Spectra RNN ###
class SpecRNN(nn.Module):
    def __init__(self, nspec, hidden_size=256, num_layers=2, bidirectional=True):
        super().__init__()

        # Local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
        )

        # RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=64,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat = self.cnn_spec(x) 
        
        feat = feat.mean(dim=-1)
        feat = feat.permute(0, 2, 1)

        # RNN along the spectral dimension
        rnn_out, _ = self.rnn_spec(feat)
        rnn_feat = rnn_out[:, -1, :]

        # Project to 512-dim feature
        out = self.proj(rnn_feat)
        return out
    
class LargeSpecRNN(nn.Module):
    def __init__(self, nspecs, hidden_size=1024, num_layers=4, bidirectional=True):
        super().__init__()

        self.nspecs = nspecs

        # Deeper local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Larger RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=256,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,  # Increased to 1024
            num_layers=num_layers,    # Increased to 4 layers
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0  # Add dropout between layers
        )

        # Larger output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512-2*self.nspecs),  # Final feature dim reduced to accommodate fiber position if needed
            nn.BatchNorm1d(512-2*self.nspecs),
            nn.ReLU(True)
        )

        # Add a linear layer to fuse the 1024 concatenated features back to 512
        self.pool_fusion = nn.Linear(1024-4*self.nspecs, 512-2*self.nspecs)  # Adjust input dim to account for removed fiber position features

    def forward(self, x):
        # x: (batch, 1, nspec, time)
        batch_size = x.size(0)
        
        # CNN processing
        x = self.cnn_spec(x)  # (batch, 256, nspec, time/2)
        
        # Reshape for RNN: merge batch and time, treat spectral bins as sequence
        b, c, nspec, t = x.size()
        x = x.permute(0, 3, 2, 1)  # (batch, time, nspec, channels)
        x = x.reshape(b * t, nspec, c)  # (batch*time, nspec, channels)
        
        # RNN processing across spectral dimension
        x, _ = self.rnn_spec(x)  # (batch*time, nspec, hidden*2)
        
        # Global pooling across spectral dimension
        x = x.mean(dim=1)  # (batch*time, hidden*2)
        
        # Projection
        x = self.proj(x)  # (batch*time, 512)
        
        # Reshape back
        x = x.view(b, t, -1)  # (batch, time, 512)

        # 1. Extract the sharp features
        x_max = x.max(dim=1)[0]   # (batch, 512)
        
        # 2. Extract the global continuum
        x_avg = x.mean(dim=1)     # (batch, 512)
        
        # 3. Combine them
        x_cat = torch.cat([x_max, x_avg], dim=-1) # (batch, 1024)
        
        # 4. Fuse back to the required 512 dimension
        x = self.pool_fusion(x_cat) # (batch, 512)
        
        # Optional: Add a non-linearity depending on your architecture design
        x = nn.functional.relu(x)
        
        return x

### Spec CNN ###
class SpecCNN(nn.Module):

    def __init__(self, nspecs):
        super(SpecCNN, self).__init__()

        self.nspecs = nspecs

        self.cnn_spec = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512-2*nspecs, kernel_size=(self.nspecs, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512-2*nspecs),
            nn.ReLU(True),
            
        )

    def forward(self, x):
        
        x = self.cnn_spec(x)
        
        return x

### Image CNN ###
class ImgCNN(nn.Module):
    def __init__(self):
        super(ImgCNN, self).__init__()

        self.cnn_img = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, 2),
            
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, 2),
            
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512, 2),
            
            nn.AvgPool2d(3),
            
        )

    def forward(self, x):
        
        x = self.cnn_img(x)
        
        return x

class ForkCNN(nn.Module):
    def __init__(self, nspecs):
        pass