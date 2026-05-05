import os
import re
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config
    from .utils import density_contour, img_to_gal_axis, resolve_feature_index
except ImportError:
    import config
    from utils import density_contour, img_to_gal_axis, resolve_feature_index


class ParameterContourPlotter:
    """Generic contour plotting for predicted parameter residuals.

    The class supports plotting any parameter in ``feature_names`` as:
    x-axis = true parameter value, y-axis = prediction residual.
    """

    DEFAULT_FIG_ROOT = "/ocean/projects/phy250048p/shared/figures"
    BASE_COLORS = ["salmon", "red"]
    MASK_COLORS = ["lightsteelblue", "cornflowerblue"]

    def __init__(self, model_name, true, pred, pred_tf=None, snr=None, feature_names=None, fig_root=None):
        self.model_name = str(model_name)
        self.true = self._validate_2d_array("true", true)
        self.pred = self._validate_2d_array("pred", pred)
        self.pred_tf = None if pred_tf is None else self._validate_2d_array("pred_tf", pred_tf)
        self.snr = None if snr is None else self._validate_1d_array("snr", snr)

        if self.true.shape != self.pred.shape:
            raise ValueError(
                f"Shape mismatch: true shape={self.true.shape} and pred shape={self.pred.shape} must match."
            )
        if self.pred_tf is not None and self.pred_tf.shape != self.true.shape:
            raise ValueError(
                f"Shape mismatch: pred_tf shape={self.pred_tf.shape} must match true shape={self.true.shape}."
            )
        if self.snr is not None and self.snr.shape[0] != self.true.shape[0]:
            raise ValueError(
                f"Shape mismatch: snr length={self.snr.shape[0]} must match nsamples={self.true.shape[0]}."
            )

        if feature_names is None:
            feature_names = list(config.train["feature_names"])
        self.feature_names = [str(name) for name in feature_names]

        if len(self.feature_names) != self.true.shape[1]:
            raise ValueError(
                "feature_names length must equal nfeatures. "
                f"Got len(feature_names)={len(self.feature_names)} and nfeatures={self.true.shape[1]}."
            )

        self.fig_root = fig_root if fig_root is not None else self.DEFAULT_FIG_ROOT

    @staticmethod
    def _validate_2d_array(name, values):
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array with shape (nsamples, nfeatures); got shape={arr.shape}.")
        return arr

    @staticmethod
    def _validate_1d_array(name, values):
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array with shape (nsamples,); got shape={arr.shape}.")
        return arr.reshape(-1)

    @staticmethod
    def _sanitize_token(token):
        normalized = str(token).strip().lower()
        special_tokens = {
            "g+": "g_plus",
            "g_plus": "g_plus",
            "gplus": "g_plus",
            "gx": "g_x",
            "g_x": "g_x",
            "gcross": "g_x",
            "g_cross": "g_x",
        }
        if normalized in special_tokens:
            return special_tokens[normalized]

        sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(token).strip())
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "value"

    @staticmethod
    def _display_name(name):
        normalized = str(name).strip().lower()
        special_tokens = {
            "g+": "g+",
            "g_plus": "g+",
            "gplus": "g+",
            "gx": "g_x",
            "g_x": "g_x",
            "gcross": "g_x",
            "g_cross": "g_x",
        }
        return special_tokens.get(normalized, str(name).strip())

    @staticmethod
    def _format_value_token(value):
        value_token = format(float(value), "g")
        value_token = value_token.replace("-", "neg_").replace(".", "p")
        return ParameterContourPlotter._sanitize_token(value_token)

    def _resolve_index(self, name):
        return resolve_feature_index(self.feature_names, name)

    def _resolve_theta_values(self):
        theta_idx = self._resolve_index("theta_int")
        return self.true[:, theta_idx]

    def _resolve_named_series(self, name, selected_pred):
        """Resolve a plot/mask series from a parameter name.

        Returns true_values, pred_values, display_name.
        """
        normalized = str(name).strip().lower()
        if normalized in {"g+", "g_plus", "gplus", "gx", "g_x", "gcross", "g_cross"}:
            theta = self._resolve_theta_values()
            g1_idx = self._resolve_index("g1")
            g2_idx = self._resolve_index("g2")
            true_g1 = self.true[:, g1_idx]
            true_g2 = self.true[:, g2_idx]
            pred_g1 = selected_pred[:, g1_idx]
            pred_g2 = selected_pred[:, g2_idx]
            true_plus, true_cross = img_to_gal_axis(true_g1, true_g2, theta)
            pred_plus, pred_cross = img_to_gal_axis(pred_g1, pred_g2, theta)
            if normalized in {"g+", "g_plus", "gplus"}:
                return true_plus, pred_plus, "g+"
            return true_cross, pred_cross, "g_x"

        idx = self._resolve_index(name)
        return self.true[:, idx], selected_pred[:, idx], self.feature_names[idx]

    def _resolve_mask_source(self, mask_param, selected_pred=None):
        """Return (array_source, label) for a given mask_param.

        Supported mask_param values:
        - 'snr' : uses the `snr` array provided at init
        - '{param}_diff' : uses (selected_pred - true) for the base parameter
        - any feature name : uses true[:, feature_idx]
        """
        mask_name = str(mask_param).strip()
        lowname = mask_name.lower()

        if lowname == "snr":
            if self.snr is None:
                raise ValueError("mask_param='snr' requires snr to be provided at initialization.")
            return self.snr, "snr"

        # support '<param>_diff' masks
        m = re.match(r"^(.+)_diff$", lowname)
        if m:
            base_name = m.group(1)
            if selected_pred is None:
                selected = self.pred
            else:
                selected = selected_pred
            true_values, pred_values, display_name = self._resolve_named_series(base_name, selected)
            resid = np.asarray(pred_values) - np.asarray(true_values)
            return resid, f"{display_name}_diff"

        # fallback: treat as a true-valued parameter name
        true_values, _, display_name = self._resolve_named_series(mask_name, self.pred if selected_pred is None else selected_pred)
        return true_values, display_name

    def _build_filename(self, param, mask_param, masking_value, flip_mask, use_tf):
        param_token = self._sanitize_token(param)
        tf_tokens = ["with", "tf"] if use_tf else []

        if mask_param is None:
            parts = ["contour", param_token, "no", "mask"] + tf_tokens
            return "_".join(parts) + ".png"

        mask_token = self._sanitize_token(mask_param)
        comparator = "lt" if flip_mask else "gt"
        value_token = self._format_value_token(masking_value)
        parts = ["contour", param_token, "with", mask_token, comparator, value_token] + tf_tokens
        return "_".join(parts) + ".png"

    def _resolve_save_path(self, filename, save_dir):
        if save_dir is None:
            target_dir = join(self.fig_root, self.model_name)
            return join(target_dir, filename)

        if str(save_dir).lower().endswith(".png"):
            return str(save_dir)

        return join(str(save_dir), filename)

    def plot(
        self,
        param,
        mask_param=None,
        masking_value=None,
        flip_mask=False,
        use_tf=False,
        nbins=25,
        save_dir=None,
    ):
        """Plot residual contours for one parameter.

        Args:
            param: Parameter name to plot on x-axis (true value).
            mask_param: Optional parameter name for threshold mask.
            masking_value: Threshold value for mask (required when mask_param is set).
            flip_mask: If False use mask_param > masking_value, else mask_param < masking_value.
            use_tf: If True, plot using pred_tf instead of pred.
            nbins: Number of bins for 2D histogram contour.
            save_dir: Optional output directory or full output .png path.
        """
        if not np.isfinite(nbins) or int(nbins) <= 1:
            raise ValueError(f"nbins must be an integer > 1; got {nbins}.")
        nbins = int(nbins)

        if use_tf and self.pred_tf is None:
            raise ValueError("use_tf=True requires pred_tf to be provided at initialization.")

        selected_pred = self.pred_tf if use_tf else self.pred
        true_values, pred_values, param_label = self._resolve_named_series(param, selected_pred)
        residuals = pred_values - true_values

        mask = None
        comparator_label = None
        if mask_param is not None:
            if masking_value is None:
                raise ValueError("masking_value is required when mask_param is provided.")
            if not np.isfinite(masking_value):
                raise ValueError(f"masking_value must be finite; got {masking_value}.")

            mask_source, mask_label = self._resolve_mask_source(mask_param, selected_pred=selected_pred)
            if flip_mask:
                mask = mask_source < masking_value
                comparator_label = "<"
            else:
                mask = mask_source > masking_value
                comparator_label = ">"

            if np.sum(mask) < 3:
                raise ValueError(
                    f"Mask selected too few samples ({np.sum(mask)}). "
                    "Choose a different masking_value or mask_param."
                )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        density_contour(true_values, residuals, nbins, nbins, ax=ax, colors=self.BASE_COLORS)

        legend_handles = []
        legend_labels = []

        if mask is not None:
            contour_mask = density_contour(
                true_values[mask],
                residuals[mask],
                nbins,
                nbins,
                ax=ax,
                colors=self.MASK_COLORS,
            )
            mask_handles, _ = contour_mask.legend_elements()
            if len(mask_handles) > 0:
                legend_handles.append(mask_handles[0])
                legend_labels.append(
                    f"{mask_label} {comparator_label} {masking_value:g}"
                )

        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{param_label} true")
        pred_label = "pred_tf" if use_tf else "pred"
        ax.set_ylabel(f"{pred_label} - true")

        if mask is None:
            title = f"{param_label} contour"
        else:
            title = f"{param_label} contour with {mask_label} {comparator_label} {masking_value:g}"
        if use_tf:
            title += " (with tf)"
        ax.set_title(title)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, frameon=False)

        fig.tight_layout()

        filename = self._build_filename(param, mask_param, masking_value, flip_mask, use_tf)
        output_path = self._resolve_save_path(filename, save_dir)
        output_dir = dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fig.savefig(output_path, dpi=300)
        return fig, ax, output_path