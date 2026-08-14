import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as so
from os.path import join


def resolve_feature_index(feature_names, target_name, aliases=None):
    """Resolve feature index by name with optional aliases."""
    names = [str(name).strip() for name in feature_names]
    target_candidates = [target_name]
    if aliases is not None:
        target_candidates.extend(list(aliases))

    normalized_candidates = {str(name).strip().lower() for name in target_candidates}
    matches = [idx for idx, name in enumerate(names) if name.lower() in normalized_candidates]

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(
            f"Could not find any of {sorted(normalized_candidates)} in feature_names={names}."
        )
    raise ValueError(
        f"Found multiple matches for {sorted(normalized_candidates)} in feature_names={names}: indices={matches}."
    )


def denormalize(samples, par_ranges, feature_names=None):
    samples = np.asarray(samples)
    denorm = np.empty_like(samples)

    if feature_names is None:
        feature_names = list(par_ranges.keys())
    else:
        feature_names = list(feature_names)

    if samples.shape[-1] != len(feature_names):
        raise ValueError(
            f"Sample feature dimension {samples.shape[-1]} does not match feature list length {len(feature_names)}."
        )

    missing = [name for name in feature_names if name not in par_ranges]
    if missing:
        raise ValueError(f"Feature names missing from par_ranges: {missing}")

    for i, name in enumerate(feature_names):
        low, high = par_ranges[name]
        values = samples[..., i].copy()
        values += 1
        values *= (high - low) / 2
        values += low
        denorm[..., i] = values
    return denorm


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level


def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs):
    """Create a density contour plot."""
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x, nbins_y), density=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y, 1))

    pdf = H * (x_bin_sizes * y_bin_sizes)
    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T

    zero = so.brentq(find_confidence_interval, 0.0, 1.0, args=(pdf, 0.0))
    one_sigma = so.brentq(find_confidence_interval, 0.0, 1.0, args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0.0, 1.0, args=(pdf, 0.95))
    levels = [zero, one_sigma, two_sigma][::-1]

    if ax is None:
        contour = plt.contourf(X, Y, Z, levels=levels, origin="lower", alpha=0.5, **contour_kwargs)
    else:
        contour = ax.contourf(X, Y, Z, levels=levels, origin="lower", alpha=0.5, **contour_kwargs)
    return contour


def make_g_scatter(
    g1,
    g2,
    g1_diff,
    g2_diff,
    c=None,
    cmap="viridis",
    s=1,
    cbar_name="SNR",
    save=True,
    filename=None,
    fig_dir=None,
    stem=None,
):
    if filename is None:
        filename = f"g_{stem}_scatter.jpg" if stem is not None else "g_scatter.jpg"

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "figure.dpi": 300})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18, 6))

    axes[0].scatter(g1, g1_diff, c=c, cmap=cmap, s=s)
    scat = axes[1].scatter(g2, g2_diff, c=c, cmap=cmap, s=s)
    axes[0].set_ylim(-0.1, 0.1)

    for i in range(2):
        axes[i].axhline(0, color="k", linestyle="--", linewidth=3)

    axes[0].set_xlabel(r"$g_+^{true}$", fontsize=20)
    axes[1].set_xlabel(r"$g_x^{true}$", fontsize=20)
    axes[0].set_ylabel(r"$g_i-g_i^{true}$", fontsize=20)

    axes[0].tick_params(axis="both", which="major", labelsize=15)
    axes[1].tick_params(axis="x", which="major", labelsize=15)

    fig.subplots_adjust(wspace=0.08, hspace=0.06)
    fig.subplots_adjust(right=0.8)

    cbar = plt.colorbar(scat, ax=axes)
    cbar.set_label(cbar_name)

    if save:
        if fig_dir is not None and stem is not None:
            plt.savefig(join(fig_dir, f"{stem}/{filename}"))
        else:
            plt.savefig(filename)


def make_g_contour(
    g1,
    g2,
    g1_diff,
    g2_diff,
    axis="12",
    nbins=25,
    mask=None,
    anti_mask=False,
    legend_str_list=None,
    save=True,
    filename=None,
    fig_dir=None,
    stem=None,
):
    if legend_str_list is None:
        legend_str_list = []
    if filename is None:
        filename = f"g_{stem}_contour.jpg" if stem is not None else "g_contour.jpg"

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "figure.dpi": 300})
    plt.figure(figsize=(8, 6))
    fig1, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 6))

    if anti_mask:
        density_contour(g1[~mask], g1_diff[~mask], nbins, nbins, ax=axes[0], colors=["salmon", "red"])
        contour = density_contour(g2[~mask], g2_diff[~mask], nbins, nbins, ax=axes[1], colors=["salmon", "red"])
        handles, _ = contour.legend_elements()
    else:
        density_contour(g1, g1_diff, nbins, nbins, ax=axes[0], colors=["salmon", "red"])
        contour = density_contour(g2, g2_diff, nbins, nbins, ax=axes[1], colors=["salmon", "red"])
        handles, _ = contour.legend_elements()

    if mask is not None:
        density_contour(g1[mask], g1_diff[mask], nbins, nbins, ax=axes[0], colors=["lightsteelblue", "cornflowerblue"])
        contour1 = density_contour(g2[mask], g2_diff[mask], nbins, nbins, ax=axes[1], colors=["lightsteelblue", "cornflowerblue"])
        handles1, _ = contour1.legend_elements()
        handles += handles1

    axes[0].set_ylim(-0.1, 0.1)
    for i in range(2):
        axes[i].axhline(0, color="k", linestyle="--", linewidth=3)

    if axis == "12":
        axes[0].set_xlabel(r"$g_1^{true}$", fontsize=20)
        axes[1].set_xlabel(r"$g_2^{true}$", fontsize=20)
    elif axis == "pc":
        axes[0].set_xlabel(r"$g_+^{true}$", fontsize=20)
        axes[1].set_xlabel(r"$g_x^{true}$", fontsize=20)
    axes[0].set_ylabel(r"$g_i-g_i^{true}$", fontsize=20)

    axes[0].tick_params(axis="both", which="major", labelsize=15)
    axes[1].tick_params(axis="x", which="major", labelsize=15)
    plt.legend(handles, legend_str_list, fontsize=16)

    fig1.subplots_adjust(wspace=0.08, hspace=0.06)
    fig1.subplots_adjust(right=0.8)
    plt.tight_layout()
    if save:
        if fig_dir is not None and stem is not None:
            plt.savefig(join(fig_dir, f"{stem}/{filename}"))
        else:
            plt.savefig(filename)


def gaussian_2d(x, y, xx, yy, A, sigma):
    coeff = A / (2 * np.pi * sigma ** 2)
    exp = -((xx - x) ** 2 + (yy - y) ** 2) / (2 * np.pi * sigma ** 2)
    return coeff * torch.exp(exp)


def add_blobs(img, nblobs, avg_amp, avg_size):
    blob_img = torch.zeros_like(img, dtype=torch.float32)
    grid_size = 48
    xx, yy = torch.meshgrid(
        torch.linspace(0, grid_size, grid_size),
        torch.linspace(0, grid_size, grid_size),
        indexing="ij",
    )
    for i in range(nblobs):
        x, y = torch.rand(2) * grid_size
        A = np.random.normal(avg_amp, 0.3 * avg_amp) * (-1) ** i
        sigma = np.random.normal(avg_size, 0.3 * avg_size)
        blob_img[0] += gaussian_2d(x, y, xx, yy, A, sigma)
    new_img = img + blob_img
    return new_img, blob_img

def img_to_gal_axis(g1, g2, theta):
    """Convert image-frame shear to the galaxy frame.

    ``theta`` follows the simulator's image-array convention: positive angles
    rotate clockwise on a displayed image (the row/y coordinate increases
    downward). Consequently, a positive ``g2`` shear is aligned with a galaxy
    at ``theta = pi / 4``. The spin-2 basis is rotated by ``2 * theta``.
    """
    # check if numpy array or torch tensor
    if isinstance(g1, torch.Tensor):
        g_plus = g1 * torch.cos(2*theta) + g2 * torch.sin(2*theta)
        g_cross = -g1 * torch.sin(2*theta) + g2 * torch.cos(2*theta)
        return g_plus, g_cross
    else:
        g_plus = g1 * np.cos(2*theta) + g2 * np.sin(2*theta)
        g_cross = -g1 * np.sin(2*theta) + g2 * np.cos(2*theta)
    return g_plus, g_cross

def gal_to_img_axis(g_plus, g_cross, theta):
    """Convert galaxy-frame shear to the image frame.

    This is the inverse of :func:`img_to_gal_axis`; ``theta`` is positive
    clockwise in the simulator's image-array convention.
    """
    # check if numpy array or torch tensor
    if isinstance(g_plus, torch.Tensor):
        g1 = g_plus * torch.cos(2*theta) - g_cross * torch.sin(2*theta)
        g2 = g_plus * torch.sin(2*theta) + g_cross * torch.cos(2*theta)
        return g1, g2
    else:
        g1 = g_plus * np.cos(2*theta) - g_cross * np.sin(2*theta)
        g2 = g_plus * np.sin(2*theta) + g_cross * np.cos(2*theta)
    return g1, g2


def saliency(ID, test_ds, model, SNR, PA, xx, yy, zz, device=None):
    if device is None:
        device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    SLC_img = torch.zeros((48, 48), dtype=torch.float).to(device)
    SLC_spec = torch.zeros((3, 64), dtype=torch.float).to(device)
    true = test_ds[ID]["fid_pars"][:2].unsqueeze(0).float().to(device)
    g1_true = true.detach().cpu().numpy()[0, 0] * 0.1
    g2_true = true.detach().cpu().numpy()[0, 1] * 0.1
    phi = 0.5 * np.arctan2(g2_true, g1_true)
    gp_true, gc_true = img_to_gal_axis(g1_true, g2_true, PA)
    Gp_diff = 0.0
    Gc_diff = 0.0
    Loss = 0.0
    G1_diff = 0.0
    G2_diff = 0.0
    N = 100
    for _ in range(N):
        img = apply_noise(test_ds[ID]["img"].unsqueeze(0).float().to(device), SNR, device=device)
        spec = apply_noise(test_ds[ID]["spec"].unsqueeze(0).float().to(device), SNR, device=device)
        img.requires_grad = True
        spec.requires_grad = True

        loss = model(img, spec, true)
        loss.backward()
        Loss += loss.item()

        log_prob = model.estimate_log_prob(img, spec, zz, 1).detach().cpu().numpy()
        index = np.argmax(log_prob)
        g1_pred = xx[index] * 0.1
        g2_pred = yy[index] * 0.1
        gp_pred, gc_pred = img_to_gal_axis(g1_pred, g2_pred, PA)
        Gc_diff += np.abs(gc_pred - gc_true)
        Gp_diff += np.abs(gp_pred - gp_true)

        slc_img, _ = torch.max(img.grad[0], dim=0)
        slc_spec, _ = torch.max(spec.grad[0], dim=0)
        slc_min = min(slc_img.min(), slc_spec.min())
        slc_max = max(slc_img.max(), slc_spec.max())
        slc_img = (slc_img - slc_min) / (slc_max - slc_min) - 0.5
        slc_spec = (slc_spec - slc_min) / (slc_max - slc_min) - 0.5
        SLC_img = SLC_img + slc_img
        SLC_spec = SLC_spec + slc_spec

    slc_img = SLC_img / N
    slc_spec = SLC_spec / N
    gp_diff = Gp_diff / N
    gc_diff = Gc_diff / N
    loss = Loss / N
    np.round(G1_diff / N, 5)
    np.round(G2_diff / N, 5)

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.1)
    slope = np.sin(PA) / np.cos(PA)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(gs[0])
    ax.imshow(np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0)))
    ax.axline((23.5, 23.5), slope=slope, color="white")
    ax.text(
        0.05,
        0.95,
        f"Loss = {np.round(loss, 5)}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=20,
        color="white",
    )
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(gs[1])
    ax.imshow(slc_img.cpu().numpy(), cmap=plt.cm.RdBu)
    ax.axline((23.5, 23.5), slope=slope, color="white")
    ax.arrow(23.5, 23.5, 10 * np.cos(phi), 10 * np.sin(phi), color="yellow", width=0.5)
    ax.text(0.05, 0.95, f"g+ = {np.round(gp_true, 5)}", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=20, color="black")
    ax.text(0.05, 0.85, f"g× = {np.round(gc_true, 5)}", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=20, color="black")
    ax.text(0.03, 0.23, f"|Δg+| = {np.round(gp_diff, 5)}", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=20, color="black")
    ax.text(0.03, 0.13, f"|Δg×| = {np.round(gc_diff, 5)}", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=20, color="black")
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(gs[2])
    plt.sca(ax)
    spec = spec.detach().cpu().numpy()[0, 0]
    ax.plot(spec[0], color="blue")
    ax.text(0.05, 0.8, "Right", horizontalalignment="left", verticalalignment="top", transform=ax.transAxes, fontsize=20, color="black")
    divider = make_axes_locatable(ax)
    texts = ["Left", "Center"]
    for j in range(2):
        ax1 = divider.append_axes("top", size="100%", pad=0, sharex=ax)
        ax1.plot(spec[j + 1], color="blue")
        ax1.text(0.05, 0.8, texts[j], horizontalalignment="left", verticalalignment="top", transform=ax1.transAxes, fontsize=20, color="black")

    ax = plt.subplot(gs[3])
    plt.sca(ax)
    slc_spec = slc_spec.cpu().numpy()
    ax.plot(slc_spec[0], color="red")
    ax.set_ylim(-0.5, 0.5)
    divider = make_axes_locatable(ax)
    for j in range(2):
        ax1 = divider.append_axes("top", size="100%", pad=0, sharex=ax)
        ax1.plot(slc_spec[j + 1], color="red")
        ax1.set_ylim(-0.5, 0.5)

    return loss


def get_quantiles(axis, marginal, quantiles=(0.16, 0.84)):
    """Helper to find parameter values at specific quantiles via the CDF."""
    cdf = np.cumsum(marginal)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, axis)


def generate_grid_plot(grid_prob, param1_axis, param2_axis, labels=None, true_point=None):
    total_prob = np.sum(grid_prob)
    normalized_prob = grid_prob / total_prob

    map_index_flat = np.argmax(grid_prob)
    map_index_p2, map_index_p1 = np.unravel_index(map_index_flat, grid_prob.shape)
    map_p1 = param1_axis[map_index_p1]
    map_p2 = param2_axis[map_index_p2]

    dx = param1_axis[1] - param1_axis[0]
    dy = param2_axis[1] - param2_axis[0]
    p1_marginal = np.sum(normalized_prob, axis=0) / dy
    p2_marginal = np.sum(normalized_prob, axis=1) / dx

    p1_low, p1_high = get_quantiles(param1_axis, p1_marginal)
    p2_low, p2_high = get_quantiles(param2_axis, p2_marginal)

    label1 = labels[0].replace("$", "").replace("\\", "") if labels else "P1"
    label2 = labels[1].replace("$", "").replace("\\", "") if labels else "P2"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7, 7),
        gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3], "hspace": 0.05, "wspace": 0.05},
    )

    ax_2d = axes[1, 0]
    ax_p1 = axes[0, 0]
    ax_p2 = axes[1, 1]
    ax_info = axes[0, 1]

    ax_2d.contourf(param1_axis, param2_axis, normalized_prob, levels=20, cmap="viridis")
    ax_2d.plot(map_p1, map_p2, "x", color="red", markersize=10, label="MAP", markeredgewidth=2)

    if true_point:
        true_p1, true_p2 = true_point
        ax_2d.plot(true_p1, true_p2, "*", color="yellow", markersize=12, label="True Value", markeredgewidth=1)
        ax_2d.legend(fontsize=8)

    ax_2d.set_xlabel(labels[0] if labels else "Parameter 1")
    ax_2d.set_ylabel(labels[1] if labels else "Parameter 2")
    ax_2d.set_xlim(-0.1, 0.1)
    ax_2d.set_ylim(-0.1, 0.1)
    ax_2d.tick_params(direction="in")

    ax_p1.plot(param1_axis, p1_marginal, color="blue")
    mask1 = (param1_axis >= p1_low) & (param1_axis <= p1_high)
    ax_p1.fill_between(param1_axis[mask1], 0, p1_marginal[mask1], color="blue", alpha=0.3)
    ax_p1.axvline(map_p1, color="red", linestyle="--", linewidth=1)
    if true_point:
        ax_p1.axvline(true_point[0], color="yellow", linestyle=":", linewidth=2)
    ax_p1.set_xlim(param1_axis.min(), param1_axis.max())
    ax_p1.tick_params(axis="x", labelbottom=False, direction="in")
    ax_p1.set_ylim(bottom=0)
    ax_p1.set_yticks([])

    ax_p2.plot(p2_marginal, param2_axis, color="red")
    mask2 = (param2_axis >= p2_low) & (param2_axis <= p2_high)
    ax_p2.fill_betweenx(param2_axis[mask2], 0, p2_marginal[mask2], color="red", alpha=0.3)
    ax_p2.axhline(map_p2, color="red", linestyle="--", linewidth=1)
    if true_point:
        ax_p2.axhline(true_point[1], color="yellow", linestyle=":", linewidth=2)
    ax_p2.set_ylim(param2_axis.min(), param2_axis.max())
    ax_p2.tick_params(axis="y", labelleft=False, direction="in")
    ax_p2.set_xlim(left=0)
    ax_p2.set_xticks([])

    ax_info.axis("off")
    map_text = f"MAP:\n{label1} = {map_p1:.3f}\n{label2} = {map_p2:.3f}"

    if true_point:
        full_text = (
            f"$\\mathbf{{MAP\\ (x)}}$\n{label1}: {map_p1:.3f}\n{label2}: {map_p2:.3f}\n\n"
            f"$\\mathbf{{True\\ (*)}}$\n{label1}: {true_point[0]:.3f}\n{label2}: {true_point[1]:.3f}"
        )
        ax_info.text(
            0.5,
            0.5,
            full_text,
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
        )
    else:
        ax_info.text(
            0.5,
            0.5,
            map_text,
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
        )

    plt.suptitle("2D Grid Posterior Plot", y=0.95, fontsize=12)


def filter_samples_sigma_clip(samples, n_sigma=5.0, min_keep=100, return_mask=False):
    """Filter rows outside +/- n_sigma per dimension for plotting stability."""
    arr = np.asarray(samples)
    if arr.ndim != 2:
        raise ValueError("samples must have shape (nsamples, nfeatures).")
    if arr.shape[0] == 0:
        raise ValueError("samples must contain at least one row.")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be > 0.")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1.")

    finite_row_mask = np.all(np.isfinite(arr), axis=1)
    finite_idx = np.flatnonzero(finite_row_mask)
    if finite_idx.size == 0:
        raise ValueError("No finite rows available after filtering NaN/Inf values.")

    finite_samples = arr[finite_row_mask]
    mean = finite_samples.mean(axis=0)
    std = finite_samples.std(axis=0)

    z = np.zeros_like(finite_samples, dtype=float)
    nonzero_std = std > 0
    if np.any(nonzero_std):
        z[:, nonzero_std] = np.abs((finite_samples[:, nonzero_std] - mean[nonzero_std]) / std[nonzero_std])

    keep_finite = np.all(z <= n_sigma, axis=1)
    if int(np.sum(keep_finite)) < min_keep:
        keep_finite = np.ones(finite_samples.shape[0], dtype=bool)

    keep_mask = np.zeros(arr.shape[0], dtype=bool)
    keep_mask[finite_idx[keep_finite]] = True

    filtered = arr[keep_mask]
    if return_mask:
        return filtered, keep_mask
    return filtered


def make_corner_plot(
    samples,
    truth=None,
    labels=None,
    names=None,
    weights=None,
    ranges=None,
    filled=True,
    bins=80,
    smooth_scale_1d=0.5,
    smooth_scale_2d=0.5,
    contour_colors=None,
    sample_label=None,
    sample_labels=None,
    truth_color="black",
    title_limit=1,
    subplot_size=2.2,
    width_inch=None,
    settings=None,
    suptitle=None,
    sigma_clip_enabled=True,
    sigma_clip_n=5.0,
    sigma_clip_min_keep=100,
):
    """Make a getdist corner plot from posterior samples.

    Parameters
    ----------
    samples : array-like or sequence of array-like
        Posterior samples. Each sample set must have shape
        ``(nsamples, nfeatures)``. If a sequence is passed, multiple sample
        sets are overplotted for comparison.
    truth : array-like, optional, shape (nfeatures,)
        Truth values to overplot as getdist markers.
    labels : sequence of str, optional
        Parameter labels shown on the axes.
    names : sequence of str, optional
        Internal parameter names used by getdist.
    weights : array-like or sequence of array-like, optional
        Sample weights. For multiple sample sets, pass one weight vector per
        set.
    sample_label : str, optional
        Backward-compatible label for a single sample set.
    sample_labels : sequence of str, optional
        Labels for multiple sample sets, used in the legend.
    ranges : sequence of tuple or dict, optional
        Plot ranges for each parameter. If a sequence is given it must have
        length ``nfeatures``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure generated by getdist.
    plotter : getdist.plots.GetDistPlotter
        Plotter used to create the triangle plot.
    mc_samples : getdist.mcsamples.MCSamples
        Wrapped getdist samples object.
    """
    try:
        getdist = importlib.import_module("getdist")
        plots = importlib.import_module("getdist.plots")
        MCSamples = getdist.MCSamples
    except ImportError as exc:
        raise ImportError("make_corner_plot requires the 'getdist' package to be installed.") from exc

    def _to_numpy_array(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Accept single-set input or multiple sets for overplotting.
    if torch.is_tensor(samples):
        sample_sets = [samples.detach().cpu().numpy()]
    elif isinstance(samples, (list, tuple)):
        if len(samples) == 0:
            raise ValueError("samples must contain at least one sample set.")
        first = _to_numpy_array(samples[0])
        if first.ndim == 1:
            sample_sets = [np.asarray(samples)]
        else:
            sample_sets = [_to_numpy_array(s) for s in samples]
    else:
        samples_arr = np.asarray(samples)
        if samples_arr.ndim == 3:
            sample_sets = [np.asarray(s) for s in samples_arr]
        else:
            sample_sets = [samples_arr]

    if len(sample_sets) == 0:
        raise ValueError("samples must contain at least one sample set.")

    for s in sample_sets:
        if s.ndim != 2:
            raise ValueError("each sample set must have shape (nsamples, nfeatures).")
        if s.shape[0] == 0:
            raise ValueError("each sample set must contain at least one sample.")

    nfeatures = sample_sets[0].shape[1]
    for s in sample_sets[1:]:
        if s.shape[1] != nfeatures:
            raise ValueError("all sample sets must have the same number of features.")

    nsets = len(sample_sets)

    if truth is not None:
        if torch.is_tensor(truth):
            truth = truth.detach().cpu().numpy()
        else:
            truth = np.asarray(truth)
        truth = np.ravel(truth)
        if truth.shape[0] != nfeatures:
            raise ValueError("truth must have shape (nfeatures,).")

    weights_sets = [None for _ in range(nsets)]
    if weights is not None:
        if nsets == 1:
            w = _to_numpy_array(weights)
            w = np.ravel(w)
            if w.shape[0] != sample_sets[0].shape[0]:
                raise ValueError("weights must have shape (nsamples,).")
            weights_sets = [w]
        else:
            if not isinstance(weights, (list, tuple)):
                raise ValueError("for multiple sample sets, weights must be a sequence with one entry per set.")
            if len(weights) != nsets:
                raise ValueError("weights must have one entry per sample set.")
            multi_weights = []
            for i, (s, w) in enumerate(zip(sample_sets, weights)):
                w_arr = _to_numpy_array(w)
                w_arr = np.ravel(w_arr)
                if w_arr.shape[0] != s.shape[0]:
                    raise ValueError(f"weights[{i}] must have shape (nsamples,).")
                multi_weights.append(w_arr)
            weights_sets = multi_weights

    if names is None:
        names = [f"p{i}" for i in range(nfeatures)]
    elif len(names) != nfeatures:
        raise ValueError("names must have length nfeatures.")

    if labels is None:
        labels = list(names)
    elif len(labels) != nfeatures:
        raise ValueError("labels must have length nfeatures.")

    getdist_ranges = None
    if ranges is not None:
        if isinstance(ranges, dict):
            getdist_ranges = ranges
        else:
            if len(ranges) != nfeatures:
                raise ValueError("ranges must have length nfeatures when passed as a sequence.")
            getdist_ranges = {name: value for name, value in zip(names, ranges)}

    sample_settings = {
        "fine_bins": bins,
        "fine_bins_2D": bins,
        "smooth_scale_1D": smooth_scale_1d,
        "smooth_scale_2D": smooth_scale_2d,
    }
    if settings is not None:
        sample_settings.update(settings)

    if sample_labels is None and sample_label is not None:
        sample_labels = [sample_label]

    if sample_labels is not None:
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]
        else:
            sample_labels = list(sample_labels)
        if len(sample_labels) != nsets:
            raise ValueError("sample_labels must have one label per sample set.")

    if sigma_clip_enabled:
        filtered_sample_sets = []
        filtered_weight_sets = []
        for s, w in zip(sample_sets, weights_sets):
            s_filtered, keep_mask = filter_samples_sigma_clip(
                s,
                n_sigma=sigma_clip_n,
                min_keep=sigma_clip_min_keep,
                return_mask=True,
            )
            filtered_sample_sets.append(s_filtered)
            if w is None:
                filtered_weight_sets.append(None)
            else:
                filtered_weight_sets.append(w[keep_mask])

        sample_sets = filtered_sample_sets
        weights_sets = filtered_weight_sets

    mc_samples_list = []
    for s, w in zip(sample_sets, weights_sets):
        mc_samples_list.append(
            MCSamples(
                samples=s,
                names=names,
                labels=labels,
                weights=w,
                ranges=getdist_ranges,
                settings=sample_settings,
            )
        )

    markers = None
    if truth is not None:
        markers = {name: value for name, value in zip(names, truth) if np.isfinite(value)}

    plotter = plots.get_subplot_plotter(subplot_size=subplot_size, width_inch=width_inch)
    plotter.settings.title_limit = title_limit
    plotter.settings.linewidth_contour = 1.2
    plotter.settings.legend_frame = False
    plotter.settings.num_plot_contours = 2

    triangle_kwargs = {
        "filled": filled,
        "markers": markers,
    }
    if contour_colors is not None:
        triangle_kwargs["contour_colors"] = contour_colors
    if sample_labels is not None:
        triangle_kwargs["legend_labels"] = sample_labels
    if truth is not None:
        triangle_kwargs["marker_args"] = {
            "color": truth_color,
            "lw": 1.2,
        }

    plotter.triangle_plot(mc_samples_list, params=list(names), **triangle_kwargs)

    fig = plotter.fig

    if suptitle is not None:
        fig.suptitle(suptitle)

    if nsets == 1:
        return fig, plotter, mc_samples_list[0]
    return fig, plotter, mc_samples_list


def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=5):
    """Truncates an existing colormap to a new range."""
    cmap = colormaps.get_cmap(cmap_name)
    new_cmap_name = f"trunc({cmap.name},{minval:.2f},{maxval:.2f})"
    colors = cmap(np.linspace(minval, maxval, n))
    new_cmap = mcolors.LinearSegmentedColormap.from_list(new_cmap_name, colors)
    return new_cmap


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings."""
    ax = plt.gca() if ax is None else ax
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    ax.autoscale()
    return lc