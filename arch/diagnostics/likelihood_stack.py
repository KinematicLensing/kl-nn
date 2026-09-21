#!/usr/bin/env python3
"""Shared-η likelihood stack versus IVW Means on repeated noises of one galaxy."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import sys
from pathlib import Path

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from data import apply_central_halpha_snr_noise, apply_image_noise_for_snr
from diagnostics.likelihood_stack_core import (
    DATA_ROOT,
    G01_DATASET,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    N_REALIZATIONS,
    N_SAMPLES,
    PREFIXES,
    PROPOSAL_PER_REALIZATION,
    SELECT_DIR,
    STACK_DIR,
    component_table,
    dumps_meta,
    galaxy_shape_variance,
    galaxies_npz,
    loads_meta,
    noise_seeds,
    prefix_ivw,
    prefix_mean_of_means,
    prefix_stack_map,
    shear_columns,
    takeaway,
    vector_median_abs_residual,
    write_json,
)
from diagnostics.likelihood_stack_select import load_galaxies
from model_registry import load_model_config
from networks import KLNPE
from train import (
    _seeded_generator,
    build_observation_levels,
    load_model,
    seed_everything,
    validate_observation_record,
)
from utils import denormalize

FIGURE_DPI = 140


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / G01_DATASET)
    parser.add_argument("--model-name", default=G01_NPE)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--checkpoint-suffix", default="best")
    parser.add_argument("--galaxies", type=Path, default=galaxies_npz(SELECT_DIR))
    parser.add_argument("--report-dir", type=Path, default=STACK_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument(
        "--proposal-per-realization",
        type=int,
        default=PROPOSAL_PER_REALIZATION,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-draws",
        action="store_true",
        help="Rebuild figures and HTML from stack.npz without sampling the NPE",
    )
    return parser.parse_args(argv)


def checkpoint_file(model_root: Path, model_name: str, suffix: str = "best") -> Path:
    return Path(model_root) / model_name / f"{model_name}{suffix}"


def stack_file(report_dir: Path) -> Path:
    return Path(report_dir) / "stack.npz"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fmt(value, digits=3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def load_selected_records(dataset, indices: np.ndarray, device: torch.device) -> dict:
    images = []
    spectra = []
    positions = []
    rmag = []
    image_snr = []
    spec_snr = []
    for index in indices:
        record = dataset[int(index)]
        rmag_i, _, image_snr_i, spec_snr_i = validate_observation_record(
            record, location=f"load {index}"
        )
        images.append(torch.as_tensor(record["img"]).float())
        spectra.append(torch.as_tensor(record["spec"]).float())
        positions.append(torch.as_tensor(record["fib_pos"]).float())
        rmag.append(float(rmag_i))
        image_snr.append(float(image_snr_i))
        spec_snr.append(float(spec_snr_i))
    image_snr_t = torch.tensor(image_snr, device=device, dtype=torch.float32)
    spec_snr_t = torch.tensor(spec_snr, device=device, dtype=torch.float32)
    image_snr_t, spec_snr_t = build_observation_levels(image_snr_t, spec_snr_t)
    return {
        "img": torch.stack(images).to(device),
        "spec": torch.stack(spectra).to(device),
        "fib_pos": torch.stack(positions).to(device),
        "rmag": torch.tensor(rmag, device=device, dtype=torch.float32),
        "image_snr": image_snr_t,
        "spec_snr": spec_snr_t,
    }


def apply_independent_noise(
    tensors: dict,
    *,
    indices: np.ndarray,
    realization: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = tensors["img"].shape[0]
    noisy_images = []
    noisy_spectra = []
    center = int(config.observation["center_fiber_index"])
    for row in range(n):
        image_seed, spec_seed = noise_seeds(seed, realization, int(indices[row]))
        image = tensors["img"][row : row + 1]
        spec = tensors["spec"][row : row + 1]
        noisy_images.append(
            apply_image_noise_for_snr(
                image,
                tensors["image_snr"][row : row + 1],
                randgen=_seeded_generator(device, image_seed),
            )
        )
        noisy_spectra.append(
            apply_central_halpha_snr_noise(
                spec,
                tensors["spec_snr"][row : row + 1],
                center_fiber_index=center,
                center_exposure_s=config.observation["center_exposure_s"],
                offset_exposure_s=config.observation["offset_exposure_s"],
                spectral_units=config.observation["spectral_units"],
                randgen=_seeded_generator(device, spec_seed),
                device=device,
            )
        )
    return torch.cat(noisy_images, dim=0), torch.cat(noisy_spectra, dim=0)


def sample_realization(
    model,
    tensors: dict,
    noisy_image: torch.Tensor,
    noisy_spec: torch.Tensor,
    *,
    n_samples: int,
    batch_size: int,
    channels_last: bool,
    names: tuple[str, ...],
    par_ranges,
    proposal_per_realization: int,
) -> dict[str, np.ndarray]:
    n = noisy_image.shape[0]
    g1_idx, g2_idx = shear_columns(names)
    means = np.empty((n, 2), dtype=np.float64)
    variances = np.empty(n, dtype=np.float64)
    proposals = np.empty(
        (n, proposal_per_realization, len(names)), dtype=np.float32
    )
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        image = noisy_image[start:stop]
        spec = noisy_spec[start:stop]
        positions = tensors["fib_pos"][start:stop]
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spec = spec.contiguous(memory_format=torch.channels_last)
        context = {
            "rmag_true": tensors["rmag"][start:stop],
            "image_snr": tensors["image_snr"][start:stop],
            "central_halpha_snr": tensors["spec_snr"][start:stop],
        }
        with torch.inference_mode():
            samples = model.sample(
                image,
                spec,
                n_samples,
                fiber_positions=positions,
                observation_context=context,
            )
        if samples.ndim != 3:
            raise RuntimeError(f"unexpected sample shape {tuple(samples.shape)}")
        physical = denormalize(
            samples.float().cpu(), par_ranges, feature_names=names
        ).numpy()
        g1 = physical[..., g1_idx]
        g2 = physical[..., g2_idx]
        means[start:stop, 0] = g1.mean(axis=1)
        means[start:stop, 1] = g2.mean(axis=1)
        variances[start:stop] = galaxy_shape_variance(
            g1.var(axis=1, ddof=0), g2.var(axis=1, ddof=0)
        )
        proposals[start:stop] = (
            samples[:, :proposal_per_realization].float().cpu().numpy()
        )
        print(f"sampled galaxies {stop}/{n}", flush=True)
    return {"mean_g": means, "variance": variances, "proposal": proposals}


def score_bank(
    model,
    tensors: dict,
    noisy_image: torch.Tensor,
    noisy_spec: torch.Tensor,
    bank: torch.Tensor,
    *,
    channels_last: bool,
) -> np.ndarray:
    if channels_last:
        noisy_image = noisy_image.contiguous(memory_format=torch.channels_last)
        noisy_spec = noisy_spec.contiguous(memory_format=torch.channels_last)
    context = {
        "rmag_true": tensors["rmag"],
        "image_snr": tensors["image_snr"],
        "central_halpha_snr": tensors["spec_snr"],
    }
    if bank.ndim != 2:
        raise ValueError("bank must have shape (n_candidates, n_features)")
    expanded = bank.unsqueeze(0).expand(noisy_image.shape[0], -1, -1)
    with torch.inference_mode():
        log_prob = model.posterior_log_prob(
            noisy_image,
            noisy_spec,
            expanded,
            tensors["fib_pos"],
            context,
        )
    return log_prob.float().cpu().numpy()


def run_stack(
    args,
    selected: dict,
    *,
    names: tuple[str, ...],
    par_ranges,
    model,
    tensors: dict,
    channels_last: bool,
    device: torch.device,
) -> dict:
    n_gal = len(selected["index"])
    n_real = args.n_realizations
    n_feat = len(names)
    proposal_n = args.proposal_per_realization
    means = np.empty((n_gal, n_real, 2), dtype=np.float64)
    variances = np.empty((n_gal, n_real), dtype=np.float64)
    proposals = np.empty((n_gal, n_real, proposal_n, n_feat), dtype=np.float32)
    g1_idx, g2_idx = shear_columns(names)

    for realization in range(n_real):
        print(f"realization {realization + 1}/{n_real}", flush=True)
        noisy_image, noisy_spec = apply_independent_noise(
            tensors,
            indices=selected["index"],
            realization=realization,
            seed=args.seed,
            device=device,
        )
        sampled = sample_realization(
            model,
            tensors,
            noisy_image,
            noisy_spec,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            channels_last=channels_last,
            names=names,
            par_ranges=par_ranges,
            proposal_per_realization=proposal_n,
        )
        means[:, realization] = sampled["mean_g"]
        variances[:, realization] = sampled["variance"]
        proposals[:, realization] = sampled["proposal"]

    stack_map = np.empty((n_gal, len(PREFIXES), 2), dtype=np.float64)
    stack_map_full = np.empty((n_gal, len(PREFIXES), n_feat), dtype=np.float64)
    for row in range(n_gal):
        print(f"stacking galaxy {row + 1}/{n_gal}", flush=True)
        row_slice = slice(row, row + 1)
        row_tensors = {
            "img": tensors["img"][row_slice],
            "spec": tensors["spec"][row_slice],
            "fib_pos": tensors["fib_pos"][row_slice],
            "rmag": tensors["rmag"][row_slice],
            "image_snr": tensors["image_snr"][row_slice],
            "spec_snr": tensors["spec_snr"][row_slice],
        }
        noisy_images = []
        noisy_spectra = []
        for realization in range(n_real):
            image, spec = apply_independent_noise(
                row_tensors,
                indices=selected["index"][row_slice],
                realization=realization,
                seed=args.seed,
                device=device,
            )
            noisy_images.append(image)
            noisy_spectra.append(spec)
        noisy_image = torch.cat(noisy_images, dim=0)
        noisy_spec = torch.cat(noisy_spectra, dim=0)
        stacked_tensors = {
            "fib_pos": tensors["fib_pos"][row]
            .unsqueeze(0)
            .expand(n_real, *tensors["fib_pos"].shape[1:])
            .contiguous(),
            "rmag": tensors["rmag"][row].expand(n_real).contiguous(),
            "image_snr": tensors["image_snr"][row].expand(n_real).contiguous(),
            "spec_snr": tensors["spec_snr"][row].expand(n_real).contiguous(),
        }
        bank = torch.as_tensor(
            proposals[row].reshape(n_real * proposal_n, n_feat),
            device=device,
            dtype=torch.float32,
        )
        log_probs = score_bank(
            model,
            stacked_tensors,
            noisy_image,
            noisy_spec,
            bank,
            channels_last=channels_last,
        )
        mapped = prefix_stack_map(
            bank.cpu().numpy(),
            log_probs,
            PREFIXES,
            proposal_per_realization=proposal_n,
        )
        physical = denormalize(mapped, par_ranges, feature_names=names)
        stack_map_full[row] = physical
        stack_map[row, :, 0] = physical[:, g1_idx]
        stack_map[row, :, 1] = physical[:, g2_idx]

    ivw = prefix_ivw(means, variances, PREFIXES)
    mom = prefix_mean_of_means(means, PREFIXES)
    return {
        "mean_g": means,
        "variance": variances,
        "ivw": ivw,
        "mean_of_means": mom,
        "stack_map": stack_map,
        "stack_map_full": stack_map_full,
        "prefixes": np.asarray(PREFIXES, dtype=np.int64),
    }


def save_stack(path: Path, selected: dict, result: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=selected["index"],
        truth_g=selected["truth_g"],
        cached_mean_g=selected["mean_g"],
        mean_g=result["mean_g"],
        variance=result["variance"],
        ivw=result["ivw"],
        mean_of_means=result["mean_of_means"],
        stack_map=result["stack_map"],
        stack_map_full=result["stack_map_full"],
        prefixes=result["prefixes"],
        meta_json=dumps_meta(meta),
    )


def load_stack(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "index": np.asarray(data["index"], dtype=np.int64),
        "truth_g": np.asarray(data["truth_g"], dtype=np.float64),
        "cached_mean_g": np.asarray(data["cached_mean_g"], dtype=np.float64),
        "mean_g": np.asarray(data["mean_g"], dtype=np.float64),
        "variance": np.asarray(data["variance"], dtype=np.float64),
        "ivw": np.asarray(data["ivw"], dtype=np.float64),
        "mean_of_means": np.asarray(data["mean_of_means"], dtype=np.float64),
        "stack_map": np.asarray(data["stack_map"], dtype=np.float64),
        "stack_map_full": np.asarray(data["stack_map_full"], dtype=np.float64),
        "prefixes": tuple(int(value) for value in np.asarray(data["prefixes"])),
        "meta": loads_meta(data["meta_json"]),
    }


def estimator_payload(result: dict) -> dict[str, np.ndarray]:
    return {
        "mean_of_means": result["mean_of_means"],
        "ivw": result["ivw"],
        "stack_map": result["stack_map"],
    }


def save_figures(result: dict, report_dir: Path) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures = {}
    truth = result["truth_g"]
    last = len(result["prefixes"]) - 1
    panels = (
        ("cached Mean", result["cached_mean_g"]),
        ("IVW Means, 16 noises", result["ivw"][:, last]),
        ("shared-parameter MAP, 16 noises", result["stack_map"][:, last]),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.2))
    for column, (title, estimate) in enumerate(panels):
        for row, component in enumerate((0, 1)):
            axis = axes[row, column]
            axis.scatter(
                truth[:, component],
                estimate[:, component],
                s=22,
                color="#1f4e79",
                alpha=0.9,
            )
            lo = float(np.min(truth[:, component]))
            hi = float(np.max(truth[:, component]))
            pad = 0.05 * max(hi - lo, 0.02)
            axis.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad], color="#bbbbbb", lw=1
            )
            name = "g1" if component == 0 else "g2"
            if row == 0:
                axis.set_title(title)
            axis.set_xlabel(f"true {name}")
            axis.set_ylabel(f"recovered {name}")
            axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path = report_dir / "recovery.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["recovery"] = path.name

    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    colors = {
        "mean_of_means": "#8a939c",
        "ivw": "#b85c38",
        "stack_map": "#1f4e79",
    }
    labels = {
        "mean_of_means": "mean of Means",
        "ivw": "inverse-variance Means",
        "stack_map": "shared-parameter MAP",
    }
    prefixes = np.asarray(result["prefixes"], dtype=np.float64)
    for name, estimates in estimator_payload(result).items():
        residuals = [
            vector_median_abs_residual(truth, estimates[:, index])
            for index in range(len(prefixes))
        ]
        axis.plot(
            prefixes,
            residuals,
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    stack_one = vector_median_abs_residual(truth, result["stack_map"][:, 0])
    axis.plot(
        prefixes,
        stack_one / np.sqrt(prefixes),
        color="#1f4e79",
        ls="--",
        lw=1,
        label="1/√N from one-draw MAP",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(result["prefixes"]))
    axis.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    axis.set_xlabel("number of independent noises")
    axis.set_ylabel("median |ĝ − g|")
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = report_dir / "residual_vs_n.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["residual_vs_n"] = path.name
    return figures


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    rows = []
    labels = {
        "mean_of_means": "mean of Means",
        "ivw": "inverse-variance Means",
        "stack_map": "shared-parameter MAP",
    }
    for row in payload["table"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[row['estimator']])}</td>"
            f"<td class='num'>{row['n_realizations']}</td>"
            f"<td class='num'>{fmt(row['g1_m'])}</td>"
            f"<td class='num'>{fmt(row['g2_m'])}</td>"
            f"<td class='num'>{fmt(row['g1_c'], 4)}</td>"
            f"<td class='num'>{fmt(row['g2_c'], 4)}</td>"
            f"<td class='num'>{fmt(row['median_abs_residual'], 4)}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Shared-parameter stack versus inverse-variance Means</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Does combining many noises of the same galaxy undo the Mean shrinkage?</h1>
<p class="lead">A real catalog galaxy is seen once. This page asks a
counterfactual: if we could redraw the noise sixteen times on the same 128
stamps, would a shared-parameter likelihood stack recover the true shear,
or would we still sit on the shrunk Mean? It is a diagnostic of the
estimator. It is not a catalog shear, and it does not replace the
multiplicative-bias tables.</p>

<p>Each galaxy keeps its true shear and its seven nuisances. The sixteen
noises are independent. The inverse-variance control averages the sixteen
posterior Means with weights 1/σ<sub>gal</sub><sup>2</sup>. The stack scores
one nine-parameter point under every noise and takes the mode of that product.
Because the training prior is flat in the shear box, that product is the
shared-parameter likelihood.</p>

<figure>
<img src="{html.escape(figures["recovery"])}" alt="Recovered versus true shear">
<figcaption>Left: the cached Mean that selected these galaxies. Middle: the
inverse-variance mean of sixteen Means. Right: the shared-parameter stack
mode after the same sixteen noises. A slope of one would track the truth.
</figcaption>
</figure>

<table>
<thead><tr>
<th>Estimator</th><th>N</th><th>m of g1</th><th>m of g2</th>
<th>c of g1</th><th>c of g2</th><th>median |ĝ − g|</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<p>These 128 galaxies were chosen because the Mean was already pulled toward
zero, so the slope here is not the catalog number. It only asks whether
combining noises on this biased set unshrinks the answer.</p>

<figure>
<img src="{html.escape(figures["residual_vs_n"])}" alt="Residual versus number of noises">
<figcaption>Median vector residual against the number of independent noises.
A calibrated shared-parameter likelihood should fall like 1/√N. An
average of shrunk Means should stay put.</figcaption>
</figure>

<p>{html.escape(payload["takeaway"])}</p>
<p class="note">Identity samples only; record-backed image and line S/N. The
training shear prior is uniform out to 0.1. Sixteen noises, 2048 posterior
draws each, with 256 draws from each noise used as the stack proposal.
</p>
</body></html>
"""
    path = report_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def write_outputs(result: dict, meta: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    table = component_table(
        result["truth_g"], estimator_payload(result), tuple(result["prefixes"])
    )
    figures = save_figures(result, report_dir)
    payload = {
        **meta,
        "table": table,
        "takeaway": takeaway(table, n_realizations=int(result["prefixes"][-1])),
        "figures": figures,
        "n_galaxies": int(len(result["truth_g"])),
        "cached_median_abs_residual": vector_median_abs_residual(
            result["truth_g"], result["cached_mean_g"]
        ),
    }
    write_html(payload, figures, report_dir)
    write_json(report_dir / "report.json", payload)
    print(f"Wrote {report_dir / 'report.json'}", flush=True)


def write_report(args) -> None:
    html_path = args.report_dir / "report.html"
    draws_path = stack_file(args.report_dir)
    if html_path.exists() and not args.overwrite and not args.reuse_draws:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    if args.n_realizations != N_REALIZATIONS:
        raise ValueError(f"n_realizations must be {N_REALIZATIONS} to match PREFIXES")
    if args.proposal_per_realization > args.n_samples:
        raise ValueError("proposal-per-realization cannot exceed n-samples")
    if PREFIXES[-1] != args.n_realizations:
        raise ValueError("PREFIXES must end at n_realizations")

    if args.reuse_draws:
        result = load_stack(draws_path)
        write_outputs(result, result["meta"], args.report_dir)
        return

    selected = load_galaxies(args.galaxies)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    configs_root = args.model_root.parent / "configs"
    model_config = load_model_config(args.model_name, configs_root=str(configs_root))
    config.set_model_config(model_config)
    names = tuple(config.TARGET_NAMES)
    par_ranges = config.par_ranges
    config.require_matching_dataset_par_ranges(args.data_dir, par_ranges)

    checkpoint = checkpoint_file(
        args.model_root, args.model_name, args.checkpoint_suffix
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = load_model(
        KLNPE,
        path=str(checkpoint),
        model_name=args.model_name,
        device=str(device),
        strict=True,
        networks_root=str(args.model_root.parent / "networks"),
    )
    model.eval()
    channels_last = bool(model_config.train.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    dataset = pxt.TorchDataset(str(args.data_dir))
    tensors = load_selected_records(dataset, selected["index"], device)
    result = run_stack(
        args,
        selected,
        names=names,
        par_ranges=par_ranges,
        model=model,
        tensors=tensors,
        channels_last=channels_last,
        device=device,
    )
    meta = {
        "model_name": args.model_name,
        "data_dir": str(args.data_dir),
        "galaxies": str(args.galaxies),
        "n_galaxies": int(len(selected["index"])),
        "n_realizations": int(args.n_realizations),
        "n_samples": int(args.n_samples),
        "proposal_per_realization": int(args.proposal_per_realization),
        "prefixes": list(PREFIXES),
        "seed": int(args.seed),
        "identity_only": True,
    }
    save_stack(draws_path, selected, result, meta)
    print(f"Wrote {draws_path}", flush=True)
    result = {
        **result,
        "truth_g": selected["truth_g"],
        "cached_mean_g": selected["mean_g"],
        "index": selected["index"],
    }
    write_outputs(result, meta, args.report_dir)


def main(argv=None) -> None:
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
