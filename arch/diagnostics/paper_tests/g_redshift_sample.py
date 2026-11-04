#!/usr/bin/env python3
"""G: NPE posteriors for the same galaxies rendered at several redshifts.

The data set (``paper_tests_G_redshift_scan_xu3like_1k_measaxis``) holds
``nbase`` Xu3 galaxies at each redshift, redshift-major. For each noise seed
and redshift, the rows of that redshift are passed to ``train.sample_density``
in base order with ``matched_group_size=1``, so base ``b`` gets image noise
seed ``noise_seed + b`` and spectral seed ``noise_seed + 101 + b`` at every
redshift (paired noise). The noise level follows each row's own clean data
and catalog SNRs. Flow draws use the same torch seed for every redshift.
10^4 identity+R90 samples and TF importance weights, as in production
(``cache_posteriors.py``).

Output (``--out``): one npz per noise seed with, per row (ID order),
16th/mean/84th TF-weighted summaries and sd for all 9 targets (H-alpha
flux in log10), truth, ESS
and noise levels; seed 0 also keeps the samples and log weights of g1, g2,
cos i and vcirc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyxis.torch as pxt
import torch

ARCH = Path("/jet/home/xwang30/kl-nn/arch")
sys.path.insert(0, str(ARCH))

import config  # noqa: E402
from cache_posteriors import summarize_posterior_samples  # noqa: E402
from model_registry import load_model_config  # noqa: E402
from networks import KLNPE  # noqa: E402
from tf_prior import TFPrior, posterior_importance_weights  # noqa: E402
from train import load_model, sample_density, seed_everything  # noqa: E402
from utils import denormalize, resolve_feature_index  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis"
NAME = "paper_tests_G_redshift_scan_xu3like_1k_measaxis"
KEEP = ("g1", "g2", "cosi", "vcirc")  # parameters whose samples are kept for seed 0


def weighted_sd(samples, weight, names):
    """Weighted standard deviation per parameter; circular for theta_int."""

    mean = weight @ samples
    resid = samples - mean
    if "theta_int" in names:
        t = names.index("theta_int")
        theta = samples[:, t]
        center = np.arctan2(weight @ np.sin(theta), weight @ np.cos(theta))
        resid[:, t] = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    return np.sqrt(weight @ resid**2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--dataset", default=NAME)
    p.add_argument("--manifest", type=Path,
                   default=S / "samples" / f"{NAME}_manifest.csv")
    p.add_argument("--noise-seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--nsamples", type=int, default=10000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or S / "cache/paper_tests_G_redshift" / args.model
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(args.model, configs_root=str(S / "configs"))
    config.set_model_config(cfg)
    names = tuple(config.train["feature_names"] if isinstance(config.train, dict)
                  else config.train.feature_names)
    vi = resolve_feature_index(names, "vcirc")
    vb = config.par_ranges["vcirc"]
    tf = TFPrior(slope=-7.22, intercept=36.0, scatter_dex=0.1,
                 vcirc_min=float(vb[0]), vcirc_max=float(vb[1]))
    data_dir = S / "datasets" / args.dataset
    config.require_matching_dataset_par_ranges(data_dir, config.par_ranges)
    base = pxt.TorchDataset(str(data_dir))

    manifest = pd.read_csv(args.manifest)
    if len(manifest) != len(base) or not (manifest["ID"].to_numpy() == np.arange(len(base))).all():
        raise ValueError("manifest IDs must be 0..N-1 and match the data set length")
    redshifts = np.unique(manifest["redshift"].to_numpy())
    groups = [manifest.index[manifest["redshift"] == z].to_numpy() for z in redshifts]
    nbase = len(groups[0])
    for z, rows in zip(redshifts, groups):
        # Every redshift lists the same bases in the same order.
        if not (manifest.loc[rows, "base_id"].to_numpy() == np.arange(nbase)).all():
            raise ValueError(f"redshift {z}: rows are not in base order")
    for i in (0, len(base) - 1):
        record_id = int(np.asarray(base[i]["id"]).reshape(-1)[0])
        if record_id != i:
            raise ValueError(f"LMDB row {i} has id {record_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = S / "models" / args.model / f"{args.model}best"
    model = load_model(KLNPE, path=str(ckpt), model_name=args.model, strict=True,
                       device=device, networks_root=str(S / "networks"))
    kw = dict(par_ranges=config.par_ranges, feature_names=names,
              target_transforms=config.TARGET_TRANSFORMS)
    keep = [names.index(k) for k in KEEP]
    fi = names.index("halpha_flux_true")

    for seed in [int(s) for s in args.noise_seeds.split(",")]:
        n = len(base)
        summary = np.empty((n, 3, len(names)))
        sd = np.empty((n, len(names)))
        truth = np.empty((n, len(names)))
        ess = np.empty(n)
        image_sigma = np.empty(n)
        spec_sigma = np.empty(n)
        if seed == 0:
            kept = np.empty((n, args.nsamples, len(keep)), dtype=np.float32)
            kept_lw = np.empty((n, args.nsamples), dtype=np.float32)
        for z, rows in zip(redshifts, groups):
            ds = torch.utils.data.Subset(base, rows.tolist())
            seed_everything(5000 + seed, deterministic=False)
            samples_n, meta = sample_density(
                model, ds, args.nsamples, device=device, matched_group_size=1,
                noise_seed=10_000 * seed, return_log_prob=False,
                return_observation_metadata=True,
            )
            samples = denormalize(np.asarray(samples_n), **kw)
            row_truth = denormalize(np.asarray(meta["truth"]), **kw)
            # Summarize the H-alpha flux in log10, the training coordinate.
            samples[..., fi] = np.log10(samples[..., fi])
            row_truth[:, fi] = np.log10(row_truth[:, fi])
            truth[rows] = row_truth
            imp = posterior_importance_weights(
                samples[..., vi], np.asarray(meta["rmag_true"], float), tf)
            lw = np.asarray(imp.log_weight, float)
            for j, row in enumerate(rows):
                w = np.exp(lw[j] - lw[j].max())
                w /= w.sum()
                summary[row] = summarize_posterior_samples(samples[j], names, w)
                sd[row] = weighted_sd(samples[j].astype(float), w, names)
            ess[rows] = np.asarray(imp.effective_sample_fraction)
            image_sigma[rows] = np.asarray(meta["image_noise_sigma"]).reshape(len(rows), -1)[:, 0]
            spec_sigma[rows] = np.asarray(meta["central_spectral_noise_sigma"]).reshape(len(rows), -1)[:, 0]
            if seed == 0:
                kept[rows] = samples[..., keep]
                kept_lw[rows] = lw
            print(json.dumps(dict(seed=seed, z=float(z), ess_median=float(np.median(ess[rows])))),
                  flush=True)
        # The image is the same at every redshift, so paired rows must get the
        # same image noise level.
        ref = image_sigma[groups[0]]
        for rows in groups[1:]:
            if not np.allclose(image_sigma[rows], ref, rtol=1e-4):
                raise RuntimeError("image noise level differs across redshifts")
        payload = dict(summary=summary, sd=sd, truth=truth, ess_fraction=ess,
                       image_noise_sigma=image_sigma, central_spectral_noise_sigma=spec_sigma,
                       redshift=manifest["redshift"].to_numpy(),
                       base_id=manifest["base_id"].to_numpy(),
                       source_id=manifest["source_id"].to_numpy(),
                       feature_names=np.array(names), noise_seed=10_000 * seed)
        if seed == 0:
            payload.update(samples=kept, log_weight=kept_lw, kept_names=np.array(KEEP))
        np.savez(out / f"seed{seed}.npz", **payload)
    (out / "meta.json").write_text(json.dumps(dict(
        model=args.model, dataset=args.dataset, nsamples=args.nsamples,
        noise_seeds=args.noise_seeds, redshifts=redshifts.tolist(), nbase=nbase,
        noise_seed_rule="image noise_seed=10000*s+b, spectral 10000*s+101+b, torch seed 5000+s",
    ), indent=1))


if __name__ == "__main__":
    main()
