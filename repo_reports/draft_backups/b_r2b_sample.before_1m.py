#!/usr/bin/env python3
"""R2b: NPE posterior of one Xu3 galaxy at two image S/N levels.

The central H-alpha S/N stays at its catalog value. Image S/N is overridden in
the record handed to ``train.sample_density`` (the noise level and the model's
observation context both follow it). The image and spectral noise seeds are the
same for every S/N level, so the noise field has the same pattern and only its
amplitude changes. 10^4 identity+R90 samples, TF importance weights as in
production (``cache_posteriors.py``).

Output: one npz per (S/N, noise seed) under
$S/cache/paper_tests_B_inference/r2b/ (a new directory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyxis.torch as pxt
import torch

ARCH = Path("/jet/home/xwang30/kl-nn/arch")
sys.path.insert(0, str(ARCH))

import config  # noqa: E402
from model_registry import load_model_config  # noqa: E402
from networks import KLNPE  # noqa: E402
from tf_prior import TFPrior, posterior_importance_weights  # noqa: E402
from train import load_model, sample_density, seed_everything  # noqa: E402
from utils import denormalize, resolve_feature_index  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
DATASET = S / "datasets/test_100k_simv3_cosi_xu3_tf"
OUT = S / "cache/paper_tests_B_inference/r2b"


class OverrideSNR(torch.utils.data.Dataset):
    """One-record view of a dataset with image_snr replaced."""

    def __init__(self, base, index, image_snr, field="image_snr"):
        self.base, self.index, self.image_snr, self.field = base, int(index), image_snr, field

    def __len__(self):
        return 1

    def __getitem__(self, i):
        record = dict(self.base[self.index])
        if self.image_snr is not None:
            ref = torch.as_tensor(record[self.field])
            record[self.field] = torch.full_like(ref, float(self.image_snr))
        return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--galaxy", type=int, default=503)
    p.add_argument("--snrs", type=str, default="50,200,catalog")
    p.add_argument("--noise-seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--nsamples", type=int, default=10000)
    p.add_argument("--field", choices=("image_snr", "central_halpha_snr"), default="image_snr",
                   help="which S/N the --snrs values override (default image S/N)")
    args = p.parse_args()

    cfg = load_model_config(MODEL, configs_root=str(S / "configs"))
    config.set_model_config(cfg)
    names = tuple(config.train["feature_names"] if isinstance(config.train, dict)
                  else config.train.feature_names)
    vi = resolve_feature_index(names, "vcirc")
    vb = config.par_ranges["vcirc"]
    tf = TFPrior(slope=-7.22, intercept=36.0, scatter_dex=0.1,
                 vcirc_min=float(vb[0]), vcirc_max=float(vb[1]))
    config.require_matching_dataset_par_ranges(DATASET, config.par_ranges)
    base = pxt.TorchDataset(str(DATASET))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = S / "models" / MODEL / f"{MODEL}best"
    model = load_model(KLNPE, path=str(ckpt), model_name=MODEL, strict=True,
                       device=device, networks_root=str(S / "networks"))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in [int(s) for s in args.noise_seeds.split(",")]:
        for snr in args.snrs.split(","):
            level = None if snr == "catalog" else float(snr)
            ds = OverrideSNR(base, args.galaxy, level, args.field)
            seed_everything(1000 + seed, deterministic=False)
            samples_n, meta = sample_density(
                model, ds, args.nsamples, device=device, matched_group_size=1,
                noise_seed=seed, spectral_noise_seed=seed + 101,
                return_log_prob=False, return_observation_metadata=True,
            )
            kw = dict(par_ranges=config.par_ranges, feature_names=names,
                      target_transforms=config.TARGET_TRANSFORMS)
            samples = denormalize(np.asarray(samples_n), **kw).astype(np.float32)
            truth = denormalize(np.asarray(meta["truth"]), **kw)[0]
            imp = posterior_importance_weights(samples[..., vi], np.asarray(meta["rmag_true"], float), tf)
            lw = np.asarray(imp.log_weight, float)[0]
            w = np.exp(lw - lw.max()); w /= w.sum()
            g = samples[0, :, :2].astype(float)
            mu = w @ g
            sd = np.sqrt(w @ (g - mu) ** 2)
            prefix = "snr" if args.field == "image_snr" else "hasnr"
            tag = f"gal{args.galaxy}_{prefix}{snr}_seed{seed}"
            np.savez(OUT / f"{tag}.npz", samples=samples[0], log_weight=lw, truth=truth,
                     image_snr=np.asarray(meta["image_snr"]), central_halpha_snr=np.asarray(meta["central_halpha_snr"]),
                     image_noise_sigma=np.asarray(meta["image_noise_sigma"]), ess_fraction=np.asarray(imp.effective_sample_fraction))
            row = dict(galaxy=args.galaxy, field=args.field, snr=snr, noise_seed=seed,
                       image_snr=float(meta["image_snr"][0]), central_halpha_snr=float(meta["central_halpha_snr"][0]),
                       mean_g=mu.tolist(), sd_g=sd.tolist(), sigma_e=float(np.hypot(*sd)),
                       ess_fraction=float(np.asarray(imp.effective_sample_fraction)[0]), truth_g=truth[:2].tolist())
            print(json.dumps(row), flush=True)
            summary.append(row)
    (OUT / f"summary_gal{args.galaxy}_{args.field}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
