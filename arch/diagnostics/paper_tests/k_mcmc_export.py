#!/usr/bin/env python3
"""Export the noisy Xu3 data vectors the 1M NPE saw, for MCMC comparison chains.

For each requested Xu3 measured-axis galaxy this reproduces the exact image and
spectral noise realization of the paper-model Xu3 cache (cache seed 42,
1000 galaxies per partition, matched_group_size 1): image seed
42 + 1_000_003 * (ID // 1000) + ID % 1000, spectral seed = image seed + 101,
torch generators on the GPU, same noise functions as ``train.sample_density``.

Writes, per galaxy, ``gal_<ID>_noisy.fits`` (the clean FITS with every data
array replaced by its noisy version and the noise RMS in each header) and the
clean FITS, plus ``truth.csv``. Checks: LMDB record == FITS data; noise RMS ==
the cache's; and a fresh NPE run on these exact data reproduces the cache's
TF-weighted posterior summaries to Monte Carlo precision (its samples are kept
in ``npe/`` for the comparison).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyxis.torch as pxt
import torch
from astropy.io import fits

ARCH = Path("/jet/home/xwang30/kl-nn/arch")
sys.path.insert(0, str(ARCH))

import config  # noqa: E402
from cache_contract import load_cache_partitions  # noqa: E402
from cache_posteriors import summarize_posterior_samples  # noqa: E402
from data import apply_central_halpha_snr_noise, apply_image_noise_for_snr  # noqa: E402
from model_registry import load_model_config  # noqa: E402
from networks import KLNPE  # noqa: E402
from tf_prior import TFPrior, posterior_importance_weights  # noqa: E402
from train import _seeded_generator, load_model, sample_density, seed_everything  # noqa: E402
from utils import denormalize  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis"
DATASET = "test_100k_simv3_cosi_xu3_tf_measaxis"
CACHE = S / "cache" / MODEL / f"{DATASET}_testset_tfweighted_v2_10k_s42_righthanded"
CACHE_SEED, NGALS = 42, 1000
NSPEC_PIX = 61


class OneRow(torch.utils.data.Dataset):
    def __init__(self, base, index):
        self.base, self.index = base, int(index)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.base[self.index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True, help="comma-separated Xu3 IDs")
    p.add_argument("--out", type=Path, default=S / "reports/paper-tests-1m/mcmc/package")
    p.add_argument("--nsamples", type=int, default=10000)
    args = p.parse_args()
    ids = [int(x) for x in args.ids.split(",")]
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: the cache noise was drawn with CUDA generators")
    device = "cuda"
    out = args.out
    (out / "npe").mkdir(parents=True, exist_ok=True)

    cfg = load_model_config(MODEL, configs_root=str(S / "configs"))
    config.set_model_config(cfg)
    obs = config.observation
    names = tuple(config.train["feature_names"] if isinstance(config.train, dict)
                  else config.train.feature_names)
    kw = dict(par_ranges=config.par_ranges, feature_names=names,
              target_transforms=config.TARGET_TRANSFORMS)
    vi = names.index("vcirc")
    vb = config.par_ranges["vcirc"]
    tf = TFPrior(slope=-7.22, intercept=36.0, scatter_dex=0.1,
                 vcirc_min=float(vb[0]), vcirc_max=float(vb[1]))
    base = pxt.TorchDataset(str(S / "datasets" / DATASET))
    model = load_model(KLNPE, path=str(S / "models" / MODEL / f"{MODEL}best"),
                       model_name=MODEL, strict=True, device=device,
                       networks_root=str(S / "networks"))
    cache = load_cache_partitions(CACHE)
    table = pd.read_csv(S / "samples" / f"{DATASET}.csv", float_precision="round_trip").set_index("ID")

    checks, truth_rows = [], []
    for gid in ids:
        part, idx = divmod(gid, NGALS)
        seed_img = CACHE_SEED + 1_000_003 * part + idx
        seed_spec = seed_img + 101
        rec = base[gid]
        if int(np.asarray(rec["id"]).reshape(-1)[0]) != gid:
            raise RuntimeError(f"LMDB row {gid} has a different id")
        clean_img = torch.as_tensor(rec["img"], dtype=torch.float32, device=device).unsqueeze(0)
        clean_spec = torch.as_tensor(rec["spec"], dtype=torch.float32, device=device).unsqueeze(0)
        rho_i = torch.as_tensor([float(np.asarray(rec["image_snr"]).reshape(-1)[0])], device=device)
        rho_s = torch.as_tensor([float(np.asarray(rec["central_halpha_snr"]).reshape(-1)[0])], device=device)
        noisy_img, img_meta = apply_image_noise_for_snr(
            clean_img, rho_i, randgen=_seeded_generator(device, seed_img), return_metadata=True)
        noisy_spec, spec_meta = apply_central_halpha_snr_noise(
            clean_spec, rho_s, center_fiber_index=obs["center_fiber_index"],
            center_exposure_s=obs["center_exposure_s"], offset_exposure_s=obs["offset_exposure_s"],
            spectral_units=obs["spectral_units"], randgen=_seeded_generator(device, seed_spec),
            device=device, return_metadata=True)
        sig_img = float(img_meta["noise_sigma"][0])
        sig_fib = [float(x) for x in spec_meta["noise_sigma"].cpu().numpy().reshape(-1)]
        sig_c = sig_fib[obs["center_fiber_index"]]

        # Clean LMDB data must equal the FITS data.
        src = S / "fits" / DATASET / f"part_{part + 1}" / f"gal_{gid}.fits"
        with fits.open(src) as h:
            fits_img = h[6].data.astype(np.float32)
            fits_spec = np.stack([h[1 + k].data for k in range(5)]).astype(np.float32)
        img_np = clean_img[0, 0].cpu().numpy()
        spec_np = clean_spec[0, 0, :, :NSPEC_PIX].cpu().numpy()
        if not (np.array_equal(img_np, fits_img) and np.array_equal(spec_np, fits_spec)):
            raise RuntimeError(f"{gid}: LMDB record differs from {src}")
        if np.any(clean_spec[0, 0, :, NSPEC_PIX:].cpu().numpy() != 0):
            raise RuntimeError(f"{gid}: unexpected non-zero spectral padding")

        # Noise RMS must equal what the cache recorded for this galaxy.
        c_img = float(np.load(cache.files["image_noise_sigma"][part]).reshape(-1)[idx])
        c_spec = float(np.load(cache.files["central_spectral_noise_sigma"][part]).reshape(-1)[idx])
        # Re-run the NPE on these exact noisy data (same seeds via sample_density).
        seed_everything(1234 + gid, deterministic=False)
        samples_n, meta = sample_density(
            model, OneRow(base, gid), args.nsamples, device=device, matched_group_size=1,
            noise_seed=seed_img, spectral_noise_seed=seed_spec,
            return_log_prob=False, return_observation_metadata=True)
        samples = denormalize(np.asarray(samples_n), **kw)[0]
        lw = np.asarray(posterior_importance_weights(
            samples[None, :, vi], np.asarray(meta["rmag_true"], float), tf).log_weight, float)[0]
        w = np.exp(lw - lw.max()); w /= w.sum()
        mine = summarize_posterior_samples(samples, names, w)
        ref = np.load(cache.files["tf_target_mean_estimates"][part])[idx]
        sd = np.sqrt(w @ (samples - mine[1]) ** 2)
        ess = 1.0 / np.sum(w ** 2)
        z_mean = (mine[1] - ref[1]) / (sd / np.sqrt(ess))
        z_mean[names.index("theta_int")] = np.nan  # circular; checked by eye below
        check = dict(ID=gid, seed_image=seed_img, seed_spectra=seed_spec,
                     image_sigma=sig_img, cache_image_sigma=c_img,
                     central_spectral_sigma=sig_c, cache_central_spectral_sigma=c_spec,
                     ess=float(ess), mean_shift_in_mc_sigma=dict(zip(names, np.round(z_mean, 2).tolist())),
                     g_mean_new=mine[1, :2].tolist(), g_mean_cache=ref[1, :2].tolist())
        print(json.dumps(check), flush=True)
        if not (np.isclose(sig_img, c_img, rtol=1e-5) and np.isclose(sig_c, c_spec, rtol=1e-5)):
            raise RuntimeError(f"{gid}: noise RMS differs from the cache")
        checks.append(check)
        np.savez(out / "npe" / f"gal_{gid}_npe_samples.npz", samples=samples.astype(np.float32),
                 log_tf_weight=lw.astype(np.float32), feature_names=np.array(names),
                 summary_16_mean_84=mine)

        # Write the clean and noisy FITS.
        shutil.copy2(src, out / f"gal_{gid}_clean.fits")
        with fits.open(src) as h:
            h[6].data = noisy_img[0, 0].cpu().numpy().astype(np.float64)
            h[6].header["NOISESIG"] = (sig_img, "white Gaussian noise RMS per pixel (added)")
            for k in range(5):
                h[1 + k].data = noisy_spec[0, 0, k, :NSPEC_PIX].cpu().numpy().astype(np.float64)
                h[1 + k].header["NOISESIG"] = (sig_fib[k], "white Gaussian noise RMS per pixel (added)")
            h[0].header["NOISY"] = (True, "noise added; see README")
            h[0].header["NOISEIMG"] = (seed_img, "torch CUDA seed, image noise")
            h[0].header["NOISESPC"] = (seed_spec, "torch CUDA seed, spectral noise")
            h.writeto(out / f"gal_{gid}_noisy.fits", overwrite=True)
        row = table.loc[gid]
        truth_rows.append(dict(ID=gid, g1=row.g1, g2=row.g2, theta_int=row.theta_int, sini=row.sini,
                               cosi=float(np.sqrt(1 - row.sini ** 2)), v0=row.v0, vcirc=row.vcirc,
                               rscale=row.rscale, hlr=row.hlr, rmag_true=row.rmag_true,
                               halpha_flux_central=row.halpha_flux_true,
                               image_snr=row.image_snr, central_halpha_snr=row.central_halpha_snr,
                               image_sigma=sig_img, **{f"fiber{k}_sigma": sig_fib[k] for k in range(5)}))
    pd.DataFrame(truth_rows).to_csv(out / "truth.csv", index=False)
    (out / "npe" / "checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
