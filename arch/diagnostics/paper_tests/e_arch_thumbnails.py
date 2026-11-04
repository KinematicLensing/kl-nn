"""Architecture figure (2026-10-02): noisy image + spectra thumbnails of Xu3 galaxy 6447 at its
catalog SNRs (noise seed 4, as in fig_posterior_snr; CPU generator, so the exact noise pattern may
differ from the GPU run). Writes thumb_image.pdf, thumb_spectra.pdf into figures_src/."""
import sys
from pathlib import Path
import numpy as np
import torch
import pyxis.torch as pxt
ARCH = Path("/jet/home/xwang30/kl-nn/arch"); sys.path.insert(0, str(ARCH))
import config
from model_registry import load_model_config
from data import (apply_image_noise_for_snr, apply_central_halpha_snr_noise,
                  image_matched_filter_norm, central_halpha_line_norm)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
DATASET = S / "datasets/test_100k_simv3_cosi_xu3_tf"
OUT = Path("/jet/home/xwang30/kl-nn/repo_reports/figures_src")
GAL, SEED = 6447, 4
config.set_model_config(load_model_config(MODEL, configs_root=str(S / "configs")))
rec = pxt.TorchDataset(str(DATASET))[GAL]
obs = config.observation
img0 = torch.as_tensor(rec["img"], dtype=torch.float32).unsqueeze(0)
spc0 = torch.as_tensor(rec["spec"], dtype=torch.float32).unsqueeze(0)
isnr = torch.as_tensor(rec["image_snr"], dtype=torch.float32).reshape(1)
ssnr = torch.as_tensor(rec["central_halpha_snr"], dtype=torch.float32).reshape(1)
img = apply_image_noise_for_snr(img0, isnr, clean_norm=image_matched_filter_norm(img0),
                                randgen=torch.Generator().manual_seed(SEED))
spc = apply_central_halpha_snr_noise(spc0, ssnr,
        clean_central_line_norm=central_halpha_line_norm(spc0, center_fiber_index=obs["center_fiber_index"]),
        center_fiber_index=obs["center_fiber_index"], center_exposure_s=obs["center_exposure_s"],
        offset_exposure_s=obs["offset_exposure_s"], spectral_units=obs["spectral_units"],
        randgen=torch.Generator().manual_seed(SEED + 101), device="cpu")
img = img[0].squeeze().numpy(); spc = spc[0].squeeze().numpy()
fib = np.asarray(rec["fib_pos"], dtype=float).reshape(-1, 2)
print("image", img.shape, "spec", spc.shape, "fib", fib.tolist(), "snr", float(isnr), float(ssnr))
np.savez(OUT / "thumb_data_gal6447.npz", img=img, spec=spc, fib=fib)

# image thumbnail with fiber footprints (positions in arcsec relative to the stamp centre)
pix = 0.2637; n = img.shape[-1]
fig = plt.figure(figsize=(1.0, 1.0)); ax = fig.add_axes([0, 0, 1, 1])
ext = [-n / 2 * pix, n / 2 * pix, -n / 2 * pix, n / 2 * pix]
ax.imshow(img, origin="lower", cmap="gray", extent=ext, interpolation="nearest")
for x, y in fib:
    ax.add_patch(plt.Circle((x, y), 0.75, fill=False, color="#d1495b", lw=0.6))
ax.set_axis_off()
fig.savefig(OUT / "thumb_image.pdf"); fig.savefig(OUT / "thumb_image.png", dpi=300)

# spectra thumbnail: five stacked rows
fig, axes = plt.subplots(spc.shape[0], 1, figsize=(1.0, 1.0), sharex=True)
fig.subplots_adjust(0.02, 0.02, 0.98, 0.98, hspace=0)
for k, ax in enumerate(axes):
    ax.plot(spc[spc.shape[0] - 1 - k][:61], color="#1b6ca8", lw=0.5)  # drop the 3 zero-padded pixels
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3); s.set_color("0.5")
fig.savefig(OUT / "thumb_spectra.pdf"); fig.savefig(OUT / "thumb_spectra.png", dpi=300)
