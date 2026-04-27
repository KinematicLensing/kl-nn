#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxis as px
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "arch"
if str(ARCH) not in sys.path:
    sys.path.insert(0, str(ARCH))

import train  # noqa: E402

SAMPLE_CSV = Path("/ocean/projects/phy250048p/shared/samples/samples_test_1m_low_hlr.csv")
DB_DIR = Path("/ocean/projects/phy250048p/shared/datasets/test_1m_low_hlr/")
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = OUT_DIR / "noise_hlr_mosaic.png"
SNRS = [10, 50, 200, 1000]


def _select_targets(samples: pd.DataFrame) -> pd.DataFrame:
    work = samples.copy()
    work["shear"] = np.hypot(work["g1"], work["g2"])

    # "Very low" selection via lower decile in sini and shear.
    sini_cut = work["sini"].quantile(0.10)
    shear_cut = work["shear"].quantile(0.10)
    candidates = work[(work["sini"] <= sini_cut) & (work["shear"] <= shear_cut)].copy()

    if candidates.empty:
        raise RuntimeError("No candidates found with low sini and low shear.")

    low_idx = (candidates["hlr"] - 0.1).abs().idxmin()
    low_row = candidates.loc[[low_idx]]

    remaining = candidates.drop(index=low_idx)
    if remaining.empty:
        raise RuntimeError("Only one candidate met the low-sini/low-shear criteria.")

    high_idx = (remaining["hlr"] - 1.0).abs().idxmin()
    high_row = remaining.loc[[high_idx]]

    picked = pd.concat([low_row, high_row])
    return picked


def _fetch_record_by_id(db, sample_id: int):
    # Fast path: key index matches sample id.
    try:
        rec = db[sample_id]
        rid = int(np.array(rec["id"]).item())
        if rid == sample_id:
            return rec
    except Exception:
        pass

    # Fallback: scan by stored id.
    for i in range(len(db)):
        rec = db[i]
        rid = int(np.array(rec["id"]).item())
        if rid == sample_id:
            return rec

    raise KeyError(f"Sample id {sample_id} not found in LMDB")


def _to_image(record) -> torch.Tensor:
    img = torch.from_numpy(record["img"]).float()
    if img.ndim == 3:
        return img
    if img.ndim == 2:
        return img.unsqueeze(0)
    raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")


def main() -> None:
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Missing sample csv: {SAMPLE_CSV}")
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Missing dataset dir: {DB_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(SAMPLE_CSV, index_col=0)
    selected = _select_targets(samples)

    row_labels = []
    clean_images = []

    with px.Reader(str(DB_DIR)) as db:
        for sample_id, row in selected.iterrows():
            rec = _fetch_record_by_id(db, int(sample_id))
            clean = _to_image(rec)
            clean_images.append(clean)
            row_labels.append(
                f"id={int(sample_id)} | hlr={row['hlr']:.3f} | sini={row['sini']:.4f} | "
                f"|g|={row['shear']:.4f}"
            )

    torch.manual_seed(42)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for r, clean in enumerate(clean_images):
        clean_b = clean.unsqueeze(0)
        clean_np = clean.squeeze(0).cpu().numpy()
        vmax = float(np.percentile(clean_np, 99.5)) if np.any(clean_np > 0) else 1.0
        levels = np.array([0.2, 0.4, 0.6, 0.8]) * max(float(clean_np.max()), 1e-8)

        for c, snr in enumerate(SNRS):
            snr_t = torch.tensor([float(snr)], dtype=torch.float32)
            noisy = train.apply_noise(clean_b, snr_t, device="cpu", use_iterative=True)
            noisy_np = noisy.squeeze(0).squeeze(0).cpu().numpy()

            ax = axes[r, c]
            im = ax.imshow(noisy_np, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
            ax.contour(clean_np, levels=levels, colors="white", linewidths=0.7, alpha=0.8)
            ax.set_title(f"SNR={snr}")
            ax.set_xticks([])
            ax.set_yticks([])

            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9)

        cbar = fig.colorbar(im, ax=axes[r, :], shrink=0.8, pad=0.01)
        cbar.set_label("Flux")

    fig.suptitle(
        "Noised Galaxy Images (rows: low/high HLR) with Noiseless Contours\n"
        "Selection: low sini + low shear from samples_test_1m_low_hlr.csv",
        fontsize=12,
    )

    fig.savefig(OUT_PATH, dpi=180)
    print("Selected rows:")
    print(selected[["g1", "g2", "sini", "hlr", "shear"]])
    print(f"Saved mosaic to: {OUT_PATH}")


if __name__ == "__main__":
    main()
