#!/usr/bin/env python3
"""Paper test T8 (Group C): NPE posterior samples on the fixed-shear benchmark.

Draws identity+R90 posterior samples (the canonical ensemble used for the Xu3
cache) with the fibrepair NPE for every (cell, galaxy, realization) of
``likelihood_stack_04_benchmark``, using the pre-noised observations stored in
``benchmark_noisy.npz``. Stores (g1, g2) samples, vcirc samples and normalized
TF log weights so that the stacking analysis (``c_stack_analysis.py``) can use
either the training nuisance prior (unweighted) or the TF prior (weighted).

Writes only to a NEW cache directory (refuses to overwrite existing cells).
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ARCH_DIR.parent
for p in (ARCH_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from data import rotate_90_datavector, rotate_90_parameters  # noqa: E402
from model_registry import load_model_config  # noqa: E402
from networks import KLNPE  # noqa: E402
from train import load_model, seed_everything  # noqa: E402
from utils import denormalize  # noqa: E402
from tf_prior import TFPrior, posterior_importance_weights  # noqa: E402
from diagnostics.likelihood_stack_benchmark import _load_cell  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
DATA = S / "datasets" / "likelihood_stack_04_benchmark"
OUT = S / "cache" / "paper_tests_C_stacking" / MODEL / "likelihood_stack_04_benchmark_r90_8192"
N_CELLS, N_GAL, N_REAL = 25, 32, 16


def parse_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=None,
                   help="default: cache/paper_tests_C_stacking/<model>/<data name>_r90_8192")
    p.add_argument("--nsamples", type=int, default=8192, help="total, half identity half R90")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--cells", type=str, default="all")
    p.add_argument("--realizations", type=int, default=N_REAL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260929)
    return p.parse_args(argv)


def _seed(seed, cell, real, rot):
    s = np.random.SeedSequence([int(seed), int(cell), int(real), int(rot), 77])
    return int(s.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


@torch.inference_mode()
def _sample(model, image, spec, pos, ctx, n, chunk, seed, channels_last):
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
        spec = spec.contiguous(memory_format=torch.channels_last)
    context = model._flow_context(model._raw_features(image, spec, pos, ctx))
    torch.manual_seed(seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(seed)
    out = []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        out.append(model.flow.sample(k, context=context))
        done += k
    return torch.cat(out, dim=1)  # (B, n, 9) normalized


def main(argv=None):
    args = parse_args(argv)
    MODEL, DATA = args.model, args.data
    if args.out is None:
        args.out = S / "cache" / "paper_tests_C_stacking" / MODEL / f"{DATA.name}_r90_8192"
    if args.nsamples % 2:
        raise ValueError("nsamples must be even")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = load_model_config(MODEL, configs_root=str(S / "configs"))
    config.set_model_config(cfg)
    names = tuple(config.TARGET_NAMES)
    par_ranges = config.par_ranges
    config.require_matching_dataset_par_ranges(DATA, par_ranges)
    ig1, ig2, iv = names.index("g1"), names.index("g2"), names.index("vcirc")
    tf = TFPrior(slope=-7.22, intercept=36.0, scatter_dex=0.1,
                 vcirc_min=float(par_ranges["vcirc"][0]), vcirc_max=float(par_ranges["vcirc"][1]))
    seed_everything(args.seed, deterministic=True)
    ckpt = S / "models" / MODEL / f"{MODEL}best"
    model = load_model(KLNPE, path=str(ckpt), model_name=MODEL, device=str(device),
                       strict=True, networks_root=str(S / "networks"))
    model.eval()
    channels_last = bool(cfg.train.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    dataset = pxt.TorchDataset(str(DATA))
    noisy = np.load(DATA / "benchmark_noisy.npz")
    images_all, spectra_all = noisy["images"], noisy["spectra"]
    cells = range(N_CELLS) if args.cells == "all" else [int(c) for c in args.cells.split(",")]
    half = args.nsamples // 2
    for cell_id in cells:
        path = args.out / f"cell{cell_id:02d}.npz"
        if path.exists():
            print(f"skip existing {path}", flush=True)
            continue
        t0 = time.time()
        cell = _load_cell(dataset, cell_id, device, names, par_ranges)
        ctx = {"rmag_true": cell["rmag"], "image_snr": cell["image_snr"],
               "central_halpha_snr": cell["spec_snr"]}
        R = args.realizations
        g = np.empty((N_GAL, R, args.nsamples, 2), np.float32)
        vc = np.empty((N_GAL, R, args.nsamples), np.float32)
        logw = np.empty_like(vc)
        ess = np.empty((N_GAL, R), np.float64)
        mean_prop = np.empty((N_GAL, R, 9), np.float64)
        mean_tf = np.empty_like(mean_prop)
        start = cell_id * N_GAL
        for r in range(R):
            image = torch.as_tensor(images_all[start:start + N_GAL, r], device=device, dtype=torch.float32)
            spec = torch.as_tensor(spectra_all[start:start + N_GAL, r], device=device, dtype=torch.float32)
            pos = cell["fib_pos"]
            s_id = _sample(model, image, spec, pos, ctx, half, args.chunk,
                           _seed(args.seed, cell_id, r, 0), channels_last)
            img_r, spec_r, _, pos_r = rotate_90_datavector(image, spec, fiber_positions=pos)
            s_r = _sample(model, img_r, spec_r, pos_r, ctx, half, args.chunk,
                          _seed(args.seed, cell_id, r, 1), channels_last)
            s_r = rotate_90_parameters(s_r, inverse=True, feature_names=names)
            bank = torch.cat((s_id, s_r), dim=1).float().cpu().numpy()
            phys = np.asarray(denormalize(bank, par_ranges, feature_names=names,
                                          target_transforms=config.TARGET_TRANSFORMS), np.float64)
            if not np.all(np.isfinite(phys)):
                raise RuntimeError("non-finite samples")
            imp = posterior_importance_weights(phys[..., iv], cell["rmag"].cpu().numpy(), tf)
            g[:, r, :, 0] = phys[..., ig1]
            g[:, r, :, 1] = phys[..., ig2]
            vc[:, r] = phys[..., iv]
            logw[:, r] = imp.log_weight
            ess[:, r] = imp.effective_sample_size
            mean_prop[:, r] = phys.mean(axis=1)
            mean_tf[:, r] = np.einsum("bs,bsk->bk", imp.weight, phys)
        np.savez(path, g=g, vcirc=vc, tf_log_weight=logw, tf_ess=ess,
                 mean_proposal=mean_prop, mean_tf=mean_tf, truth_full=cell["truth"],
                 rmag_true=cell["rmag"].cpu().numpy(),
                 image_snr=cell["image_snr"].cpu().numpy(),
                 central_halpha_snr=cell["spec_snr"].cpu().numpy())
        print(f"cell {cell_id} done in {time.time() - t0:.1f}s; median TF ESS frac "
              f"{np.median(ess) / args.nsamples:.3f}; mean g {mean_prop[..., :2].mean(axis=(0, 1))}",
              flush=True)
    meta = {"schema": "paper-tests-C-stacking-samples-v1", "model": MODEL,
            "checkpoint": str(ckpt), "dataset": str(DATA),
            "noise": str(DATA / "benchmark_noisy.npz"), "nsamples": args.nsamples,
            "ensemble": "identity+R90 equal mixture (half each)", "seed": args.seed,
            "feature_names": list(names), "tf_prior": tf.to_dict(),
            "arrays": "g (gal, real, S, 2) physical; vcirc; tf_log_weight normalized per obs"}
    mp = args.out / "meta.json"
    if not mp.exists() and args.cells == "all" and args.realizations == N_REAL:
        mp.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
