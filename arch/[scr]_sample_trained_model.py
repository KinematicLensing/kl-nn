import argparse
import json
import os
import re
from os.path import basename, join

import numpy as np
import pyxis.torch as pxt
import torch
from torch.utils.data import Subset

import config
from train import load_model, sample_density
from utils import resolve_feature_index


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Load a trained density model, draw posterior samples for a test dataset, "
			"and cache results as NumPy files."
		)
	)
	parser.add_argument("--model-name", required=True, help="Model directory name under --model-root.")
	parser.add_argument("--dataset-name", required=True, help="Test dataset directory name under --data-root.")
	parser.add_argument(
		"--samples-per-galaxy",
		type=int,
		required=True,
		help="Number of posterior samples to draw for each galaxy.",
	)
	parser.add_argument(
		"--checkpoint-suffix",
		default="latest",
		help=(
			"Suffix of checkpoint filename after model name (e.g. 119 for model_name119). "
			"Use 'latest' to auto-pick the highest numeric suffix."
		),
	)
	parser.add_argument(
		"--checkpoint-path",
		default=None,
		help="Optional explicit checkpoint path. Overrides --checkpoint-suffix.",
	)
	parser.add_argument(
		"--model-root",
		default=config.train["model_path"],
		help="Root folder containing model subdirectories.",
	)
	parser.add_argument(
		"--data-root",
		default="/ocean/projects/phy250048p/shared/datasets",
		help="Root folder containing test dataset subdirectories.",
	)
	parser.add_argument(
		"--cache-dir",
		default="/ocean/projects/phy250048p/shared/cache",
		help="Directory where sample and SNR cache files are written.",
	)
	parser.add_argument(
		"--output-stem",
		default=None,
		help=(
			"Optional output name stem. Defaults to '<model-name>_<dataset-name>_ns<samples-per-galaxy>_n<ngals>'."
		),
	)
	parser.add_argument(
		"--n-galaxies",
		type=int,
		default=10000,
		help="Number of test galaxies to process. Use -1 to process the full test dataset.",
	)
	parser.add_argument(
		"--split-start",
		type=int,
		default=0,
		help="Start index (inclusive) of the selected galaxy range after --n-galaxies truncation.",
	)
	parser.add_argument(
		"--split-end",
		type=int,
		default=-1,
		help="End index (exclusive) of the selected galaxy range after --n-galaxies truncation. -1 means end.",
	)
	parser.add_argument(
		"--mode",
		type=int,
		default=config.train.get("mode", 2),
		help="Model mode used by load_model (0/1/2).",
	)
	parser.add_argument(
		"--device",
		choices=["auto", "cpu", "cuda"],
		default="auto",
		help="Execution device. 'auto' chooses CUDA when available.",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting existing cache files.",
	)
	return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device_arg == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("--device cuda requested but CUDA is not available")
	return torch.device(device_arg)


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
	if args.checkpoint_path is not None:
		if not os.path.isfile(args.checkpoint_path):
			raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
		return args.checkpoint_path

	model_dir = join(args.model_root, args.model_name)
	if not os.path.isdir(model_dir):
		raise FileNotFoundError(f"Model directory not found: {model_dir}")

	if args.checkpoint_suffix != "latest":
		ckpt_name = (
			args.checkpoint_suffix
			if args.checkpoint_suffix.startswith(args.model_name)
			else f"{args.model_name}{args.checkpoint_suffix}"
		)
		ckpt_path = join(model_dir, ckpt_name)
		if not os.path.isfile(ckpt_path):
			raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
		return ckpt_path

	pat = re.compile(rf"^{re.escape(args.model_name)}(\d+)$")
	candidates = []
	for fname in os.listdir(model_dir):
		match = pat.match(fname)
		if match is None:
			continue
		candidates.append((int(match.group(1)), join(model_dir, fname)))

	if not candidates:
		raise FileNotFoundError(
			f"No checkpoints matching '{args.model_name}<number>' found in {model_dir}"
		)

	candidates.sort(key=lambda x: x[0])
	return candidates[-1][1]


def build_subset(test_ds, n_galaxies: int, split_start: int, split_end: int):
	total = len(test_ds)
	limit = total if n_galaxies < 0 else min(n_galaxies, total)

	if split_start < 0:
		raise ValueError("--split-start must be >= 0")
	if split_start > limit:
		raise ValueError(f"--split-start ({split_start}) exceeds selected dataset size ({limit})")

	effective_end = limit if split_end < 0 else min(split_end, limit)
	if effective_end < split_start:
		raise ValueError("--split-end must be >= --split-start")

	indices = np.arange(split_start, effective_end, dtype=int)
	if len(indices) == 0:
		raise ValueError(
			"Selected split is empty. Adjust --split-start/--split-end (or job array split settings)."
		)

	return Subset(test_ds, indices.tolist()), limit, split_start, effective_end


def get_vcirc_mu(test_ds, device: torch.device) -> torch.Tensor:
	vcirc_idx = resolve_feature_index(config.train["feature_names"], "vcirc", aliases=("v_circ",))
	vcirc_mu = torch.zeros((len(test_ds),), dtype=torch.float32, device=device)
	for i in range(len(test_ds)):
		vcirc_mu[i] = float(test_ds[i]["fid_pars"][vcirc_idx])
	return vcirc_mu


def ensure_writable(path: str, overwrite: bool) -> None:
	if os.path.exists(path) and not overwrite:
		raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def main() -> None:
	args = parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = resolve_device(args.device)
	data_dir = join(args.data_root, args.dataset_name)
	if not os.path.isdir(data_dir):
		raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

	checkpoint_path = resolve_checkpoint_path(args)
	print(f"Loading checkpoint: {checkpoint_path}")
	model = load_model(mode=args.mode, path=checkpoint_path, strict=True, assign=True, device=str(device))

	print(f"Loading test dataset: {data_dir}")
	test_ds = pxt.TorchDataset(data_dir)
	test_subset, limited_size, split_start, split_end = build_subset(
		test_ds,
		args.n_galaxies,
		args.split_start,
		args.split_end,
	)
	ngals = len(test_subset)

	vcirc_mu = get_vcirc_mu(test_subset, device=device) if args.mode == 2 else None

	print(
		"Sampling posterior with "
		f"{ngals} galaxies and {args.samples_per_galaxy} samples/galaxy on device {device}"
	)
	samples, snr = sample_density(
		model,
		test_subset,
		args.samples_per_galaxy,
		vcirc_mu=vcirc_mu,
		device=str(device),
	)

	os.makedirs(args.cache_dir, exist_ok=True)
	dataset_tag = basename(args.dataset_name.rstrip("/"))
	split_tag = f"s{split_start}e{split_end}"
	default_stem = (
		f"{args.model_name}_{dataset_tag}_ns{args.samples_per_galaxy}_n{ngals}_{split_tag}"
	)
	out_stem = args.output_stem if args.output_stem else default_stem

	sample_path = join(args.cache_dir, f"sample_{out_stem}.npy")
	snr_path = join(args.cache_dir, f"snr_{out_stem}.npy")
	meta_path = join(args.cache_dir, f"meta_{out_stem}.json")

	ensure_writable(sample_path, args.overwrite)
	ensure_writable(snr_path, args.overwrite)
	ensure_writable(meta_path, args.overwrite)

	np.save(sample_path, samples)
	np.save(snr_path, snr)

	meta = {
		"model_name": args.model_name,
		"checkpoint_path": checkpoint_path,
		"dataset_name": args.dataset_name,
		"dataset_dir": data_dir,
		"mode": args.mode,
		"limited_size": limited_size,
		"split_start": split_start,
		"split_end": split_end,
		"n_galaxies": ngals,
		"samples_per_galaxy": args.samples_per_galaxy,
		"device": str(device),
		"sample_path": sample_path,
		"snr_path": snr_path,
		"seed": args.seed,
	}
	with open(meta_path, "w", encoding="utf-8") as fobj:
		json.dump(meta, fobj, indent=2)

	print(f"Saved samples to: {sample_path}")
	print(f"Saved SNRs to: {snr_path}")
	print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
	main()
