from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyxis.torch as pxt
import torch
from torch import nn

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from data import apply_noise
from model_registry import load_model_config
from networks import CCLPretrain
from train import load_model, seed_everything


DEFAULT_OUTPUT_ROOT = Path("/ocean/projects/phy250048p/shared/figures")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen CCL backbone with a deterministic MLP probe."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-suffix", default="latest")
    parser.add_argument("--model-root")
    parser.add_argument("--train-data")
    parser.add_argument("--valid-data")
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--valid-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 256))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-mode",
        choices=("log_uniform", "none"),
        default="log_uniform",
    )
    parser.add_argument("--snr-log-min", type=float, default=0.0)
    parser.add_argument("--snr-log-max", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_checkpoint(args, model_root):
    if args.checkpoint_path:
        path = Path(args.checkpoint_path)
    else:
        model_dir = Path(model_root) / args.model_name
        if args.checkpoint_suffix != "latest":
            path = model_dir / f"{args.model_name}{args.checkpoint_suffix}"
        else:
            pattern = re.compile(rf"^{re.escape(args.model_name)}(\d+)$")
            candidates = []
            for candidate in model_dir.iterdir():
                match = pattern.match(candidate.name)
                if match:
                    candidates.append((int(match.group(1)), candidate))
            if not candidates:
                raise FileNotFoundError(
                    f"No numeric checkpoints found for {args.model_name} in {model_dir}"
                )
            path = max(candidates, key=lambda item: item[0])[1]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def choose_indices(dataset_size, requested, seed):
    if requested <= 0:
        raise ValueError("sample counts must be positive")
    count = min(requested, dataset_size)
    if count == dataset_size:
        return np.arange(dataset_size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=count, replace=False))


@torch.inference_mode()
def extract_features(
    model,
    dataset,
    indices,
    *,
    batch_size,
    device,
    seed,
    noise_mode,
    snr_log_min,
    snr_log_max,
    channels_last,
    split_name,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if snr_log_max <= snr_log_min:
        raise ValueError("snr_log_max must exceed snr_log_min")

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    snr_generator = torch.Generator(device=generator_device).manual_seed(seed + 11)
    img_generator = torch.Generator(device=generator_device).manual_seed(seed + 23)
    spec_generator = torch.Generator(device=generator_device).manual_seed(seed + 37)

    all_features = []
    all_labels = []
    total_batches = math.ceil(len(indices) / batch_size)
    for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
        batch_indices = indices[start : start + batch_size]
        rows = [dataset[int(index)] for index in batch_indices]
        img = torch.stack([torch.as_tensor(row["img"]) for row in rows]).float().to(device)
        spec = torch.stack([torch.as_tensor(row["spec"]) for row in rows]).float().to(device)
        fp = torch.stack([torch.as_tensor(row["fib_pos"]) for row in rows]).float().to(device)
        labels = torch.stack(
            [torch.as_tensor(row["fid_pars"][:8]) for row in rows]
        ).float()

        if noise_mode == "log_uniform":
            log_snr = torch.rand(
                (len(rows),),
                device=device,
                generator=snr_generator,
            )
            log_snr = log_snr * (snr_log_max - snr_log_min) + snr_log_min
            snr = torch.pow(10.0, log_snr)
            img = apply_noise(
                img,
                snr,
                device=device,
                randgen=img_generator,
            )
            spec = apply_noise(
                spec,
                snr,
                device=device,
                randgen=spec_generator,
            )

        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last)
            spec = spec.contiguous(memory_format=torch.channels_last)

        all_features.append(model.extract_features(img, spec, fp).float().cpu())
        all_labels.append(labels)
        if batch_number == 1 or batch_number == total_batches or batch_number % 20 == 0:
            print(f"{split_name}: extracted batch {batch_number}/{total_batches}", flush=True)

    return torch.cat(all_features), torch.cat(all_labels)


def encode_probe_targets(labels, feature_names):
    columns = []
    encoded_names = []
    for index, name in enumerate(feature_names):
        if name == "theta_int":
            columns.extend(
                (
                    torch.sin(math.pi * labels[:, index]),
                    torch.cos(math.pi * labels[:, index]),
                )
            )
            encoded_names.extend(("theta_int_sin", "theta_int_cos"))
        else:
            columns.append(labels[:, index])
            encoded_names.append(name)
    return torch.stack(columns, dim=1), encoded_names


class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(512, 256)):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers = []
        width_in = input_dim
        for width_out in hidden_dims:
            layers.extend((nn.Linear(width_in, width_out), nn.ReLU()))
            width_in = width_out
        layers.append(nn.Linear(width_in, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def fit_mlp_probe(
    train_features,
    train_targets,
    valid_features,
    *,
    hidden_dims,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    seed,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be non-negative and finite")

    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_scale = train_features.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    x_train = ((train_features - feature_mean) / feature_scale).to(device)
    x_valid = ((valid_features - feature_mean) / feature_scale).to(device)
    y_train = train_targets.to(device)

    # Reset immediately before constructing the probe so its initialization and
    # minibatch order do not depend on how many random draws feature extraction used.
    seed_everything(seed, deterministic=True)
    probe = MLPProbe(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=tuple(hidden_dims),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    training_losses = []
    report_every = max(1, epochs // 10)

    for epoch in range(epochs):
        probe.train()
        order = torch.randperm(x_train.shape[0], device=device, generator=generator)
        summed_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            batch_indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(probe(x_train[batch_indices]), y_train[batch_indices])
            loss.backward()
            optimizer.step()
            summed_loss += loss.detach().item() * len(batch_indices)

        mean_loss = summed_loss / x_train.shape[0]
        training_losses.append(mean_loss)
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % report_every == 0:
            print(
                f"probe epoch {epoch + 1:4d}/{epochs}: train_mse={mean_loss:.7f}",
                flush=True,
            )

    probe.eval()
    with torch.inference_mode():
        predictions = probe(x_valid).float().cpu()
    return predictions, training_losses


def coefficient_of_determination(truth, prediction):
    residual = np.sum((truth - prediction) ** 2)
    total = np.sum((truth - np.mean(truth)) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def linear_calibration(truth, prediction):
    design = np.column_stack((truth, np.ones_like(truth)))
    slope, intercept = np.linalg.lstsq(design, prediction, rcond=None)[0]
    return float(slope), float(intercept)


def evaluate_probe(labels, predictions, feature_names, encoded_names):
    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    encoded_index = {name: index for index, name in enumerate(encoded_names)}
    metrics = {}

    for parameter_index, name in enumerate(feature_names):
        truth = labels_np[:, parameter_index]
        if name == "theta_int":
            pred_sin = predictions_np[:, encoded_index["theta_int_sin"]]
            pred_cos = predictions_np[:, encoded_index["theta_int_cos"]]
            prediction = np.arctan2(pred_sin, pred_cos) / np.pi
            residual = (prediction - truth + 1.0) % 2.0 - 1.0
            circular_bias = np.arctan2(
                np.mean(np.sin(np.pi * residual)),
                np.mean(np.cos(np.pi * residual)),
            ) / np.pi
            metrics[name] = {
                "circular_rmse": float(np.sqrt(np.mean(residual ** 2))),
                "circular_mae": float(np.mean(np.abs(residual))),
                "circular_bias": float(circular_bias),
                "mean_cosine_alignment": float(np.mean(np.cos(np.pi * residual))),
                "sin_r2": coefficient_of_determination(
                    np.sin(np.pi * truth), pred_sin
                ),
                "cos_r2": coefficient_of_determination(
                    np.cos(np.pi * truth), pred_cos
                ),
            }
            continue

        prediction = predictions_np[:, encoded_index[name]]
        residual = prediction - truth
        slope, intercept = linear_calibration(truth, prediction)
        metrics[name] = {
            "r2": coefficient_of_determination(truth, prediction),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        }
    return metrics


def print_metrics(metrics):
    print("parameter                 r2       rmse       bias      slope")
    for name, values in metrics.items():
        if name == "theta_int":
            theta_r2 = 0.5 * (values["sin_r2"] + values["cos_r2"])
            print(
                f"{name:20s} {theta_r2:8.4f} "
                f"{values['circular_rmse']:10.5f} "
                f"{values['circular_bias']:10.5f} {'circular':>10s}"
            )
        else:
            print(
                f"{name:20s} {values['r2']:8.4f} {values['rmse']:10.5f} "
                f"{values['bias']:10.5f} {values['calibration_slope']:10.5f}"
            )


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed, deterministic=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    model_config = load_model_config(args.model_name, allow_fallback_current=False)
    config.set_model_config(model_config)
    model_root = args.model_root or model_config.pretrain.model_path
    checkpoint = resolve_checkpoint(args, model_root)

    train_data = args.train_data or model_config.data.data_dir
    valid_data = args.valid_data or model_config.test.data_dir
    print(f"checkpoint={checkpoint}")
    print(f"train_data={train_data}")
    print(f"valid_data={valid_data}")

    model = load_model(
        config.pretrain,
        Model=CCLPretrain,
        path=str(checkpoint),
        strict=True,
        assign=True,
        GPUs=1,
        device=str(device),
        model_name=args.model_name,
        use_compile=False,
    )
    model.eval()

    train_dataset = pxt.TorchDataset(train_data)
    valid_dataset = pxt.TorchDataset(valid_data)
    train_indices = choose_indices(len(train_dataset), args.train_samples, args.seed)
    valid_indices = choose_indices(len(valid_dataset), args.valid_samples, args.seed + 1)

    extraction_kwargs = {
        "batch_size": args.batch_size,
        "device": device,
        "noise_mode": args.noise_mode,
        "snr_log_min": args.snr_log_min,
        "snr_log_max": args.snr_log_max,
        "channels_last": bool(model_config.pretrain.channels_last),
    }
    train_features, train_labels = extract_features(
        model,
        train_dataset,
        train_indices,
        seed=args.seed + 101,
        split_name="train",
        **extraction_kwargs,
    )
    valid_features, valid_labels = extract_features(
        model,
        valid_dataset,
        valid_indices,
        seed=args.seed + 202,
        split_name="valid",
        **extraction_kwargs,
    )

    feature_names = list(model_config.train.feature_names)
    train_targets, encoded_names = encode_probe_targets(
        train_labels[:, : len(feature_names)], feature_names
    )
    predictions, training_losses = fit_mlp_probe(
        train_features,
        train_targets,
        valid_features,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed + 303,
    )
    metrics = evaluate_probe(
        valid_labels[:, : len(feature_names)],
        predictions,
        feature_names,
        encoded_names,
    )
    print_metrics(metrics)

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_ROOT
        / args.model_name
        / f"ccl_mlp_probe_{checkpoint.name}.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "train_data": train_data,
        "valid_data": valid_data,
        "train_samples": len(train_indices),
        "valid_samples": len(valid_indices),
        "noise_mode": args.noise_mode,
        "snr_log_min": args.snr_log_min,
        "snr_log_max": args.snr_log_max,
        "probe": {
            "type": "mlp",
            "hidden_dims": list(args.hidden_dims),
            "activation": "ReLU",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "initial_train_mse": training_losses[0],
            "final_train_mse": training_losses[-1],
        },
        "feature_dimension": int(train_features.shape[1]),
        "metrics": metrics,
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
