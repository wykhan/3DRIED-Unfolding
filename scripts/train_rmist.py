#!/usr/bin/env python3
"""Strict RMIST M6-R synthetic training for Eqs. (27)--(31)."""

import argparse
import csv
import json
import math
import platform
import shlex
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import RMISTNet
from three_dried.solvers.rmist_ista import rmist_ista
from three_dried.training.rmist_training import (
    FixedRangeRMISTOperator,
    RMISTSyntheticDataset,
    averaged_complex_l2_loss,
    batch_metrics,
    make_fixed_mask,
    mask_sha256,
)


def environment_text():
    return "\n".join(
        [
            "Python: " + platform.python_version(),
            "Platform: " + platform.platform(),
            "NumPy: " + np.__version__,
            "PyTorch: " + torch.__version__,
            "CUDA runtime: " + str(torch.version.cuda),
            "CUDA available: " + str(torch.cuda.is_available()),
            "GPUs: " + str(
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            ),
        ]
    ) + "\n"


def validate_strict_config(config):
    geometry = config["geometry"]
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    required = {
        "grid_size": (int(geometry["grid_size"]), 512),
        "range_m": (float(geometry["range_m"]), 0.5),
        "training_pairs": (int(dataset["training_pairs"]), 1000),
        "validation_pairs": (int(dataset["validation_pairs"]), 200),
        "scatterers_per_image": (int(dataset["scatterers_per_image"]), 100),
        "sampling_rate": (float(dataset["sampling_rate"]), 0.5),
        "snr_db": (float(dataset["snr_db"]), 20.0),
        "phases": (int(model["phases"]), 9),
        "tau_init": (float(model["tau_init"]), 1e-4),
        "theta_init": (float(model["theta_init"]), 1e-3),
        "learning_rate": (float(training["learning_rate"]), 1e-4),
        "epochs": (int(training["epochs"]), 100),
        "batch_size": (int(training["batch_size"]), 8),
    }
    failures = [name for name, (actual, expected) in required.items() if actual != expected]
    if failures:
        raise ValueError("strict M6 configuration mismatch: " + ", ".join(failures))
    if training["optimizer"] != "Adam" or not bool(dataset["fixed_mask"]):
        raise ValueError("strict M6 requires Adam and one fixed mask")


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def evaluate_model(model, dataset, operator, mask, batch_size, device):
    totals = {"loss": 0.0, "nrmse": 0.0, "psnr_db": 0.0, "tbr_db": 0.0, "snr_db": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for index_batch in batches(list(range(dataset.pair_count)), batch_size):
            truth, echo, achieved_snr = dataset.make_batch(
                index_batch, operator, mask, device
            )
            reconstruction = model(echo, dataset.range_m)
            loss = averaged_complex_l2_loss(reconstruction, truth)
            metrics = batch_metrics(reconstruction, truth)
            size = len(index_batch)
            totals["loss"] += float(loss) * size
            for name in ("nrmse", "psnr_db", "tbr_db"):
                totals[name] += float(metrics[name].sum())
            totals["snr_db"] += float(achieved_snr.sum())
            count += size
    return {name: value / count for name, value in totals.items()}


def evaluate_ista(dataset, operator, mask, batch_size, device):
    totals = {"loss": 0.0, "nrmse": 0.0, "psnr_db": 0.0, "tbr_db": 0.0}
    count = 0
    with torch.no_grad():
        for index_batch in batches(list(range(dataset.pair_count)), batch_size):
            truth, echo, _ = dataset.make_batch(index_batch, operator, mask, device)
            reconstruction, _ = rmist_ista(
                echo,
                operator,
                dataset.range_m,
                tau=1e-4,
                theta=1e-3,
                iterations=9,
            )
            loss = averaged_complex_l2_loss(reconstruction, truth)
            metrics = batch_metrics(reconstruction, truth)
            size = len(index_batch)
            totals["loss"] += float(loss) * size
            for name in ("nrmse", "psnr_db", "tbr_db"):
                totals[name] += float(metrics[name].sum())
            count += size
    return {name: value / count for name, value in totals.items()}


def preflight(config, base_operator, cached_operator, mask, train_dataset, device):
    training = config["training"]
    model_cfg = config["model"]
    truth_a, echo_a, snr_a = train_dataset.make_batch(
        [0], cached_operator, mask, device
    )
    truth_b, echo_b, snr_b = train_dataset.make_batch(
        [0], cached_operator, mask, device
    )
    deterministic = bool(
        torch.equal(truth_a, truth_b)
        and torch.equal(echo_a, echo_b)
        and torch.equal(snr_a, snr_b)
    )
    mask_binary = bool(torch.all((mask == 0) | (mask == 1)).item())

    model = RMISTNet(
        cached_operator,
        phases=int(model_cfg["phases"]),
        tau_init=float(model_cfg["tau_init"]),
        theta_init=float(model_cfg["theta_init"]),
    ).to(device)
    truth, echo, achieved = train_dataset.make_batch(
        list(range(int(training["batch_size"]))), cached_operator, mask, device
    )
    loss = averaged_complex_l2_loss(model(echo, train_dataset.range_m), truth)
    loss.backward()
    tau_gradient = model.tau.grad.detach()
    theta_gradient = model.theta.grad.detach()
    backward_finite = bool(
        torch.isfinite(loss).item()
        and torch.isfinite(tau_gradient).all().item()
        and torch.isfinite(theta_gradient).all().item()
    )

    tiny_model = RMISTNet(
        cached_operator,
        phases=int(model_cfg["phases"]),
        tau_init=float(model_cfg["tau_init"]),
        theta_init=float(model_cfg["theta_init"]),
    ).to(device)
    tiny_optimizer = torch.optim.Adam(
        tiny_model.parameters(), lr=float(training["learning_rate"])
    )
    tiny_truth, tiny_echo, _ = train_dataset.make_batch(
        list(range(int(training["batch_size"]))), cached_operator, mask, device
    )
    tiny_losses = []
    for _ in range(int(training["tiny_overfit_steps"])):
        tiny_optimizer.zero_grad(set_to_none=True)
        tiny_loss = averaged_complex_l2_loss(
            tiny_model(tiny_echo, train_dataset.range_m), tiny_truth
        )
        tiny_loss.backward()
        tiny_optimizer.step()
        tiny_losses.append(float(tiny_loss.detach()))

    # The cache must contain the exact accepted M3 kernel values.
    cache_exact = bool(
        torch.equal(
            cached_operator.phi,
            base_operator.phi(0.5, torch.complex64, device=device),
        )
        and torch.equal(
            cached_operator.phi_dagger,
            base_operator.phi_dagger(0.5, torch.complex64, device=device),
        )
    )
    result = {
        "dataset_deterministic": deterministic,
        "mask_binary": mask_binary,
        "mask_realized_sampling_rate": float(mask.mean()),
        "requested_snr_db": train_dataset.snr_db,
        "achieved_snr_db": float(achieved.mean()),
        "one_batch_loss": float(loss.detach()),
        "all_tau_gradients_finite": bool(torch.isfinite(tau_gradient).all().item()),
        "all_theta_gradients_finite": bool(torch.isfinite(theta_gradient).all().item()),
        "all_18_gradients_finite": backward_finite,
        "tiny_overfit_initial_loss": tiny_losses[0],
        "tiny_overfit_final_loss": tiny_losses[-1],
        "tiny_overfit_reduced": tiny_losses[-1] < tiny_losses[0],
        "tiny_overfit_steps": len(tiny_losses),
        "cached_kernels_exact": cache_exact,
    }
    required = (
        result["dataset_deterministic"],
        result["mask_binary"],
        abs(result["achieved_snr_db"] - train_dataset.snr_db) < 1e-3,
        result["all_18_gradients_finite"],
        result["tiny_overfit_reduced"],
        result["cached_kernels_exact"],
    )
    if not all(required):
        raise RuntimeError("M6 preflight failed: " + json.dumps(result, sort_keys=True))
    return result


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_curves(review_dir, history, parameters):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Eq. (27) averaged L2 loss", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(review_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for phase in range(9):
        axes[0].plot(epochs, [row["tau_{}".format(phase + 1)] for row in parameters])
        axes[1].plot(epochs, [row["theta_{}".format(phase + 1)] for row in parameters])
    axes[0].set(xlabel="Epoch", ylabel="tau_t", title="Phase step sizes")
    axes[1].set(xlabel="Epoch", ylabel="theta_t", title="Phase thresholds")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(review_dir / "parameter_curves.png", dpi=160)
    plt.close(fig)


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_train.yaml"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_strict_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("strict 512-square M6 training requires an available CUDA device")

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    output_dir = Path(config["output_dir"])
    review_dir = Path(config["review_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output_dir / "config_used.yaml")
    shutil.copyfile(args.config, review_dir / "config_used.yaml")

    geometry = config["geometry"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    grid_size = int(geometry["grid_size"])
    range_m = float(geometry["range_m"])
    base_operator = RMISTKernel(
        float(geometry["center_frequency_hz"]),
        float(geometry["dx_m"]),
        float(geometry["dz_m"]),
        n_fft_space=grid_size,
        ky_epsilon=float(geometry["ky_epsilon"]),
    )
    operator = FixedRangeRMISTOperator(
        base_operator, range_m, torch.complex64, device
    )
    mask_cpu = make_fixed_mask(
        (grid_size, grid_size),
        float(dataset_cfg["sampling_rate"]),
        int(dataset_cfg["mask_seed"]),
    )
    mask = mask_cpu.to(device)
    common_dataset = {
        "grid_size": grid_size,
        "scatterers_per_image": int(dataset_cfg["scatterers_per_image"]),
        "amplitude_min": float(dataset_cfg["amplitude_min"]),
        "amplitude_max": float(dataset_cfg["amplitude_max"]),
        "snr_db": float(dataset_cfg["snr_db"]),
        "range_m": range_m,
    }
    train_dataset = RMISTSyntheticDataset(
        pair_count=int(dataset_cfg["training_pairs"]),
        seed=int(dataset_cfg["training_seed"]),
        **common_dataset
    )
    validation_dataset = RMISTSyntheticDataset(
        pair_count=int(dataset_cfg["validation_pairs"]),
        seed=int(dataset_cfg["validation_seed"]),
        **common_dataset
    )

    dataset_spec = {
        "classification": "PAPER UNSPECIFIED IMPLEMENTATION CHOICE where noted",
        "equations": "RMIST Eq. (28)-(31)",
        "grid_shape": [grid_size, grid_size],
        "range_m": range_m,
        "training_pairs": train_dataset.pair_count,
        "validation_pairs": validation_dataset.pair_count,
        "training_seed": train_dataset.seed,
        "validation_seed": validation_dataset.seed,
        "seeds_disjoint": train_dataset.seed != validation_dataset.seed,
        "scatterers_per_image": train_dataset.scatterers_per_image,
        "oracle": "100 unique discrete impulses; no blur PSF",
        "coordinate_distribution": "uniform without replacement",
        "amplitude_distribution": "Uniform(0.5, 1.0)",
        "phase_distribution": "Uniform(0, 2*pi)",
        "echo": "fixed_mask * accepted strict RM(alpha_gt) + circular complex AWGN",
        "requested_snr_db": train_dataset.snr_db,
        "hammer_rma_used_as_label": False,
        "paper_unspecified_choices": [
            "validation_pairs=200 with disjoint seed",
            "100 scatterers in every training oracle",
            "amplitude Uniform(0.5,1.0)",
            "one fixed SR=0.5 mask/model",
            "random seeds",
            "deterministic epoch shuffling",
        ],
    }
    mask_summary = {
        "shape": [grid_size, grid_size],
        "seed": int(dataset_cfg["mask_seed"]),
        "target_sampling_rate": float(dataset_cfg["sampling_rate"]),
        "realized_sampling_rate": float(mask_cpu.mean()),
        "binary": bool(torch.all((mask_cpu == 0) | (mask_cpu == 1)).item()),
        "fixed_for_training_and_validation": True,
        "trainable": False,
        "sha256_uint8": mask_sha256(mask_cpu),
        "classification": "PAPER UNSPECIFIED IMPLEMENTATION CHOICE",
    }
    (review_dir / "dataset_spec.json").write_text(
        json.dumps(dataset_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (review_dir / "mask_summary.json").write_text(
        json.dumps(mask_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preflight_result = preflight(
        config, base_operator, operator, mask, train_dataset, device
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PREFLIGHT " + json.dumps(preflight_result, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    model = RMISTNet(
        operator,
        phases=int(model_cfg["phases"]),
        tau_init=float(model_cfg["tau_init"]),
        theta_init=float(model_cfg["theta_init"]),
    ).to(device)
    initial_tau = model.tau.detach().cpu().tolist()
    initial_theta = model.theta.detach().cpu().tolist()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg["learning_rate"])
    )
    batch_size = int(training_cfg["batch_size"])

    untrained_metrics = evaluate_model(
        model, validation_dataset, operator, mask, batch_size, device
    )
    history = []
    parameter_history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    total_peak_memory = 0.0
    epochs = int(training_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        shuffle_generator = torch.Generator().manual_seed(
            int(training_cfg["shuffle_seed"]) + epoch
        )
        order = torch.randperm(train_dataset.pair_count, generator=shuffle_generator).tolist()
        model.train()
        loss_sum = 0.0
        tau_grad_sum = 0.0
        theta_grad_sum = 0.0
        batch_count = 0
        for index_batch in batches(order, batch_size):
            truth, echo, _ = train_dataset.make_batch(
                index_batch, operator, mask, device
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(echo, range_m)
            loss = averaged_complex_l2_loss(reconstruction, truth)
            loss.backward()
            if not (
                torch.isfinite(loss).item()
                and torch.isfinite(model.tau.grad).all().item()
                and torch.isfinite(model.theta.grad).all().item()
            ):
                raise RuntimeError("non-finite loss or gradient during strict training")
            tau_grad_sum += float(torch.linalg.vector_norm(model.tau.grad).detach())
            theta_grad_sum += float(torch.linalg.vector_norm(model.theta.grad).detach())
            optimizer.step()
            if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise RuntimeError("non-finite strict RMIST parameter")
            loss_sum += float(loss.detach()) * len(index_batch)
            batch_count += 1
        train_loss = loss_sum / train_dataset.pair_count
        validation_metrics = evaluate_model(
            model, validation_dataset, operator, mask, batch_size, device
        )
        epoch_seconds = time.perf_counter() - epoch_started
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        total_peak_memory = max(total_peak_memory, peak_memory_mb)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "tau_grad_norm": tau_grad_sum / batch_count,
            "theta_grad_norm": theta_grad_sum / batch_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": epoch_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
        }
        history.append(row)
        parameter_row = {"epoch": epoch}
        for phase, value in enumerate(model.tau.detach().cpu().tolist(), start=1):
            parameter_row["tau_{}".format(phase)] = value
        for phase, value in enumerate(model.theta.detach().cpu().tolist(), start=1):
            parameter_row["theta_{}".format(phase)] = value
        parameter_history.append(parameter_row)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "validation_loss": best_loss,
                    "config": config,
                },
                checkpoint_dir / "best_validation.pt",
            )
        print(
            "EPOCH {}/{} train={:.8g} val={:.8g} tau0={:.8g} theta0={:.8g} sec={:.2f}".format(
                epoch,
                epochs,
                train_loss,
                validation_metrics["loss"],
                float(model.tau[0].detach()),
                float(model.theta[0].detach()),
                epoch_seconds,
            ),
            flush=True,
        )

    final_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    torch.save(
        {"epoch": epochs, "model_state_dict": final_state, "config": config},
        checkpoint_dir / "epoch_100.pt",
    )
    final_metrics = evaluate_model(
        model, validation_dataset, operator, mask, batch_size, device
    )
    best_model = RMISTNet(
        operator,
        phases=int(model_cfg["phases"]),
        tau_init=float(model_cfg["tau_init"]),
        theta_init=float(model_cfg["theta_init"]),
    ).to(device)
    best_model.load_state_dict(best_state)
    trained_metrics = evaluate_model(
        best_model, validation_dataset, operator, mask, batch_size, device
    )
    ini_ista_metrics = evaluate_ista(
        validation_dataset, operator, mask, batch_size, device
    )
    total_runtime = time.perf_counter() - run_started

    training_fields = [
        "epoch", "train_loss", "val_loss", "tau_grad_norm", "theta_grad_norm",
        "learning_rate", "epoch_time_seconds", "peak_gpu_memory_mb",
    ]
    parameter_fields = ["epoch"] + ["tau_{}".format(i) for i in range(1, 10)] + [
        "theta_{}".format(i) for i in range(1, 10)
    ]
    save_csv(review_dir / "training_history.csv", history, training_fields)
    save_csv(review_dir / "parameter_history.csv", parameter_history, parameter_fields)
    save_curves(review_dir, history, parameter_history)

    summary = {
        "status": "PASS"
        if trained_metrics["nrmse"] < untrained_metrics["nrmse"]
        else "FAILED",
        "method": "strict RMIST synthetic training Eq. (27)-(31)",
        "training_pairs": train_dataset.pair_count,
        "validation_pairs": validation_dataset.pair_count,
        "batch_size": batch_size,
        "epochs_completed": epochs,
        "optimizer": "Adam",
        "learning_rate": float(training_cfg["learning_rate"]),
        "scheduler": None,
        "weight_decay": 0.0,
        "gradient_clipping": False,
        "early_stopping": False,
        "loss": "mean over batch of per-sample complex L2 reconstruction norm (displayed Eq. 27)",
        "mask": mask_summary,
        "noise": {
            "requested_snr_db": train_dataset.snr_db,
            "achieved_validation_mean_snr_db": untrained_metrics["snr_db"],
        },
        "preflight": preflight_result,
        "initial_parameters": {"tau": initial_tau, "theta": initial_theta},
        "final_parameters": {
            "tau": model.tau.detach().cpu().tolist(),
            "theta": model.theta.detach().cpu().tolist(),
        },
        "best_parameters": {
            "tau": best_model.tau.detach().cpu().tolist(),
            "theta": best_model.theta.detach().cpu().tolist(),
        },
        "best_epoch": best_epoch,
        "train_loss_first_epoch": history[0]["train_loss"],
        "train_loss_final_epoch": history[-1]["train_loss"],
        "validation_loss_initial": untrained_metrics["loss"],
        "validation_loss_best": best_loss,
        "validation_loss_final": final_metrics["loss"],
        "untrained": untrained_metrics,
        "trained_best_validation": trained_metrics,
        "ini_ista_9_step": ini_ista_metrics,
        "improves_nrmse_over_untrained": trained_metrics["nrmse"] < untrained_metrics["nrmse"],
        "total_runtime_seconds": total_runtime,
        "peak_gpu_memory_mb": total_peak_memory,
        "device": torch.cuda.get_device_name(device),
        "kernel_cache": "IMPLEMENTATION PERFORMANCE OPTIMIZATION — algorithm unchanged",
        "best_checkpoint": str(checkpoint_dir / "best_validation.pt"),
        "epoch_100_checkpoint": str(checkpoint_dir / "epoch_100.pt"),
        "checkpoints_committed": False,
        "hammer_inference_performed": False,
        "command": command,
        "environment": environment_text().splitlines(),
    }
    (review_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "PASS":
        raise RuntimeError("trained model did not improve held-out NRMSE")


if __name__ == "__main__":
    main()
