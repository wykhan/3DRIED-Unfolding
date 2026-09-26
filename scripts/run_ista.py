#!/usr/bin/env python3
"""Run the M4 noise-free complex sparse Kernel-ISTA validation."""

import argparse
import json
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
from three_dried.imaging.holography import HolographyGeometry, HolographyOperator
from three_dried.simulation.synthetic import sparse_complex_scene
from three_dried.solvers.kernel_ista import estimate_lipschitz, kernel_ista


def nmse(estimate, truth):
    return float((torch.sum(torch.abs(estimate - truth) ** 2) / torch.sum(torch.abs(truth) ** 2)).cpu())


def normalized_db(value):
    magnitude = torch.abs(value).detach().cpu().numpy().astype(np.float64)
    return np.maximum(20 * np.log10(magnitude / (magnitude.max() + 1e-15) + 1e-12), -60)


def environment_text():
    return "\n".join([
        "Python: " + platform.python_version(), "Platform: " + platform.platform(),
        "PyTorch: " + torch.__version__, "CUDA runtime: " + str(torch.version.cuda),
        "CUDA available: " + str(torch.cuda.is_available()),
        "GPUs: " + str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/kernel_ista_baseline.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(v) for v in [sys.executable] + sys.argv)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output / "config_used.yaml")

    requested_device = config["device"]
    device = requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = {"complex64": torch.complex64, "complex128": torch.complex128}[config["dtype"]]
    geometry = HolographyGeometry(**{
        **config["geometry"],
        "image_shape": tuple(config["geometry"]["image_shape"]),
        "aperture_shape": tuple(config["geometry"]["aperture_shape"]),
    })
    operator = HolographyOperator(geometry)
    truth_cpu, scene_metadata = sparse_complex_scene(
        geometry.image_shape, int(config["synthetic"]["num_scatterers"]), int(config["seed"]),
        margin=int(config["synthetic"]["margin"]), dtype=dtype,
    )
    truth = truth_cpu.to(device)
    measurement = operator.forward(truth)
    if config["synthetic"]["noise_snr_db"] is not None:
        raise ValueError("M4 baseline must validate the configured noise-free path first")

    started = time.perf_counter()
    lipschitz_power, power_history = estimate_lipschitz(
        operator, int(config["power_iteration"]["iterations"]),
        int(config["power_iteration"]["seed"]), dtype=dtype, device=device,
    )
    kernel_reference = torch.zeros(geometry.image_shape, dtype=dtype, device=device)
    # A = C U^H diag(conj(H)) U, where U is unitary up to cancelling FFT
    # scale factors and ||C||=1. Therefore ||A||² <= max|H|².
    lipschitz_upper = float(torch.max(torch.abs(operator.spectral_kernel(kernel_reference)) ** 2).cpu())
    if lipschitz_power > lipschitz_upper * (1.0 + 1e-5):
        raise FloatingPointError("power estimate exceeds analytical operator-norm bound")
    step = float(config["ista"]["step_scale"]) / lipschitz_upper
    backprojection = operator.adjoint(measurement)
    initial = step * backprojection
    regularization = float(config["ista"]["lambda_fraction_of_max_adjoint"]) * float(torch.abs(backprojection).max().cpu())
    result = kernel_ista(
        operator, measurement, regularization, step,
        int(config["ista"]["iterations"]), initial,
    )
    runtime = time.perf_counter() - started

    # Independent stability run: perturb initialization without using truth.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 99)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    perturbation = torch.complex(
        torch.randn(geometry.image_shape, dtype=real_dtype, device=device, generator=generator),
        torch.randn(geometry.image_shape, dtype=real_dtype, device=device, generator=generator),
    )
    perturbation = perturbation / torch.linalg.vector_norm(perturbation)
    perturbation = perturbation * float(config["ista"]["perturbation_fraction"]) * torch.linalg.vector_norm(initial)
    perturbed = kernel_ista(
        operator, measurement, regularization, step,
        int(config["ista"]["iterations"]), initial + perturbation,
    )

    initial_nmse = nmse(initial, truth)
    final_nmse = nmse(result.image, truth)
    perturbed_nmse = nmse(perturbed.image, truth)
    histories = result.history
    objective = [item["objective"] for item in histories]
    fidelity = [item["data_fidelity"] for item in histories]
    monotone_tolerance = max(objective[0], 1.0) * 1e-6
    monotone = all(b <= a + monotone_tolerance for a, b in zip(objective, objective[1:]))

    arrays = {
        "ground_truth.npy": truth.detach().cpu().numpy(),
        "measurement.npy": measurement.detach().cpu().numpy(),
        "adjoint_initialization.npy": initial.detach().cpu().numpy(),
        "reconstruction.npy": result.image.detach().cpu().numpy(),
    }
    for name, value in arrays.items():
        np.save(output / name, value)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, value, title in zip(
        axes, [truth, initial, result.image],
        ["ground truth", "scaled adjoint initialization", "Kernel-ISTA"],
    ):
        shown = axis.imshow(normalized_db(value), origin="lower", cmap="magma", vmin=-50, vmax=0)
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("z index")
    fig.colorbar(shown, ax=axes, label="normalized magnitude (dB)", shrink=0.8)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    iterations = [item["iteration"] for item in histories]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].semilogy(iterations, objective)
    axes[0].set(title="ISTA objective", xlabel="iteration", ylabel="objective")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy(iterations, fidelity)
    axes[1].set(title="data fidelity", xlabel="iteration", ylabel="0.5 ||Ax-y||²")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)

    metrics = {
        "status": "PASS" if final_nmse < 0.5 * initial_nmse and monotone and torch.isfinite(result.image).all() else "FAILED",
        "device": device,
        "dtype": str(dtype),
        "seed": int(config["seed"]),
        "scene": scene_metadata,
        "noise": "none",
        "lipschitz_power_estimate": lipschitz_power,
        "lipschitz_analytic_upper_bound": lipschitz_upper,
        "power_to_upper_bound_ratio": lipschitz_power / lipschitz_upper,
        "power_iteration_history": power_history,
        "step_size": step,
        "step_scale": float(config["ista"]["step_scale"]),
        "regularization_lambda": regularization,
        "soft_threshold": regularization * step,
        "iterations": int(config["ista"]["iterations"]),
        "initial_nmse": initial_nmse,
        "final_nmse": final_nmse,
        "nmse_improvement_factor": initial_nmse / final_nmse,
        "perturbed_initialization_final_nmse": perturbed_nmse,
        "perturbation_fraction": float(config["ista"]["perturbation_fraction"]),
        "initial_objective": objective[0],
        "final_objective": objective[-1],
        "initial_data_fidelity": fidelity[0],
        "final_data_fidelity": fidelity[-1],
        "objective_monotone_with_tolerance": monotone,
        "runtime_seconds_including_power_iteration": runtime,
        "finite": bool(torch.isfinite(result.image).all().item()),
        "history": histories,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key not in ("history", "power_iteration_history")}, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise SystemExit("M4 acceptance metrics failed")


if __name__ == "__main__":
    main()
