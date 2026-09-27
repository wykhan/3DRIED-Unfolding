#!/usr/bin/env python3
"""Run strict RMIST Eq. (16)--(19) synthetic M4-R diagnostics."""

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
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.simulation.rmist_synthetic import (
    add_complex_awgn,
    point_scatterer_scene,
    random_mask,
)
from three_dried.solvers.rmist_ista import rmist_ista


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


def normalized_error(estimate, truth):
    return float(torch.linalg.vector_norm(estimate - truth) / torch.linalg.vector_norm(truth))


def normalized_residual(estimate, measurement, operator, range_m):
    residual = measurement - operator.rm(estimate, range_m)
    return float(
        torch.linalg.vector_norm(residual)
        / torch.clamp(
            torch.linalg.vector_norm(measurement),
            min=torch.finfo(measurement.real.dtype).eps,
        )
    )


def run_mode(measurement, truth, operator, range_m, params):
    started = time.perf_counter()
    estimate, history = rmist_ista(
        measurement,
        operator,
        range_m,
        tau=float(params["tau"]),
        theta=float(params["theta"]),
        iterations=int(params["iterations"]),
    )
    return estimate, history, {
        "tau": float(params["tau"]),
        "theta": float(params["theta"]),
        "iterations": int(params["iterations"]),
        "nrmse": normalized_error(estimate, truth),
        "normalized_measurement_residual": normalized_residual(
            estimate, measurement, operator, range_m
        ),
        "runtime_seconds": time.perf_counter() - started,
        "residual_curve": history["residual_norm"],
        "finite": bool(torch.isfinite(estimate).all()),
    }


def save_image(path, value, title):
    magnitude = torch.abs(value).cpu().numpy()
    peak = max(float(magnitude.max()), np.finfo(np.float32).eps)
    db = np.maximum(20.0 * np.log10(magnitude / peak + 1e-12), -60.0)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(db, origin="lower", cmap="jet", vmin=-60.0, vmax=0.0)
    ax.set(title=title, xlabel="x index", ylabel="z index")
    fig.colorbar(image, ax=ax, label="Normalized magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_ista.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    shutil.copyfile(args.config, output / "config_used.yaml")
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")

    geometry = config["geometry"]
    grid_size = int(geometry["grid_size"])
    range_m = float(geometry["range_m"])
    operator = RMISTKernel(
        geometry["center_frequency_hz"],
        geometry["dx_m"],
        geometry["dz_m"],
        grid_size,
        geometry["ky_epsilon"],
    )
    truth = point_scatterer_scene(
        (grid_size, grid_size), int(config["synthetic"]["point_count"]), seed
    )
    full_mask = random_mask((grid_size, grid_size), 1.0, seed + 1)
    sparse_mask = random_mask(
        (grid_size, grid_size),
        float(config["synthetic"]["sparse_sampling_rate"]),
        seed + 2,
    )
    clean_echo = operator.rm(truth, range_m, mask=full_mask)
    noisy_echo = add_complex_awgn(
        clean_echo, float(config["synthetic"]["snr_db"]), seed + 3
    )
    sparse_clean = clean_echo * sparse_mask
    sparse_noisy = add_complex_awgn(
        sparse_clean, float(config["synthetic"]["snr_db"]), seed + 4
    )

    # This diagnostic grid search is explicitly not attributed to RMIST.
    tuning_records = []
    tuned_cfg = config["tuned_diagnostic"]
    for tau in tuned_cfg["tau_candidates"]:
        for theta in tuned_cfg["theta_candidates"]:
            params = {
                "tau": tau,
                "theta": theta,
                "iterations": tuned_cfg["iterations"],
            }
            _, _, result = run_mode(noisy_echo, truth, operator, range_m, params)
            tuning_records.append(result)
    best = min(tuning_records, key=lambda value: value["nrmse"])
    tuned_params = {
        "tau": best["tau"],
        "theta": best["theta"],
        "iterations": int(tuned_cfg["iterations"]),
    }
    (output / "tuning.json").write_text(
        json.dumps(
            {
                "classification": "3DRIED-specific diagnostic; not RMIST-reported",
                "selection_case": tuned_cfg["selection_case"],
                "selection_metric": "minimum final NRMSE",
                "candidates": tuning_records,
                "selected": tuned_params,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = {
        "noise_free_full": clean_echo,
        "snr20_full": noisy_echo,
        "snr20_sampling50": sparse_noisy,
    }
    modes = {
        "ini": config["ini_ista"],
        "tuned": tuned_params,
    }
    results = {}
    estimates = {}
    for case_name, measurement in cases.items():
        results[case_name] = {}
        for mode_name, params in modes.items():
            estimate, _, metrics = run_mode(
                measurement, truth, operator, range_m, params
            )
            results[case_name][mode_name] = metrics
            estimates[(case_name, mode_name)] = estimate
            np.savez(
                output / (case_name + "_" + mode_name + ".npz"),
                reconstruction=estimate.numpy(),
                measurement=measurement.numpy(),
            )

    np.save(output / "ground_truth.npy", truth.numpy())
    np.save(output / "full_mask.npy", full_mask.numpy())
    np.save(output / "sampling50_mask.npy", sparse_mask.numpy())
    save_image(output / "ground_truth_db.png", truth, "Complex point-scatterer oracle")
    save_image(
        output / "snr20_tuned_db.png",
        estimates[("snr20_full", "tuned")],
        "Tuned diagnostic, 20 dB full data",
    )

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for case_name, mode_values in results.items():
        for mode_name, values in mode_values.items():
            ax.plot(
                range(1, len(values["residual_curve"]) + 1),
                values["residual_curve"],
                marker="o",
                label=case_name + "/" + mode_name,
            )
    ax.set(
        xlabel="Iteration",
        ylabel="Pre-update normalized residual",
        title="Strict RM-kernel ISTA residual curves",
        yscale="log",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(output / "residual_curves.png", dpi=160)
    plt.close(fig)

    signal_noise = noisy_echo - clean_echo
    achieved_snr = 10.0 * torch.log10(
        torch.mean(torch.abs(clean_echo) ** 2)
        / torch.mean(torch.abs(signal_noise) ** 2)
    )
    metrics = {
        "status": "PASS"
        if all(
            value["finite"]
            for modes_for_case in results.values()
            for value in modes_for_case.values()
        )
        else "FAILED",
        "method": "strict RMIST Eq. (16)-(19) conventional RM-kernel ISTA",
        "initialization": "RM_dagger(measurement)",
        "ini_parameter_provenance": "RMIST-reported network initialization values; not reported optimal ISTA values",
        "ini_parameters": config["ini_ista"],
        "tuned_parameter_provenance": "3DRIED-specific diagnostic grid search; paper does not report final Tuned-ISTA values",
        "tuned_parameters": tuned_params,
        "scene": {
            "shape": [grid_size, grid_size],
            "point_count": int(config["synthetic"]["point_count"]),
            "random_phase": "uniform [0, 2*pi)",
            "range_m": range_m,
        },
        "noise": {
            "requested_snr_db": float(config["synthetic"]["snr_db"]),
            "achieved_full_data_snr_db": float(achieved_snr),
        },
        "sampling50_realized_rate": float(sparse_mask.mean()),
        "results": results,
        "paper_style_512_benchmark": "deferred; this M4-R run is a separate 128x128 smoke/diagnostic experiment",
        "total_runtime_seconds": time.perf_counter() - run_started,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise SystemExit("M4-R validation failed")


if __name__ == "__main__":
    main()
