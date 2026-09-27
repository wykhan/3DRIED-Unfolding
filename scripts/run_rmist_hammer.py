#!/usr/bin/env python3
"""Run inference-only strict RMIST Eq. (32) on real single-Rx Hammer data."""

import argparse
import csv
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
from three_dried.data.three_dried import load_single_rx_aperture
from three_dried.imaging.rma import image_coordinates_mm, matlab_bin_range_m, range_fft
from three_dried.imaging.rmist_hammer import (
    accepted_effective_mask,
    load_accepted_model,
    observed_sample_consistency,
    reconstruct_sparse_methods,
)
from three_dried.imaging.rmist_kernel import RMISTKernel


METHODS = ("m2_full_rma", "rma50", "ini_ista50", "rmist50")
DISPLAY_NAMES = {
    "m2_full_rma": "Full-data M2 RMA reference",
    "rma50": "50% masked RMA",
    "ini_ista50": "50% masked Ini-ISTA (9 steps)",
    "rmist50": "50% masked trained RMIST",
}


def environment_text() -> str:
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


def validate_config(config: dict) -> None:
    data = config["data"]
    inference = config["inference"]
    required = {
        "rx_index": (int(data["rx_index"]), 0),
        "nx": (int(data["nx"]), 407),
        "nz": (int(data["nz"]), 200),
        "motion_error_samples": (int(data["motion_error_samples"]), 43),
        "serpentine_correct": (bool(data["serpentine_correct"]), True),
        "n_range_fft": (int(inference["n_range_fft"]), 256),
        "n_fft_space": (int(inference["n_fft_space"]), 512),
        "range_bins": (
            [int(value) for value in inference["matlab_range_bins"]],
            list(range(12, 25)),
        ),
        "best_epoch": (int(config["model"]["best_epoch"]), 99),
        "mask_seed": (int(config["mask"]["seed"]), 20260928),
        "mask_sampling_rate": (float(config["mask"]["sampling_rate"]), 0.5),
    }
    failures = [name for name, pair in required.items() if pair[0] != pair[1]]
    if failures:
        raise ValueError("strict M7 configuration mismatch: " + ", ".join(failures))
    if config["reporting"]["real_data_outcome"] not in {
        "SUCCESS", "PARTIAL_DEGRADED", "FAILURE"
    }:
        raise ValueError("invalid M7 real-data outcome classification")


def normalized_db(value: np.ndarray, common_peak: float, floor_db: float) -> np.ndarray:
    if not np.isfinite(common_peak) or common_peak <= 0:
        raise ValueError("invalid common visualization peak")
    return np.maximum(20.0 * np.log10(np.abs(value) / common_peak + 1e-12), floor_db)


def save_comparison_mip(
    path: Path,
    volumes: dict,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    x_keep: np.ndarray,
    z_keep: np.ndarray,
    dynamic_range_db: float,
) -> float:
    roi_magnitudes = {
        name: np.abs(value[:, z_keep][:, :, x_keep]) for name, value in volumes.items()
    }
    common_peak = max(float(value.max()) for value in roi_magnitudes.values())
    extent = [x_mm[x_keep][0], x_mm[x_keep][-1], z_mm[z_keep][0], z_mm[z_keep][-1]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    image = None
    for axis, name in zip(axes, METHODS):
        mip = np.max(roi_magnitudes[name], axis=0)
        image = axis.imshow(
            normalized_db(mip, common_peak, -dynamic_range_db),
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="jet",
            vmin=-dynamic_range_db,
            vmax=0,
        )
        axis.set(title=DISPLAY_NAMES[name], xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(image, ax=axes, shrink=0.75, label="Common-peak normalized magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return common_peak


def save_representative_slices(
    path: Path,
    volumes: dict,
    ranges_m: np.ndarray,
    bins: list,
    representative_indices: list,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    x_keep: np.ndarray,
    z_keep: np.ndarray,
    common_peak: float,
    dynamic_range_db: float,
) -> None:
    extent = [x_mm[x_keep][0], x_mm[x_keep][-1], z_mm[z_keep][0], z_mm[z_keep][-1]]
    fig, axes = plt.subplots(
        len(representative_indices), 4,
        figsize=(20, 5 * len(representative_indices)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    image = None
    for row, index in enumerate(representative_indices):
        for column, name in enumerate(METHODS):
            plane = volumes[name][index][np.ix_(z_keep, x_keep)]
            image = axes[row, column].imshow(
                normalized_db(plane, common_peak, -dynamic_range_db),
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="jet",
                vmin=-dynamic_range_db,
                vmax=0,
            )
            axes[row, column].set(
                title="{}\nBin {} ({:.1f} mm)".format(
                    DISPLAY_NAMES[name], bins[index], ranges_m[index] * 1e3
                ),
                xlabel="x (mm)",
                ylabel="z (mm)",
            )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75,
                 label="Common-peak normalized magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_volume_comparison(
    path: Path,
    m2_volume: np.ndarray,
    rmist_volume: np.ndarray,
    ranges_m: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    x_keep: np.ndarray,
    z_keep: np.ndarray,
    common_peak: float,
    threshold_db: float,
) -> None:
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    for subplot, (title, volume) in enumerate(
        (("Full-data M2 RMA", m2_volume), ("50% trained RMIST", rmist_volume)), start=1
    ):
        axis = fig.add_subplot(1, 2, subplot, projection="3d")
        db = normalized_db(volume, common_peak, -80.0)
        roi = np.zeros(volume.shape[-2:], dtype=bool)
        roi[np.ix_(z_keep, x_keep)] = True
        points = np.argwhere((db >= threshold_db) & roi[None])
        if len(points) > 60000:
            generator = np.random.default_rng(20260927)
            points = points[generator.choice(len(points), 60000, replace=False)]
        if len(points):
            ri, zi, xi = points.T
            values = db[ri, zi, xi]
            scatter = axis.scatter(
                x_mm[xi], ranges_m[ri] * 1e3, z_mm[zi], c=values, s=1.0,
                cmap="jet", vmin=threshold_db, vmax=0, alpha=0.35,
            )
            fig.colorbar(scatter, ax=axis, shrink=0.55, label="Common-peak dB")
        axis.set(title=title, xlabel="x (mm)", ylabel="range y (mm)", zlabel="z (mm)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_hammer_inference.yaml"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    requested_device = str(config["inference"]["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA inference device is unavailable")
    device = torch.device(requested_device)
    output_dir = Path(config["output_dir"])
    review_dir = Path(config["review_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output_dir / "config_used.yaml")

    data = config["data"]
    radar = config["radar"]
    aperture = config["aperture"]
    inference = config["inference"]
    operator = RMISTKernel(
        float(radar["center_frequency_hz"]),
        float(aperture["dx_m"]),
        float(aperture["dz_m"]),
        n_fft_space=int(inference["n_fft_space"]),
    )
    _, effective_mask, mask_summary = accepted_effective_mask(
        operator,
        (int(data["nz"]), int(data["nx"])),
        int(config["mask"]["seed"]),
        float(config["mask"]["sampling_rate"]),
        str(config["mask"]["expected_sha256"]),
    )
    model, provenance = load_accepted_model(
        operator,
        Path(config["model"]["summary_path"]),
        Path(config["model"]["checkpoint_path"]),
        device,
    )
    (review_dir / "mask_summary.json").write_text(
        json.dumps(mask_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (review_dir / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parameter_values = {
        "best_epoch": 99,
        "tau": model.tau.detach().cpu().tolist(),
        "theta": model.theta.detach().cpu().tolist(),
        "trainable_during_m7": False,
    }
    (review_dir / "parameter_values.json").write_text(
        json.dumps(parameter_values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    echo, load_metadata = load_single_rx_aperture(
        Path(config["dataset_dir"]) / "output_rawdata.mat",
        rx_index=int(data["rx_index"]),
        nx=int(data["nx"]),
        nz=int(data["nz"]),
        motion_error_samples=int(data["motion_error_samples"]),
        serpentine_correct=bool(data["serpentine_correct"]),
    )
    focused = range_fft(echo, int(inference["n_range_fft"]))
    bins = [int(value) for value in inference["matlab_range_bins"]]
    ranges_m = np.asarray(
        [
            matlab_bin_range_m(
                value,
                float(radar["sample_rate_hz"]),
                float(radar["chirp_slope_hz_s"]),
                int(inference["n_range_fft"]),
                float(radar["instrument_delay_s"]),
            )
            for value in bins
        ],
        dtype=np.float64,
    )

    representative = bins.index(int(inference["representative_matlab_bin"]))
    physical = torch.from_numpy(focused[:, :, bins[representative] - 1]).to(device)
    padded = operator.pad_measurement(physical)
    outputs = reconstruct_sparse_methods(
        padded, effective_mask, operator, model, float(ranges_m[representative])
    )
    preflight_residuals = {
        name: observed_sample_consistency(
            estimate, padded, effective_mask, operator, float(ranges_m[representative])
        )
        for name, estimate in outputs.items()
    }
    preflight = {
        "status": "PASS"
        if all(torch.isfinite(value).all().item() for value in outputs.values())
        and all(np.isfinite(value) for value in preflight_residuals.values())
        else "FAIL",
        "matlab_range_bin": bins[representative],
        "range_m": float(ranges_m[representative]),
        "all_outputs_finite": all(
            torch.isfinite(value).all().item() for value in outputs.values()
        ),
        "observed_residual": preflight_residuals,
        "rx_index_python": load_metadata["rx_index_python"],
        "rx_index_matlab": load_metadata["rx_index_matlab"],
        "mask_hash": mask_summary["full_grid_sha256_uint8"],
        "effective_physical_sampling_rate": mask_summary[
            "effective_physical_sampling_rate"
        ],
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PREFLIGHT " + json.dumps(preflight, sort_keys=True), flush=True)
    if preflight["status"] != "PASS":
        raise RuntimeError("M7 one-slice preflight failed")
    if args.preflight_only:
        return

    sparse = {
        name: np.empty((len(bins), 512, 512), dtype=np.complex64)
        for name in ("rma50", "ini_ista50", "rmist50")
    }
    residual_rows = []
    for index, (matlab_bin, range_m) in enumerate(zip(bins, ranges_m)):
        physical = torch.from_numpy(focused[:, :, matlab_bin - 1]).to(device)
        padded = operator.pad_measurement(physical)
        slice_outputs = reconstruct_sparse_methods(
            padded, effective_mask, operator, model, float(range_m)
        )
        row = {"matlab_range_bin": matlab_bin, "range_m": float(range_m)}
        for name, estimate in slice_outputs.items():
            if not torch.isfinite(estimate).all().item():
                raise FloatingPointError(name + " produced a non-finite slice")
            sparse[name][index] = estimate.detach().cpu().numpy()
            row[name + "_observed_residual"] = observed_sample_consistency(
                estimate, padded, effective_mask, operator, float(range_m)
            )
        residual_rows.append(row)
        print("SLICE {}/{} bin={} range_m={:.9f}".format(index + 1, len(bins), matlab_bin, range_m), flush=True)

    reference = np.load(config["reference"]["volume_path"])
    coordinates = np.load(config["reference"]["coordinates_path"])
    if reference.shape != (len(bins), 512, 512):
        raise RuntimeError("accepted M2 reference has unexpected shape")
    if not np.array_equal(coordinates["matlab_range_bins"], np.asarray(bins)):
        raise RuntimeError("accepted M2 bins differ from M7")
    if not np.allclose(coordinates["range_m"], ranges_m, rtol=0.0, atol=1e-14):
        raise RuntimeError("accepted M2 ranges differ from M7")
    volumes = {"m2_full_rma": reference, **sparse}
    for name, value in sparse.items():
        np.save(output_dir / (name + "_volume.npy"), value)

    fields = [
        "matlab_range_bin", "range_m", "rma50_observed_residual",
        "ini_ista50_observed_residual", "rmist50_observed_residual",
    ]
    with (review_dir / "per_slice_consistency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(residual_rows)

    x_mm, z_mm = image_coordinates_mm(
        512,
        float(aperture["dx_m"]),
        float(aperture["dz_m"]),
        float(config["reference"]["x_offset_mm"]),
        float(config["reference"]["z_offset_mm"]),
    )
    crop_half = float(config["visualization"]["crop_size_mm"]) / 2.0
    x_keep = np.abs(x_mm) <= crop_half
    z_keep = np.abs(z_mm) <= crop_half
    dynamic_range = float(config["visualization"]["dynamic_range_db"])
    common_peak = save_comparison_mip(
        review_dir / "comparison_mip.png", volumes, x_mm, z_mm,
        x_keep, z_keep, dynamic_range,
    )
    peak_range_index = int(np.unravel_index(np.argmax(np.abs(reference)), reference.shape)[0])
    representative_indices = sorted(
        set(max(0, min(len(bins) - 1, peak_range_index + offset)) for offset in (-1, 0, 1))
    )
    save_representative_slices(
        review_dir / "representative_slices.png", volumes, ranges_m, bins,
        representative_indices, x_mm, z_mm, x_keep, z_keep, common_peak, dynamic_range,
    )
    save_volume_comparison(
        output_dir / "volume_3d_comparison.png", reference, sparse["rmist50"],
        ranges_m, x_mm, z_mm, x_keep, z_keep, common_peak,
        float(config["visualization"]["point_cloud_threshold_db"]),
    )

    aggregate = {}
    for name in ("rma50", "ini_ista50", "rmist50"):
        values = np.asarray([row[name + "_observed_residual"] for row in residual_rows])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    comparison_diagnostics = {
        "classification": "non-GT method-to-method complex relative difference",
        "rma50_vs_ini_ista50": float(
            np.linalg.norm((sparse["rma50"] - sparse["ini_ista50"]).ravel())
            / np.linalg.norm(sparse["rma50"].ravel())
        ),
        "rma50_vs_rmist50": float(
            np.linalg.norm((sparse["rma50"] - sparse["rmist50"]).ravel())
            / np.linalg.norm(sparse["rma50"].ravel())
        ),
        "ini_ista50_vs_rmist50": float(
            np.linalg.norm((sparse["ini_ista50"] - sparse["rmist50"]).ravel())
            / np.linalg.norm(sparse["ini_ista50"].ravel())
        ),
    }
    summary = {
        "technical_pipeline": "PASS",
        "real_data_outcome": config["reporting"]["real_data_outcome"],
        "qualitative_assessment": config["reporting"]["qualitative_assessment"],
        "method": "strict RMIST Eq. (32) range-slice inference",
        "model": provenance,
        "mask": mask_summary,
        "rx_index_python": load_metadata["rx_index_python"],
        "rx_index_matlab": load_metadata["rx_index_matlab"],
        "raw_selected_shape": list(echo.shape),
        "matlab_range_bins": bins,
        "range_m": ranges_m.tolist(),
        "volume_shape": list(reference.shape),
        "all_outputs_finite": True,
        "observed_sample_consistency": aggregate,
        "observed_sample_consistency_aggregate_definition": "arithmetic mean/std/min/max of the per-slice E_obs values",
        "method_comparison_diagnostics": comparison_diagnostics,
        "visualization": {
            "normalization": "one common ROI peak across all four methods",
            "dynamic_range_db": dynamic_range,
            "representative_matlab_bins": [bins[index] for index in representative_indices],
            "point_cloud_threshold_db": float(config["visualization"]["point_cloud_threshold_db"]),
        },
        "preflight": preflight,
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "environment": environment_text().splitlines(),
        "training_or_parameter_updates_performed": False,
        "ground_truth_metrics_computed": False,
    }
    (review_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
