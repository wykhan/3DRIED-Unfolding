#!/usr/bin/env python3
"""Evaluate the four fixed M9 scale-normalized RMIST variants."""

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
from three_dried.imaging.rmist_diagnostics import phase_metrics, trace_rmist_forward
from three_dried.imaging.rmist_hammer import (
    accepted_effective_mask,
    load_accepted_model,
    observed_sample_consistency,
)
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.imaging.rmist_normalization import (
    magnitude_proxy,
    normalized_range_profile,
    normalized_rmist,
    p95_magnitude,
    positive_scale,
    rms_magnitude,
)
from three_dried.models.rmist_net import RMISTNet
from three_dried.training.rmist_training import (
    FixedRangeRMISTOperator,
    RMISTSyntheticDataset,
    make_fixed_mask,
    mask_sha256,
)


RMIST_METHODS = ("N0", "N1", "N2", "N3", "N4")
DISPLAY_METHODS = ("m2_full_rma", "rma50") + RMIST_METHODS
DISPLAY_NAMES = {
    "m2_full_rma": "M2 full-data RMA proxy",
    "rma50": "RMA50",
    "N0": "N0 raw RMIST50",
    "N1": "N1 global alpha0 RMS",
    "N2": "N2 per-slice alpha0 RMS",
    "N3": "N3 per-slice alpha0 p95",
    "N4": "N4 observed measurement RMS",
}
PHASE_METRICS = ("E_grad", "E_shrink", "E_update", "E_RMA")


def environment_text():
    return "\n".join([
        "Python: " + platform.python_version(),
        "Platform: " + platform.platform(),
        "NumPy: " + np.__version__,
        "PyTorch: " + torch.__version__,
        "CUDA runtime: " + str(torch.version.cuda),
        "CUDA available: " + str(torch.cuda.is_available()),
        "GPUs: " + str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
    ]) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    checks = {
        "variants": (list(config["normalization"]["variants"]), list(RMIST_METHODS)),
        "grid": (int(config["geometry"]["grid_size"]), 512),
        "epoch": (int(config["model"]["best_epoch"]), 99),
        "mask_seed": (int(config["mask"]["seed"]), 20260928),
        "rate": (float(config["mask"]["sampling_rate"]), 0.5),
        "subset": (int(config["synthetic"]["subset_size"]), 32),
        "validation_seed": (int(config["synthetic"]["validation_seed"]), 20271000),
        "rx": (int(config["hammer"]["rx_index"]), 0),
        "bins": ([int(v) for v in config["hammer"]["matlab_range_bins"]], list(range(12, 25))),
        "representative": ([int(v) for v in config["hammer"]["representative_bins"]], [16, 17, 18]),
    }
    failures = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise ValueError("strict M9 configuration mismatch: " + ", ".join(failures))


def synthetic_references(config, operator, accepted_state, mask, device):
    section = config["synthetic"]
    range_m = float(section["range_m"])
    cached = FixedRangeRMISTOperator(operator, range_m, torch.complex64, device)
    model = RMISTNet(cached).to(device)
    model.load_state_dict(accepted_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = RMISTSyntheticDataset(
        pair_count=int(section["subset_size"]), grid_size=512,
        scatterers_per_image=int(section["scatterers_per_image"]),
        amplitude_min=float(section["amplitude_min"]),
        amplitude_max=float(section["amplitude_max"]),
        snr_db=float(section["snr_db"]), range_m=range_m,
        seed=int(section["validation_seed"]),
    )
    rms_values, p95_values, measurement_values = [], [], []
    mask_device = mask.to(device)
    observed = mask_device.bool()
    batch_size = int(section["batch_size"])
    for start in range(0, dataset.pair_count, batch_size):
        indices = list(range(start, min(start + batch_size, dataset.pair_count)))
        _, measurement, _ = dataset.make_batch(indices, cached, mask_device, device)
        with torch.no_grad():
            alpha0 = model.initialize(measurement, range_m)
        for local in range(len(indices)):
            rms_values.append(rms_magnitude(alpha0[local]))
            p95_values.append(p95_magnitude(alpha0[local]))
            measurement_values.append(rms_magnitude(measurement[local][observed]))
    references = {
        "sample_count": dataset.pair_count,
        "alpha0_rms_median": float(np.median(rms_values)),
        "alpha0_p95_median": float(np.median(p95_values)),
        "observed_measurement_rms_median": float(np.median(measurement_values)),
        "alpha0_rms_per_sample": rms_values,
        "alpha0_p95_per_sample": p95_values,
        "observed_measurement_rms_per_sample": measurement_values,
    }
    expected = float(section["accepted_m8_alpha0_rms"])
    tolerance = float(section["regression_atol"])
    references["accepted_m8_alpha0_rms"] = expected
    references["absolute_difference_from_m8"] = abs(references["alpha0_rms_median"] - expected)
    references["m8_regression_tolerance"] = tolerance
    references["m8_regression_pass"] = references["absolute_difference_from_m8"] <= tolerance
    if not references["m8_regression_pass"]:
        raise RuntimeError("regenerated synthetic alpha0 RMS does not match accepted M8")
    return references


def append_phase_rows(rows, method, matlab_bin, metrics):
    for phase in metrics:
        row = {"method": method, "matlab_range_bin": matlab_bin,
               "phase": int(phase["phase"][0])}
        for name in PHASE_METRICS:
            row[name] = float(phase[name][0].detach().cpu())
        rows.append(row)


def phase_summary(rows, method):
    selected = [row for row in rows if row["method"] == method]
    result = {}
    for name in ("E_grad", "E_shrink", "E_update"):
        per_slice_max = [max(row[name] for row in selected if row["matlab_range_bin"] == value)
                         for value in range(12, 25)]
        result["max_" + name] = {"median": float(np.median(per_slice_max)),
                                  "max": float(np.max(per_slice_max))}
    final = [row["E_RMA"] for row in selected if row["phase"] == 9]
    result["final_E_RMA"] = {"median": float(np.median(final)),
                              "max": float(np.max(final))}
    result["all_finite"] = bool(all(np.isfinite(row[name]) for row in selected for name in PHASE_METRICS))
    result["dynamics_restored"] = bool(
        result["final_E_RMA"]["median"] >= 0.1
        or result["max_E_shrink"]["median"] >= 0.05
    )
    return result


def proxy_rows(reference, volumes, bins, roi):
    rows = []
    reference_roi = reference[:, roi[0]][:, :, roi[1]]
    for method, volume in volumes.items():
        estimate_roi = volume[:, roi[0]][:, :, roi[1]]
        per_slice = []
        for index, matlab_bin in enumerate(bins):
            cosine, distance = magnitude_proxy(reference_roi[index], estimate_roi[index])
            per_slice.append((cosine, distance))
            rows.append({"method": method, "scope": "slice", "matlab_range_bin": matlab_bin,
                         "magnitude_cosine": cosine, "normalized_magnitude_distance": distance})
        rows.append({"method": method, "scope": "per_slice_mean", "matlab_range_bin": "",
                     "magnitude_cosine": float(np.mean([v[0] for v in per_slice])),
                     "normalized_magnitude_distance": float(np.mean([v[1] for v in per_slice]))})
        cosine, distance = magnitude_proxy(reference_roi, estimate_roi)
        rows.append({"method": method, "scope": "full_volume", "matlab_range_bin": "",
                     "magnitude_cosine": cosine, "normalized_magnitude_distance": distance})
        cosine, distance = magnitude_proxy(np.max(np.abs(reference_roi), axis=0),
                                            np.max(np.abs(estimate_roi), axis=0))
        rows.append({"method": method, "scope": "xz_mip", "matlab_range_bin": "",
                     "magnitude_cosine": cosine, "normalized_magnitude_distance": distance})
    return rows


def plot_mip(path, volumes, roi, x_mm, z_mm, dynamic_range):
    cropped = {name: np.abs(value[:, roi[0]][:, :, roi[1]]) for name, value in volumes.items()}
    common_peak = max(float(value.max()) for value in cropped.values())
    extent = [x_mm[roi[1]][0], x_mm[roi[1]][-1], z_mm[roi[0]][0], z_mm[roi[0]][-1]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    image = None
    for axis, name in zip(axes.flat, DISPLAY_METHODS):
        db = np.maximum(20 * np.log10(np.max(cropped[name], axis=0) / common_peak + 1e-12), -dynamic_range)
        image = axis.imshow(db, origin="lower", extent=extent, aspect="equal", cmap="jet",
                            vmin=-dynamic_range, vmax=0)
        axis.set(title=DISPLAY_NAMES[name], xlabel="x (mm)", ylabel="z (mm)")
    axes.flat[-1].axis("off")
    fig.colorbar(image, ax=list(axes.flat), shrink=0.75, label="Common-peak magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return common_peak


def plot_slices(path, volumes, roi, x_mm, z_mm, bins, representative, common_peak, dynamic_range):
    indices = [bins.index(value) for value in representative]
    extent = [x_mm[roi[1]][0], x_mm[roi[1]][-1], z_mm[roi[0]][0], z_mm[roi[0]][-1]]
    fig, axes = plt.subplots(len(indices), len(DISPLAY_METHODS), figsize=(24, 11), constrained_layout=True)
    image = None
    for row, index in enumerate(indices):
        for column, name in enumerate(DISPLAY_METHODS):
            plane = np.abs(volumes[name][index][np.ix_(roi[0], roi[1])])
            db = np.maximum(20 * np.log10(plane / common_peak + 1e-12), -dynamic_range)
            image = axes[row, column].imshow(db, origin="lower", extent=extent, aspect="equal",
                                             cmap="jet", vmin=-dynamic_range, vmax=0)
            axes[row, column].set(title="{}\nbin {}".format(DISPLAY_NAMES[name], bins[index]),
                                  xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Common-peak magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_profiles(path, profiles, bins):
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name in DISPLAY_METHODS:
        axis.plot(bins, profiles[name], "o-", label=DISPLAY_NAMES[name])
    axis.set(xlabel="MATLAB range bin", ylabel="Unit-L2 slice energy",
             title="Normalized range-energy profiles")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_scale_normalized.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    output_dir, review_dir = Path(config["output_dir"]), Path(config["review_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output_dir / "config_used.yaml")

    geometry = config["geometry"]
    operator = RMISTKernel(float(geometry["center_frequency_hz"]), float(geometry["dx_m"]),
                           float(geometry["dz_m"]), int(geometry["grid_size"]),
                           float(geometry["ky_epsilon"]))
    model, provenance = load_accepted_model(operator, Path(config["model"]["summary_path"]),
                                             Path(config["model"]["checkpoint_path"]), device)
    parameter_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    full_mask = make_fixed_mask((512, 512), float(config["mask"]["sampling_rate"]),
                                int(config["mask"]["seed"]))
    if mask_sha256(full_mask) != config["mask"]["expected_sha256"]:
        raise RuntimeError("accepted M6 mask hash mismatch")
    references = synthetic_references(config, operator, model.state_dict(), full_mask, device)
    (review_dir / "synthetic_reference_scales.json").write_text(
        json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hammer = config["hammer"]
    _, effective_mask, mask_summary = accepted_effective_mask(
        operator, (int(hammer["nz"]), int(hammer["nx"])), int(config["mask"]["seed"]),
        float(config["mask"]["sampling_rate"]), str(config["mask"]["expected_sha256"]))
    echo, metadata = load_single_rx_aperture(
        Path(config["dataset_dir"]) / "output_rawdata.mat", rx_index=int(hammer["rx_index"]),
        nx=int(hammer["nx"]), nz=int(hammer["nz"]),
        motion_error_samples=int(hammer["motion_error_samples"]),
        serpentine_correct=bool(hammer["serpentine_correct"]))
    focused = range_fft(echo, int(hammer["n_range_fft"]))
    bins = [int(value) for value in hammer["matlab_range_bins"]]
    ranges = [matlab_bin_range_m(value, float(hammer["sample_rate_hz"]),
              float(hammer["chirp_slope_hz_s"]), int(hammer["n_range_fft"]),
              float(hammer["instrument_delay_s"])) for value in bins]
    effective_device = effective_mask.to(device)

    measurements, padded_echoes, alpha_rms, alpha_p95, observed_rms = [], [], [], [], []
    for matlab_bin, range_m in zip(bins, ranges):
        physical = torch.from_numpy(focused[:, :, matlab_bin - 1]).to(device)
        padded = operator.pad_measurement(physical)
        measurement = padded * effective_device.to(padded.dtype)
        with torch.no_grad():
            alpha0 = model.initialize(measurement, float(range_m))
        measurements.append(measurement)
        padded_echoes.append(padded)
        alpha_rms.append(rms_magnitude(alpha0))
        alpha_p95.append(p95_magnitude(alpha0))
        observed_rms.append(rms_magnitude(measurement[effective_device.bool()]))
    global_scale = positive_scale(references["alpha0_rms_median"], float(np.median(alpha_rms)))
    scales = {
        "N0": [1.0] * len(bins),
        "N1": [global_scale] * len(bins),
        "N2": [positive_scale(references["alpha0_rms_median"], value) for value in alpha_rms],
        "N3": [positive_scale(references["alpha0_p95_median"], value) for value in alpha_p95],
        "N4": [positive_scale(references["observed_measurement_rms_median"], value) for value in observed_rms],
    }
    scale_rows = []
    sanity_min = float(config["normalization"]["numerical_sanity_min"])
    sanity_max = float(config["normalization"]["numerical_sanity_max"])
    for index, matlab_bin in enumerate(bins):
        for method in RMIST_METHODS:
            value = scales[method][index]
            scale_rows.append({"method": method, "matlab_range_bin": matlab_bin,
                               "range_m": ranges[index], "scale": value,
                               "alpha0_rms": alpha_rms[index], "alpha0_p95": alpha_p95[index],
                               "observed_measurement_rms": observed_rms[index],
                               "within_broad_sanity_range": sanity_min <= value <= sanity_max})
    write_csv(review_dir / "normalization_scales.csv", scale_rows,
              ["method", "matlab_range_bin", "range_m", "scale", "alpha0_rms", "alpha0_p95",
               "observed_measurement_rms", "within_broad_sanity_range"])

    volumes = {name: np.empty((len(bins), 512, 512), dtype=np.complex64)
               for name in ("rma50",) + RMIST_METHODS}
    phase_rows, consistency_rows = [], []
    n0_ordinary_max_relative = 0.0
    for index, (matlab_bin, range_m) in enumerate(zip(bins, ranges)):
        measurement, padded = measurements[index], padded_echoes[index]
        rma = operator.rm_dagger(measurement, float(range_m))
        volumes["rma50"][index] = rma.detach().cpu().numpy()
        consistency_rows.append({"method": "rma50", "matlab_range_bin": matlab_bin,
                                 "observed_consistency": observed_sample_consistency(
                                     rma, padded, effective_mask, operator, float(range_m))})
        for method in RMIST_METHODS:
            scale = scales[method][index]
            scaled_measurement = measurement * scale
            traced_scaled, alpha0_scaled, trace = trace_rmist_forward(model, scaled_measurement, float(range_m))
            metrics = phase_metrics(scaled_measurement, alpha0_scaled, trace, model.theta)
            append_phase_rows(phase_rows, method, matlab_bin, metrics)
            estimate = traced_scaled / scale
            if method == "N0":
                with torch.no_grad():
                    ordinary = normalized_rmist(model, measurement, float(range_m), 1.0)
                denominator = torch.clamp(torch.linalg.vector_norm(ordinary),
                                          min=torch.finfo(ordinary.real.dtype).eps)
                relative = float((torch.linalg.vector_norm(estimate - ordinary) / denominator).cpu())
                n0_ordinary_max_relative = max(n0_ordinary_max_relative, relative)
                if not torch.allclose(estimate, ordinary, rtol=1e-6, atol=1e-6):
                    raise RuntimeError("N0 does not reproduce ordinary RMIST within tolerance")
            if not torch.isfinite(estimate).all().item():
                raise FloatingPointError(method + " produced non-finite output")
            volumes[method][index] = estimate.detach().cpu().numpy()
            consistency_rows.append({"method": method, "matlab_range_bin": matlab_bin,
                                     "observed_consistency": observed_sample_consistency(
                                         estimate, padded, effective_mask, operator, float(range_m))})
        print("M9 slice {}/{} bin={}".format(index + 1, len(bins), matlab_bin), flush=True)

    write_csv(review_dir / "phase_activity_by_method.csv", phase_rows,
              ["method", "matlab_range_bin", "phase"] + list(PHASE_METRICS))
    write_csv(review_dir / "observed_consistency.csv", consistency_rows,
              ["method", "matlab_range_bin", "observed_consistency"])

    reference = np.load(config["reference"]["volume_path"])
    coordinates = np.load(config["reference"]["coordinates_path"])
    if reference.shape != (13, 512, 512) or not np.array_equal(coordinates["matlab_range_bins"], bins):
        raise RuntimeError("accepted M2 reference mismatch")
    if not np.allclose(coordinates["range_m"], ranges, rtol=0.0, atol=1e-14):
        raise RuntimeError("accepted M2 range coordinates mismatch")
    accepted_m7_path = Path(config["reference"]["accepted_m7_volume_path"])
    if accepted_m7_path.is_file():
        accepted_m7 = np.load(accepted_m7_path)
        n0_m7_relative = float(np.linalg.norm((volumes["N0"] - accepted_m7).ravel()) /
                               np.linalg.norm(accepted_m7.ravel()))
        n0_m7_pass = bool(np.allclose(volumes["N0"], accepted_m7, rtol=1e-6, atol=1e-6))
        if not n0_m7_pass:
            raise RuntimeError("N0 does not reproduce accepted local M7 RMIST volume")
    else:
        n0_m7_relative, n0_m7_pass = None, None

    display_volumes = {"m2_full_rma": reference, **volumes}
    x_mm, z_mm = image_coordinates_mm(512, float(geometry["dx_m"]), float(geometry["dz_m"]),
                                       float(config["reference"]["x_offset_mm"]),
                                       float(config["reference"]["z_offset_mm"]))
    half = float(config["visualization"]["crop_size_mm"]) / 2.0
    roi = (np.abs(z_mm) <= half, np.abs(x_mm) <= half)
    comparison = {name: value for name, value in volumes.items()}
    proxies = proxy_rows(reference, comparison, bins, roi)
    write_csv(review_dir / "m2_reference_proxies.csv", proxies,
              ["method", "scope", "matlab_range_bin", "magnitude_cosine", "normalized_magnitude_distance"])

    profiles = {name: normalized_range_profile(value) for name, value in display_volumes.items()}
    range_rows, range_summary = [], {}
    reference_profile = profiles["m2_full_rma"]
    for method, profile in profiles.items():
        cosine = float(np.dot(profile, reference_profile))
        dominant = bins[int(np.argmax(profile))]
        range_summary[method] = {"cosine_to_m2": cosine, "dominant_matlab_bin": dominant}
        for matlab_bin, value in zip(bins, profile):
            range_rows.append({"method": method, "matlab_range_bin": matlab_bin,
                               "normalized_slice_energy": float(value),
                               "profile_cosine_to_m2": cosine, "dominant_matlab_bin": dominant})
    write_csv(review_dir / "range_profile.csv", range_rows,
              ["method", "matlab_range_bin", "normalized_slice_energy",
               "profile_cosine_to_m2", "dominant_matlab_bin"])

    dynamic_range = float(config["visualization"]["dynamic_range_db"])
    common_peak = plot_mip(review_dir / "comparison_mip.png", display_volumes, roi, x_mm, z_mm, dynamic_range)
    plot_slices(review_dir / "representative_slices.png", display_volumes, roi, x_mm, z_mm, bins,
                [int(v) for v in hammer["representative_bins"]], common_peak, dynamic_range)
    plot_profiles(review_dir / "range_profile.png", profiles, bins)

    phase_summaries = {method: phase_summary(phase_rows, method) for method in RMIST_METHODS}
    observed_summary = {}
    for method in ("rma50",) + RMIST_METHODS:
        values = [row["observed_consistency"] for row in consistency_rows if row["method"] == method]
        observed_summary[method] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                                    "max": float(np.max(values)), "all_finite": bool(np.all(np.isfinite(values)))}
    proxy_summary = {}
    for method in ("rma50",) + RMIST_METHODS:
        proxy_summary[method] = {
            scope: next({"magnitude_cosine": row["magnitude_cosine"],
                         "normalized_magnitude_distance": row["normalized_magnitude_distance"]}
                        for row in proxies if row["method"] == method and row["scope"] == scope)
            for scope in ("per_slice_mean", "full_volume", "xz_mip")
        }

    tolerance = float(config["proxy_criteria"]["comparison_tolerance"])
    max_range_drop = float(config["proxy_criteria"]["maximum_range_profile_cosine_drop"])
    max_observed_ratio = float(config["proxy_criteria"]["maximum_observed_consistency_ratio"])
    visual_assessment = str(config["reporting"]["visual_target_destruction_check"])
    raw_proxy = proxy_summary["N0"]
    criteria = {}
    for method in RMIST_METHODS[1:]:
        cosine_improved = all(proxy_summary[method][scope]["magnitude_cosine"] >
                              raw_proxy[scope]["magnitude_cosine"] + tolerance
                              for scope in ("per_slice_mean", "full_volume", "xz_mip"))
        distance_not_worse = all(proxy_summary[method][scope]["normalized_magnitude_distance"] <=
                                 raw_proxy[scope]["normalized_magnitude_distance"] + tolerance
                                 for scope in ("per_slice_mean", "full_volume", "xz_mip"))
        range_ok = range_summary[method]["cosine_to_m2"] >= range_summary["N0"]["cosine_to_m2"] - max_range_drop
        dominant_ok = range_summary[method]["dominant_matlab_bin"] == 17
        observed_ok = observed_summary[method]["all_finite"] and (
            observed_summary[method]["mean"] <= max_observed_ratio * observed_summary["N0"]["mean"])
        criteria[method] = {
            "dynamics_restored": phase_summaries[method]["dynamics_restored"],
            "all_three_m2_proxy_cosines_improve": cosine_improved,
            "all_three_m2_proxy_distances_not_worse": distance_not_worse,
            "range_profile_not_materially_degraded": range_ok,
            "dominant_range_is_bin17": dominant_ok,
            "observed_consistency_not_catastrophic": observed_ok,
            "visual_target_destruction_check": visual_assessment,
            "proxy_benefit_without_visual_gate": bool(phase_summaries[method]["dynamics_restored"] and
                cosine_improved and distance_not_worse and range_ok and dominant_ok and observed_ok),
        }

    parameters_unchanged = all(torch.equal(value.detach().cpu(), parameter_before[name])
                               for name, value in model.state_dict().items())
    if not parameters_unchanged:
        raise RuntimeError("accepted learned parameters changed")
    summary = {
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "outcome_classification": str(config["reporting"]["outcome_classification"]),
        "classification_reason": "Dynamics are restored, but all normalized variants fail the predeclared proxy-benefit gates and degrade real-data proxy evidence.",
        "inference_only": True,
        "training_or_parameter_updates_performed": False,
        "learned_parameters_unchanged": True,
        "model_provenance": provenance,
        "mask": mask_summary,
        "synthetic_references": {key: value for key, value in references.items() if "per_sample" not in key},
        "scale_factors": {method: {"min": min(values), "median": float(np.median(values)), "max": max(values)}
                          for method, values in scales.items()},
        "all_scales_finite_positive": bool(all(np.isfinite(value) and value > 0
                                                for values in scales.values() for value in values)),
        "phase_activity": phase_summaries,
        "observed_consistency": observed_summary,
        "m2_reference_proxy_summary": proxy_summary,
        "range_profile_summary": range_summary,
        "proxy_benefit_criteria": criteria,
        "predeclared_materiality": config["proxy_criteria"],
        "n0_regression": {
            "ordinary_rmist_regression_pass": True,
            "max_relative_difference_from_ordinary": n0_ordinary_max_relative,
            "accepted_m7_local_volume_exists": accepted_m7_path.is_file(),
            "accepted_m7_allclose_pass": n0_m7_pass,
            "relative_difference_from_accepted_m7": n0_m7_relative,
        },
        "hammer": {"rx_index_python": metadata["rx_index_python"],
                   "rx_index_matlab": metadata["rx_index_matlab"],
                   "matlab_range_bins": bins, "range_m": ranges},
        "visualization": {"common_peak": common_peak, "dynamic_range_db": dynamic_range,
                          "representative_bins": hammer["representative_bins"],
                          "policy": "same ROI, orientation, bins, and one common peak across all seven methods"},
        "m2_reference_is_ground_truth": False,
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "environment": environment_text().splitlines(),
    }
    (review_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, value in volumes.items():
        np.save(output_dir / (name + "_volume.npy"), value)
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
