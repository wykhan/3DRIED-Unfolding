#!/usr/bin/env python3
"""Diagnose fixed-model RMIST synthetic-to-Hammer phase inactivity."""

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
from three_dried.imaging.rma import matlab_bin_range_m, range_fft
from three_dried.imaging.rmist_diagnostics import (
    magnitude_statistics,
    phase_metrics,
    trace_rmist_forward,
)
from three_dried.imaging.rmist_hammer import (
    accepted_effective_mask,
    load_accepted_model,
)
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import RMISTNet
from three_dried.training.rmist_training import (
    FixedRangeRMISTOperator,
    RMISTSyntheticDataset,
    make_fixed_mask,
    mask_sha256,
)


METRIC_NAMES = (
    "E_res", "E_grad", "E_shrink", "E_update", "E_RMA",
    "theta_over_p50", "theta_over_p90", "theta_over_p99",
    "fraction_le_theta", "fraction_le_2theta", "fraction_le_5theta",
)
STAT_NAMES = ("rms", "mean", "median", "p90", "p95", "p99", "max")


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
    expected = {
        "grid_size": (int(config["geometry"]["grid_size"]), 512),
        "best_epoch": (int(config["model"]["best_epoch"]), 99),
        "mask_seed": (int(config["mask"]["seed"]), 20260928),
        "sampling_rate": (float(config["mask"]["sampling_rate"]), 0.5),
        "synthetic_range": (float(config["synthetic"]["range_m"]), 0.5),
        "validation_seed": (int(config["synthetic"]["validation_seed"]), 20271000),
        "subset_size": (int(config["synthetic"]["subset_size"]), 32),
        "scatterers": (int(config["synthetic"]["scatterers_per_image"]), 100),
        "snr_db": (float(config["synthetic"]["snr_db"]), 20.0),
        "rx_index": (int(config["hammer"]["rx_index"]), 0),
        "bins": (
            [int(value) for value in config["hammer"]["matlab_range_bins"]],
            list(range(12, 25)),
        ),
        "primary_bin": (int(config["hammer"]["primary_matlab_bin"]), 17),
    }
    failures = [name for name, pair in expected.items() if pair[0] != pair[1]]
    if failures:
        raise ValueError("strict M8 configuration mismatch: " + ", ".join(failures))


def aggregate_stat_rows(rows):
    return {
        name: {
            "median_across_samples": float(np.median([row[name] for row in rows])),
            "mean_across_samples": float(np.mean([row[name] for row in rows])),
            "std_across_samples": float(np.std([row[name] for row in rows])),
        }
        for name in STAT_NAMES
    }


def append_phase_rows(destination, metrics, identity_name, identities):
    for phase_row in metrics:
        count = int(phase_row["phase"].shape[0])
        for index in range(count):
            row = {identity_name: identities[index], "phase": int(phase_row["phase"][index])}
            for name in METRIC_NAMES:
                row[name] = float(phase_row[name][index].detach().cpu())
            destination.append(row)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase_summary(rows, selector=None):
    selected = rows if selector is None else [row for row in rows if selector(row)]
    result = {}
    for name in ("E_grad", "E_shrink", "E_update"):
        values = np.asarray([row[name] for row in selected], dtype=np.float64)
        result[name] = {"max": float(values.max()), "median": float(np.median(values))}
    final = [row["E_RMA"] for row in selected if row["phase"] == 9]
    result["final_E_RMA"] = {
        "max": float(np.max(final)), "median": float(np.median(final)),
        "mean": float(np.mean(final)),
    }
    return result


def ordinary_trace_regression(model, measurement, range_m, rtol, atol):
    with torch.no_grad():
        ordinary = model(measurement, range_m)
    traced, alpha0, trace = trace_rmist_forward(model, measurement, range_m)
    difference = torch.linalg.vector_norm(traced - ordinary)
    denominator = torch.clamp(
        torch.linalg.vector_norm(ordinary), min=torch.finfo(ordinary.real.dtype).eps
    )
    relative = float((difference / denominator).detach().cpu())
    if not torch.allclose(traced, ordinary, rtol=rtol, atol=atol):
        raise RuntimeError("tracing altered accepted RMIST output")
    return traced, alpha0, trace, relative


def plot_phase_activity(path, synthetic_rows, hammer_rows, primary_bin):
    phases = np.arange(1, 10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, ("E_grad", "E_shrink", "E_update")):
        synth = [np.mean([row[metric] for row in synthetic_rows if row["phase"] == phase]) for phase in phases]
        real = [next(row[metric] for row in hammer_rows if row["matlab_range_bin"] == primary_bin and row["phase"] == phase) for phase in phases]
        axis.semilogy(phases, np.maximum(synth, 1e-20), "o-", label="Synthetic mean")
        axis.semilogy(phases, np.maximum(real, 1e-20), "s-", label="Hammer bin 17")
        axis.set(xlabel="Phase", ylabel=metric, title=metric)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_threshold_activity(path, synthetic_rows, hammer_rows, primary_bin):
    phases = np.arange(1, 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for metric, style in (("fraction_le_theta", "-"), ("fraction_le_2theta", "--"), ("fraction_le_5theta", ":")):
        synth = [np.mean([row[metric] for row in synthetic_rows if row["phase"] == phase]) for phase in phases]
        real = [next(row[metric] for row in hammer_rows if row["matlab_range_bin"] == primary_bin and row["phase"] == phase) for phase in phases]
        axes[0].plot(phases, synth, "o" + style, label="Synthetic " + metric.replace("fraction_le_", "≤"))
        axes[0].plot(phases, real, "s" + style, label="Hammer " + metric.replace("fraction_le_", "≤"))
    axes[0].set(xlabel="Phase", ylabel="Fraction of pixels", title="Thresholded-pixel activity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    for metric in ("theta_over_p50", "theta_over_p90", "theta_over_p99"):
        synth = [np.mean([row[metric] for row in synthetic_rows if row["phase"] == phase]) for phase in phases]
        real = [next(row[metric] for row in hammer_rows if row["matlab_range_bin"] == primary_bin and row["phase"] == phase) for phase in phases]
        axes[1].semilogy(phases, np.maximum(synth, 1e-20), "o-", label="Synthetic " + metric)
        axes[1].semilogy(phases, np.maximum(real, 1e-20), "s--", label="Hammer " + metric)
    axes[1].set(xlabel="Phase", ylabel="Ratio", title="Threshold / rough magnitude percentile")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scale_sweep(path, rows):
    ordered = sorted(rows, key=lambda row: row["scale"])
    scales = np.asarray([row["scale"] for row in ordered])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].loglog(scales, [row["final_E_RMA"] for row in ordered], "o-", label="Final E_RMA")
    axes[0].loglog(scales, [row["max_E_update"] for row in ordered], "s-", label="Max E_update")
    axes[0].set(xlabel="Counterfactual input scale c", ylabel="Relative activity", title="Movement from RMA")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].loglog(scales, [row["max_E_shrink"] for row in ordered], "o-", label="Max E_shrink")
    axes[1].semilogx(scales, [row["max_fraction_le_theta"] for row in ordered], "s-", label="Max fraction ≤ theta")
    axes[1].set(xlabel="Counterfactual input scale c", ylabel="Activity", title="Shrinkage activation")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def classify_root_cause(
    synthetic_summary, hammer_summary, synthetic_rows, hammer_rows, sweep_rows
):
    original = next(row for row in sweep_rows if row["label"] == "original_c1")
    aligned = next(row for row in sweep_rows if row["label"] == "1.0_c_align")
    activity_gain = aligned["final_E_RMA"] / max(original["final_E_RMA"], 1e-20)
    synth_reference = synthetic_summary["final_E_RMA"]["median"]
    scale_material = activity_gain >= 10.0 and aligned["final_E_RMA"] >= 0.1 * synth_reference
    synthetic_phase1 = [row for row in synthetic_rows if row["phase"] == 1]
    hammer_phase1 = next(
        row for row in hammer_rows
        if row["matlab_range_bin"] == 17 and row["phase"] == 1
    )
    phase1_grad_ratio = hammer_phase1["E_grad"] / max(
        np.mean([row["E_grad"] for row in synthetic_phase1]), 1e-20
    )
    phase1_residual_ratio = hammer_phase1["E_res"] / max(
        np.mean([row["E_res"] for row in synthetic_phase1]), 1e-20
    )
    phase1_shrink_ratio = hammer_phase1["E_shrink"] / max(
        np.mean([row["E_shrink"] for row in synthetic_phase1]), 1e-20
    )
    residual_degeneracy = (
        not scale_material
        and hammer_phase1["E_grad"] < 0.1 * np.mean(
            [row["E_grad"] for row in synthetic_phase1]
        )
    )
    if scale_material and residual_degeneracy:
        classification = "MULTIPLE_FACTORS_SUPPORTED"
    elif scale_material:
        classification = "SCALE_MISMATCH_SUPPORTED"
    elif residual_degeneracy:
        classification = "RESIDUAL_DEGENERACY_DOMINANT"
    elif aligned["final_E_RMA"] < 0.1 * synth_reference:
        classification = "SCALE_MISMATCH_NOT_SUFFICIENT / BROADER_DISTRIBUTION_MISMATCH_SUPPORTED"
    else:
        classification = "INCONCLUSIVE"
    return classification, {
        "aligned_to_original_final_E_RMA_ratio": activity_gain,
        "aligned_final_E_RMA_to_synthetic_median_ratio": aligned["final_E_RMA"] / max(synth_reference, 1e-20),
        "hammer_to_synthetic_phase1_E_grad_ratio": phase1_grad_ratio,
        "hammer_to_synthetic_phase1_E_res_ratio": phase1_residual_ratio,
        "hammer_to_synthetic_phase1_E_shrink_ratio": phase1_shrink_ratio,
        "classification_rule": "scale support requires >=10x aligned activity gain and aligned final E_RMA >=0.1x synthetic median; residual degeneracy is dominant only when scale is insufficient and Hammer phase-1 E_grad is <0.1x synthetic",
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_sim2real_diagnostic.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA diagnostic device is unavailable")
    output_dir = Path(config["output_dir"])
    review_dir = Path(config["review_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output_dir / "config_used.yaml")

    geometry = config["geometry"]
    base_operator = RMISTKernel(
        float(geometry["center_frequency_hz"]), float(geometry["dx_m"]),
        float(geometry["dz_m"]), int(geometry["grid_size"]),
        float(geometry["ky_epsilon"]),
    )
    hammer_model, provenance = load_accepted_model(
        base_operator, Path(config["model"]["summary_path"]),
        Path(config["model"]["checkpoint_path"]), device,
    )
    parameter_before = {name: value.detach().cpu().clone() for name, value in hammer_model.state_dict().items()}
    rtol = float(config["numerics"]["trace_output_rtol"])
    atol = float(config["numerics"]["trace_output_atol"])

    synthetic_cfg = config["synthetic"]
    synthetic_range = float(synthetic_cfg["range_m"])
    cached_operator = FixedRangeRMISTOperator(
        base_operator, synthetic_range, torch.complex64, device
    )
    synthetic_model = RMISTNet(cached_operator).to(device)
    synthetic_model.load_state_dict(hammer_model.state_dict(), strict=True)
    synthetic_model.eval()
    for parameter in synthetic_model.parameters():
        parameter.requires_grad_(False)
    full_mask = make_fixed_mask(
        (512, 512), float(config["mask"]["sampling_rate"]), int(config["mask"]["seed"])
    )
    actual_mask_hash = mask_sha256(full_mask)
    if actual_mask_hash != config["mask"]["expected_sha256"]:
        raise RuntimeError("accepted M6 mask hash mismatch")
    full_mask_device = full_mask.to(device)
    dataset = RMISTSyntheticDataset(
        pair_count=int(synthetic_cfg["subset_size"]), grid_size=512,
        scatterers_per_image=int(synthetic_cfg["scatterers_per_image"]),
        amplitude_min=float(synthetic_cfg["amplitude_min"]),
        amplitude_max=float(synthetic_cfg["amplitude_max"]),
        snr_db=float(synthetic_cfg["snr_db"]), range_m=synthetic_range,
        seed=int(synthetic_cfg["validation_seed"]),
    )
    synthetic_phase_rows = []
    synthetic_measurement_all = []
    synthetic_measurement_observed = []
    synthetic_alpha0_all = []
    trace_differences = []
    batch_size = int(synthetic_cfg["batch_size"])
    for start in range(0, dataset.pair_count, batch_size):
        indices = list(range(start, min(start + batch_size, dataset.pair_count)))
        _, measurement, _ = dataset.make_batch(indices, cached_operator, full_mask_device, device)
        output, alpha0, trace, difference = ordinary_trace_regression(
            synthetic_model, measurement, synthetic_range, rtol, atol
        )
        del output
        trace_differences.append(difference)
        metrics = phase_metrics(measurement, alpha0, trace, synthetic_model.theta)
        append_phase_rows(synthetic_phase_rows, metrics, "sample_index", indices)
        observed = full_mask_device.bool()
        for local_index in range(len(indices)):
            synthetic_measurement_all.append(magnitude_statistics(measurement[local_index]))
            synthetic_measurement_observed.append(magnitude_statistics(measurement[local_index][observed]))
            synthetic_alpha0_all.append(magnitude_statistics(alpha0[local_index]))
        del measurement, alpha0, trace, metrics

    hammer_cfg = config["hammer"]
    _, effective_mask, mask_summary = accepted_effective_mask(
        base_operator, (int(hammer_cfg["nz"]), int(hammer_cfg["nx"])),
        int(config["mask"]["seed"]), float(config["mask"]["sampling_rate"]),
        str(config["mask"]["expected_sha256"]),
    )
    echo, load_metadata = load_single_rx_aperture(
        Path(hammer_cfg["dataset_dir"]) / "output_rawdata.mat",
        rx_index=int(hammer_cfg["rx_index"]), nx=int(hammer_cfg["nx"]),
        nz=int(hammer_cfg["nz"]),
        motion_error_samples=int(hammer_cfg["motion_error_samples"]),
        serpentine_correct=bool(hammer_cfg["serpentine_correct"]),
    )
    focused = range_fft(echo, int(hammer_cfg["n_range_fft"]))
    bins = [int(value) for value in hammer_cfg["matlab_range_bins"]]
    ranges = [
        matlab_bin_range_m(
            matlab_bin, float(hammer_cfg["sample_rate_hz"]),
            float(hammer_cfg["chirp_slope_hz_s"]), int(hammer_cfg["n_range_fft"]),
            float(hammer_cfg["instrument_delay_s"]),
        )
        for matlab_bin in bins
    ]
    effective_device = effective_mask.to(device)
    hammer_phase_rows = []
    hammer_measurement_all = []
    hammer_measurement_observed = []
    hammer_alpha0_all = []
    primary_measurement = None
    primary_alpha0_rms = None
    for matlab_bin, range_m in zip(bins, ranges):
        physical = torch.from_numpy(focused[:, :, matlab_bin - 1]).to(device)
        padded = base_operator.pad_measurement(physical)
        measurement = padded * effective_device.to(padded.dtype)
        output, alpha0, trace, difference = ordinary_trace_regression(
            hammer_model, measurement, float(range_m), rtol, atol
        )
        del output
        trace_differences.append(difference)
        metrics = phase_metrics(measurement, alpha0, trace, hammer_model.theta)
        append_phase_rows(hammer_phase_rows, metrics, "matlab_range_bin", [matlab_bin])
        hammer_measurement_all.append(magnitude_statistics(measurement))
        hammer_measurement_observed.append(magnitude_statistics(measurement[effective_device.bool()]))
        alpha_stats = magnitude_statistics(alpha0)
        hammer_alpha0_all.append(alpha_stats)
        if matlab_bin == int(hammer_cfg["primary_matlab_bin"]):
            primary_measurement = measurement.detach().clone()
            primary_alpha0_rms = alpha_stats["rms"]
            primary_range = float(range_m)
        del padded, measurement, alpha0, trace, metrics

    synthetic_alpha0_rms_median = float(np.median([row["rms"] for row in synthetic_alpha0_all]))
    c_align = synthetic_alpha0_rms_median / primary_alpha0_rms
    scale_rows = []
    scales = [("original_c1", 1.0)] + [
        ("{}_c_align".format(relative), float(relative) * c_align)
        for relative in config["scale_sweep"]["relative_to_align"]
    ]
    for label, scale in scales:
        scaled = primary_measurement * scale
        output, alpha0, trace, difference = ordinary_trace_regression(
            hammer_model, scaled, primary_range, rtol, atol
        )
        trace_differences.append(difference)
        metrics = phase_metrics(scaled, alpha0, trace, hammer_model.theta)
        single_rows = []
        append_phase_rows(single_rows, metrics, "scale_label", [label])
        output_original_scale = output / scale
        reference_alpha0 = alpha0 / scale
        epsilon = torch.finfo(output.real.dtype).eps
        relative_from_rma = float(
            (torch.linalg.vector_norm(output_original_scale - reference_alpha0)
             / (torch.linalg.vector_norm(reference_alpha0) + epsilon)).cpu()
        )
        scale_rows.append(
            {
                "label": label,
                "scale": scale,
                "max_E_grad": max(row["E_grad"] for row in single_rows),
                "max_E_shrink": max(row["E_shrink"] for row in single_rows),
                "max_E_update": max(row["E_update"] for row in single_rows),
                "final_E_RMA": next(row["E_RMA"] for row in single_rows if row["phase"] == 9),
                "relative_output_from_original_scale_RMA": relative_from_rma,
                "max_fraction_le_theta": max(row["fraction_le_theta"] for row in single_rows),
                "max_fraction_le_2theta": max(row["fraction_le_2theta"] for row in single_rows),
                "max_fraction_le_5theta": max(row["fraction_le_5theta"] for row in single_rows),
                "trace_relative_difference": difference,
            }
        )
        del scaled, output, alpha0, trace, metrics

    phase_fields_synthetic = ["sample_index", "phase"] + list(METRIC_NAMES)
    phase_fields_hammer = ["matlab_range_bin", "phase"] + list(METRIC_NAMES)
    scale_fields = [
        "label", "scale", "max_E_grad", "max_E_shrink", "max_E_update",
        "final_E_RMA", "relative_output_from_original_scale_RMA",
        "max_fraction_le_theta", "max_fraction_le_2theta", "max_fraction_le_5theta",
        "trace_relative_difference",
    ]
    write_csv(review_dir / "phase_metrics_synthetic.csv", synthetic_phase_rows, phase_fields_synthetic)
    write_csv(review_dir / "phase_metrics_hammer.csv", hammer_phase_rows, phase_fields_hammer)
    write_csv(review_dir / "scale_sweep.csv", scale_rows, scale_fields)

    scale_statistics = {
        "statistic_definitions": {
            "rms": "sqrt(mean(|x|^2)) over stated pixels",
            "percentiles": "linear quantiles of |x| over stated pixels",
            "active_or_observed_support": "accepted binary measurement mask equals one",
        },
        "synthetic_32_samples": {
            "measurement_all_pixels": aggregate_stat_rows(synthetic_measurement_all),
            "measurement_observed_support": aggregate_stat_rows(synthetic_measurement_observed),
            "alpha0_all_pixels": aggregate_stat_rows(synthetic_alpha0_all),
        },
        "hammer_13_slices": {
            "measurement_all_pixels": aggregate_stat_rows(hammer_measurement_all),
            "measurement_observed_support": aggregate_stat_rows(hammer_measurement_observed),
            "alpha0_all_pixels": aggregate_stat_rows(hammer_alpha0_all),
        },
        "hammer_bin17": {
            "measurement_all_pixels": hammer_measurement_all[bins.index(17)],
            "measurement_observed_support": hammer_measurement_observed[bins.index(17)],
            "alpha0_all_pixels": hammer_alpha0_all[bins.index(17)],
        },
        "real_bin17_to_synthetic_median_ratios": {
            "measurement_all_pixels": {
                name: hammer_measurement_all[bins.index(17)][name]
                / max(np.median([row[name] for row in synthetic_measurement_all]), 1e-300)
                for name in STAT_NAMES
            },
            "measurement_observed_support": {
                name: hammer_measurement_observed[bins.index(17)][name]
                / max(np.median([row[name] for row in synthetic_measurement_observed]), 1e-300)
                for name in STAT_NAMES
            },
            "alpha0_all_pixels": {
                name: hammer_alpha0_all[bins.index(17)][name]
                / max(np.median([row[name] for row in synthetic_alpha0_all]), 1e-300)
                for name in STAT_NAMES
            },
        },
        "c_align": c_align,
    }
    (review_dir / "scale_statistics.json").write_text(
        json.dumps(scale_statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    synthetic_summary = phase_summary(synthetic_phase_rows)
    hammer_primary_summary = phase_summary(
        hammer_phase_rows, lambda row: row["matlab_range_bin"] == 17
    )
    hammer_all_summary = phase_summary(hammer_phase_rows)
    classification, classification_evidence = classify_root_cause(
        synthetic_summary, hammer_primary_summary, synthetic_phase_rows,
        hammer_phase_rows, scale_rows
    )
    plot_phase_activity(review_dir / "phase_activity.png", synthetic_phase_rows, hammer_phase_rows, 17)
    plot_threshold_activity(review_dir / "threshold_activity.png", synthetic_phase_rows, hammer_phase_rows, 17)
    plot_scale_sweep(review_dir / "scale_comparison.png", scale_rows)

    parameters_unchanged = all(
        torch.equal(value.detach().cpu(), parameter_before[name])
        for name, value in hammer_model.state_dict().items()
    )
    summary = {
        "status": "PASS",
        "diagnostic_only": True,
        "root_cause_classification": classification,
        "classification_evidence": classification_evidence,
        "model_provenance": provenance,
        "learned_parameters_unchanged": parameters_unchanged,
        "mask_sha256": actual_mask_hash,
        "mask_effective_physical_sampling_rate": mask_summary["effective_physical_sampling_rate"],
        "synthetic_subset_size": dataset.pair_count,
        "synthetic_alpha0_rms_median": synthetic_alpha0_rms_median,
        "hammer_bin17_alpha0_rms": primary_alpha0_rms,
        "hammer_to_synthetic_alpha0_rms_ratio": primary_alpha0_rms / synthetic_alpha0_rms_median,
        "c_align": c_align,
        "synthetic_phase_activity": synthetic_summary,
        "hammer_bin17_phase_activity": hammer_primary_summary,
        "hammer_all_slices_phase_activity": hammer_all_summary,
        "scale_sweep": scale_rows,
        "instrumentation_output_regression": {
            "status": "PASS",
            "maximum_complex_relative_difference": max(trace_differences),
            "rtol": rtol,
            "atol": atol,
        },
        "numerical_safety_epsilon": "torch.finfo(real dtype).eps added only to diagnostic denominators",
        "hammer": {
            "rx_index_python": load_metadata["rx_index_python"],
            "rx_index_matlab": load_metadata["rx_index_matlab"],
            "matlab_range_bins": bins,
            "range_m": ranges,
            "primary_bin": 17,
            "primary_range_m": primary_range,
        },
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "environment": environment_text().splitlines(),
        "production_normalization_added": False,
        "parameter_updates_performed": False,
    }
    if not parameters_unchanged:
        raise RuntimeError("accepted learned parameters changed during diagnosis")
    (review_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
