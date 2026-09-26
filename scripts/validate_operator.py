#!/usr/bin/env python3
"""Validate the holography operator and its equivalence to M2 RMA."""

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
from three_dried.data.three_dried import load_single_rx_aperture
from three_dried.imaging.holography import HolographyGeometry, HolographyOperator
from three_dried.imaging.rma import matlab_bin_range_m, range_fft, reconstruct_volume


def random_complex(shape, dtype, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    return torch.complex(
        torch.randn(shape, dtype=real_dtype, device=device, generator=generator),
        torch.randn(shape, dtype=real_dtype, device=device, generator=generator),
    )


def adjoint_error(operator, x, y):
    lhs = torch.vdot(operator.forward(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), operator.adjoint(y).reshape(-1))
    return float((torch.abs(lhs - rhs) / torch.clamp(torch.abs(lhs), min=1e-12)).cpu())


def wrong_adjoint_error(operator, x, y):
    """Sensitivity control: deliberately omit conjugation in the forward kernel."""
    kernel = operator.spectral_kernel(x)
    broken_forward = operator._crop(operator._propagate(x, kernel))
    lhs = torch.vdot(broken_forward.reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), operator.adjoint(y).reshape(-1))
    return float((torch.abs(lhs - rhs) / torch.clamp(torch.abs(lhs), min=1e-12)).cpu())


def environment_text():
    return "\n".join([
        "Python: " + platform.python_version(),
        "Platform: " + platform.platform(),
        "PyTorch: " + torch.__version__,
        "CUDA runtime: " + str(torch.version.cuda),
        "CUDA available: " + str(torch.cuda.is_available()),
        "GPUs: " + str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
    ]) + "\n"


def normalized_db(value):
    magnitude = np.abs(value).astype(np.float64)
    return np.maximum(20 * np.log10(magnitude / magnitude.max() + 1e-12), -50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m3_operator.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(v) for v in [sys.executable] + sys.argv)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")
    shutil.copyfile(args.config, output / "config_used.yaml")

    image_shape = tuple(config["operator"]["image_shape"])
    aperture_shape = (int(config["data"]["nz"]), int(config["data"]["nx"]))
    reference_range = matlab_bin_range_m(
        17, config["radar"]["sample_rate_hz"], config["radar"]["chirp_slope_hz_s"],
        256, config["radar"]["instrument_delay_s"],
    )
    geometry = HolographyGeometry(
        image_shape=image_shape, aperture_shape=aperture_shape,
        range_m=reference_range, center_frequency_hz=config["radar"]["center_frequency_hz"],
        dx_m=config["aperture"]["dx_m"], dz_m=config["aperture"]["dz_m"],
    )
    operator = HolographyOperator(geometry)

    errors = []
    for dtype in (torch.complex64, torch.complex128):
        for seed in config["operator"]["adjoint_test_seeds"]:
            x = random_complex(image_shape, dtype, "cpu", int(seed))
            y = random_complex(aperture_shape, dtype, "cpu", int(seed) + 1000)
            errors.append({"device": "cpu", "dtype": str(dtype), "seed": seed,
                           "relative_error": adjoint_error(operator, x, y)})
    if torch.cuda.is_available():
        seed = int(config["operator"]["adjoint_test_seeds"][0])
        x = random_complex(image_shape, torch.complex64, "cuda", seed)
        y = random_complex(aperture_shape, torch.complex64, "cuda", seed + 1000)
        errors.append({"device": "cuda", "dtype": str(torch.complex64), "seed": seed,
                       "relative_error": adjoint_error(operator, x, y)})

    seed = int(config["operator"]["sensitivity_seed"])
    x = random_complex(image_shape, torch.complex128, "cpu", seed)
    y = random_complex(aperture_shape, torch.complex128, "cpu", seed + 1)
    sensitivity = {
        "correct_relative_error": adjoint_error(operator, x, y),
        "wrong_phase_relative_error": wrong_adjoint_error(operator, x, y),
    }

    started = time.perf_counter()
    echo, load_metadata = load_single_rx_aperture(
        Path(config["dataset_dir"]) / "output_rawdata.mat",
        rx_index=int(config["data"]["rx_index"]), nx=int(config["data"]["nx"]),
        nz=int(config["data"]["nz"]), motion_error_samples=int(config["data"]["motion_error_samples"]),
        serpentine_correct=bool(config["data"]["serpentine_correct"]),
    )
    spectrum = range_fft(echo, 256)
    bins = [int(v) for v in config["operator"]["matlab_range_bins"]]
    rma_params = {**config["radar"], **config["aperture"], "n_range_fft": 256,
                  "n_fft_space": image_shape[0]}
    m2_fresh, ranges = reconstruct_volume(echo, bins, rma_params)
    operator_volume = np.empty_like(m2_fresh)
    for index, (matlab_bin, range_m) in enumerate(zip(bins, ranges)):
        slice_operator = HolographyOperator(HolographyGeometry(
            image_shape=image_shape, aperture_shape=aperture_shape, range_m=float(range_m),
            center_frequency_hz=config["radar"]["center_frequency_hz"],
            dx_m=config["aperture"]["dx_m"], dz_m=config["aperture"]["dz_m"],
        ))
        measurement = torch.from_numpy(spectrum[:, :, matlab_bin - 1])
        operator_volume[index] = slice_operator.adjoint(measurement).numpy()
    elapsed = time.perf_counter() - started

    reference_path = Path(config["m2_results_dir"]) / "volume.npy"
    m2_saved = np.load(reference_path) if reference_path.exists() else m2_fresh
    complex_relative = float(np.linalg.norm(operator_volume - m2_saved) / np.linalg.norm(m2_saved))
    magnitude_relative = float(np.linalg.norm(np.abs(operator_volume) - np.abs(m2_saved)) / np.linalg.norm(np.abs(m2_saved)))
    correlation = float(np.corrcoef(np.abs(operator_volume).ravel(), np.abs(m2_saved).ravel())[0, 1])
    alpha = np.vdot(operator_volume.ravel(), m2_saved.ravel()) / np.vdot(operator_volume.ravel(), operator_volume.ravel())
    aligned_relative = float(np.linalg.norm(alpha * operator_volume - m2_saved) / np.linalg.norm(m2_saved))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    reference_peak = float(np.abs(m2_saved).max())
    images_db = [
        normalized_db(np.max(np.abs(m2_saved), axis=0)),
        normalized_db(np.max(np.abs(operator_volume), axis=0)),
        np.maximum(20 * np.log10(np.max(np.abs(operator_volume - m2_saved), axis=0) / reference_peak + 1e-15), -140),
    ]
    titles = ["M2 RMA MIP", "M3 adjoint MIP", "difference relative to M2 peak"]
    for index, (axis, image_db, title) in enumerate(zip(axes, images_db, titles)):
        vmin = -40 if index < 2 else -140
        shown = axis.imshow(image_db, origin="lower", cmap="jet", vmin=vmin, vmax=0)
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("z index")
        fig.colorbar(shown, ax=axis, label="dB", shrink=0.8)
    fig.savefig(output / "m2_consistency.png", dpi=150)
    plt.close(fig)

    metrics = {
        "status": "PASS" if max(item["relative_error"] for item in errors) < 1e-5 and complex_relative < 1e-5 else "FAILED",
        "adjoint_tests": errors,
        "max_adjoint_relative_error": max(item["relative_error"] for item in errors),
        "sensitivity_control": sensitivity,
        "m2_consistency": {
            "complex_relative_error_unaligned": complex_relative,
            "magnitude_relative_error": magnitude_relative,
            "magnitude_pearson_correlation": correlation,
            "least_squares_global_scalar_real": float(alpha.real),
            "least_squares_global_scalar_imag": float(alpha.imag),
            "complex_relative_error_after_scalar_alignment": aligned_relative,
            "m2_source": str(reference_path if reference_path.exists() else "fresh reconstruction"),
            "bins": bins,
            "ranges_m": ranges.tolist(),
        },
        "runtime_seconds": elapsed,
        "load_metadata": load_metadata,
        "geometry": config,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise SystemExit("M3 validation failed")


if __name__ == "__main__":
    main()
