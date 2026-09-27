#!/usr/bin/env python3
"""Validate strict RMIST kernels, including equivalence with Hammer M2."""

import argparse
import json
import platform
import shlex
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_dried.data.three_dried import load_single_rx_aperture
from three_dried.imaging.rma import matlab_bin_range_m, range_fft
from three_dried.imaging.rmist_kernel import RMISTKernel


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rmist_kernel.yaml"))
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

    rm_cfg = config["rmist"]
    operator = RMISTKernel(
        config["radar"]["center_frequency_hz"],
        config["aperture"]["dx_m"],
        config["aperture"]["dz_m"],
        int(rm_cfg["n_fft_space"]),
        float(rm_cfg["ky_epsilon"]),
    )
    test_range_m = matlab_bin_range_m(
        17,
        config["radar"]["sample_rate_hz"],
        config["radar"]["chirp_slope_hz_s"],
        int(rm_cfg["n_range_fft"]),
        config["radar"]["instrument_delay_s"],
    )
    started = time.perf_counter()
    ky, support = operator.spectral_geometry(shifted=True)
    reciprocal = operator.phi(test_range_m, torch.complex128) * operator.phi_dagger(
        test_range_m, torch.complex128
    )
    reciprocal_error = torch.abs(reciprocal[support] - 1.0)

    synthetic = {}
    generator = torch.Generator().manual_seed(seed)
    for dtype, name in ((torch.complex64, "complex64"), (torch.complex128, "complex128")):
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        spectrum = torch.complex(
            torch.randn(512, 512, generator=generator, dtype=real_dtype),
            torch.randn(512, 512, generator=generator, dtype=real_dtype),
        )
        scene = torch.fft.ifft2(spectrum * support)
        recovered = operator.rm_dagger(operator.rm(scene, test_range_m), test_range_m)
        synthetic[name] = float(
            torch.linalg.vector_norm(recovered - scene)
            / torch.linalg.vector_norm(scene)
        )

    padding_probe = torch.arange(200 * 407).reshape(200, 407)
    padded_probe = operator.pad_measurement(padding_probe)
    padding_exact = bool(
        torch.equal(padded_probe[156:356, 52:459], padding_probe)
        and torch.equal(operator.crop_measurement(padded_probe), padding_probe)
    )

    data_cfg = config["data"]
    echo, _ = load_single_rx_aperture(
        Path(config["dataset_dir"]) / "output_rawdata.mat",
        rx_index=int(data_cfg["rx_index"]),
        nx=int(data_cfg["nx"]),
        nz=int(data_cfg["nz"]),
        motion_error_samples=int(data_cfg["motion_error_samples"]),
        serpentine_correct=bool(data_cfg["serpentine_correct"]),
    )
    focused = range_fft(echo, int(rm_cfg["n_range_fft"]))
    bins = [int(value) for value in rm_cfg["matlab_range_bins"]]
    expected_volume = np.load(config["m2_volume"], mmap_mode="r")
    reproduced = np.empty_like(expected_volume)
    per_slice = []
    for index, matlab_bin in enumerate(bins):
        range_m = matlab_bin_range_m(
            matlab_bin,
            config["radar"]["sample_rate_hz"],
            config["radar"]["chirp_slope_hz_s"],
            int(rm_cfg["n_range_fft"]),
            config["radar"]["instrument_delay_s"],
        )
        measurement = torch.from_numpy(focused[:, :, matlab_bin - 1])
        padded = operator.pad_measurement(measurement)
        actual = operator.rm_dagger(padded, range_m).numpy().astype(np.complex64)
        reproduced[index] = actual
        per_slice.append(relative_error(actual, expected_volume[index]))
    np.save(output / "hammer_rm_dagger_volume.npy", reproduced)
    m2_relative = relative_error(reproduced, expected_volume)

    cuda = {"available": torch.cuda.is_available(), "passed": None}
    if torch.cuda.is_available():
        probe = torch.zeros(512, 512, dtype=torch.complex64, device="cuda")
        probe[256, 256] = 1.0 + 1.0j
        cuda_result = operator.rm_dagger(operator.rm(probe, test_range_m), test_range_m)
        cuda["passed"] = bool(torch.isfinite(cuda_result).all().item())

    metrics = {
        "status": "PASS"
        if (
            float(reciprocal_error.max()) < 1e-12
            and synthetic["complex64"] < 2e-5
            and synthetic["complex128"] < 1e-12
            and m2_relative < 1e-5
            and padding_exact
            and (cuda["passed"] is not False)
        )
        else "FAILED",
        "runtime_seconds": time.perf_counter() - started,
        "test_range_m": test_range_m,
        "fft_normalization": "backward (MATLAB-compatible; forward unnormalized, inverse 1/N)",
        "support": {
            "valid_count": int(support.sum()),
            "invalid_count": int((~support).sum()),
            "ky_min_valid": float(ky[support].min()),
            "ky_epsilon": operator.ky_epsilon,
        },
        "reciprocal": {
            "max_absolute_error": float(reciprocal_error.max()),
            "rms_absolute_error": float(torch.sqrt(torch.mean(reciprocal_error**2))),
        },
        "synthetic_full_domain_round_trip_relative_error": synthetic,
        "padding_200x407_to_512x512_exact": padding_exact,
        "padding_offsets": {"z_before_after": [156, 156], "x_before_after": [52, 53]},
        "m2_hammer": {
            "complex_relative_difference": m2_relative,
            "per_slice_relative_difference": per_slice,
            "target": 1e-5,
            "bins": bins,
        },
        "cuda_smoke": cuda,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "PASS":
        raise SystemExit("M3-R validation failed")


if __name__ == "__main__":
    main()
