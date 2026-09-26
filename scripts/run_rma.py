#!/usr/bin/env python3
"""Run the reproducible single-Rx Hammer conventional RMA baseline."""

import argparse
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_dried.data.three_dried import load_single_rx_aperture
from three_dried.imaging.rma import image_coordinates_mm, reconstruct_volume


def db_normalized(array: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    magnitude = np.abs(array).astype(np.float64)
    peak = float(magnitude.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("reconstruction has no finite positive peak")
    return np.maximum(20.0 * np.log10(magnitude / peak + 1e-12), floor_db)


def environment_text() -> str:
    try:
        import torch
        torch_info = "PyTorch: {}\nCUDA runtime: {}\nCUDA available: {}\nGPUs: {}".format(
            torch.__version__, torch.version.cuda, torch.cuda.is_available(),
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
    except Exception as exc:
        torch_info = "PyTorch probe error: {!r}".format(exc)
    return "\n".join([
        "Python: " + platform.python_version(),
        "Platform: " + platform.platform(),
        "NumPy: " + np.__version__,
        "SciPy: " + scipy.__version__,
        "Matplotlib: " + matplotlib.__version__,
        torch_info,
    ]) + "\n"


def save_plane(path: Path, data_db: np.ndarray, extent, title: str, xlabel: str, ylabel: str, dynamic_range_db: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(data_db, origin="lower", extent=extent, aspect="equal", cmap="jet", vmin=-dynamic_range_db, vmax=0)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(image, ax=ax, label="Normalized magnitude (dB)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hammer.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    slices_dir = output / "representative_slices"
    slices_dir.mkdir(exist_ok=True)

    log = logging.getLogger("rma")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(output / "run.log", mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        log.addHandler(handler)

    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    shutil.copyfile(args.config, output / "config_used.yaml")
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")
    log.info("Command: %s", command)
    log.info("Selected receiver: Python index %d / MATLAB Rx%d", config["data"]["rx_index"], config["data"]["rx_index"] + 1)

    started = time.perf_counter()
    mat_path = Path(config["dataset_dir"]) / "output_rawdata.mat"
    echo, load_metadata = load_single_rx_aperture(
        mat_path,
        rx_index=int(config["data"]["rx_index"]),
        nx=int(config["data"]["nx"]),
        nz=int(config["data"]["nz"]),
        motion_error_samples=int(config["data"]["motion_error_samples"]),
        serpentine_correct=bool(config["data"]["serpentine_correct"]),
    )
    log.info("Single-Rx corrected echo: shape=%s dtype=%s finite=%s", echo.shape, echo.dtype, np.isfinite(echo).all())

    rma_cfg = config["rma"]
    params = {
        **config["radar"],
        **config["aperture"],
        "n_range_fft": rma_cfg["n_range_fft"],
        "n_fft_space": rma_cfg["n_fft_space"],
    }
    bins = [int(value) for value in rma_cfg["matlab_range_bins"]]
    volume, ranges_m = reconstruct_volume(echo, bins, params)
    elapsed = time.perf_counter() - started
    finite = bool(np.isfinite(volume).all())
    if not finite:
        raise FloatingPointError("RMA output contains NaN or Inf")
    np.save(output / "volume.npy", volume)

    x_mm, z_mm = image_coordinates_mm(
        int(rma_cfg["n_fft_space"]), config["aperture"]["dx_m"], config["aperture"]["dz_m"],
        rma_cfg["x_offset_mm"], rma_cfg["z_offset_mm"],
    )
    np.savez(output / "coordinates.npz", x_mm=x_mm, z_mm=z_mm, range_m=ranges_m, matlab_range_bins=np.asarray(bins))
    dynamic = float(rma_cfg["display_dynamic_range_db"])
    crop_half = float(rma_cfg["crop_size_mm"]) / 2.0
    x_keep = np.abs(x_mm) <= crop_half
    z_keep = np.abs(z_mm) <= crop_half
    x_roi = x_mm[x_keep]
    z_roi = z_mm[z_keep]

    # x-z MIP over physical range; this is the view shown by the supplied 2-D reference.
    mip = np.max(np.abs(volume), axis=0)
    mip_db = db_normalized(mip[np.ix_(z_keep, x_keep)])
    save_plane(output / "mip_xz.png", mip_db, [x_roi[0], x_roi[-1], z_roi[0], z_roi[-1]], "Hammer RMA: x-z maximum-intensity projection", "Horizontal x (mm)", "Vertical z (mm)", dynamic)

    representative = sorted(set([0, len(bins) // 3, len(bins) // 2, 2 * len(bins) // 3, len(bins) - 1]))
    for index in representative:
        save_plane(
            slices_dir / "bin_{:03d}_{:.1f}mm.png".format(bins[index], ranges_m[index] * 1e3),
            db_normalized(volume[index][np.ix_(z_keep, x_keep)]), [x_roi[0], x_roi[-1], z_roi[0], z_roi[-1]],
            "MATLAB range bin {} ({:.1f} mm)".format(bins[index], ranges_m[index] * 1e3),
            "Horizontal x (mm)", "Vertical z (mm)", dynamic,
        )

    volume_db = db_normalized(volume)
    threshold = float(rma_cfg["point_cloud_threshold_db"])
    roi_mask = np.zeros(volume.shape[1:], dtype=bool)
    roi_mask[np.ix_(z_keep, x_keep)] = True
    points = np.argwhere((volume_db >= threshold) & roi_mask[None, :, :])
    # Deterministic cap keeps the plotted artifact compact while metrics retain all points.
    if len(points) > 60_000:
        choice = np.random.choice(len(points), 60_000, replace=False)
        plot_points = points[choice]
    else:
        plot_points = points
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    if len(plot_points):
        ri, zi, xi = plot_points.T
        values = volume_db[ri, zi, xi]
        scatter = ax.scatter(x_mm[xi], ranges_m[ri] * 1e3, z_mm[zi], c=values, s=1.0, cmap="jet", vmin=threshold, vmax=0, alpha=0.35)
        fig.colorbar(scatter, ax=ax, shrink=0.65, label="Normalized magnitude (dB)")
    ax.set(xlabel="x (mm)", ylabel="range y (mm)", zlabel="z (mm)", title="Hammer RMA thresholded volume")
    fig.savefig(output / "volume_3d.png", dpi=160)
    plt.close(fig)

    peak_index = np.unravel_index(np.argmax(np.abs(volume)), volume.shape)
    peak_range_index, peak_z_index, peak_x_index = [int(value) for value in peak_index]
    metrics = {
        "status": "PASS" if finite else "FAILED",
        "runtime_seconds": elapsed,
        "raw_selected_shape": list(echo.shape),
        "range_focused_shape": [*echo.shape[:2], int(rma_cfg["n_range_fft"])],
        "volume_shape": list(volume.shape),
        "volume_dtype": str(volume.dtype),
        "finite": finite,
        "nan_count": int(np.isnan(volume).sum()),
        "inf_count": int(np.isinf(volume).sum()),
        "matlab_range_bins": bins,
        "range_m": ranges_m.tolist(),
        "x_range_mm": [float(x_mm[0]), float(x_mm[-1])],
        "z_range_mm": [float(z_mm[0]), float(z_mm[-1])],
        "visualization_roi_mm": {
            "x": [float(x_roi[0]), float(x_roi[-1])],
            "z": [float(z_roi[0]), float(z_roi[-1])],
        },
        "peak": {
            "matlab_range_bin": bins[peak_range_index],
            "range_m": float(ranges_m[peak_range_index]),
            "x_mm": float(x_mm[peak_x_index]),
            "z_mm": float(z_mm[peak_z_index]),
            "magnitude": float(np.abs(volume[peak_index])),
        },
        "point_count_at_threshold": int(len(points)),
        "point_cloud_threshold_db": threshold,
        "strong_return_range_m_at_threshold": (
            [float(ranges_m[points[:, 0]].min()), float(ranges_m[points[:, 0]].max())]
            if len(points) else None
        ),
        "load_metadata": load_metadata,
        "fft_normalization": "NumPy default (MATLAB-compatible: forward unnormalized, inverse 1/N)",
    }
    (output / "metrics_or_stats.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("RMA volume: shape=%s dtype=%s finite=%s", volume.shape, volume.dtype, finite)
    log.info("Physical coordinates: x=%s mm, z=%s mm, range=%s m", metrics["x_range_mm"], metrics["z_range_mm"], metrics["range_m"])
    log.info("Peak: %s", metrics["peak"])
    log.info("Runtime: %.3f s", elapsed)


if __name__ == "__main__":
    main()
