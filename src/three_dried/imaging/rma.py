"""Reference-compatible 2-D phase-compensation RMA applied by range bin."""

from typing import Dict, Iterable, Tuple

import numpy as np


C0 = 299_792_458.0


def range_fft(echo: np.ndarray, n_fft: int = 256) -> np.ndarray:
    """Range FFT over complex fast time, preserving phase."""
    if echo.ndim != 3:
        raise ValueError("echo must have shape [vertical, horizontal, fast_time]")
    return np.fft.fft(echo, n=n_fft, axis=-1).astype(np.complex64)


def matlab_bin_range_m(
    matlab_bin: int,
    sample_rate_hz: float,
    chirp_slope_hz_s: float,
    n_range_fft: int,
    instrument_delay_s: float,
) -> float:
    """Range convention used verbatim by the bundled MATLAB RMA demo."""
    return C0 / 2.0 * (
        matlab_bin * sample_rate_hz / (chirp_slope_hz_s * n_range_fft)
        - instrument_delay_s
    )


def reconstruct_slice(
    range_focused_slice: np.ndarray,
    range_m: float,
    center_frequency_hz: float,
    dx_m: float,
    dz_m: float,
    n_fft_space: int = 512,
) -> np.ndarray:
    """Reproduce the bundled MATLAB plane-wave-spectrum phase compensation."""
    if range_focused_slice.ndim != 2:
        raise ValueError("slice must have shape [vertical, horizontal]")
    nz, nx = range_focused_slice.shape
    if nz > n_fft_space or nx > n_fft_space:
        raise ValueError("n_fft_space must cover both aperture axes")

    k = 2.0 * np.pi * center_frequency_hz / C0
    kx = np.linspace(-np.pi / dx_m, np.pi / dx_m, n_fft_space)
    kz = np.linspace(-np.pi / dz_m, np.pi / dz_m, n_fft_space)[:, None]
    radial_sq = (2.0 * k) ** 2 - kx[None, :] ** 2 - kz**2
    propagating = radial_sq >= 0.0
    ky = np.sqrt(np.maximum(radial_sq, 0.0))
    phase = ky * np.exp(-1j * range_m * ky)
    phase[~propagating] = 0.0
    phase = np.fft.fftshift(phase)

    before_z = (n_fft_space - nz) // 2
    before_x = (n_fft_space - nx) // 2
    padded = np.pad(
        range_focused_slice,
        ((before_z, n_fft_space - nz - before_z),
         (before_x, n_fft_space - nx - before_x)),
    )
    # NumPy's default convention matches MATLAB: unnormalized FFT and 1/N IFFT.
    image = np.fft.ifft2(np.fft.fft2(padded) * phase)
    return image.astype(np.complex64)


def image_coordinates_mm(
    n_fft_space: int, dx_m: float, dz_m: float, x_offset_mm: float, z_offset_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_fft_space) - (n_fft_space - 1) / 2.0
    x_mm = indices * dx_m * 1e3 + x_offset_mm
    z_mm = indices * dz_m * 1e3 + z_offset_mm
    return x_mm, z_mm


def reconstruct_volume(
    echo: np.ndarray,
    matlab_bins: Iterable[int],
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack complex RMA slices as ``[range_y, vertical_z, horizontal_x]``."""
    spectrum = range_fft(echo, int(params["n_range_fft"]))
    bins = list(matlab_bins)
    volume = np.empty(
        (len(bins), int(params["n_fft_space"]), int(params["n_fft_space"])),
        dtype=np.complex64,
    )
    ranges = np.empty(len(bins), dtype=np.float64)
    for out_index, matlab_bin in enumerate(bins):
        if matlab_bin < 1 or matlab_bin > spectrum.shape[-1]:
            raise ValueError("MATLAB range bin outside FFT")
        ranges[out_index] = matlab_bin_range_m(
            matlab_bin,
            params["sample_rate_hz"],
            params["chirp_slope_hz_s"],
            int(params["n_range_fft"]),
            params["instrument_delay_s"],
        )
        volume[out_index] = reconstruct_slice(
            spectrum[:, :, matlab_bin - 1],
            ranges[out_index],
            params["center_frequency_hz"],
            params["dx_m"],
            params["dz_m"],
            int(params["n_fft_space"]),
        )
    return volume, ranges

