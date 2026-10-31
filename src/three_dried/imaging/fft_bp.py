"""Exact FFT evaluation of fixed-depth, phase-only planar backprojection.

On the shared regular image/aperture lattice, the spherical phase depends
only on coordinate differences. Linear convolution therefore evaluates the
same discrete sum as basic_bp.direct_slice_bp. There is no 1/R weighting,
sensitivity normalization, far-field approximation, or circular wraparound.
This is a narrowband slice operator, not FMCW range-migration BP.
"""
import numpy as np
from scipy import fft

from .basic_bp import C0


def phase_kernel(shape, dx_m, dz_m, depth_m, center_frequency_hz):
    if len(shape) != 2 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError('positive two-dimensional image shape required')
    if any(not np.isfinite(v) or v <= 0 for v in
           (dx_m, dz_m, depth_m, center_frequency_hz)):
        raise ValueError('positive finite spacings, depth and carrier required')
    nz, nx = map(int, shape)
    x = np.arange(1 - nx, nx) * dx_m
    z = np.arange(1 - nz, nz) * dz_m
    distance = np.sqrt(depth_m**2 + z[:, None]**2 + x[None, :]**2)
    return np.exp(-1j * (4 * np.pi * center_frequency_hz / C0) * distance)


class FFTSliceBP:
    """Cached complex128 CPU BP; input and output share a regular lattice.

    Embed the physical aperture at its accepted indices; multiply by its
    binary observation mask before calling. Zeros simply omit missing terms
    from the BP sum. No interpolation of missing aperture positions occurs.
    """
    def __init__(self, shape, dx_m, dz_m, depth_m, center_frequency_hz, workers=1):
        kernel = phase_kernel(shape, dx_m, dz_m, depth_m, center_frequency_hz)
        self.shape = tuple(map(int, shape))
        self.fft_shape = tuple(fft.next_fast_len(3 * n - 2) for n in self.shape)
        self.crop = tuple(slice(n - 1, 2 * n - 1) for n in self.shape)
        self.workers = workers
        self.kernel_fft = fft.fft2(kernel, s=self.fft_shape, norm='backward', workers=workers)

    def __call__(self, echo):
        echo = np.asarray(echo)
        if echo.shape != self.shape or not np.iscomplexobj(echo):
            raise ValueError('complex echo on the full padded image lattice required')
        spectrum = fft.fft2(echo, s=self.fft_shape, norm='backward', workers=self.workers)
        image = fft.ifft2(spectrum * self.kernel_fft, norm='backward', workers=self.workers)
        return image[self.crop].copy()


class TorchFFTSliceBP:
    """Same cached sum in PyTorch complex128; CUDA FFTs when device is CUDA.

    FFT backward normalization matches SciPy's linear convolution. Double
    precision is deliberate: acceleration is tested without lowering phase
    or accumulation precision. The returned image stays on the chosen device.
    """
    def __init__(self, shape, dx_m, dz_m, depth_m, center_frequency_hz, device='cuda:0'):
        import torch
        kernel = phase_kernel(shape, dx_m, dz_m, depth_m, center_frequency_hz)
        self.shape = tuple(map(int, shape))
        self.fft_shape = tuple(fft.next_fast_len(3 * n - 2) for n in self.shape)
        self.crop = tuple(slice(n - 1, 2 * n - 1) for n in self.shape)
        self.device = torch.device(device)
        self.kernel_fft = torch.fft.fft2(
            torch.as_tensor(kernel, device=self.device), s=self.fft_shape, norm='backward')

    def __call__(self, echo):
        import torch
        echo = torch.as_tensor(echo, device=self.device)
        if tuple(echo.shape) != self.shape or not torch.is_complex(echo):
            raise ValueError('complex echo on the full padded image lattice required')
        echo = echo.to(torch.complex128)
        spectrum = torch.fft.fft2(echo, s=self.fft_shape, norm='backward')
        image = torch.fft.ifft2(spectrum * self.kernel_fft, norm='backward')
        return image[self.crop].contiguous()
