"""Strict RMIST range-migration kernels mapped to the 3DRIED convention.

``rm_dagger`` is RMIST's inverse RM kernel, not a Hilbert adjoint.  The
operational kernels are fft-shifted to match the human-approved M2 RMA.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as torch_functional


C0 = 299_792_458.0


@dataclass(frozen=True)
class RMISTKernel:
    """RMIST Eq. (13)--(15) on a square 3DRIED spatial FFT grid."""

    center_frequency_hz: float
    dx_m: float
    dz_m: float
    n_fft_space: int = 512
    ky_epsilon: float = 1e-12

    def _geometry(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate geometry in float64 even for complex64 operation.  This
        # follows M2's NumPy construction and avoids phase error from forming
        # r*ky in float32 before the final kernel cast.
        k = 2.0 * torch.pi * self.center_frequency_hz / C0
        kx = torch.linspace(
            -torch.pi / self.dx_m,
            torch.pi / self.dx_m,
            self.n_fft_space,
            dtype=torch.float64,
            device=device,
        )
        kz = torch.linspace(
            -torch.pi / self.dz_m,
            torch.pi / self.dz_m,
            self.n_fft_space,
            dtype=torch.float64,
            device=device,
        )[:, None]
        radial_sq = (2.0 * k) ** 2 - kx[None, :] ** 2 - kz**2
        ky = torch.sqrt(torch.clamp(radial_sq, min=0.0))
        valid = (radial_sq >= 0.0) & (ky > self.ky_epsilon)
        return ky, valid

    def spectral_geometry(
        self,
        device: Optional[torch.device] = None,
        shifted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``ky`` and valid propagating support."""
        actual_device = torch.device("cpu") if device is None else torch.device(device)
        ky, valid = self._geometry(actual_device)
        if shifted:
            ky = torch.fft.fftshift(ky)
            valid = torch.fft.fftshift(valid)
        return ky, valid

    @staticmethod
    def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in (torch.complex64, torch.float32):
            return torch.complex64
        if dtype in (torch.complex128, torch.float64):
            return torch.complex128
        raise TypeError("RMIST kernels require float32/64 or complex64/128 dtype")

    def phi(
        self,
        range_m: float,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        shifted: bool = True,
    ) -> torch.Tensor:
        """3DRIED forward kernel ``ky^-1 * exp(+j*ky*r)``."""
        actual_device = torch.device("cpu") if device is None else torch.device(device)
        complex_dtype = self._complex_dtype(dtype)
        ky, valid = self._geometry(actual_device)
        safe_ky = torch.where(valid, ky, torch.ones_like(ky))
        kernel = torch.exp(1j * ky * float(range_m)) / safe_ky
        kernel = torch.where(valid, kernel, torch.zeros_like(kernel))
        kernel = kernel.to(complex_dtype)
        return torch.fft.fftshift(kernel) if shifted else kernel

    def phi_dagger(
        self,
        range_m: float,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        shifted: bool = True,
    ) -> torch.Tensor:
        """3DRIED inverse kernel ``ky * exp(-j*ky*r)``."""
        actual_device = torch.device("cpu") if device is None else torch.device(device)
        complex_dtype = self._complex_dtype(dtype)
        ky, valid = self._geometry(actual_device)
        kernel = ky * torch.exp(-1j * ky * float(range_m))
        kernel = torch.where(valid, kernel, torch.zeros_like(kernel))
        kernel = kernel.to(complex_dtype)
        return torch.fft.fftshift(kernel) if shifted else kernel

    def pad_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        """Center a physical ``[..., z, x]`` echo in the square RM domain."""
        if measurement.ndim < 2:
            raise ValueError("measurement must have at least two dimensions")
        nz, nx = measurement.shape[-2:]
        if nz > self.n_fft_space or nx > self.n_fft_space:
            raise ValueError("measurement exceeds the RM domain")
        before_z = (self.n_fft_space - nz) // 2
        before_x = (self.n_fft_space - nx) // 2
        after_z = self.n_fft_space - nz - before_z
        after_x = self.n_fft_space - nx - before_x
        return torch_functional.pad(
            measurement, (before_x, after_x, before_z, after_z)
        )

    def crop_measurement(
        self, padded_echo: torch.Tensor, physical_shape: Tuple[int, int] = (200, 407)
    ) -> torch.Tensor:
        """Invert centered padding for a requested physical ``(z, x)`` shape."""
        if padded_echo.shape[-2:] != (self.n_fft_space, self.n_fft_space):
            raise ValueError("padded_echo does not match the configured RM domain")
        nz, nx = physical_shape
        if nz > self.n_fft_space or nx > self.n_fft_space:
            raise ValueError("physical_shape exceeds the RM domain")
        before_z = (self.n_fft_space - nz) // 2
        before_x = (self.n_fft_space - nx) // 2
        return padded_echo[..., before_z : before_z + nz, before_x : before_x + nx]

    def rm(
        self,
        image: torch.Tensor,
        range_m: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply strict RMIST forward RM; an optional mask acts in echo space."""
        self._validate_domain(image, "image")
        kernel = self.phi(range_m, image.dtype, image.device)
        echo = torch.fft.ifft2(torch.fft.fft2(image) * kernel)
        if mask is not None:
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError("mask spatial shape must match the RM domain")
            echo = echo * mask.to(device=image.device, dtype=image.dtype)
        return echo

    def rm_dagger(self, padded_echo: torch.Tensor, range_m: float) -> torch.Tensor:
        """Apply RMIST's inverse RM kernel (not a Hilbert adjoint)."""
        self._validate_domain(padded_echo, "padded_echo")
        kernel = self.phi_dagger(
            range_m, padded_echo.dtype, padded_echo.device
        )
        return torch.fft.ifft2(torch.fft.fft2(padded_echo) * kernel)

    def _validate_domain(self, value: torch.Tensor, name: str) -> None:
        if not torch.is_complex(value):
            raise TypeError("{} must be complex-valued".format(name))
        if value.shape[-2:] != (self.n_fft_space, self.n_fft_space):
            raise ValueError(
                "{0} must end in [{1}, {1}]".format(name, self.n_fft_space)
            )
