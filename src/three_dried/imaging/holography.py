"""Reusable near-field plane-wave holography operator.

The adjoint is intentionally the validated M2 RMA slice reconstruction:

    A^H y = F^-1 [ H F(P y) ]

where P center-pads the physical aperture and
``H = fftshift(ky * exp(-1j * range_m * ky))``.  Consequently, the exact
forward operator is its Hilbert adjoint:

    A x = P^H F^-1 [ conj(H) F(x) ].

PyTorch's explicit ``norm="backward"`` convention matches NumPy/MATLAB's
unnormalized forward FFT and inverse FFT scaled by the number of samples.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as functional


C0 = 299_792_458.0


@dataclass(frozen=True)
class HolographyGeometry:
    """Physical geometry for one fixed-range holographic slice."""

    image_shape: Tuple[int, int]
    aperture_shape: Tuple[int, int]
    range_m: float
    center_frequency_hz: float
    dx_m: float
    dz_m: float


class HolographyOperator:
    """Linear complex image-to-aperture operator and its exact adjoint."""

    def __init__(self, geometry: HolographyGeometry) -> None:
        self.geometry = geometry
        image_z, image_x = geometry.image_shape
        aperture_z, aperture_x = geometry.aperture_shape
        if min(image_z, image_x, aperture_z, aperture_x) <= 0:
            raise ValueError("all spatial dimensions must be positive")
        if aperture_z > image_z or aperture_x > image_x:
            raise ValueError("aperture_shape must fit inside image_shape")
        if geometry.range_m <= 0 or geometry.center_frequency_hz <= 0:
            raise ValueError("range and center frequency must be positive")
        if geometry.dx_m <= 0 or geometry.dz_m <= 0:
            raise ValueError("spatial sampling intervals must be positive")

        self._before_z = (image_z - aperture_z) // 2
        self._before_x = (image_x - aperture_x) // 2
        self._after_z = image_z - aperture_z - self._before_z
        self._after_x = image_x - aperture_x - self._before_x

    @property
    def image_shape(self) -> Tuple[int, int]:
        return self.geometry.image_shape

    @property
    def measurement_shape(self) -> Tuple[int, int]:
        return self.geometry.aperture_shape

    def _check_complex(self, value: torch.Tensor, expected_shape: Tuple[int, int], name: str) -> None:
        if not torch.is_complex(value):
            raise TypeError("{} must be a complex tensor".format(name))
        if tuple(value.shape[-2:]) != expected_shape:
            raise ValueError(
                "{} trailing shape {} does not match {}".format(
                    name, tuple(value.shape[-2:]), expected_shape
                )
            )

    def spectral_kernel(self, reference: torch.Tensor) -> torch.Tensor:
        """Return M2's shifted propagation/obliquity kernel."""
        if reference.dtype not in (torch.complex64, torch.complex128):
            raise TypeError("operator supports complex64 and complex128")
        # M2 constructs the physical wavenumber grid in NumPy float64 even
        # when the measured echo is complex64.  Keeping this calculation in
        # float64 avoids range*ky phase loss, then the final kernel is cast to
        # the requested complex dtype so operator I/O dtype is preserved.
        real_dtype = torch.float64
        nz, nx = self.image_shape
        kx = torch.linspace(
            -torch.pi / self.geometry.dx_m,
            torch.pi / self.geometry.dx_m,
            nx,
            dtype=real_dtype,
            device=reference.device,
        )
        kz = torch.linspace(
            -torch.pi / self.geometry.dz_m,
            torch.pi / self.geometry.dz_m,
            nz,
            dtype=real_dtype,
            device=reference.device,
        )[:, None]
        k = 2.0 * torch.pi * self.geometry.center_frequency_hz / C0
        radial_sq = (2.0 * k) ** 2 - kx[None, :] ** 2 - kz**2
        propagating = radial_sq >= 0.0
        ky = torch.sqrt(torch.clamp(radial_sq, min=0.0))
        kernel = ky.to(torch.complex128) * torch.exp(
            -1j * self.geometry.range_m * ky
        )
        kernel = torch.where(propagating, kernel, torch.zeros_like(kernel))
        return torch.fft.fftshift(kernel, dim=(-2, -1)).to(reference.dtype)

    @staticmethod
    def _propagate(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(value, dim=(-2, -1), norm="backward")
        return torch.fft.ifft2(spectrum * kernel, dim=(-2, -1), norm="backward")

    def _pad(self, measurement: torch.Tensor) -> torch.Tensor:
        return functional.pad(
            measurement,
            (self._before_x, self._after_x, self._before_z, self._after_z),
        )

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        z0, x0 = self._before_z, self._before_x
        nz, nx = self.measurement_shape
        return image[..., z0 : z0 + nz, x0 : x0 + nx]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map a full image plane to the centered physical aperture."""
        self._check_complex(image, self.image_shape, "image")
        kernel = self.spectral_kernel(image)
        return self._crop(self._propagate(image, torch.conj(kernel)))

    def adjoint(self, measurement: torch.Tensor) -> torch.Tensor:
        """Backproject an aperture measurement exactly as the M2 RMA slice."""
        self._check_complex(measurement, self.measurement_shape, "measurement")
        padded = self._pad(measurement)
        kernel = self.spectral_kernel(padded)
        return self._propagate(padded, kernel)

    def normal(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A``."""
        return self.adjoint(self.forward(image))
