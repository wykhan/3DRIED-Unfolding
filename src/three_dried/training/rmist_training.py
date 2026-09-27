"""Dataset, loss, metrics, and fixed-range cache for strict RMIST M6-R."""

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from three_dried.imaging.rmist_kernel import RMISTKernel


def make_fixed_mask(
    shape: Tuple[int, int], sampling_rate: float, seed: int
) -> torch.Tensor:
    """Create the one deterministic Bernoulli echo-domain mask for M6-R."""
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1]")
    generator = torch.Generator().manual_seed(int(seed))
    return (torch.rand(shape, generator=generator) < sampling_rate).to(torch.float32)


def mask_sha256(mask: torch.Tensor) -> str:
    """Hash a CPU copy of a binary mask for provenance."""
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


class FixedRangeRMISTOperator:
    """Cache accepted non-trainable RM kernels for one range/device/dtype.

    This is an implementation performance optimization only.  Kernel values
    are obtained directly from the accepted :class:`RMISTKernel` methods.
    """

    def __init__(
        self,
        operator: RMISTKernel,
        range_m: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.operator = operator
        self.range_m = float(range_m)
        self.n_fft_space = operator.n_fft_space
        self.phi = operator.phi(self.range_m, dtype=dtype, device=device)
        self.phi_dagger = operator.phi_dagger(
            self.range_m, dtype=dtype, device=device
        )
        self.phi.requires_grad_(False)
        self.phi_dagger.requires_grad_(False)

    def _validate(self, value: torch.Tensor, range_m: float, name: str) -> None:
        if float(range_m) != self.range_m:
            raise ValueError("cached operator is valid only at its fixed range")
        if not torch.is_complex(value):
            raise TypeError(name + " must be complex-valued")
        if value.shape[-2:] != (self.n_fft_space, self.n_fft_space):
            raise ValueError(name + " has the wrong spatial shape")
        if value.dtype != self.phi.dtype or value.device != self.phi.device:
            raise ValueError(name + " dtype/device does not match cached kernels")

    def rm(
        self,
        image: torch.Tensor,
        range_m: float,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self._validate(image, range_m, "image")
        echo = torch.fft.ifft2(torch.fft.fft2(image) * self.phi)
        if mask is not None:
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError("mask spatial shape must match the RM domain")
            echo = echo * mask.to(device=image.device, dtype=image.dtype)
        return echo

    def rm_dagger(self, padded_echo: torch.Tensor, range_m: float) -> torch.Tensor:
        self._validate(padded_echo, range_m, "padded_echo")
        return torch.fft.ifft2(torch.fft.fft2(padded_echo) * self.phi_dagger)


@dataclass(frozen=True)
class RMISTSyntheticDataset:
    """Procedural deterministic Eq. (28)--(31) oracle/echo pairs."""

    pair_count: int
    grid_size: int
    scatterers_per_image: int
    amplitude_min: float
    amplitude_max: float
    snr_db: float
    range_m: float
    seed: int

    def __post_init__(self) -> None:
        if self.pair_count < 1:
            raise ValueError("pair_count must be positive")
        if not 0 < self.scatterers_per_image <= self.grid_size**2:
            raise ValueError("scatterer count must fit in the oracle grid")
        if not 0.0 <= self.amplitude_min <= self.amplitude_max:
            raise ValueError("invalid amplitude interval")

    def _oracle(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.pair_count:
            raise IndexError("sample index outside dataset")
        rng = np.random.default_rng(self.seed + int(index))
        positions = rng.choice(
            self.grid_size * self.grid_size,
            size=self.scatterers_per_image,
            replace=False,
        )
        amplitudes = rng.uniform(
            self.amplitude_min, self.amplitude_max, self.scatterers_per_image
        )
        phases = rng.uniform(0.0, 2.0 * np.pi, self.scatterers_per_image)
        values = amplitudes * np.exp(1j * phases)
        oracle = np.zeros(self.grid_size * self.grid_size, dtype=np.complex64)
        oracle[positions] = values.astype(np.complex64)
        return torch.from_numpy(oracle.reshape(self.grid_size, self.grid_size))

    def oracle_batch(
        self, indices: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        return torch.stack([self._oracle(int(index)) for index in indices]).to(device)

    def make_batch(
        self,
        indices: Sequence[int],
        operator: FixedRangeRMISTOperator,
        mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fixed oracle labels, noisy echoes, and achieved sample SNRs."""
        truth = self.oracle_batch(indices, device)
        clean = operator.rm(truth, self.range_m) * mask.to(
            device=device, dtype=truth.dtype
        )
        noises = []
        for row, index in enumerate(indices):
            generator = torch.Generator(device=device).manual_seed(
                self.seed + 10_000_000 + int(index)
            )
            raw = torch.complex(
                torch.randn(
                    (self.grid_size, self.grid_size),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.randn(
                    (self.grid_size, self.grid_size),
                    generator=generator,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            signal_power = torch.mean(torch.abs(clean[row]) ** 2)
            raw_power = torch.mean(torch.abs(raw) ** 2)
            target_ratio = 10.0 ** (self.snr_db / 10.0)
            noises.append(raw * torch.sqrt(signal_power / (target_ratio * raw_power)))
        noise = torch.stack(noises)
        echo = clean + noise
        achieved = 10.0 * torch.log10(
            torch.mean(torch.abs(clean) ** 2, dim=(-2, -1))
            / torch.mean(torch.abs(noise) ** 2, dim=(-2, -1))
        )
        return truth, echo, achieved


def averaged_complex_l2_loss(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """RMIST displayed Eq. (27): mean of per-sample complex L2 norms."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target shapes must match")
    if not torch.is_complex(reconstruction) or not torch.is_complex(target):
        raise TypeError("Eq. (27) inputs must be complex-valued")
    if reconstruction.ndim < 3:
        raise ValueError("Eq. (27) expects a batch dimension")
    difference = (reconstruction - target).reshape(reconstruction.shape[0], -1)
    return torch.linalg.vector_norm(difference, dim=1).mean()


def batch_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Per-sample evaluation-only NRMSE, PSNR, and paper Eq. (33) TBR."""
    difference = (reconstruction - target).reshape(reconstruction.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    flat_reconstruction = reconstruction.reshape(reconstruction.shape[0], -1)
    error_norm = torch.linalg.vector_norm(difference, dim=1)
    target_norm = torch.linalg.vector_norm(flat_target, dim=1)
    nrmse = error_norm / torch.clamp(target_norm, min=1e-12)
    mse = torch.mean(torch.abs(difference) ** 2, dim=1)
    peak = torch.amax(torch.abs(flat_target), dim=1)
    psnr = 10.0 * torch.log10(
        torch.clamp(peak**2 / torch.clamp(mse, min=1e-20), min=1e-20)
    )
    support = torch.abs(flat_target) > 0
    target_energy = torch.sum(torch.abs(flat_reconstruction) ** 2 * support, dim=1)
    background_energy = torch.sum(
        torch.abs(flat_reconstruction) ** 2 * (~support), dim=1
    )
    tbr = 10.0 * torch.log10(
        torch.clamp(target_energy / torch.clamp(background_energy, min=1e-20), min=1e-20)
    )
    return {"nrmse": nrmse, "psnr_db": psnr, "tbr_db": tbr}
