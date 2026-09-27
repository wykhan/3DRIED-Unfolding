"""Small deterministic synthetic cases for strict RMIST validation."""

from typing import Tuple

import torch


def point_scatterer_scene(
    shape: Tuple[int, int], count: int, seed: int, dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """Create unit-amplitude point targets with phases uniform on [0, 2*pi)."""
    if count < 1 or count > shape[0] * shape[1]:
        raise ValueError("count must fit in the scene")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(shape[0] * shape[1], generator=generator)[:count]
    phases = 2.0 * torch.pi * torch.rand(count, generator=generator, dtype=torch.float64)
    values = torch.exp(1j * phases).to(dtype)
    scene = torch.zeros(shape, dtype=dtype)
    scene.reshape(-1)[indices] = values
    return scene


def random_mask(
    shape: Tuple[int, int], sampling_rate: float, seed: int, dtype=torch.float32
) -> torch.Tensor:
    """Return a deterministic Bernoulli spatial sampling mask."""
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1]")
    if sampling_rate == 1.0:
        return torch.ones(shape, dtype=dtype)
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator) < sampling_rate).to(dtype)


def add_complex_awgn(signal: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    """Add circular complex AWGN at a measured mean-power SNR."""
    generator = torch.Generator().manual_seed(seed)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    real_dtype = signal.real.dtype
    noise = torch.complex(
        torch.randn(signal.shape, generator=generator, dtype=real_dtype),
        torch.randn(signal.shape, generator=generator, dtype=real_dtype),
    ) * torch.sqrt(noise_power / 2.0)
    return signal + noise
