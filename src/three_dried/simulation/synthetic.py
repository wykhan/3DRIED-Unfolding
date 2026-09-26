"""Reproducible complex sparse scenes for conventional solver validation."""

from typing import Dict, Tuple

import torch


def sparse_complex_scene(
    shape: Tuple[int, int],
    num_scatterers: int,
    seed: int,
    margin: int = 4,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Create independent point scatterers with random amplitude and phase."""
    nz, nx = shape
    if margin < 0 or nz <= 2 * margin or nx <= 2 * margin:
        raise ValueError("margin leaves no valid scene support")
    available = (nz - 2 * margin) * (nx - 2 * margin)
    if not 0 < num_scatterers <= available:
        raise ValueError("invalid number of scatterers")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    flat = torch.randperm(available, generator=generator)[:num_scatterers]
    width = nx - 2 * margin
    z_index = flat // width + margin
    x_index = flat % width + margin
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    amplitudes = 0.5 + 0.5 * torch.rand(num_scatterers, dtype=real_dtype, generator=generator)
    phases = 2.0 * torch.pi * torch.rand(num_scatterers, dtype=real_dtype, generator=generator)
    values = amplitudes.to(dtype) * torch.exp(1j * phases)
    scene = torch.zeros(shape, dtype=dtype)
    scene[z_index, x_index] = values
    metadata = {
        "seed": seed,
        "shape": list(shape),
        "num_scatterers": num_scatterers,
        "margin": margin,
        "positions_zx": [[int(z), int(x)] for z, x in zip(z_index, x_index)],
        "amplitudes": [float(v) for v in amplitudes],
        "phases_rad": [float(v) for v in phases],
    }
    return scene, metadata
