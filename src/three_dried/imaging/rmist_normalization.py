"""Deterministic inference-only scale normalization for accepted RMIST."""

from typing import Tuple

import numpy as np
import torch

from three_dried.models.rmist_net import RMISTNet


def rms_magnitude(value: torch.Tensor) -> float:
    """Return ``sqrt(mean(|value|^2))`` as a host scalar."""
    magnitude = torch.abs(value).reshape(-1).to(torch.float64)
    return float(torch.sqrt(torch.mean(magnitude.square())).detach().cpu())


def p95_magnitude(value: torch.Tensor) -> float:
    """Return the linear 95th percentile of magnitude as a host scalar."""
    magnitude = torch.abs(value).reshape(-1).to(torch.float64)
    return float(torch.quantile(magnitude, 0.95).detach().cpu())


def positive_scale(reference: float, observed: float) -> float:
    """Compute an unclipped positive scale, rejecting invalid inputs."""
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference statistic must be finite and positive")
    if not np.isfinite(observed) or observed <= 0.0:
        raise ValueError("observed statistic must be finite and positive")
    epsilon = np.finfo(np.float64).eps
    scale = float(reference / (observed + epsilon))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization scale must be finite and positive")
    return scale


def normalized_rmist(
    model: RMISTNet,
    measurement: torch.Tensor,
    range_m: float,
    scale: float,
    return_scaled_output: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Apply ``RMIST(c*S)/c`` without changing the accepted model."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization scale must be finite and positive")
    scaled_output = model(measurement * scale, range_m)
    output = scaled_output / scale
    if return_scaled_output:
        return output, scaled_output
    return output


def magnitude_proxy(reference: np.ndarray, estimate: np.ndarray) -> Tuple[float, float]:
    """Return magnitude cosine and unit-L2 magnitude distance."""
    reference_mag = np.abs(reference).astype(np.float64, copy=False).ravel()
    estimate_mag = np.abs(estimate).astype(np.float64, copy=False).ravel()
    reference_norm = float(np.linalg.norm(reference_mag))
    estimate_norm = float(np.linalg.norm(estimate_mag))
    if reference_norm <= 0.0 or estimate_norm <= 0.0:
        raise ValueError("proxy inputs must have nonzero magnitude norm")
    cosine = float(np.dot(reference_mag, estimate_mag) / (reference_norm * estimate_norm))
    distance = float(np.linalg.norm(reference_mag / reference_norm - estimate_mag / estimate_norm))
    return cosine, distance


def normalized_range_profile(volume: np.ndarray) -> np.ndarray:
    """Return the unit-L2 vector of per-slice magnitude energies."""
    profile = np.linalg.norm(np.abs(volume).reshape(volume.shape[0], -1), axis=1)
    norm = float(np.linalg.norm(profile))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("range profile must be finite and nonzero")
    return profile / norm
