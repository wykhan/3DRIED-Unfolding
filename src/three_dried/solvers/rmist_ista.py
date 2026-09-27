"""Conventional strict RM-kernel ISTA from RMIST Eq. (16)--(19)."""

from typing import Dict, List, Optional, Tuple

import torch

from three_dried.imaging.rmist_kernel import RMISTKernel


def complex_soft_threshold(
    value: torch.Tensor, threshold: float, epsilon: float = 1e-12
) -> torch.Tensor:
    """RMIST magnitude shrinkage with one shared real/imaginary scale."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    magnitude = torch.abs(value)
    scale = torch.relu(magnitude - threshold) / (magnitude + epsilon)
    return value * scale


def rmist_ista(
    measurement: torch.Tensor,
    operator: RMISTKernel,
    range_m: float,
    tau: float = 1e-4,
    theta: float = 1e-3,
    iterations: int = 9,
    initial: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Apply RMIST Eq. (19) without a Hilbert-adjoint substitution.

    The default initialization is RMIST's inverse-kernel reconstruction,
    ``RM_dagger(S)``. The returned history is observational only and never
    changes an update.
    """
    if tau < 0 or theta < 0:
        raise ValueError("tau and theta must be non-negative")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if initial is None:
        alpha = operator.rm_dagger(measurement, range_m)
    else:
        if initial.shape != measurement.shape:
            raise ValueError("initial and measurement shapes must match")
        alpha = initial.clone()

    history = {"residual_norm": [], "estimate_norm": []}
    measurement_norm = torch.linalg.vector_norm(measurement)
    normalizer = torch.clamp(measurement_norm, min=torch.finfo(measurement.real.dtype).eps)
    for _ in range(iterations):
        # RMIST Eq. (19), preserved in the paper's stated order.
        residual = measurement - operator.rm(alpha, range_m)
        rough = alpha + tau * operator.rm_dagger(residual, range_m)
        alpha = complex_soft_threshold(rough, theta)
        history["residual_norm"].append(
            float((torch.linalg.vector_norm(residual) / normalizer).detach().cpu())
        )
        history["estimate_norm"].append(
            float(torch.linalg.vector_norm(alpha).detach().cpu())
        )
    return alpha, history
