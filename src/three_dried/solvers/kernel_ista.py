"""Non-learned Kernel-ISTA for complex-valued holographic inversion."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


def complex_soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Shrink complex magnitude while preserving phase."""
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    magnitude = torch.abs(value)
    eps = torch.finfo(magnitude.dtype).eps
    scale = torch.clamp(magnitude - threshold, min=0.0) / (magnitude + eps)
    return value * scale


def estimate_lipschitz(
    operator,
    iterations: int,
    seed: int,
    dtype: torch.dtype = torch.complex64,
    device: str = "cpu",
) -> Tuple[float, List[float]]:
    """Estimate ``||A||_2^2`` by power iteration on ``A^H A``."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    vector = torch.complex(
        torch.randn(operator.image_shape, dtype=real_dtype, device=device, generator=generator),
        torch.randn(operator.image_shape, dtype=real_dtype, device=device, generator=generator),
    )
    vector = vector / torch.linalg.vector_norm(vector)
    history = []
    with torch.no_grad():
        for _ in range(iterations):
            normal_vector = operator.normal(vector)
            norm = torch.linalg.vector_norm(normal_vector)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("power iteration produced an invalid norm")
            rayleigh = torch.real(torch.vdot(vector.reshape(-1), normal_vector.reshape(-1)))
            history.append(float(rayleigh.cpu()))
            vector = normal_vector / norm
        normal_vector = operator.normal(vector)
        estimate = torch.real(torch.vdot(vector.reshape(-1), normal_vector.reshape(-1)))
    value = float(estimate.cpu())
    if not value > 0:
        raise FloatingPointError("non-positive Lipschitz estimate")
    return value, history


def objective_terms(operator, image: torch.Tensor, measurement: torch.Tensor, regularization: float) -> Dict[str, float]:
    residual = operator.forward(image) - measurement
    data_fidelity = 0.5 * torch.sum(torch.abs(residual) ** 2)
    l1_norm = torch.sum(torch.abs(image))
    objective = data_fidelity + regularization * l1_norm
    return {
        "objective": float(objective.detach().cpu()),
        "data_fidelity": float(data_fidelity.detach().cpu()),
        "l1_norm": float(l1_norm.detach().cpu()),
        "residual_norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
    }


@dataclass
class IstaResult:
    image: torch.Tensor
    history: List[Dict[str, float]]


def kernel_ista(
    operator,
    measurement: torch.Tensor,
    regularization: float,
    step_size: float,
    iterations: int,
    initial: Optional[torch.Tensor] = None,
) -> IstaResult:
    """Solve ``0.5||Ax-y||² + regularization*||x||_1`` with ISTA."""
    if regularization < 0 or step_size <= 0 or iterations < 0:
        raise ValueError("regularization/iterations must be nonnegative and step_size positive")
    if initial is None:
        image = torch.zeros(operator.image_shape, dtype=measurement.dtype, device=measurement.device)
    else:
        image = initial.clone()
    threshold = regularization * step_size
    history = []
    with torch.no_grad():
        initial_terms = objective_terms(operator, image, measurement, regularization)
        history.append({"iteration": 0, **initial_terms})
        for iteration in range(1, iterations + 1):
            residual = measurement - operator.forward(image)
            image = complex_soft_threshold(
                image + step_size * operator.adjoint(residual), threshold
            )
            terms = objective_terms(operator, image, measurement, regularization)
            if not all(torch.isfinite(torch.tensor(v)).item() for v in terms.values()):
                raise FloatingPointError("ISTA produced NaN or Inf")
            history.append({"iteration": iteration, **terms})
    return IstaResult(image=image, history=history)
