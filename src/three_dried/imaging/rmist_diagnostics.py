"""Non-invasive diagnostics for a fixed, accepted RMIST model."""

from typing import Dict, List, Tuple

import torch

from three_dried.models.rmist_net import RMISTNet


def magnitude_statistics(value: torch.Tensor) -> Dict[str, float]:
    """Return required magnitude-scale statistics over all supplied values."""
    if not torch.is_complex(value):
        raise TypeError("scale statistics require complex-valued data")
    magnitude = torch.abs(value).reshape(-1).to(torch.float64)
    quantiles = torch.quantile(
        magnitude, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64,
                                device=magnitude.device)
    )
    return {
        "rms": float(torch.sqrt(torch.mean(magnitude**2)).cpu()),
        "mean": float(torch.mean(magnitude).cpu()),
        "median": float(quantiles[0].cpu()),
        "p90": float(quantiles[1].cpu()),
        "p95": float(quantiles[2].cpu()),
        "p99": float(quantiles[3].cpu()),
        "max": float(torch.max(magnitude).cpu()),
    }


def trace_rmist_forward(
    model: RMISTNet,
    padded_measurement: torch.Tensor,
    range_m: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Trace every Eq. (20) term without calling or changing model state."""
    with torch.no_grad():
        alpha0 = model.initialize(padded_measurement, range_m)
        alpha = alpha0
        trace = []
        for phase in range(model.phases):
            alpha_before = alpha
            residual = padded_measurement - model.operator.rm(alpha_before, range_m)
            correction = model.tau[phase] * model.operator.rm_dagger(
                residual, range_m
            )
            rough = alpha_before + correction
            magnitude = torch.abs(rough)
            scale = torch.relu(magnitude - model.theta[phase]) / (
                magnitude + 1e-12
            )
            alpha_after = rough * scale
            trace.append(
                {
                    "phase": torch.tensor(phase + 1, device=alpha.device),
                    "alpha_before": alpha_before,
                    "residual": residual,
                    "correction": correction,
                    "rough": rough,
                    "alpha_after": alpha_after,
                }
            )
            alpha = alpha_after
    return alpha, alpha0, trace


def _per_sample_norm(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=1)


def phase_metrics(
    padded_measurement: torch.Tensor,
    alpha0: torch.Tensor,
    trace: List[Dict[str, torch.Tensor]],
    theta: torch.Tensor,
) -> List[Dict[str, torch.Tensor]]:
    """Compute all M8 phase diagnostics, retaining a per-sample dimension."""
    measurement = padded_measurement
    initial = alpha0
    if measurement.ndim == 2:
        measurement = measurement.unsqueeze(0)
        initial = initial.unsqueeze(0)
    epsilon = torch.finfo(measurement.real.dtype).eps
    measurement_norm = _per_sample_norm(measurement)
    initial_norm = _per_sample_norm(initial)
    rows = []
    for phase_index, record in enumerate(trace):
        values = {
            name: value if value.ndim > 2 else value.unsqueeze(0)
            for name, value in record.items()
            if name != "phase"
        }
        alpha_before = values["alpha_before"]
        residual = values["residual"]
        correction = values["correction"]
        rough = values["rough"]
        alpha_after = values["alpha_after"]
        rough_magnitude = torch.abs(rough).reshape(rough.shape[0], -1)
        percentiles = torch.quantile(
            rough_magnitude.to(torch.float64),
            torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64,
                         device=rough.device),
            dim=1,
        ).to(rough.real.dtype)
        threshold = theta[phase_index].detach().to(rough.real.dtype)
        rows.append(
            {
                "phase": torch.full(
                    (rough.shape[0],), phase_index + 1, dtype=torch.int64,
                    device=rough.device,
                ),
                "E_res": _per_sample_norm(residual) / (measurement_norm + epsilon),
                "E_grad": _per_sample_norm(correction)
                / (_per_sample_norm(alpha_before) + epsilon),
                "E_shrink": _per_sample_norm(alpha_after - rough)
                / (_per_sample_norm(rough) + epsilon),
                "E_update": _per_sample_norm(alpha_after - alpha_before)
                / (_per_sample_norm(alpha_before) + epsilon),
                "E_RMA": _per_sample_norm(alpha_after - initial)
                / (initial_norm + epsilon),
                "theta_over_p50": threshold / (percentiles[0] + epsilon),
                "theta_over_p90": threshold / (percentiles[1] + epsilon),
                "theta_over_p99": threshold / (percentiles[2] + epsilon),
                "fraction_le_theta": torch.mean(
                    (rough_magnitude <= threshold).to(rough.real.dtype), dim=1
                ),
                "fraction_le_2theta": torch.mean(
                    (rough_magnitude <= 2.0 * threshold).to(rough.real.dtype), dim=1
                ),
                "fraction_le_5theta": torch.mean(
                    (rough_magnitude <= 5.0 * threshold).to(rough.real.dtype), dim=1
                ),
            }
        )
    return rows
