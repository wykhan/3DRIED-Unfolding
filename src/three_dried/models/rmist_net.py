"""Strict nine-phase RMIST-Net from RMIST Eqs. (20)--(26).

The network keeps PyTorch complex tensors internally.  This is a
paper-unspecified representation choice equivalent to the paper's separate
real and imaginary paths: both components are processed by the same complex
RM kernels and share the magnitude shrinkage factor in every phase.
"""

from typing import Dict, List, Tuple, Union

import torch
from torch import nn

from three_dried.imaging.rmist_kernel import RMISTKernel


def complex_to_two_channel(value: torch.Tensor) -> torch.Tensor:
    """Map complex ``[..., H, W]`` data to real ``[..., 2, H, W]`` data."""
    if not torch.is_complex(value):
        raise TypeError("value must be complex-valued")
    if value.ndim < 2:
        raise ValueError("value must have at least two spatial dimensions")
    return torch.stack((value.real, value.imag), dim=-3)


def two_channel_to_complex(value: torch.Tensor) -> torch.Tensor:
    """Invert :func:`complex_to_two_channel` for ``[..., 2, H, W]`` data."""
    if torch.is_complex(value):
        raise TypeError("value must be real-valued")
    if value.ndim < 3 or value.shape[-3] != 2:
        raise ValueError("value must have a size-2 real/imaginary channel axis")
    return torch.complex(value.select(-3, 0), value.select(-3, 1))


class RMISTNet(nn.Module):
    """Literal architecture-only unfolding of RMIST Eq. (20).

    The strict M5-R model always contains nine phases.  Its only trainable
    tensors are the independently addressable real vectors ``tau`` and
    ``theta``.  The supplied RMIST operator remains fixed and is not an
    ``nn.Module`` parameter.
    """

    def __init__(
        self,
        operator: RMISTKernel,
        phases: int = 9,
        tau_init: float = 1e-4,
        theta_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if phases != 9:
            raise ValueError("strict RMIST-Net requires exactly 9 phases")
        self.operator = operator
        self.phases = phases
        self.tau = nn.Parameter(torch.full((phases,), float(tau_init)))
        self.theta = nn.Parameter(torch.full((phases,), float(theta_init)))

    def initialize(
        self, padded_measurement: torch.Tensor, range_m: float
    ) -> torch.Tensor:
        """Apply the exact Eq. (21) initialization ``RM_dagger(S)``."""
        return self.operator.rm_dagger(padded_measurement, range_m)

    def phase_update(
        self,
        alpha: torch.Tensor,
        padded_measurement: torch.Tensor,
        range_m: float,
        phase: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one literal Eq. (20) phase and return output plus residual."""
        if phase < 0 or phase >= self.phases:
            raise IndexError("phase index is outside the strict 9-phase model")

        # Stage 1, Eq. (20)/(23): residual error computation.
        residual = padded_measurement - self.operator.rm(alpha, range_m)
        # Stage 2, Eq. (20)/(24): rough estimate with phase-specific tau_t.
        rough = alpha + self.tau[phase] * self.operator.rm_dagger(
            residual, range_m
        )
        # Stage 3, Eq. (25)--(26): one magnitude scale shared by real/imag.
        magnitude = torch.abs(rough)
        scale = torch.relu(magnitude - self.theta[phase]) / (magnitude + 1e-12)
        return rough * scale, residual

    def forward(
        self,
        padded_measurement: torch.Tensor,
        range_m: float,
        return_trace: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, float]]]]:
        """Reconstruct one slice or a same-range batch with nine phases."""
        alpha = self.initialize(padded_measurement, range_m)
        trace: List[Dict[str, float]] = []
        for phase in range(self.phases):
            alpha, residual = self.phase_update(
                alpha, padded_measurement, range_m, phase
            )
            if return_trace:
                trace.append(
                    {
                        "phase": phase + 1,
                        "tau_t": float(self.tau[phase].detach().cpu()),
                        "theta_t": float(self.theta[phase].detach().cpu()),
                        "residual_norm": float(
                            torch.linalg.vector_norm(residual).detach().cpu()
                        ),
                        "estimate_norm": float(
                            torch.linalg.vector_norm(alpha).detach().cpu()
                        ),
                    }
                )
        if return_trace:
            return alpha, trace
        return alpha
