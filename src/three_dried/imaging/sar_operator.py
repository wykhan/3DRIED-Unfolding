"""Differentiable measured RM and its Hilbert adjoint, for generic ISTA only.

The strict RMIST RM_dagger implementation is intentionally independent.
One constant scales both A and y to RMS spectral gain one; image units stay
unchanged. This rescales the least-squares objective, not its minimizers.
"""
import torch
from torch import nn

from three_dried.imaging.rmist_kernel import RMISTKernel


class MeasuredSAR(nn.Module):
    def __init__(self, kernel, range_m, mask, dtype=torch.complex64, device='cpu'):
        super().__init__()
        phi = kernel.phi(range_m, dtype=dtype, device=torch.device(device))
        mask = torch.as_tensor(mask, device=device)
        if tuple(mask.shape) != tuple(phi.shape) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('binary mask must cover the padded RM grid')
        self.register_buffer('mask', mask.bool())
        self.register_buffer('gain', phi.abs().square().mean().sqrt().reciprocal())
        self.register_buffer('phi', phi * self.gain)
        self.register_buffer('lipschitz', self.phi.abs().square().max())
        self.range_m = float(range_m)

    @staticmethod
    def apply(value, multiplier):
        if not torch.is_complex(value):
            raise TypeError('SAR data must retain real and imaginary parts')
        if value.shape[-2:] != multiplier.shape[-2:]:
            raise ValueError('image / measurement shape differs from SAR grid')
        return torch.fft.ifft2(torch.fft.fft2(value, norm='ortho') * multiplier, norm='ortho')

    def forward(self, image):
        return self.mask * self.apply(image, self.phi)

    def adjoint(self, measurement):
        return self.apply(self.mask * measurement, self.phi.conj())

    def encode_physical_measurement(self, measurement):
        return self.gain * self.mask * measurement


def complex_shrink(value, threshold):
    magnitude = value.abs()
    return value * (magnitude - threshold).clamp_min(0) / magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny)


@torch.no_grad()
def classical_ista(measurement, operator, iterations=200, lambda_relative=.01):
    """Unaccelerated complex L1 ISTA with the same A/A^H as ISTA-Net."""
    ahy = operator.adjoint(measurement)
    lam = lambda_relative * ahy.abs().amax(dim=(-2, -1), keepdim=True)
    step = operator.lipschitz.reciprocal()
    image = ahy.clone()
    history = []
    for _ in range(iterations):
        image = complex_shrink(image - step * operator.adjoint(operator(image) - measurement), step * lam)
        objective = .5 * (operator(image) - measurement).abs().square().sum() + (lam * image.abs()).sum()
        history.append(float(objective))
    return image, history
