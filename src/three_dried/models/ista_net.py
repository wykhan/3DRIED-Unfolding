"""Original ISTA-Net phase topology (CVPR 2018), adapted to complex SAR.

Reference: jianzhangcs/ISTA-Net-PyTorch / Train_CS_ISTA_Net.py.
Each unshared phase uses two analysis convolutions with ReLU, feature soft
thresholding, two synthesis convolutions with ReLU, and a symmetry residual
computed WITHOUT thresholding. There is no ISTA-Net+ residual connection.
The CS matrix products are replaced by the supplied SAR A and A^H.
"""
import torch
from torch import nn
from torch.nn import functional as F


def to_channels(value):
    if value.ndim != 3 or not torch.is_complex(value):
        raise ValueError('expected complex [batch,z,x]')
    return torch.stack((value.real, value.imag), dim=1)


def to_complex(value):
    return torch.complex(value[:, 0], value[:, 1])


class ISTAPhase(nn.Module):
    def __init__(self, features=32, theta_init=.01, initialization='identity_perturbed'):
        super().__init__()
        if features < 4:
            raise ValueError('at least four features required for two signed channels')
        self.step = nn.Parameter(torch.tensor(.5))
        self.threshold = nn.Parameter(torch.tensor(float(theta_init)))
        self.analysis1 = nn.Conv2d(2, features, 3, padding=1, bias=False)
        self.analysis2 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.synthesis1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.synthesis2 = nn.Conv2d(features, 2, 3, padding=1, bias=False)
        for layer in (self.analysis1, self.analysis2, self.synthesis1, self.synthesis2):
            nn.init.xavier_normal_(layer.weight)
        if initialization == 'identity_perturbed':
            # A documented initialization adaptation for sparse complex SAR.
            # Preserve signed Re/Im through ReLU; small noise activates all
            # feature channels. Architecture and loss remain the original ones.
            with torch.no_grad():
                for layer in (self.analysis1, self.analysis2, self.synthesis1, self.synthesis2):
                    layer.weight.normal_(0, .001)
                for channel in range(2):
                    for sign_index, sign in enumerate((1., -1.)):
                        f = 2 * channel + sign_index
                        self.analysis1.weight[f, channel, 1, 1] += sign
                        self.analysis2.weight[f, f, 1, 1] += 1
                        self.synthesis1.weight[f, f, 1, 1] += 1
                        self.synthesis2.weight[channel, f, 1, 1] += sign
        elif initialization != 'xavier':
            raise ValueError('unknown initialization')

    def analyze(self, value):
        return self.analysis2(F.relu(self.analysis1(value)))

    def synthesize(self, value):
        return self.synthesis2(F.relu(self.synthesis1(value)))

    def forward(self, image, measurement, operator):
        rough = image - (self.step / operator.lipschitz) * operator.adjoint(operator(image) - measurement)
        inputs = to_channels(rough)
        features = self.analyze(inputs)
        shrunk = features.sign() * F.relu(features.abs() - self.threshold)
        prediction = self.synthesize(shrunk)
        symmetry = self.synthesize(features) - inputs
        return to_complex(prediction), symmetry


class ISTANet(nn.Module):
    def __init__(self, phases=9, features=32, theta_init=.01, initialization='identity_perturbed'):
        super().__init__()
        self.phases = nn.ModuleList([ISTAPhase(features, theta_init, initialization) for _ in range(phases)])

    def forward(self, measurement, operator, return_symmetry=False):
        image = operator.adjoint(measurement)
        symmetry = []
        for phase in self.phases:
            image, residual = phase(image, measurement, operator)
            if return_symmetry:
                symmetry.append(residual)
        return (image, symmetry) if return_symmetry else image


def ista_net_loss(prediction, target, symmetry, gamma=.01):
    energy = target.abs().square().mean().clamp_min(1e-12)
    discrepancy = (prediction - target).abs().square().mean()
    # Real/imag mean has the factor 1/2 of complex squared magnitude.
    constraint = sum(residual.square().mean() for residual in symmetry)
    return (discrepancy + gamma * constraint) / energy
