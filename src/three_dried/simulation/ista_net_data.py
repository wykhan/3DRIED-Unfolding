"""Disjoint procedural point scenes; no measured image is a training label."""
import hashlib
import numpy as np
import torch

from three_dried.baseline.config import coordinates, ranges_m
from three_dried.baseline.operators import SliceRM
from three_dried.baseline.sampling import nested_masks
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.imaging.sar_operator import MeasuredSAR


def make_operators(acquisition, rate, pattern, device='cpu'):
    n = acquisition['imaging']['n_fft_space']
    physical = nested_masks((acquisition['data']['nz'], acquisition['data']['nx']), [rate],
                            acquisition['sampling']['seed'], pattern)[rate]
    mask = SliceRM(acquisition, ranges_m(acquisition)[0]).pad(physical)
    kernel = RMISTKernel(acquisition['radar']['center_frequency_hz'], acquisition['aperture']['dx_m'],
                         acquisition['aperture']['dz_m'], n)
    return {int(b): MeasuredSAR(kernel, float(y), mask, device=device)
            for b, y in zip(acquisition['imaging']['matlab_range_bins'], ranges_m(acquisition))}


class PointScenes:
    def __init__(self, acquisition, settings, split):
        self.acquisition, self.settings = acquisition, settings
        self.split = split
        self.seed = int(settings['split_seeds'][split])
        self.count = int(settings['split_counts'][split])
        x, z = coordinates(acquisition)
        rx, rz = settings['roi_x_m'], settings['roi_z_m']
        self.x_indices = np.flatnonzero((x >= rx[0]) & (x <= rx[1]))
        self.z_indices = np.flatnonzero((z >= rz[0]) & (z <= rz[1]))
        if not len(self.x_indices) or not len(self.z_indices):
            raise ValueError('training ROI misses reconstruction grid')

    def __len__(self):
        return self.count

    def scene(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        n = self.acquisition['imaging']['n_fft_space']
        gt = np.zeros((n, n), dtype=np.complex64)
        count = int(rng.integers(self.settings['points_min'], self.settings['points_max'] + 1))
        locations = rng.choice(len(self.x_indices) * len(self.z_indices), size=count, replace=False)
        iz, ix = np.unravel_index(locations, (len(self.z_indices), len(self.x_indices)))
        gt[self.z_indices[iz], self.x_indices[ix]] = rng.uniform(.5, 1., count) * np.exp(1j * rng.uniform(0, 2*np.pi, count))
        bins = self.acquisition['imaging']['matlab_range_bins']
        matlab_bin = int(bins[int(rng.integers(len(bins)))])
        fingerprint = hashlib.sha256(gt.tobytes() + str(matlab_bin).encode()).hexdigest()
        return torch.from_numpy(gt).unsqueeze(0), matlab_bin, fingerprint

    def pair(self, index, operators):
        target, matlab_bin, fingerprint = self.scene(index)
        op = operators[matlab_bin]
        target = target.to(op.phi.device)
        with torch.no_grad():
            measurement = op(target)
            snr = self.settings.get('snr_db')
            if snr is not None:
                gen = torch.Generator(device=target.device).manual_seed(self.seed + 1000000 + index)
                noise = torch.complex(torch.randn(target.shape, generator=gen, device=target.device),
                                      torch.randn(target.shape, generator=gen, device=target.device)) * op.mask
                noise *= torch.linalg.vector_norm(measurement) / torch.linalg.vector_norm(noise).clamp_min(1e-20) * 10**(-float(snr)/20)
                measurement = measurement + noise
        return measurement, target, op, fingerprint
