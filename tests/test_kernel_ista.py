import unittest

import torch

from three_dried.imaging.holography import HolographyGeometry, HolographyOperator
from three_dried.simulation.synthetic import sparse_complex_scene
from three_dried.solvers.kernel_ista import complex_soft_threshold, estimate_lipschitz, kernel_ista


class KernelIstaTests(unittest.TestCase):
    def test_complex_soft_threshold_preserves_phase(self):
        value = torch.tensor([3 + 4j, 0.3 + 0.4j, 0j], dtype=torch.complex64)
        result = complex_soft_threshold(value, 1.0)
        self.assertTrue(torch.allclose(torch.abs(result), torch.tensor([4.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(result[0] / torch.abs(result[0]), value[0] / torch.abs(value[0]), atol=1e-6))

    def test_sparse_scene_is_reproducible_and_complex(self):
        first, metadata_first = sparse_complex_scene((24, 20), 5, 99, margin=2)
        second, metadata_second = sparse_complex_scene((24, 20), 5, 99, margin=2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(metadata_first, metadata_second)
        self.assertTrue(torch.is_complex(first))
        self.assertEqual(torch.count_nonzero(first).item(), 5)

    def test_ista_objective_decreases(self):
        operator = HolographyOperator(HolographyGeometry(
            image_shape=(24, 24), aperture_shape=(18, 20), range_m=0.6143,
            center_frequency_hz=78.8e9, dx_m=1e-3, dz_m=2e-3,
        ))
        truth, _ = sparse_complex_scene(operator.image_shape, 3, 17, margin=3)
        measurement = operator.forward(truth)
        lipschitz, _ = estimate_lipschitz(operator, 30, 18)
        step = 0.9 / lipschitz
        adjoint = step * operator.adjoint(measurement)
        regularization = 0.001 * float(torch.abs(operator.adjoint(measurement)).max())
        result = kernel_ista(operator, measurement, regularization, step, 20, adjoint)
        objectives = [item["objective"] for item in result.history]
        self.assertLess(objectives[-1], objectives[0])
        self.assertTrue(torch.isfinite(result.image).all().item())


if __name__ == "__main__":
    unittest.main()
