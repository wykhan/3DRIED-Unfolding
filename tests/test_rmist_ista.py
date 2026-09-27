import unittest

import torch

from three_dried.simulation.rmist_synthetic import (
    add_complex_awgn,
    point_scatterer_scene,
    random_mask,
)
from three_dried.solvers.rmist_ista import complex_soft_threshold, rmist_ista


class IdentityRM:
    def rm(self, value, range_m):
        return value

    def rm_dagger(self, value, range_m):
        return value


class RMISTISTATests(unittest.TestCase):
    def test_complex_threshold_magnitude_and_phase(self):
        values = torch.tensor([3.0 + 4.0j, -2.0j, 0.0j], dtype=torch.complex128)
        actual = complex_soft_threshold(values, 1.0)
        torch.testing.assert_close(
            torch.abs(actual), torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64)
        )
        nonzero = torch.abs(actual) > 0
        torch.testing.assert_close(
            actual[nonzero] / torch.abs(actual[nonzero]),
            values[nonzero] / torch.abs(values[nonzero]),
        )

    def test_one_iteration_is_literal_eq19(self):
        measurement = torch.tensor([[2.0 + 1.0j]], dtype=torch.complex64)
        initial = torch.tensor([[0.5 - 0.25j]], dtype=torch.complex64)
        actual, _ = rmist_ista(
            measurement, IdentityRM(), 0.5, tau=0.2, theta=0.1,
            iterations=1, initial=initial,
        )
        residual = measurement - initial
        rough = initial + 0.2 * residual
        expected = complex_soft_threshold(rough, 0.1)
        torch.testing.assert_close(actual, expected)

    def test_reported_initialization_mode_runs_nine_iterations(self):
        measurement = torch.ones(4, 4, dtype=torch.complex64)
        _, history = rmist_ista(measurement, IdentityRM(), 0.5)
        self.assertEqual(len(history["residual_norm"]), 9)

    def test_synthetic_helpers_are_complex_and_reproducible(self):
        scene_a = point_scatterer_scene((16, 16), 8, 7)
        scene_b = point_scatterer_scene((16, 16), 8, 7)
        self.assertTrue(torch.equal(scene_a, scene_b))
        self.assertTrue(torch.is_complex(scene_a))
        self.assertEqual(int(torch.count_nonzero(scene_a)), 8)
        mask = random_mask((16, 16), 0.5, 8)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)).item())
        noisy_a = add_complex_awgn(scene_a, 20.0, 9)
        noisy_b = add_complex_awgn(scene_a, 20.0, 9)
        self.assertTrue(torch.equal(noisy_a, noisy_b))


if __name__ == "__main__":
    unittest.main()
