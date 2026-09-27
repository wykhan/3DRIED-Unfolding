import unittest

import torch

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.training.rmist_training import (
    FixedRangeRMISTOperator,
    RMISTSyntheticDataset,
    averaged_complex_l2_loss,
    batch_metrics,
    make_fixed_mask,
)


class RMISTTrainingTests(unittest.TestCase):
    def setUp(self):
        self.operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=16)
        self.cached = FixedRangeRMISTOperator(
            self.operator, 0.5, torch.complex64, torch.device("cpu")
        )
        self.mask = make_fixed_mask((16, 16), 0.5, 10)
        self.dataset = RMISTSyntheticDataset(
            pair_count=4,
            grid_size=16,
            scatterers_per_image=8,
            amplitude_min=0.5,
            amplitude_max=1.0,
            snr_db=20.0,
            range_m=0.5,
            seed=20,
        )

    def test_fixed_mask_is_binary_and_deterministic(self):
        repeated = make_fixed_mask((16, 16), 0.5, 10)
        self.assertTrue(torch.equal(self.mask, repeated))
        self.assertTrue(torch.all((self.mask == 0) | (self.mask == 1)).item())

    def test_cached_operator_exactly_matches_accepted_operator(self):
        generator = torch.Generator().manual_seed(30)
        value = torch.complex(
            torch.randn(2, 16, 16, generator=generator),
            torch.randn(2, 16, 16, generator=generator),
        )
        torch.testing.assert_close(
            self.cached.phi,
            self.operator.phi(0.5, torch.complex64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            self.cached.phi_dagger,
            self.operator.phi_dagger(0.5, torch.complex64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            self.cached.rm(value, 0.5), self.operator.rm(value, 0.5)
        )
        torch.testing.assert_close(
            self.cached.rm_dagger(value, 0.5), self.operator.rm_dagger(value, 0.5)
        )
        self.assertFalse(self.cached.phi.requires_grad)
        self.assertFalse(self.cached.phi_dagger.requires_grad)

    def test_dataset_is_deterministic_complex_and_has_exact_snr(self):
        first = self.dataset.make_batch([0, 1], self.cached, self.mask, torch.device("cpu"))
        second = self.dataset.make_batch([0, 1], self.cached, self.mask, torch.device("cpu"))
        for actual, repeated in zip(first, second):
            torch.testing.assert_close(actual, repeated, rtol=0.0, atol=0.0)
        truth, echo, achieved = first
        self.assertTrue(torch.is_complex(truth))
        self.assertTrue(torch.is_complex(echo))
        self.assertTrue(torch.all(torch.count_nonzero(truth, dim=(-2, -1)) == 8).item())
        torch.testing.assert_close(achieved, torch.full((2,), 20.0), atol=1e-4, rtol=0.0)

    def test_eq27_is_mean_per_sample_complex_l2_norm(self):
        estimate = torch.tensor(
            [[[3.0 + 4.0j]], [[0.0 + 2.0j]]], dtype=torch.complex64
        )
        target = torch.zeros_like(estimate)
        torch.testing.assert_close(
            averaged_complex_l2_loss(estimate, target), torch.tensor(3.5)
        )

    def test_metrics_are_evaluation_only_and_finite(self):
        truth, _, _ = self.dataset.make_batch(
            [0, 1], self.cached, self.mask, torch.device("cpu")
        )
        metrics = batch_metrics(0.9 * truth, truth)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(value).all().item())


if __name__ == "__main__":
    unittest.main()
