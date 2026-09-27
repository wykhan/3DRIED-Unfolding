import unittest

import torch

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import (
    RMISTNet,
    complex_to_two_channel,
    two_channel_to_complex,
)
from three_dried.solvers.rmist_ista import complex_soft_threshold, rmist_ista


class ScaledRM:
    """Deterministic differentiable test double with nonzero phase residuals."""

    def rm(self, value, range_m):
        return 0.5 * value

    def rm_dagger(self, value, range_m):
        return 1.25 * value


class RMISTNetTests(unittest.TestCase):
    def setUp(self):
        self.range_m = 0.5
        self.operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=16)

    @staticmethod
    def measurement(dtype, batch=False, device="cpu"):
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        generator = torch.Generator(device=device).manual_seed(20260927)
        shape = (2, 16, 16) if batch else (16, 16)
        return torch.complex(
            torch.randn(shape, generator=generator, dtype=real_dtype, device=device),
            torch.randn(shape, generator=generator, dtype=real_dtype, device=device),
        )

    def test_parameter_structure_and_initialization(self):
        model = RMISTNet(self.operator)
        self.assertEqual(tuple(model.tau.shape), (9,))
        self.assertEqual(tuple(model.theta.shape), (9,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 18)
        self.assertEqual([name for name, _ in model.named_parameters()], ["tau", "theta"])
        torch.testing.assert_close(model.tau, torch.full((9,), 1e-4))
        torch.testing.assert_close(model.theta, torch.full((9,), 1e-3))
        with torch.no_grad():
            model.tau[3] = 0.25
            model.theta[7] = 0.5
        self.assertEqual(float(model.tau[3]), 0.25)
        self.assertEqual(int(torch.count_nonzero(model.tau == 1e-4)), 8)
        self.assertEqual(float(model.theta[7]), 0.5)
        self.assertEqual(int(torch.count_nonzero(model.theta == 1e-3)), 8)
        with self.assertRaises(ValueError):
            RMISTNet(self.operator, phases=8)

    def test_alpha0_is_exact_rm_dagger(self):
        measurement = self.measurement(torch.complex64)
        model = RMISTNet(self.operator)
        torch.testing.assert_close(
            model.initialize(measurement, self.range_m),
            self.operator.rm_dagger(measurement, self.range_m),
            rtol=0.0,
            atol=0.0,
        )

    def test_one_phase_is_literal_eq20(self):
        model = RMISTNet(ScaledRM())
        measurement = torch.tensor([[2.0 + 1.0j]], dtype=torch.complex64)
        alpha = torch.tensor([[0.5 - 0.25j]], dtype=torch.complex64)
        actual, residual = model.phase_update(alpha, measurement, self.range_m, 0)
        expected_residual = measurement - 0.5 * alpha
        rough = alpha + model.tau[0] * 1.25 * expected_residual
        expected = complex_soft_threshold(rough, model.theta[0])
        torch.testing.assert_close(residual, expected_residual, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_m4_to_m5_regression_complex64_and_complex128(self):
        for dtype, tolerance in ((torch.complex64, 1e-5), (torch.complex128, 1e-12)):
            measurement = self.measurement(dtype)
            model = RMISTNet(self.operator).to(dtype=measurement.real.dtype)
            actual = model(measurement, self.range_m)
            expected, _ = rmist_ista(
                measurement,
                self.operator,
                self.range_m,
                tau=1e-4,
                theta=1e-3,
                iterations=9,
            )
            relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
            self.assertLess(float(relative), tolerance)

    def test_complex_two_channel_round_trip(self):
        for batch in (False, True):
            value = self.measurement(torch.complex128, batch=batch)
            channels = complex_to_two_channel(value)
            self.assertEqual(channels.shape[-3], 2)
            recovered = two_channel_to_complex(channels)
            torch.testing.assert_close(recovered, value, rtol=0.0, atol=0.0)

    def test_gradient_reaches_every_phase_without_optimizer(self):
        model = RMISTNet(self.operator)
        measurement = self.measurement(torch.complex64)
        target = torch.zeros_like(measurement)
        reconstruction = model(measurement, self.range_m)
        loss = torch.mean(torch.abs(reconstruction - target) ** 2)
        loss.backward()
        for gradient in (model.tau.grad, model.theta.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertTrue(torch.all(torch.abs(gradient) > 0).item())

    def test_single_batch_cpu_and_trace(self):
        model = RMISTNet(self.operator)
        single = self.measurement(torch.complex64)
        batch = self.measurement(torch.complex64, batch=True)
        single_output, trace = model(single, self.range_m, return_trace=True)
        batch_output = model(batch, self.range_m)
        self.assertEqual(single_output.shape, single.shape)
        self.assertEqual(batch_output.shape, batch.shape)
        self.assertEqual(len(trace), 9)
        self.assertEqual([entry["phase"] for entry in trace], list(range(1, 10)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_complex64(self):
        model = RMISTNet(self.operator).cuda()
        measurement = self.measurement(torch.complex64, device="cuda")
        output = model(measurement, self.range_m)
        self.assertEqual(output.device.type, "cuda")
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
