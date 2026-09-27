import unittest

import numpy as np
import torch

from three_dried.imaging.rmist_kernel import C0, RMISTKernel


class RMISTKernelTests(unittest.TestCase):
    def setUp(self):
        self.operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=32)
        self.range_m = 0.6143103953

    def test_exact_3dried_kernel_formulas(self):
        ky, valid = self.operator.spectral_geometry(shifted=False)
        phi = self.operator.phi(self.range_m, torch.complex128, shifted=False)
        phi_dagger = self.operator.phi_dagger(
            self.range_m, torch.complex128, shifted=False
        )
        expected_phi = torch.zeros_like(phi)
        expected_dagger = torch.zeros_like(phi_dagger)
        expected_phi[valid] = torch.exp(1j * ky[valid] * self.range_m) / ky[valid]
        expected_dagger[valid] = ky[valid] * torch.exp(
            -1j * ky[valid] * self.range_m
        )
        torch.testing.assert_close(phi, expected_phi, rtol=0.0, atol=0.0)
        torch.testing.assert_close(phi_dagger, expected_dagger, rtol=0.0, atol=0.0)

    def test_reciprocal_product_on_valid_support(self):
        _, valid = self.operator.spectral_geometry(shifted=False)
        product = self.operator.phi(
            self.range_m, torch.complex128, shifted=False
        ) * self.operator.phi_dagger(
            self.range_m, torch.complex128, shifted=False
        )
        torch.testing.assert_close(
            product[valid], torch.ones_like(product[valid]), rtol=1e-12, atol=1e-12
        )

    def test_nonpropagating_and_zero_ky_are_explicitly_zero(self):
        wavelength = C0 / 78.8e9
        operator = RMISTKernel(
            78.8e9, wavelength / 4.0, wavelength / 4.0, n_fft_space=3,
            ky_epsilon=1e-8,
        )
        ky, valid = operator.spectral_geometry(shifted=False)
        phi = operator.phi(0.5, torch.complex128, shifted=False)
        dagger = operator.phi_dagger(0.5, torch.complex128, shifted=False)
        self.assertTrue(torch.any(~valid).item())
        self.assertTrue(torch.any((ky <= operator.ky_epsilon) & (~valid)).item())
        self.assertTrue(torch.all(phi[~valid] == 0).item())
        self.assertTrue(torch.all(dagger[~valid] == 0).item())
        self.assertTrue(torch.isfinite(phi).all().item())

    def test_physical_padding_and_crop(self):
        operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=512)
        physical = torch.arange(200 * 407).reshape(200, 407)
        padded = operator.pad_measurement(physical)
        self.assertEqual(tuple(padded.shape), (512, 512))
        self.assertTrue(torch.equal(padded[156:356, 52:459], physical))
        self.assertEqual(int(torch.count_nonzero(padded)), 200 * 407 - 1)
        self.assertTrue(torch.equal(operator.crop_measurement(padded), physical))

    def test_full_domain_round_trip_complex64_and_complex128(self):
        generator = torch.Generator().manual_seed(20260927)
        for dtype, tolerance in ((torch.complex64, 2e-5), (torch.complex128, 1e-12)):
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            spectrum = torch.complex(
                torch.randn(32, 32, generator=generator, dtype=real_dtype),
                torch.randn(32, 32, generator=generator, dtype=real_dtype),
            )
            _, support = self.operator.spectral_geometry(shifted=True)
            scene = torch.fft.ifft2(spectrum * support)
            recovered = self.operator.rm_dagger(
                self.operator.rm(scene, self.range_m), self.range_m
            )
            relative = torch.linalg.vector_norm(recovered - scene) / torch.linalg.vector_norm(scene)
            self.assertLess(float(relative), tolerance)

    def test_m2_numpy_formula_equivalence_complex64(self):
        operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=32)
        rng = np.random.RandomState(20260927)
        physical = (rng.randn(20, 27) + 1j * rng.randn(20, 27)).astype(np.complex64)
        padded = np.pad(physical, ((6, 6), (2, 3)))
        k = 2.0 * np.pi * 78.8e9 / C0
        kx = np.linspace(-np.pi / 1e-3, np.pi / 1e-3, 32)
        kz = np.linspace(-np.pi / 2e-3, np.pi / 2e-3, 32)[:, None]
        radial_sq = (2.0 * k) ** 2 - kx[None, :] ** 2 - kz**2
        support = radial_sq >= 0.0
        ky = np.sqrt(np.maximum(radial_sq, 0.0))
        kernel = ky * np.exp(-1j * self.range_m * ky)
        kernel[~support] = 0.0
        expected = np.fft.ifft2(np.fft.fft2(padded) * np.fft.fftshift(kernel)).astype(np.complex64)
        actual = operator.rm_dagger(torch.from_numpy(padded), self.range_m).numpy()
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertLess(relative, 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_smoke(self):
        scene = torch.zeros(32, 32, dtype=torch.complex64, device="cuda")
        scene[16, 16] = 1.0 + 0.5j
        result = self.operator.rm_dagger(self.operator.rm(scene, self.range_m), self.range_m)
        self.assertEqual(result.device.type, "cuda")
        self.assertTrue(torch.isfinite(result).all().item())


if __name__ == "__main__":
    unittest.main()
