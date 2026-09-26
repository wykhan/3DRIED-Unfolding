import unittest

import torch

from three_dried.imaging.holography import HolographyGeometry, HolographyOperator


def random_complex(shape, dtype, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    imag = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    return torch.complex(real, imag)


def relative_adjoint_error(operator, x, y):
    left = torch.vdot(operator.forward(x).reshape(-1), y.reshape(-1))
    right = torch.vdot(x.reshape(-1), operator.adjoint(y).reshape(-1))
    return (torch.abs(left - right) / torch.clamp(torch.abs(left), min=1e-12)).item()


class HolographyOperatorTests(unittest.TestCase):
    @staticmethod
    def operator():
        return HolographyOperator(HolographyGeometry(
            image_shape=(32, 40), aperture_shape=(19, 27), range_m=0.6143103953,
            center_frequency_hz=78.8e9, dx_m=1e-3, dz_m=2e-3,
        ))

    def test_shapes_dtype_and_finite_cpu(self):
        operator = self.operator()
        x = random_complex(operator.image_shape, torch.complex64, "cpu", 1)
        y = operator.forward(x)
        back = operator.adjoint(y)
        self.assertEqual(tuple(y.shape), operator.measurement_shape)
        self.assertEqual(tuple(back.shape), operator.image_shape)
        self.assertEqual(y.dtype, torch.complex64)
        self.assertEqual(back.dtype, torch.complex64)
        self.assertTrue(torch.isfinite(back).all().item())

    def test_complex128_adjoint_identity(self):
        operator = self.operator()
        x = random_complex(operator.image_shape, torch.complex128, "cpu", 7)
        y = random_complex(operator.measurement_shape, torch.complex128, "cpu", 8)
        self.assertLess(relative_adjoint_error(operator, x, y), 1e-12)

    def test_complex64_adjoint_identity_multiple_seeds(self):
        operator = self.operator()
        for seed in (2, 11, 29):
            x = random_complex(operator.image_shape, torch.complex64, "cpu", seed)
            y = random_complex(operator.measurement_shape, torch.complex64, "cpu", seed + 1)
            self.assertLess(relative_adjoint_error(operator, x, y), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_smoke_and_adjoint(self):
        operator = self.operator()
        x = random_complex(operator.image_shape, torch.complex64, "cuda", 41)
        y = random_complex(operator.measurement_shape, torch.complex64, "cuda", 42)
        self.assertEqual(operator.forward(x).device.type, "cuda")
        self.assertLess(relative_adjoint_error(operator, x, y), 1e-5)


if __name__ == "__main__":
    unittest.main()

