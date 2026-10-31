"""The acceleration must preserve the literal complex BP sum and indexing."""
import unittest
import numpy as np
import torch

from three_dried.imaging.basic_bp import direct_slice_bp
from three_dried.imaging.fft_bp import FFTSliceBP, TorchFFTSliceBP
from three_dried.baseline.sampling import nested_masks


class FFTBPTests(unittest.TestCase):
    def test_rectangular_shifted_grid_all_rates_both_masks(self):
        rng = np.random.default_rng(20261027)
        x = -.05 + np.arange(12) * .001
        z = .12 + np.arange(10) * .002
        echo = rng.normal(size=(6, 7)) + 1j * rng.normal(size=(6, 7))
        op = FFTSliceBP((10, 12), .001, .002, .614, 78.8e9)
        for pattern in ('height_lines', 'random_2d'):
            for rate, mask in nested_masks(echo.shape, [1., .7, .5, .2], 17, pattern).items():
                with self.subTest(pattern=pattern, rate=rate):
                    padded = np.zeros((10, 12), complex)
                    padded[2:8, 2:9] = echo * mask
                    expected = direct_slice_bp(echo, x[2:9], z[2:8], x, z,
                                               .614, 78.8e9, mask)
                    actual = op(padded)
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 2e-11)

    def test_edges_have_linear_convolution_not_circular_wrap(self):
        x = np.arange(12)*.001
        z = np.arange(10)*.002
        echo = np.zeros((10, 12), complex)
        echo[0, 0] = .3 + .7j
        echo[-1, -1] = -.4 + .2j
        mask = echo != 0
        expected = direct_slice_bp(echo, x, z, x, z, .6, 78.8e9, mask)
        actual = FFTSliceBP(echo.shape, .001, .002, .6, 78.8e9)(echo)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_torch_cpu_matches_scipy(self):
        rng = np.random.default_rng(5)
        echo = rng.normal(size=(10, 12)) + 1j*rng.normal(size=(10, 12))
        expected = FFTSliceBP(echo.shape, .001, .002, .6, 78.8e9)(echo)
        actual = TorchFFTSliceBP(echo.shape, .001, .002, .6, 78.8e9, 'cpu')(echo).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_complex128_matches_scipy(self):
        echo = np.zeros((16, 16), complex)
        echo[3:13:2, 2:14] = 1 + .5j
        expected = FFTSliceBP(echo.shape, .001, .002, .6, 78.8e9)(echo)
        actual = TorchFFTSliceBP(echo.shape, .001, .002, .6, 78.8e9)(echo).cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_input_shape_and_complex_values_are_required(self):
        op = FFTSliceBP((10, 12), .001, .002, .6, 78.8e9)
        for echo in (np.ones((10, 12)), np.ones((10, 11), complex)):
            with self.assertRaises(ValueError):
                op(echo)


if __name__ == '__main__':
    unittest.main()
