from pathlib import Path
import unittest

import h5py
import numpy as np

from three_dried.data.three_dried import inspect_raw_mat
from three_dried.imaging.rma import matlab_bin_range_m, reconstruct_slice


DATA = Path("/home/superws/dataset/3DRIED/Hammer/Hammer (1)/output_rawdata.mat")


class BaselineTests(unittest.TestCase):
    def test_dataset_is_readable_and_layout_matches_reference(self):
        result = inspect_raw_mat(DATA)
        self.assertEqual(result["dimensions"]["numFrames"], 82278)
        self.assertEqual(result["dimensions"]["numRxChan"], 4)
        self.assertEqual(result["dimensions"]["numSamples"], 256)
        self.assertTrue(result["is_complex"])

    def test_rma_slice_is_complex_finite(self):
        rng = np.random.RandomState(20260926)
        echo = (rng.randn(8, 10) + 1j * rng.randn(8, 10)).astype(np.complex64)
        image = reconstruct_slice(echo, 0.614, 78.8e9, 1e-3, 2e-3, 16)
        self.assertEqual(image.shape, (16, 16))
        self.assertTrue(np.iscomplexobj(image))
        self.assertTrue(np.isfinite(image).all())

    def test_reference_bin_17_range(self):
        value = matlab_bin_range_m(17, 5e6, 70.295e12, 256, 6.2516e-10)
        self.assertLess(abs(value - 0.6143103953), 1e-9)


if __name__ == "__main__":
    unittest.main()
