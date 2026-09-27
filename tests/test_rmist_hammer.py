import ast
import tempfile
import unittest
from pathlib import Path

import torch

from three_dried.imaging.rmist_hammer import (
    accepted_effective_mask,
    observed_sample_consistency,
    physical_support_mask,
)
from three_dried.imaging.rmist_kernel import RMISTKernel


class TestRMISTHammerInference(unittest.TestCase):
    def setUp(self):
        self.operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=16)

    def test_physical_support_is_centered_and_exact(self):
        support = physical_support_mask(self.operator, (6, 9))
        self.assertEqual(int(support.sum()), 54)
        self.assertTrue(torch.equal(support[5:11, 3:12], torch.ones(6, 9)))
        outside = support.clone()
        outside[5:11, 3:12] = 0
        self.assertEqual(int(outside.sum()), 0)

    def test_accepted_mask_hash_and_physical_rate(self):
        operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=512)
        full, effective, summary = accepted_effective_mask(
            operator,
            (200, 407),
            20260928,
            0.5,
            "43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2",
        )
        self.assertEqual(tuple(full.shape), (512, 512))
        self.assertEqual(int(effective.sum()), 40682)
        self.assertAlmostEqual(summary["effective_physical_sampling_rate"], 40682 / 81400)

    def test_observed_consistency_is_zero_for_exact_echo(self):
        image = torch.zeros(16, 16, dtype=torch.complex64)
        image[7, 8] = 1 + 2j
        range_m = 0.5
        echo = self.operator.rm(image, range_m)
        mask = torch.ones(16, 16)
        value = observed_sample_consistency(
            image, echo, mask, self.operator, range_m
        )
        self.assertLess(value, 1e-5)

    def test_m7_script_contains_no_optimizer_or_training_call(self):
        source = Path("scripts/run_rmist_hammer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called)
        self.assertNotIn("step", called)
        self.assertNotIn("train", called)
        self.assertNotIn("Adam", source)


if __name__ == "__main__":
    unittest.main()
