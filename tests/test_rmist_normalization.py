import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.imaging.rmist_normalization import (
    magnitude_proxy,
    normalized_range_profile,
    normalized_rmist,
    positive_scale,
)
from three_dried.models.rmist_net import RMISTNet


class TestRMISTNormalization(unittest.TestCase):
    def setUp(self):
        self.operator = RMISTKernel(78.8e9, 0.001, 0.002, 16)
        self.model = RMISTNet(self.operator).eval()
        generator = torch.Generator().manual_seed(71)
        self.measurement = torch.complex(
            torch.randn(16, 16, generator=generator),
            torch.randn(16, 16, generator=generator),
        )

    def test_unit_scale_exactly_reproduces_ordinary_rmist(self):
        with torch.no_grad():
            ordinary = self.model(self.measurement, 0.5)
            normalized = normalized_rmist(self.model, self.measurement, 0.5, 1.0)
        self.assertTrue(torch.allclose(ordinary, normalized, rtol=1e-6, atol=1e-6))

    def test_linear_rm_dagger_normalize_denormalize(self):
        scale = 2.75e-9
        direct = self.operator.rm_dagger(self.measurement, 0.5)
        restored = self.operator.rm_dagger(scale * self.measurement, 0.5) / scale
        relative = torch.linalg.vector_norm(direct - restored) / torch.linalg.vector_norm(direct)
        self.assertLess(float(relative), 1e-6)

    def test_scale_validation(self):
        self.assertAlmostEqual(positive_scale(2.0, 4.0), 0.5)
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                positive_scale(1.0, value)

    def test_proxy_identities(self):
        value = np.arange(1, 13, dtype=np.float32).reshape(3, 2, 2)
        cosine, distance = magnitude_proxy(value, value)
        self.assertAlmostEqual(cosine, 1.0)
        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(normalized_range_profile(value))), 1.0)

    def test_execution_path_contains_no_training_calls(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "scripts" / "run_rmist_scale_normalized.py",
            root / "src" / "three_dried" / "imaging" / "rmist_normalization.py",
        ]
        forbidden_attributes = {"backward", "step", "train"}
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, forbidden_attributes)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.assertNotIn("torch.optim", ast.get_source_segment(path.read_text(encoding="utf-8"), node))


if __name__ == "__main__":
    unittest.main()
