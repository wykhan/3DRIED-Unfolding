import ast
import unittest
from pathlib import Path

import torch

from three_dried.imaging.rmist_diagnostics import (
    magnitude_statistics,
    phase_metrics,
    trace_rmist_forward,
)
from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import RMISTNet


class TestRMISTSim2RealDiagnostic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=16)
        self.model = RMISTNet(self.operator)
        self.measurement = torch.complex(
            torch.randn(16, 16), torch.randn(16, 16)
        )

    def test_traced_forward_matches_ordinary_output_and_preserves_parameters(self):
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        ordinary = self.model(self.measurement, 0.5)
        traced, _, trace = trace_rmist_forward(self.model, self.measurement, 0.5)
        self.assertTrue(torch.equal(traced, ordinary))
        self.assertEqual(len(trace), 9)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_phase_decomposition_and_metrics_are_consistent(self):
        output, alpha0, trace = trace_rmist_forward(self.model, self.measurement, 0.5)
        for record in trace:
            self.assertTrue(torch.equal(
                record["rough"], record["alpha_before"] + record["correction"]
            ))
        self.assertTrue(torch.equal(output, trace[-1]["alpha_after"]))
        rows = phase_metrics(self.measurement, alpha0, trace, self.model.theta)
        self.assertEqual(len(rows), 9)
        for row in rows:
            for name, value in row.items():
                if name != "phase":
                    self.assertTrue(torch.isfinite(value).all())

    def test_magnitude_statistics_reproduce_toy_values(self):
        value = torch.tensor([0 + 0j, 3 + 0j, 4 + 0j, 0 + 0j])
        stats = magnitude_statistics(value)
        self.assertAlmostEqual(stats["rms"], 2.5)
        self.assertAlmostEqual(stats["mean"], 1.75)
        self.assertAlmostEqual(stats["median"], 1.5)
        self.assertAlmostEqual(stats["max"], 4.0)

    def test_execution_path_has_no_update_calls(self):
        paths = [
            Path("scripts/diagnose_rmist_sim2real.py"),
            Path("src/three_dried/imaging/rmist_diagnostics.py"),
        ]
        forbidden_attributes = {"backward", "step", "train"}
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            called = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            self.assertTrue(forbidden_attributes.isdisjoint(called))
            self.assertNotIn("torch.optim", source)


if __name__ == "__main__":
    unittest.main()
