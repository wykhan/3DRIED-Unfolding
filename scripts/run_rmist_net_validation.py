#!/usr/bin/env python3
"""Create lightweight deterministic M5-R architecture review evidence."""

import argparse
import json
import os
import platform
import shlex
import sys
from pathlib import Path

import torch

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import RMISTNet
from three_dried.solvers.rmist_ista import rmist_ista


def deterministic_measurement(dtype, batch=False, device="cpu"):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    generator = torch.Generator(device=device).manual_seed(20260927)
    shape = (2, 32, 32) if batch else (32, 32)
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=real_dtype, device=device),
        torch.randn(shape, generator=generator, dtype=real_dtype, device=device),
    )


def relative_difference(actual, expected):
    return float(
        (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
        .detach()
        .cpu()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/review/M5")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    range_m = 0.5
    operator = RMISTKernel(78.8e9, 1e-3, 2e-3, n_fft_space=32)
    metrics = {}
    trace = None

    for dtype, name in ((torch.complex64, "complex64"), (torch.complex128, "complex128")):
        measurement = deterministic_measurement(dtype)
        model = RMISTNet(operator).to(dtype=measurement.real.dtype)
        output, current_trace = model(measurement, range_m, return_trace=True)
        reference, _ = rmist_ista(
            measurement, operator, range_m, tau=1e-4, theta=1e-3, iterations=9
        )
        alpha0 = model.initialize(measurement, range_m)
        expected_alpha0 = operator.rm_dagger(measurement, range_m)
        metrics[name] = {
            "alpha0_relative_difference": relative_difference(alpha0, expected_alpha0),
            "m4_to_m5_relative_difference": relative_difference(output, reference),
            "finite": bool(torch.isfinite(output).all().item()),
        }
        if name == "complex64":
            trace = current_trace

    batch = deterministic_measurement(torch.complex64, batch=True)
    batch_output = RMISTNet(operator)(batch, range_m)
    metrics["batch"] = {
        "input_shape": list(batch.shape),
        "output_shape": list(batch_output.shape),
        "passed": list(batch.shape) == list(batch_output.shape),
    }

    gradient_model = RMISTNet(operator)
    gradient_measurement = deterministic_measurement(torch.complex64)
    gradient_output = gradient_model(gradient_measurement, range_m)
    torch.mean(torch.abs(gradient_output) ** 2).backward()
    tau_gradient = gradient_model.tau.grad.detach()
    theta_gradient = gradient_model.theta.grad.detach()
    metrics["gradient"] = {
        "test_operator": "strict RMISTKernel, deterministic complex64 input",
        "tau_finite": bool(torch.isfinite(tau_gradient).all().item()),
        "theta_finite": bool(torch.isfinite(theta_gradient).all().item()),
        "tau_all_nonzero": bool(torch.all(torch.abs(tau_gradient) > 0).item()),
        "theta_all_nonzero": bool(torch.all(torch.abs(theta_gradient) > 0).item()),
        "tau_values": tau_gradient.tolist(),
        "theta_values": theta_gradient.tolist(),
        "optimizer_created": False,
        "optimizer_step_taken": False,
    }

    cuda_available = torch.cuda.is_available()
    cuda_passed = None
    cuda_device = None
    if cuda_available:
        cuda_device = torch.cuda.get_device_name(0)
        cuda_model = RMISTNet(operator).cuda()
        cuda_measurement = deterministic_measurement(torch.complex64, device="cuda")
        cuda_output = cuda_model(cuda_measurement, range_m)
        cuda_passed = bool(torch.isfinite(cuda_output).all().item())
    metrics["cuda"] = {
        "available": cuda_available,
        "device": cuda_device,
        "complex64_passed": cuda_passed,
    }

    model = RMISTNet(operator)
    metrics.update(
        {
            "seed": 20260927,
            "phases": model.phases,
            "trainable_tensor_names": [name for name, _ in model.named_parameters()],
            "trainable_scalar_count": sum(value.numel() for value in model.parameters()),
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
            "m6_training_started": False,
            "hammer_inference_started": False,
        }
    )

    parameters = {
        "tau": model.tau.detach().tolist(),
        "theta": model.theta.detach().tolist(),
    }
    architecture = """Strict RMIST-Net Eq. (20)-(26)\nphases: 9\ninitialization: alpha0 = RM_dagger(S)\nphase: v=S-RM(alpha); r=alpha+tau_t*RM_dagger(v); alpha=F_st(r,theta_t)\ntrainable tensors: tau[9], theta[9]\ntrainable scalar count: 18\ninternal representation: PyTorch complex (equivalent shared complex processing of separate real/imag paths)\noptimizer/training loop: none\n"""

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "parameter_values.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "phase_trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "architecture.txt").write_text(architecture)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
