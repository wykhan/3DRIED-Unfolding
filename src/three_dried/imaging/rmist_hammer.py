"""Inference-only helpers for strict RMIST Eq. (32) on Hammer slices."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.models.rmist_net import RMISTNet
from three_dried.solvers.rmist_ista import rmist_ista
from three_dried.training.rmist_training import make_fixed_mask, mask_sha256


def physical_support_mask(
    operator: RMISTKernel,
    physical_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the centered physical-aperture support in the RM grid."""
    physical = torch.ones(physical_shape, dtype=dtype)
    return operator.pad_measurement(physical)


def accepted_effective_mask(
    operator: RMISTKernel,
    physical_shape: Tuple[int, int],
    seed: int,
    sampling_rate: float,
    expected_sha256: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Recreate and validate the accepted M6 mask and Hammer support mapping."""
    shape = (operator.n_fft_space, operator.n_fft_space)
    full_mask = make_fixed_mask(shape, sampling_rate, seed)
    actual_hash = mask_sha256(full_mask)
    if actual_hash != expected_sha256:
        raise RuntimeError("accepted M6 mask hash mismatch: " + actual_hash)
    support = physical_support_mask(operator, physical_shape)
    effective = full_mask * support
    support_count = int(support.sum().item())
    sampled_count = int(effective.sum().item())
    summary = {
        "seed": int(seed),
        "target_sampling_rate": float(sampling_rate),
        "full_grid_shape": list(shape),
        "full_grid_sha256_uint8": actual_hash,
        "full_grid_realized_sampling_rate": float(full_mask.mean().item()),
        "physical_shape_zx": list(physical_shape),
        "physical_support_samples": support_count,
        "effective_observed_samples": sampled_count,
        "effective_physical_sampling_rate": sampled_count / support_count,
        "definition": "Mask_eff = Mask_M6 * centered physical-support mask",
    }
    return full_mask, effective, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accepted_model(
    operator: RMISTKernel,
    summary_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[RMISTNet, Dict[str, object]]:
    """Load epoch-99 M6 scalars or reconstruct them exactly from Git data."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if int(summary["best_epoch"]) != 99:
        raise RuntimeError("accepted M6 best epoch is not 99")
    accepted = summary["best_parameters"]
    accepted_state = {
        "tau": torch.tensor(accepted["tau"], dtype=torch.float32),
        "theta": torch.tensor(accepted["theta"], dtype=torch.float32),
    }
    model = RMISTNet(operator)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["model_state_dict"]
        parameter_match = bool(
            int(checkpoint["epoch"]) == 99
            and torch.equal(state["tau"], accepted_state["tau"])
            and torch.equal(state["theta"], accepted_state["theta"])
        )
        if not parameter_match:
            raise RuntimeError("checkpoint does not match accepted M6 parameters")
        model.load_state_dict(state, strict=True)
        provenance = {
            "model_source": str(checkpoint_path),
            "source_kind": "local accepted best-validation checkpoint",
            "checkpoint_exists": True,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "best_epoch": 99,
            "parameter_match": True,
        }
    else:
        model.load_state_dict(accepted_state, strict=True)
        provenance = {
            "model_source": str(summary_path) + " -> best_parameters",
            "source_kind": "exact reconstruction from committed 18 scalars",
            "checkpoint_exists": False,
            "checkpoint_sha256": None,
            "best_epoch": 99,
            "parameter_match": True,
        }
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, provenance


def observed_sample_consistency(
    estimate: torch.Tensor,
    padded_physical_echo: torch.Tensor,
    effective_mask: torch.Tensor,
    operator: RMISTKernel,
    range_m: float,
) -> float:
    """Compute the M7 observed-sample residual diagnostic."""
    mask = effective_mask.to(device=estimate.device, dtype=estimate.dtype)
    residual = mask * (operator.rm(estimate, range_m) - padded_physical_echo)
    denominator = torch.linalg.vector_norm(mask * padded_physical_echo)
    epsilon = torch.finfo(estimate.real.dtype).eps
    value = torch.linalg.vector_norm(residual) / torch.clamp(denominator, min=epsilon)
    return float(value.detach().cpu())


def reconstruct_sparse_methods(
    padded_physical_echo: torch.Tensor,
    effective_mask: torch.Tensor,
    operator: RMISTKernel,
    model: RMISTNet,
    range_m: float,
) -> Dict[str, torch.Tensor]:
    """Run the three required sparse methods on one exact range slice."""
    mask = effective_mask.to(
        device=padded_physical_echo.device, dtype=padded_physical_echo.dtype
    )
    measurement = padded_physical_echo * mask
    rma50 = operator.rm_dagger(measurement, range_m)
    ini_ista50, _ = rmist_ista(
        measurement, operator, range_m, tau=1e-4, theta=1e-3, iterations=9
    )
    with torch.no_grad():
        rmist50 = model(measurement, range_m)
    return {"rma50": rma50, "ini_ista50": ini_ista50, "rmist50": rmist50}
