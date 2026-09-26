"""Read-only loader for the MATLAB v7.3 3DRIED raw capture."""

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


def _matlab_complex(dataset: h5py.Dataset) -> np.ndarray:
    """Convert MATLAB's HDF5 compound complex representation."""
    value = np.asarray(dataset)
    fields = value.dtype.fields
    if fields and "real" in fields and "imag" in fields:
        return value["real"] + 1j * value["imag"]
    return value.astype(np.complex64)


def _scalar(group: h5py.Group, key: str) -> float:
    return float(np.asarray(group[key]).reshape(-1)[0])


def inspect_raw_mat(path: Path) -> Dict[str, object]:
    """Return the concise structural metadata needed to interpret the capture."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        raw = handle["adcRawData"]
        dim = raw["dim"]
        rf = raw["rfParams"]
        refs = raw["data"]
        first = handle[refs[0, 0]]
        first_value = _matlab_complex(first)
        return {
            "mat_format": "MATLAB v7.3 (HDF5)",
            "root_keys": sorted(handle.keys()),
            "variable": "adcRawData",
            "cell_array_stored_shape": list(refs.shape),
            "frame_matlab_shape": [1, int(_scalar(dim, "numRxChan")), int(_scalar(dim, "numSamples"))],
            "frame_hdf5_shape": list(first.shape),
            "frame_storage_dtype": str(first.dtype),
            "logical_dtype": "complex64",
            "is_complex": bool(np.iscomplexobj(first_value)),
            "dimensions": {key: _scalar(dim, key) for key in sorted(dim.keys())},
            "rf_params_file_units": {key: _scalar(rf, key) for key in sorted(rf.keys())},
        }


def load_single_rx_aperture(
    path: Path,
    rx_index: int = 0,
    nx: int = 407,
    nz: int = 200,
    motion_error_samples: int = 43,
    serpentine_correct: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load one Rx as ``[vertical, horizontal, fast_time]`` complex64.

    This reproduces ``Demo_RMA2D_Imaging.m`` without materializing the other
    receivers. MATLAB's row ``ii`` starts at
    ``(ii-1)*nx + floor(ii/nz*err)`` in the chosen Rx frame stream.
    """
    if rx_index < 0:
        raise ValueError("rx_index must be non-negative")
    path = Path(path)
    with h5py.File(path, "r") as handle:
        raw = handle["adcRawData"]
        refs = raw["data"]
        dim = raw["dim"]
        n_frames = int(_scalar(dim, "numFrames"))
        n_rx = int(_scalar(dim, "numRxChan"))
        n_fast = int(_scalar(dim, "numSamples"))
        if rx_index >= n_rx:
            raise ValueError("rx_index {} outside 0..{}".format(rx_index, n_rx - 1))

        selected = np.empty((nz, nx, n_fast), dtype=np.complex64)
        source_intervals = []
        for iz in range(nz):
            matlab_row = iz + 1
            offset = int(np.floor(matlab_row / nz * motion_error_samples))
            start = iz * nx + offset
            stop = start + nx
            if stop > n_frames:
                raise ValueError("motion-corrected aperture exceeds available frames")
            row_refs = refs[start:stop, 0]
            for ix, ref in enumerate(row_refs):
                stored = _matlab_complex(handle[ref])
                selected[iz, ix] = stored[:, rx_index, 0]
            source_intervals.append([start, stop - 1])

    if serpentine_correct:
        selected[1::2] = selected[1::2, ::-1]
    metadata = {
        "output_layout": ["vertical_z", "horizontal_x", "fast_time"],
        "output_shape": list(selected.shape),
        "output_dtype": str(selected.dtype),
        "rx_index_python": rx_index,
        "rx_index_matlab": rx_index + 1,
        "available_rx": n_rx,
        "motion_error_samples": motion_error_samples,
        "source_frame_first_last_zero_based": [source_intervals[0], source_intervals[-1]],
        "serpentine_corrected": serpentine_correct,
    }
    return selected, metadata

