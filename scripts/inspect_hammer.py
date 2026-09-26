#!/usr/bin/env python3
"""Produce a machine-readable, read-only Hammer dataset audit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_dried.data.three_dried import inspect_raw_mat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/superws/dataset/3DRIED/Hammer/Hammer (1)"))
    parser.add_argument("--output", type=Path, default=Path("results/hammer_audit.json"))
    parser.add_argument("--hash-files", action="store_true", help="Compute SHA-256 for immutable-input verification")
    args = parser.parse_args()
    files = []
    for path in sorted(args.dataset.iterdir()):
        item = {"name": path.name, "bytes": path.stat().st_size, "format": path.suffix.lower()}
        if path.suffix.lower() in (".jpg", ".jpeg"):
            with Image.open(path) as image:
                item.update(image_size=list(image.size), image_mode=image.mode)
        if args.hash_files:
            item["sha256"] = sha256(path)
        files.append(item)
    mat = inspect_raw_mat(args.dataset / "output_rawdata.mat")
    summary = {
        "dataset": str(args.dataset),
        "read_only_inspection": True,
        "files": files,
        "mat": mat,
        "axis_interpretation": {
            "logical_frame": ["chirp", "rx", "adc_fast_time"],
            "aperture_after_reference_motion_correction": ["vertical_z", "horizontal_x", "adc_fast_time"],
            "horizontal_samples": 407,
            "vertical_samples": 200,
            "horizontal_spacing_m": 0.001,
            "vertical_spacing_m": 0.002,
        },
        "single_rx": {"selected_python_index": 0, "selected_matlab_index": 1, "justification": "Bundled Demo_RMA2D_Imaging.m uses Raw_echo(1:4:end,:), i.e. Rx1."},
        "nearby_reference_code": [
            "/home/superws/dataset/3DRIED/Demo_datapreprocessing.m",
            "/home/superws/dataset/3DRIED/Demo_RMA2D_Imaging.m",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

