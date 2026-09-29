#!/usr/bin/env python
"""Write a read-only contract report for official THINGS-EEG2 sub-08 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from nvol_retrieval.data import SAMPLING_RATE_HZ, load_sub08_split  # noqa: E402


def _name_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _summary(split: str, data_root: Path) -> dict:
    item = load_sub08_split(data_root, split)  # type: ignore[arg-type]
    image_count = len(item.names) // item.repeats if split == "train" else len(item.names)
    official_shape = [image_count, item.repeats, *item.eeg.shape[-2:]]
    return {
        "official_array_shape": official_shape,
        "eeg_shape": list(item.eeg.shape),
        "dtype": str(item.eeg.dtype),
        "images": image_count,
        "trials_per_image": item.repeats,
        "eeg_inputs": len(item.image_idx),
        "unique_names": len(set(item.names)),
        "ordered_names_sha256": _name_hash(item.names[::item.repeats] if split == "train" else item.names),
    }


def _safe_output(data_root: Path, output: Path) -> Path:
    resolved_root = data_root.resolve()
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(resolved_root):
        raise ValueError("--output must be outside --data-root to preserve raw data")
    return resolved_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", default="8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports official sub-08 data.")
    output = _safe_output(args.data_root, args.output)
    report = {
        "subject": "sub-08",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "caption_path_rule": "train_image.npy and test_image.npy contain adjacent duplicate paths for two captions, not EEG repeats.",
        "train": _summary("train", args.data_root),
        "test": _summary("test", args.data_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
