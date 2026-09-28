"""Validated access to the official THINGS-EEG2 sub-08 arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Split = Literal["train", "test"]
EXPECTED_SHAPES = {
    "train": (16540, 4, 63, 250),
    "test": (200, 80, 63, 250),
}
SAMPLING_RATE_HZ = 250


@dataclass(frozen=True)
class ThingsEEGSplit:
    eeg: np.ndarray
    names: list[str]
    image_idx: np.ndarray
    repeats: int


def _subject_dir(subject: int | str) -> str:
    if subject not in (8, "8", "08", "sub-08"):
        raise ValueError("This reproduction only supports official sub-08 data.")
    return "sub-08"


def _payload(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing EEG payload: {path}")
    try:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, dict):
            payload = loaded
        elif isinstance(loaded, np.lib.npyio.NpzFile):
            payload = dict(loaded.items())
            loaded.close()
        elif isinstance(loaded, np.ndarray) and loaded.shape == ():
            payload = loaded.item()
        else:
            raise ValueError("payload is not a pickled mapping")
    except (OSError, ValueError, EOFError, AttributeError) as error:
        raise ValueError(f"Invalid EEG payload {path}: {error}") from error
    if not isinstance(payload, dict) or "preprocessed_eeg_data" not in payload:
        raise ValueError(f"Invalid EEG payload {path}: missing preprocessed_eeg_data")
    eeg = payload["preprocessed_eeg_data"]
    if not isinstance(eeg, np.ndarray) or eeg.dtype != np.dtype(np.float64):
        raise ValueError(f"Invalid EEG payload {path}: preprocessed_eeg_data must have dtype float64")
    return payload


def _names(root: Path, split: Split, image_count: int) -> list[str]:
    return [Path(path).stem for path in _image_paths(root, split, image_count)]


def _image_paths(root: Path, split: Split, image_count: int) -> list[str]:
    array_path = root / "image_set" / "img_path" / f"{split}_image.npy"
    try:
        paths = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid image-path array {array_path}: {error}") from error
    if paths.ndim != 1 or len(paths) != image_count * 2:
        raise ValueError(f"Invalid image-path array {array_path}: expected {image_count * 2} caption paths")
    path_strings = [str(path) for path in paths.tolist()]
    if any(path_strings[index] != path_strings[index + 1] for index in range(0, len(path_strings), 2)):
        raise ValueError(f"Invalid image-path array {array_path}: captions must be adjacent duplicate paths")
    unique_paths = path_strings[::2]
    if len(set(unique_paths)) != image_count:
        raise ValueError(f"Invalid image-path array {array_path}: expected {image_count} unique stimulus paths")
    return unique_paths


def canonical_image_ids(data_root: str | Path, split: Split, image_count: int | None = None) -> list[str]:
    """Return cache-compatible ordered stimulus identities relative to data_root."""
    root = Path(data_root)
    count = EXPECTED_SHAPES[split][0] if image_count is None else image_count
    ids = []
    for value in _image_paths(root, split, count):
        parts = Path(value).parts
        try:
            relative = Path("image_set", *parts[parts.index("image_set") + 1 :])
        except ValueError as error:
            raise ValueError(f"Image manifest entry has no image_set component: {value}") from error
        ids.append(relative.as_posix())
    return ids


def load_sub08_split(data_root: str | Path, split: Split, subject: int | str = 8) -> ThingsEEGSplit:
    """Load one official split without averaging the 80 held-out trials."""
    if split not in EXPECTED_SHAPES:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_root)
    subject_dir = _subject_dir(subject)
    filename = "preprocessed_eeg_training.npy" if split == "train" else "preprocessed_eeg_test.npy"
    eeg = _payload(root / "Preprocessed_data_250Hz" / subject_dir / filename)["preprocessed_eeg_data"]
    expected_shape = EXPECTED_SHAPES[split]
    if eeg.shape != expected_shape:
        raise ValueError(f"Unexpected {split} EEG shape {eeg.shape}; expected {expected_shape}")
    image_count, repeats = eeg.shape[:2]
    names = _names(root, split, image_count)
    image_idx = np.repeat(np.arange(image_count, dtype=np.int64), repeats) if split == "train" else np.arange(image_count, dtype=np.int64)
    return ThingsEEGSplit(
        eeg=eeg.reshape(image_count * repeats, *eeg.shape[2:]) if split == "train" else eeg,
        names=[name for name in names for _ in range(repeats)] if split == "train" else names,
        image_idx=image_idx,
        repeats=repeats,
    )
