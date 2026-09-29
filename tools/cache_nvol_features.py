"""Cache ordered frozen NVOL visual targets from local THINGS-EEG2 images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.data import load_sub08_split
from nvol_retrieval.vision import FeatureCache, load_openclip_teacher, preflight_openclip


def _image_paths(data_root: Path, split: str) -> list[Path]:
    source = np.load(data_root / "image_set" / "img_path" / f"{split}_image.npy", allow_pickle=False)
    paths = [str(value) for value in source[::2]]
    resolved = []
    for value in paths:
        parts = Path(value).parts
        try:
            relative = Path(*parts[parts.index("image_set") + 1 :])
        except ValueError as error:
            raise ValueError(f"Image manifest entry has no image_set component: {value}") from error
        path = data_root / "image_set" / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing stimulus image: {path}")
        resolved.append(path)
    return resolved


def _canonical_image_ids(data_root: Path, paths: list[Path]) -> list[str]:
    return [path.resolve().relative_to(data_root.resolve()).as_posix() for path in paths]


def _validate_extraction_args(data_root: Path, output_dir: Path, batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if output_dir.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("--output-dir must be outside --data-root to protect raw data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance(device: torch.device, disable_cudnn_requested: bool, source_checkpoint: dict[str, str]) -> dict[str, object]:
    return {
        "disable_cudnn_requested": disable_cudnn_requested,
        "cudnn_enabled_effective": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "identity_hash_kind": "canonical_name_and_image_id_sha256",
        "source_checkpoint": source_checkpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--subject", default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    _validate_extraction_args(args.data_root, args.output_dir, args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but torch.cuda.is_available() is false")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled by --disable-cudnn")
    preflight_openclip()
    source_checkpoint = {"resolved_path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint)}
    teacher, preprocess = load_openclip_teacher(args.checkpoint, args.device)
    teacher.eval()
    for split in (("train", "test") if args.split == "both" else (args.split,)):
        manifest = load_sub08_split(args.data_root, split, subject=8)
        names = manifest.names[::manifest.repeats] if split == "train" else manifest.names
        paths = _image_paths(args.data_root, split)
        if [path.stem for path in paths] != names:
            raise ValueError("Image manifest order differs from EEG image order")
        batches = []
        for start in range(0, len(paths), args.batch_size):
            images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths[start : start + args.batch_size]]).to(args.device)
            with torch.no_grad():
                batches.append(teacher.encode_pooled(images).cpu())
        cache = FeatureCache(names, _canonical_image_ids(args.data_root, paths), torch.cat(batches))
        cache_path = args.output_dir / f"nvol_{split}_features.pt"
        cache.save(cache_path)
        provenance = _runtime_provenance(device, args.disable_cudnn, source_checkpoint) | {"split": split, "cache_file": cache_path.name, "cache_file_sha256": _sha256(cache_path), "cache_identity_hash": cache.name_hash}
        (args.output_dir / f"nvol_{split}_features.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"{split}: {tuple(cache.features.shape)} hash={cache.name_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
