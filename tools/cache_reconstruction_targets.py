"""Cache local-only, provenance-bound targets for sub-08 reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.reconstruction_data import ReconstructionTargetCache
from nvol_retrieval.vision import FEATURE_DIM, FeatureCache, load_openclip_teacher, preflight_openclip


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(data_root: Path, output: Path, batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if output.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("--output must be outside --data-root to protect raw data")


def _load_validated_feature_cache(path: Path, *, split: str, names: list[str], image_ids: list[str],
                                  checkpoint: Path) -> FeatureCache:
    cache = FeatureCache.load(path, expected_names=names, expected_image_ids=image_ids)
    provenance_path = path.with_name(f"{path.stem}.provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing adjacent feature-cache provenance: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid feature-cache provenance: {provenance_path}") from error
    expected_checkpoint = {"resolved_path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)}
    if not isinstance(provenance, dict) or any((
        provenance.get("split") != split,
        provenance.get("cache_file") != path.name,
        provenance.get("cache_file_sha256") != _sha256(path),
        provenance.get("cache_identity_hash") != cache.name_hash,
        provenance.get("source_checkpoint") != expected_checkpoint,
    )):
        raise ValueError("Feature-cache provenance does not bind this split, cache identity, file SHA256, and visual checkpoint")
    return cache


def _semantic_targets(teacher: torch.nn.Module, preprocess: object, paths: list[Path], batch_size: int,
                      device: torch.device) -> torch.Tensor:
    batches = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths[start : start + batch_size]]).to(device)
        with torch.no_grad():
            semantic = teacher.backbone(images)
        if semantic.ndim != 2 or semantic.shape != (len(images), FEATURE_DIM):
            raise ValueError("Local OpenCLIP visual forward must return final semantic targets [B,1024]")
        batches.append(semantic.cpu())
    return torch.cat(batches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True, help="nvol_<split>_features.pt from cache_nvol_features.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="local ViT-H/14 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--subject", default="8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    if not args.feature_cache.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("--feature-cache and --checkpoint must be existing local files")
    _validate_args(args.data_root, args.output, args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA target extraction but CUDA is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled by --disable-cudnn")
    manifest = load_sub08_split(args.data_root, args.split, subject=8)
    names = manifest.names[::manifest.repeats] if args.split == "train" else manifest.names
    image_ids = canonical_image_ids(args.data_root, args.split)
    feature_cache = _load_validated_feature_cache(
        args.feature_cache, split=args.split, names=names, image_ids=image_ids, checkpoint=args.checkpoint,
    )
    paths = [args.data_root / image_id for image_id in image_ids]
    if not all(path.is_file() for path in paths) or [path.stem for path in paths] != names:
        raise ValueError("Image manifest order differs from the EEG image order")
    preflight_openclip()
    teacher, preprocess = load_openclip_teacher(args.checkpoint, device)
    teacher.eval()
    targets = ReconstructionTargetCache.from_feature_cache(
        feature_cache, _semantic_targets(teacher, preprocess, paths, args.batch_size, device), split=args.split,
        source_feature_cache_sha256=_sha256(args.feature_cache), visual_checkpoint_sha256=_sha256(args.checkpoint),
        visual_checkpoint_path=str(args.checkpoint.resolve()),
    )
    targets.save(args.output)
    print(f"{args.split}: layer12={tuple(targets.layer12_pooled.shape)} semantic={tuple(targets.final_semantic.shape)} hash={targets.name_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
