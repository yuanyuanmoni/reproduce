"""Evaluate a final sub-08 ATM checkpoint on the fixed 200-image test bank."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch
from torch.torch_version import TorchVersion

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.atm import ATMAlignmentModel
from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.evaluation import evaluate_exact_200, persist_result
from nvol_retrieval.vision import FeatureCache, NVOLLayerFusion


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    # Train metadata records torch.__version__, which is a TorchVersion on PyTorch 2.7.
    # This is the sole non-tensor type accepted from our locally produced checkpoint.
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict) or not {"atm", "fusion", "logit_scale", "metadata"} <= set(value):
        raise ValueError("Invalid final-epoch ATM checkpoint")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or metadata.get("subject") != 8 or metadata.get("checkpoint_selection") != "final_epoch":
        raise ValueError("Checkpoint is not a final-epoch sub-08 run")
    logit_scale = value["logit_scale"]
    if (not isinstance(logit_scale, torch.Tensor) or not torch.is_floating_point(logit_scale)
            or logit_scale.shape != torch.Size([]) or not torch.isfinite(logit_scale).item()):
        raise ValueError("Checkpoint logit_scale must be a finite floating-point scalar")
    return value


def _configure_cudnn(disable_cudnn: bool) -> None:
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled by --disable-cudnn")


def _runtime_provenance(device: torch.device, disable_cudnn_requested: bool) -> dict[str, object]:
    return {
        "disable_cudnn_requested": disable_cudnn_requested,
        "cudnn_enabled_effective": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-visual-cache", type=Path, required=True, help="cache used by train.py")
    parser.add_argument("--test-visual-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="8")
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    if not args.checkpoint.is_file() or not args.train_visual_cache.is_file() or not args.test_visual_cache.is_file():
        raise FileNotFoundError("checkpoint and both visual caches must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA evaluation but CUDA is unavailable")
    _configure_cudnn(args.disable_cudnn)
    split = load_sub08_split(args.data_root, "test", subject=8)
    ids = canonical_image_ids(args.data_root, "test")
    cache = FeatureCache.load(args.test_visual_cache, expected_names=split.names, expected_image_ids=ids)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    metadata = checkpoint["metadata"]
    if metadata.get("cache_hash") != FeatureCache.load(args.train_visual_cache).name_hash or metadata.get("cache_file_sha256") != _hash_file(args.train_visual_cache):
        raise ValueError("Training cache does not match the checkpoint metadata")
    model, fusion = ATMAlignmentModel().to(device).eval(), NVOLLayerFusion().to(device).eval()
    model.load_state_dict(checkpoint["atm"]); fusion.load_state_dict(checkpoint["fusion"])
    with torch.no_grad():
        # The 80 raw EEG trials are averaged before the ATM, never after ranking.
        queries = model(torch.from_numpy(split.eeg).float().mean(dim=1).to(device))
        gallery = fusion(cache.features.float().to(device))
    result = evaluate_exact_200(queries, gallery, args.csls_k, candidate_names=split.names,
                                candidate_image_ids=ids, trial_aggregation="mean_80_raw_eeg_before_atm")
    persist_result(result, args.output_dir, split.names, ids, _runtime_provenance(device, args.disable_cudnn))
    print(f"raw Top-1/Top-5: {result.raw_top1:.4f}/{result.raw_top5:.4f}; CSLS: {result.csls_top1:.4f}/{result.csls_top5:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
