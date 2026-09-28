"""Train ATM against cached frozen NVOL block features; final epoch is the only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.atm import ATMAlignmentModel
from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.loss import symmetric_multi_positive_infonce
from nvol_retrieval.vision import FeatureCache, NVOLLayerFusion


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


class GroupedTrialBatchSampler(Sampler[list[int]]):
    """Shuffle image IDs, never separating their four flattened train trials."""

    def __init__(self, image_idx: torch.Tensor, batch_size: int, repeats: int = 4,
                 incomplete_policy: str = "keep", generator: torch.Generator | None = None) -> None:
        if batch_size <= 0 or batch_size % repeats:
            raise ValueError("batch_size must be divisible by the four EEG trials per image")
        if incomplete_policy not in {"keep", "error"}:
            raise ValueError("incomplete_policy must be 'keep' or 'error'")
        groups = [torch.nonzero(image_idx == image, as_tuple=False).flatten().tolist() for image in torch.unique(image_idx, sorted=True)]
        if any(len(group) != repeats for group in groups):
            raise ValueError("Training EEG must contain exactly four trials for every image ID")
        self.groups, self.images_per_batch = groups, batch_size // repeats
        self.incomplete_policy, self.generator = incomplete_policy, generator

    def __iter__(self):
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for offset in range(0, len(order), self.images_per_batch):
            selected = order[offset : offset + self.images_per_batch]
            if len(selected) < self.images_per_batch and self.incomplete_policy == "error":
                raise ValueError("Final image group batch is incomplete; use --incomplete-batch keep to retain it")
            yield [sample for group in selected for sample in self.groups[group]]

    def __len__(self) -> int:
        quotient, remainder = divmod(len(self.groups), self.images_per_batch)
        if remainder and self.incomplete_policy == "error":
            return quotient + 1  # iteration raises on this final batch
        return quotient + bool(remainder)


def build_optimizer(model: ATMAlignmentModel, fusion: NVOLLayerFusion, logit_scale: torch.nn.Parameter,
                    lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([*model.parameters(), *fusion.parameters(), logit_scale], lr=lr)


def _validate_grouped_batch(image_idx: torch.Tensor, repeats: int = 4) -> None:
    _, counts = torch.unique(image_idx, return_counts=True)
    if len(image_idx) % repeats or not torch.all(counts == repeats):
        raise ValueError("Training batch must contain every image ID exactly four times")


def train_epoch(model: ATMAlignmentModel, fusion: NVOLLayerFusion, loader: DataLoader, features: torch.Tensor,
                optimizer: torch.optim.Optimizer, logit_scale: torch.nn.Parameter, device: torch.device,
                max_train_batches: int | None = None) -> float:
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive")
    model.train(); fusion.train()
    total = examples = 0.0
    for batch_index, (eeg, image_idx) in enumerate(loader):
        if max_train_batches is not None and batch_index >= max_train_batches:
            break
        _validate_grouped_batch(image_idx)
        eeg, image_idx = eeg.to(device), image_idx.to(device)
        loss = symmetric_multi_positive_infonce(model(eeg), fusion(features[image_idx]), image_idx, logit_scale)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.detach().item() * len(eeg)
        examples += len(eeg)
    return total / examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True, help="nvol_train_features.pt from cache_nvol_features.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="8")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--incomplete-batch", choices=("keep", "error"), default="keep")
    parser.add_argument("--max-train-batches", type=int, help="Diagnostic-only cap on complete grouped batches per epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or (args.max_train_batches is not None and args.max_train_batches <= 0):
        raise ValueError("--epochs, --batch-size, --lr, and --max-train-batches must be positive")
    if not args.visual_cache.is_file():
        raise FileNotFoundError("Missing prerequisite visual cache; run tools/cache_nvol_features.py for train first")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training but CUDA is unavailable")
    _configure_cudnn(args.disable_cudnn)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    split = load_sub08_split(args.data_root, "train", subject=8)
    ordered_names = split.names[::split.repeats]
    cache = FeatureCache.load(
        args.visual_cache, expected_names=ordered_names,
        expected_image_ids=canonical_image_ids(args.data_root, "train"),
    )
    features = cache.features.float().to(device)
    if features.shape[1:] != (3, 1280):
        raise ValueError("Visual cache must contain frozen [N,3,1280] features")
    model, fusion = ATMAlignmentModel().to(device), NVOLLayerFusion().to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), device=device, dtype=torch.float32))
    optimizer = build_optimizer(model, fusion, logit_scale, args.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(split.eeg).float(), torch.from_numpy(split.image_idx)),
        batch_sampler=GroupedTrialBatchSampler(torch.from_numpy(split.image_idx), args.batch_size, split.repeats, args.incomplete_batch),
    )
    losses = [train_epoch(model, fusion, loader, features, optimizer, logit_scale, device, args.max_train_batches) for _ in range(args.epochs)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "final_epoch.pt"
    metadata = {"subject": 8, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                "cache_hash": cache.name_hash, "cache_file_sha256": _hash_file(args.visual_cache), "checkpoint_selection": "final_epoch", "incomplete_batch": args.incomplete_batch, "max_train_batches": args.max_train_batches, "losses": losses} | _runtime_provenance(device, args.disable_cudnn)
    metadata["atm_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    torch.save({"atm": model.state_dict(), "fusion": fusion.state_dict(), "logit_scale": logit_scale.detach().cpu(), "metadata": metadata}, checkpoint)
    (args.output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved final epoch checkpoint: {checkpoint}; final_loss={losses[-1]:.6f}; logit_scale={logit_scale.item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
