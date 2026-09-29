"""Train the final-epoch, train-only sub-08 reconstruction feature decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.atm import ATMAlignmentModel
from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.reconstruction import ConditionalNVOLPrior, FrozenLayer12Projection, SemanticAdapter, adapter_loss, prior_loss
from nvol_retrieval.reconstruction_data import ReconstructionTargetCache
from nvol_retrieval.vision import NVOLLayerFusion


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_retrieval_checkpoint(path: Path, device: torch.device) -> dict:
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict) or not {"atm", "fusion", "metadata"} <= set(value):
        raise ValueError("Invalid ATM retrieval checkpoint")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or metadata.get("subject") != 8 or metadata.get("checkpoint_selection") != "final_epoch":
        raise ValueError("Retrieval checkpoint must be the final epoch for sub-08")
    return value


def _load_train_targets(path: Path, names: list[str], image_ids: list[str]) -> ReconstructionTargetCache:
    cache = ReconstructionTargetCache.load(path, expected_names=names, expected_image_ids=image_ids)
    if cache.split != "train":
        raise ValueError("Reconstruction training hard-rejects test target caches")
    return cache


def _validate_provenance(retrieval: dict, targets: ReconstructionTargetCache) -> None:
    metadata = retrieval["metadata"]
    if metadata.get("cache_file_sha256") != targets.source_feature_cache_sha256 or metadata.get("cache_hash") != targets.name_hash:
        raise ValueError("ATM retrieval checkpoint and train target cache do not have the same source-cache provenance")


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def train_epoch(atm: torch.nn.Module, prior: ConditionalNVOLPrior, adapter: SemanticAdapter, loader: DataLoader,
                layer12: torch.Tensor, semantic: torch.Tensor, prior_optimizer: torch.optim.Optimizer,
                adapter_optimizer: torch.optim.Optimizer, device: torch.device, sample_seed: int,
                inference_steps: int, max_train_batches: int | None = None) -> dict[str, float]:
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive")
    prior.train(); adapter.train()
    totals = {"prior": 0.0, "adapter": 0.0}; examples = 0
    for batch_index, (eeg, image_idx) in enumerate(loader):
        if max_train_batches is not None and batch_index >= max_train_batches:
            break
        eeg, image_idx = eeg.to(device), image_idx.to(device)
        with torch.no_grad():
            condition = atm(eeg)
            target_layer12 = adapter.layer12_projection(layer12[image_idx])
        loss, _ = prior_loss(prior, condition, target_layer12)
        prior_optimizer.zero_grad(set_to_none=True); loss.backward(); prior_optimizer.step()
        with torch.no_grad():
            generated = prior.sample(condition, seed=sample_seed + batch_index, inference_steps=inference_steps)
        loss_adapter, _ = adapter_loss(adapter, layer12[image_idx], generated, semantic[image_idx])
        adapter_optimizer.zero_grad(set_to_none=True); loss_adapter.backward(); adapter_optimizer.step()
        totals["prior"] += loss.detach().item() * len(eeg); totals["adapter"] += loss_adapter.detach().item() * len(eeg); examples += len(eeg)
    if not examples:
        raise ValueError("No reconstruction training batches were run")
    return {key: value / examples for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--train-target-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="8")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prior-hidden-dim", type=int, default=1024)
    parser.add_argument("--prior-timesteps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=512)
    parser.add_argument("--max-train-batches", type=int, help="Diagnostic-only batch cap per epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    if min(args.epochs, args.batch_size, args.prior_hidden_dim, args.prior_timesteps, args.inference_steps, args.adapter_bottleneck_dim) <= 0 or args.lr <= 0:
        raise ValueError("Training dimensions, epochs, batch size, inference steps, and lr must be positive")
    if not args.retrieval_checkpoint.is_file() or not args.train_target_cache.is_file():
        raise FileNotFoundError("--retrieval-checkpoint and --train-target-cache must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA reconstruction training but CUDA is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled by --disable-cudnn")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    split = load_sub08_split(args.data_root, "train", subject=8)
    names, image_ids = split.names[::split.repeats], canonical_image_ids(args.data_root, "train")
    targets = _load_train_targets(args.train_target_cache, names, image_ids)
    retrieval = _load_retrieval_checkpoint(args.retrieval_checkpoint, device)
    _validate_provenance(retrieval, targets)
    atm, fusion = ATMAlignmentModel().to(device), NVOLLayerFusion().to(device)
    atm.load_state_dict(retrieval["atm"]); fusion.load_state_dict(retrieval["fusion"])
    _freeze(atm); _freeze(fusion)
    prior = ConditionalNVOLPrior(hidden_dim=args.prior_hidden_dim, timesteps=args.prior_timesteps).to(device)
    adapter = SemanticAdapter(FrozenLayer12Projection.from_retrieval_fusion(fusion), bottleneck_dim=args.adapter_bottleneck_dim).to(device)
    layer12, semantic = targets.layer12_pooled.float().to(device), targets.final_semantic.float().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(split.eeg).float(), torch.from_numpy(split.image_idx)), batch_size=args.batch_size, shuffle=True)
    prior_optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr)
    adapter_optimizer = torch.optim.AdamW(adapter.adapter.parameters(), lr=args.lr)
    losses = [train_epoch(atm, prior, adapter, loader, layer12, semantic, prior_optimizer, adapter_optimizer, device,
                          args.seed + epoch * 10_000, args.inference_steps, args.max_train_batches) for epoch in range(args.epochs)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "final_epoch_reconstruction.pt"
    metadata = {
        "subject": 8, "checkpoint_selection": "final_epoch", "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "prior_hidden_dim": args.prior_hidden_dim, "prior_timesteps": args.prior_timesteps,
        "inference_steps": args.inference_steps, "adapter_bottleneck_dim": args.adapter_bottleneck_dim,
        "max_train_batches": args.max_train_batches, "losses": losses,
        "retrieval_checkpoint_sha256": _sha256(args.retrieval_checkpoint),
        "retrieval_train_cache_sha256": retrieval["metadata"]["cache_file_sha256"],
        "train_target_cache_sha256": _sha256(args.train_target_cache), "train_target_identity_hash": targets.name_hash,
        "visual_checkpoint_sha256": targets.visual_checkpoint_sha256,
        "disable_cudnn_requested": args.disable_cudnn, "cudnn_enabled_effective": torch.backends.cudnn.enabled,
    }
    torch.save({"prior": prior.state_dict(), "adapter": adapter.state_dict(), "metadata": metadata}, checkpoint)
    (args.output_dir / "reconstruction_train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved final reconstruction epoch: {checkpoint}; prior={losses[-1]['prior']:.6f}; adapter={losses[-1]['adapter']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
