"""Export 200 test-only reconstruction features from a final decoder checkpoint."""

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
from nvol_retrieval.reconstruction import ConditionalNVOLPrior, FrozenLayer12Projection, SemanticAdapter
from nvol_retrieval.reconstruction_data import mean_80_raw_test_trials
from nvol_retrieval.vision import NVOLLayerFusion, name_hash
from train_reconstruction import _load_retrieval_checkpoint, _sha256


def _load_decoder(path: Path, device: torch.device) -> dict:
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict) or set(value) != {"prior", "adapter", "metadata"}:
        raise ValueError("Invalid reconstruction decoder checkpoint")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or metadata.get("subject") != 8 or metadata.get("checkpoint_selection") != "final_epoch":
        raise ValueError("Reconstruction decoder must be a final-epoch sub-08 checkpoint")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", default="8")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use native CUDA kernels when this host's cuDNN has no compatible convolution engine")
    args = parser.parse_args()
    if str(args.subject) not in {"8", "08", "sub-08"}:
        raise ValueError("This reproduction only supports sub-08")
    if args.batch_size <= 0 or not args.retrieval_checkpoint.is_file() or not args.decoder_checkpoint.is_file():
        raise ValueError("--batch-size must be positive and both checkpoints must exist")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable prediction bundle: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA export but CUDA is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled by --disable-cudnn")
    retrieval, decoder = _load_retrieval_checkpoint(args.retrieval_checkpoint, device), _load_decoder(args.decoder_checkpoint, device)
    metadata = decoder["metadata"]
    if metadata.get("retrieval_checkpoint_sha256") != _sha256(args.retrieval_checkpoint):
        raise ValueError("Decoder checkpoint was not trained from this ATM retrieval checkpoint")
    if metadata.get("retrieval_train_cache_sha256") != retrieval["metadata"].get("cache_file_sha256"):
        raise ValueError("Decoder and ATM retrieval checkpoint do not have matching train-cache provenance")
    test = load_sub08_split(args.data_root, "test", subject=8)
    names, image_ids = test.names, canonical_image_ids(args.data_root, "test")
    if len(names) != 200 or len(image_ids) != 200 or len(set(names)) != 200 or len(set(image_ids)) != 200:
        raise ValueError("Test manifest must contain 200 unique ordered stimulus identities")
    raw_mean = mean_80_raw_test_trials(test.eeg)
    atm, fusion = ATMAlignmentModel().to(device).eval(), NVOLLayerFusion().to(device).eval()
    atm.load_state_dict(retrieval["atm"]); fusion.load_state_dict(retrieval["fusion"])
    frozen_projection = FrozenLayer12Projection.from_retrieval_fusion(fusion)
    prior = ConditionalNVOLPrior(hidden_dim=metadata["prior_hidden_dim"], timesteps=metadata["prior_timesteps"]).to(device).eval()
    adapter = SemanticAdapter(frozen_projection, bottleneck_dim=metadata["adapter_bottleneck_dim"]).to(device).eval()
    prior.load_state_dict(decoder["prior"]); adapter.load_state_dict(decoder["adapter"])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(raw_mean), args.batch_size):
            condition = atm(torch.from_numpy(raw_mean[start:start + args.batch_size]).float().to(device))
            generated = prior.sample(condition, seed=args.seed + start, inference_steps=metadata["inference_steps"])
            predictions.append(adapter.from_generated(generated).cpu())
    predicted = torch.cat(predictions)
    if predicted.shape != (200, 1024) or not torch.isfinite(predicted).all():
        raise ValueError("Expected 200 finite [1024] reconstruction features")
    bundle_metadata = {"subject": 8, "count": 200, "trial_aggregation": "mean_80_raw_eeg_before_atm_once",
                       "retrieval_checkpoint_sha256": _sha256(args.retrieval_checkpoint), "decoder_checkpoint_sha256": _sha256(args.decoder_checkpoint),
                       "decoder_train_target_cache_sha256": metadata["train_target_cache_sha256"], "seed": args.seed,
                       "identity_hash": name_hash(names, image_ids), "disable_cudnn_requested": args.disable_cudnn,
                       "cudnn_enabled_effective": torch.backends.cudnn.enabled}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "image_ids": image_ids, "predicted_features": predicted, "metadata": bundle_metadata}, args.output)
    print(f"saved 200 ordered reconstruction features: {args.output}; hash={bundle_metadata['identity_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
