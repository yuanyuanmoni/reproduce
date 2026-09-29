"""Generate sub-08 reconstructions from a saved EEG prediction bundle only.

The command deliberately has no image-input argument.  Held-out stimulus pixels
are never opened here: the only SDXL/IP-Adapter condition is a row from the
immutable Task-3 ``predicted_features`` tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.torch_version import TorchVersion

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.vision import FEATURE_DIM, name_hash


CONDITION_SOURCE = "prediction_bundle.predicted_features"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def load_prediction_bundle(path: Path) -> dict[str, Any]:
    """Load and validate the immutable Task-3 payload, without image access."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction bundle not found: {path}")
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    fields = {"names", "image_ids", "predicted_features", "metadata"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Invalid reconstruction prediction bundle")
    names = payload["names"]
    image_ids = payload["image_ids"]
    features = payload["predicted_features"]
    metadata = payload["metadata"]
    if (not isinstance(names, list) or not isinstance(image_ids, list) or not isinstance(metadata, dict)
            or not isinstance(features, torch.Tensor) or features.shape != (200, FEATURE_DIM)
            or not torch.is_floating_point(features) or not torch.isfinite(features).all()):
        raise ValueError("Prediction bundle requires 200 finite [1024] features")
    if len(set(names)) != 200 or len(set(image_ids)) != 200:
        raise ValueError("Prediction bundle requires 200 unique ordered identities")
    expected = name_hash(names, image_ids)
    if metadata.get("subject") != 8 or metadata.get("count") != 200 or metadata.get("identity_hash") != expected:
        raise ValueError("Prediction bundle metadata identity is invalid")
    if metadata.get("trial_aggregation") != "mean_80_raw_eeg_before_atm_once":
        raise ValueError("Prediction bundle does not document the required raw-80 trial aggregation")
    if not all(isinstance(metadata.get(key), str) and len(metadata[key]) == 64 for key in
               ("retrieval_checkpoint_sha256", "decoder_checkpoint_sha256")):
        raise ValueError("Prediction bundle checkpoint provenance is invalid")
    return {"names": names, "image_ids": image_ids, "predicted_features": features, "metadata": metadata}


def validate_bundle_against_test_manifest(bundle: dict[str, Any], names: list[str], image_ids: list[str]) -> None:
    """Fail closed if test EEG/image manifest order differs from Task-3 output."""
    if len(names) != 200 or len(image_ids) != 200 or len(set(names)) != 200 or len(set(image_ids)) != 200:
        raise ValueError("Official test manifest must contain 200 unique ordered identities")
    if bundle["names"] != names or bundle["image_ids"] != image_ids:
        raise ValueError("Prediction bundle identities/order do not match the official test manifest")


def condition_from_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """The sole condition routing boundary; it accepts no image/oracle argument."""
    if not isinstance(prediction, torch.Tensor) or prediction.shape != (FEATURE_DIM,):
        raise ValueError("Each saved prediction must be a finite [1024] feature")
    if not torch.is_floating_point(prediction) or not torch.isfinite(prediction).all():
        raise ValueError("Each saved prediction must be finite")
    return prediction.detach().clone().reshape(1, 1, FEATURE_DIM)


def assert_non_oracle_condition(record: dict[str, Any]) -> None:
    if record.get("condition_source") != CONDITION_SOURCE:
        raise ValueError("Generation manifest condition source is not the saved prediction bundle")
    forbidden = {"original_feature_path", "oracle_feature_path", "condition_image_path", "test_visual_feature"}
    if forbidden.intersection(record):
        raise ValueError("Generation manifest contains an oracle visual condition")


def _preflight_assets(args: argparse.Namespace) -> None:
    required = [args.sdxl_path / "model_index.json", args.ip_adapter_image_encoder / "config.json", args.ip_adapter_weights]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("All generation assets must be supplied locally:\n" + "\n".join(missing))
    try:
        args.ip_adapter_weights.resolve().relative_to(args.ip_adapter_root.resolve())
    except ValueError as error:
        raise ValueError("--ip-adapter-weights must be inside --ip-adapter-root") from error


def _load_local_pipeline(args: argparse.Namespace, dtype: torch.dtype):
    """Load only paths provided by the caller; every HF access is offline."""
    try:
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPVisionModelWithProjection
    except ImportError as error:
        raise RuntimeError("Generation requires locally installed diffusers and transformers") from error
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.ip_adapter_image_encoder, local_files_only=True, torch_dtype=dtype
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_path, image_encoder=encoder, local_files_only=True, torch_dtype=dtype,
    )
    relative_parent = args.ip_adapter_weights.parent.resolve().relative_to(args.ip_adapter_root.resolve())
    pipe.load_ip_adapter(
        str(args.ip_adapter_root), subfolder=str(relative_parent) if str(relative_parent) != "." else None,
        weight_name=args.ip_adapter_weights.name, image_encoder_folder=None, local_files_only=True,
    )
    return pipe.to(args.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Used only for canonical test manifest/reference paths.")
    parser.add_argument("--prediction-bundle", type=Path, required=True)
    parser.add_argument("--sdxl-path", type=Path, required=True)
    parser.add_argument("--ip-adapter-root", type=Path, required=True)
    parser.add_argument("--ip-adapter-image-encoder", type=Path, required=True)
    parser.add_argument("--ip-adapter-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--limit", type=int, default=200, help="Generate the first N ordered test items; use 200 for the formal run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps <= 0 or args.height <= 0 or args.width <= 0 or not 1 <= args.limit <= 200:
        raise ValueError("--steps, --height, --width must be positive and --limit must be in [1,200]")
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite an immutable reconstruction generation directory")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA generation but CUDA is unavailable")
    _preflight_assets(args)
    bundle = load_prediction_bundle(args.prediction_bundle)
    test = load_sub08_split(args.data_root, "test", subject=8)
    image_ids = canonical_image_ids(args.data_root, "test")
    validate_bundle_against_test_manifest(bundle, test.names, image_ids)

    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    pipe = _load_local_pipeline(args, dtype)
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    asset_hashes = {"sdxl_model_index_sha256": sha256_file(args.sdxl_path / "model_index.json"),
                    "ip_adapter_image_encoder_config_sha256": sha256_file(args.ip_adapter_image_encoder / "config.json"),
                    "ip_adapter_weights_sha256": sha256_file(args.ip_adapter_weights)}
    try:
        with manifest_path.open("x", encoding="utf-8") as manifest:
            for index, (name, image_id, prediction) in enumerate(zip(bundle["names"], bundle["image_ids"], bundle["predicted_features"])):
                if index >= args.limit:
                    break
                condition = condition_from_prediction(prediction).to(args.device, dtype=dtype)
                sample_seed = args.seed + index
                image = pipe(
                    prompt="", negative_prompt="", num_inference_steps=args.steps, guidance_scale=0.0,
                    height=args.height, width=args.width,
                    generator=torch.Generator(device=args.device).manual_seed(sample_seed),
                    ip_adapter_image_embeds=[condition],
                ).images[0]
                output = args.output_dir / "images" / f"{index:03d}.png"
                output.parent.mkdir(exist_ok=True)
                image.save(output)
                record = {"index": index, "name": name, "image_id": image_id,
                          "reference_path": str(args.data_root / image_id), "generated_path": str(output.resolve()),
                          "seed": sample_seed, "condition_source": CONDITION_SOURCE, "condition_feature_sha256": tensor_sha256(prediction),
                          "prediction_bundle": str(args.prediction_bundle.resolve()), "prediction_bundle_sha256": sha256_file(args.prediction_bundle),
                          "prediction_identity_hash": bundle["metadata"]["identity_hash"],
                          "retrieval_checkpoint_sha256": bundle["metadata"]["retrieval_checkpoint_sha256"],
                          "decoder_checkpoint_sha256": bundle["metadata"]["decoder_checkpoint_sha256"],
                          "trial_aggregation": bundle["metadata"]["trial_aggregation"], "steps": args.steps,
                          "guidance_scale": 0.0, "height": args.height, "width": args.width,
                          "prompt": "", "negative_prompt": "", "asset_paths": {"sdxl": str(args.sdxl_path), "ip_adapter_image_encoder": str(args.ip_adapter_image_encoder), "ip_adapter_weights": str(args.ip_adapter_weights)},
                          "asset_hashes": asset_hashes}
                assert_non_oracle_condition(record)
                manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # Do not claim partial images are a completed immutable generation run.
        raise
    print(f"saved {args.limit} prediction-conditioned reconstructions and immutable manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
