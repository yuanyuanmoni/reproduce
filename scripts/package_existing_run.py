#!/usr/bin/env python
"""Bind completed real-data artifacts into a fail-closed sub-08 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nvol_retrieval.vision import name_hash
from verify_run import verify_run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _command(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])


def package(run: Path, contract: dict[str, Any], environment: str, *, data_root: Path | None,
            checkpoint_path: Path, device: str, seed: int, disable_cudnn: bool) -> None:
    """Write provenance only after existing artifacts bind to one another."""
    required = [run / "final_epoch.pt", run / "train_metadata.json", run / "cache/nvol_train_features.pt",
                run / "cache/nvol_test_features.pt", run / "evaluation/metrics.json",
                run / "evaluation/query_names.json", run / "evaluation/rankings.pt", run / "evaluation/scores.pt",
                run / "cache/nvol_train_features.provenance.json", run / "cache/nvol_test_features.provenance.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing completed artifacts: " + ", ".join(missing))
    target_names = ("data_contract.json", "environment.yml", "commands.txt", "run_metadata.json")
    existing = [name for name in target_names if (run / name).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite evidence: " + ", ".join(existing))
    if contract.get("subject") != "sub-08" or contract.get("train", {}).get("official_array_shape") != [16540, 4, 63, 250] or contract.get("test", {}).get("official_array_shape") != [200, 80, 63, 250]:
        raise ValueError("data contract is not official sub-08 train/test data")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError("visual checkpoint must be an existing regular file")
    checkpoint = _load(run / "final_epoch.pt")
    train = _read(run / "train_metadata.json")
    if checkpoint.get("metadata") != train or train.get("subject") != 8 or train.get("checkpoint_selection") != "final_epoch":
        raise ValueError("checkpoint/train metadata does not establish a final sub-08 run")
    if train.get("seed") != seed or train.get("device") != device or train.get("disable_cudnn_requested") is not disable_cudnn:
        raise ValueError("provided seed/device/cuDNN settings do not match training metadata")
    caches = {split: _load(run / "cache" / f"nvol_{split}_features.pt") for split in ("train", "test")}
    provenance = {split: _read(run / "cache" / f"nvol_{split}_features.provenance.json") for split in ("train", "test")}
    for split, expected_count in (("train", 16540), ("test", 200)):
        cache = caches[split]
        if (not isinstance(cache.get("names"), list) or not isinstance(cache.get("image_ids"), list)
                or len(cache["names"]) != expected_count or name_hash(cache["names"], cache["image_ids"]) != cache.get("name_hash")
                or _names_hash(cache["names"]) != contract[split]["ordered_names_sha256"]):
            raise ValueError(f"{split} cache does not bind to the supplied data contract")
        if provenance[split].get("cache_identity_hash") != cache["name_hash"]:
            raise ValueError(f"{split} cache provenance does not bind to its cache")
        if provenance[split].get("source_checkpoint") != {"resolved_path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)}:
            raise ValueError(f"{split} cache uses a different visual checkpoint")
        if provenance[split].get("disable_cudnn_requested") is not disable_cudnn:
            raise ValueError(f"{split} cache cuDNN request differs from packaging input")
    metrics = _read(run / "evaluation/metrics.json")
    identities = _read(run / "evaluation/query_names.json")
    if name_hash(identities.get("names", []), identities.get("image_ids", [])) != caches["test"]["name_hash"]:
        raise ValueError("evaluation identities do not bind to the test cache")
    cudnn = ["--disable-cudnn"] if disable_cudnn else []
    py, root = sys.executable, ROOT
    data = str(data_root.resolve()) if data_root else "<supplied-data-contract>"
    commands = [
        _command([py, root / "tools/inspect_sub08_data.py", "--data-root", data, "--subject", "8", "--output", run / "data_contract.json"]),
        _command([py, root / "tools/cache_nvol_features.py", "--data-root", data, "--checkpoint", checkpoint_path, "--output-dir", run / "cache", "--subject", "8", "--device", device, *cudnn]),
        _command([py, root / "train.py", "--data-root", data, "--visual-cache", run / "cache/nvol_train_features.pt", "--output-dir", run, "--subject", "8", "--epochs", train["epochs"], "--batch-size", train["batch_size"], "--lr", train["lr"], "--incomplete-batch", train["incomplete_batch"], "--seed", seed, "--device", device, *cudnn]),
        _command([py, root / "evaluate.py", "--data-root", data, "--checkpoint", run / "final_epoch.pt", "--train-visual-cache", run / "cache/nvol_train_features.pt", "--test-visual-cache", run / "cache/nvol_test_features.pt", "--output-dir", run / "evaluation", "--subject", "8", "--csls-k", metrics["csls_k"], "--device", device, *cudnn]),
    ]
    record = {"real_data": data_root is not None, "subject": 8, "checkpoint_selection": "final_epoch",
              "trial_aggregation": "mean_80_raw_eeg_before_atm", "ground_truth": "exact_image_identity",
              "commands": commands, "environment_export": "environment.yml", "device": device, "seed": seed,
              "data_contract_sha256": hashlib.sha256(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n").hexdigest(),
              "data_contract_ordered_names_sha256": {split: contract[split]["ordered_names_sha256"] for split in ("train", "test")},
              "cache_identity_hash_kind": "canonical_name_and_image_id_sha256",
              "visual_checkpoint": provenance["train"]["source_checkpoint"],
              "train_cache_hash": caches["train"]["name_hash"], "test_cache_hash": caches["test"]["name_hash"],
              "train_cache_file_sha256": _sha256(run / "cache/nvol_train_features.pt"), "test_cache_file_sha256": _sha256(run / "cache/nvol_test_features.pt"),
              "cache_provenance": provenance,
              "metrics": {key: metrics[key] for key in ("raw_top1", "raw_top5", "csls_top1", "csls_top5")}}
    (run / "data_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run / "environment.yml").write_text(environment, encoding="utf-8")
    (run / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    record["data_contract_sha256"] = _sha256(run / "data_contract.json")
    (run / "run_metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--data-contract", type=Path, help="read-only precomputed contract; cannot claim real_data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--environment-file", type=Path, help="existing environment export to copy")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.data_root:
        with tempfile.TemporaryDirectory() as temporary:
            contract_path = Path(temporary) / "data_contract.json"
            subprocess.run([sys.executable, str(ROOT / "tools/inspect_sub08_data.py"), "--data-root", str(args.data_root.resolve()), "--subject", "8", "--output", str(contract_path)], check=True)
            contract = _read(contract_path)
    else:
        contract = _read(args.data_contract)
    if args.environment_file:
        environment = args.environment_file.read_text(encoding="utf-8")
    else:
        environment = subprocess.run(["conda", "env", "export", "--prefix", sys.prefix], check=True, text=True, capture_output=True).stdout
    if not environment.strip():
        raise ValueError("environment export is empty")
    package(run, contract, environment, data_root=args.data_root, checkpoint_path=args.checkpoint,
            device=args.device, seed=args.seed, disable_cudnn=args.disable_cudnn)
    errors = verify_run(run)
    if errors:
        print("FAILED\n" + "\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: existing real-data artifacts are now a complete verified evidence package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
