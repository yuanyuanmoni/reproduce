#!/usr/bin/env python
"""Fail-closed protocol verification for a completed sub-08 retrieval run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import torch
from torch.torch_version import TorchVersion

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from nvol_retrieval.vision import name_hash

COUNT = 200
AGGREGATION = "mean_80_raw_eeg_before_atm"
METRICS = ("raw_top1", "raw_top5", "csls_top1", "csls_top5")
ALLOWED_FILES = {
    "commands.txt", "data_contract.json", "environment.yml", "final_epoch.pt",
    "run_metadata.json", "train_metadata.json", "cache/nvol_train_features.pt",
    "cache/nvol_test_features.pt", "evaluation/metrics.json", "evaluation/query_names.json",
    "evaluation/rankings.pt", "evaluation/scores.pt",
    "cache/nvol_train_features.provenance.json", "cache/nvol_test_features.provenance.json",
}


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid or missing {path.name}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _load_torch(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        # Formal train metadata contains torch.__version__ (TorchVersion in PyTorch 2.7).
        # Keep all artifact loads restricted to tensors and this explicit metadata scalar.
        with torch.serialization.safe_globals([TorchVersion]):
            value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"invalid or missing {path.name}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a mapping")
        return {}
    return value


def _checked_ranking(scores: Any, stored_order: Any) -> tuple[torch.Tensor, dict[str, float]] | None:
    """Validate a complete descending order, allowing only exactly equal ties."""
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float16, torch.float32, torch.float64)
            or tuple(scores.shape) != (COUNT, COUNT) or not torch.isfinite(scores).all()):
        return None
    if (not isinstance(stored_order, torch.Tensor) or stored_order.dtype not in (torch.int32, torch.int64)
            or tuple(stored_order.shape) != (COUNT, COUNT)):
        return None
    order = stored_order.to(torch.int64)
    if not torch.equal(order.sort(dim=1).values, torch.arange(COUNT).expand(COUNT, -1)):
        return None
    ordered_scores = scores.gather(1, order)
    # Different PyTorch backends may order equal values differently.  This is
    # deliberately exact: a near tie is still an unequal score and may not swap.
    if torch.any(ordered_scores[:, 1:] > ordered_scores[:, :-1]):
        return None
    ranks = order.argsort(dim=1).diagonal() + 1
    return ranks, {"top1": (ranks <= 1).float().mean().item(), "top5": (ranks <= 5).float().mean().item()}


def _check_matrices(evaluation: Path, metrics: dict[str, Any], record: dict[str, Any], errors: list[str]) -> None:
    scores = _load_torch(evaluation / "scores.pt", errors)
    rankings = _load_torch(evaluation / "rankings.pt", errors)
    computed: dict[str, float] = {}
    for prefix in ("raw", "csls"):
        value = scores.get(f"{prefix}_scores")
        if (not isinstance(value, torch.Tensor) or value.dtype not in (torch.float16, torch.float32, torch.float64)
                or tuple(value.shape) != (COUNT, COUNT) or not torch.isfinite(value).all()):
            errors.append(f"{prefix}_scores must be a finite floating [200,200] matrix")
            continue
        if prefix == "raw" and (value.min().item() < -1.00001 or value.max().item() > 1.00001):
            errors.append("raw_scores must be cosine similarities in [-1,1]")
        if prefix == "csls" and (value.min().item() < -4.00001 or value.max().item() > 4.00001):
            errors.append("csls_scores must be in the theoretical [-4,4] range")
        result = _checked_ranking(value, rankings.get(f"{prefix}_order"))
        if result is None:
            errors.append(f"{prefix}_order is not a complete descending score order except for exactly equal ties")
            continue
        ranks, result_metrics = result
        if not isinstance(rankings.get(f"{prefix}_ranks"), torch.Tensor) or not torch.equal(rankings[f"{prefix}_ranks"], ranks):
            errors.append(f"{prefix}_ranks does not match stored score matrix")
        computed.update({f"{prefix}_top1": result_metrics["top1"], f"{prefix}_top5": result_metrics["top5"]})
    for key, expected in computed.items():
        for source, payload in (("metrics.json", metrics), ("run metadata", record.get("metrics", {}))):
            value = payload.get(key) if isinstance(payload, dict) else None
            if not isinstance(value, (int, float)) or abs(value - expected) > 1e-7:
                errors.append(f"{source} {key} does not match stored score matrix")


def _check_cache(path: Path, expected_count: int, expected_hash: Any, expected_sha: Any, errors: list[str]) -> None:
    payload = _load_torch(path, errors)
    names, image_ids = payload.get("names"), payload.get("image_ids")
    if not isinstance(names, list) or not isinstance(image_ids, list) or len(names) != expected_count or len(image_ids) != expected_count:
        errors.append(f"{path.name} must record {expected_count} ordered image identities")
        return
    observed = name_hash(names, image_ids)
    if payload.get("name_hash") != observed or expected_hash != observed:
        errors.append(f"{path.name} ordered identity hash mismatch")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or not torch.is_floating_point(features) or tuple(features.shape) != (expected_count, 3, 1280):
        errors.append(f"{path.name} must contain finite [N,3,1280] frozen visual features")
    elif not torch.isfinite(features).all():
        errors.append(f"{path.name} contains non-finite visual features")
    if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
        errors.append(f"{path.name} file SHA-256 mismatch")


def _check_cache_provenance(run: Path, record: dict[str, Any], errors: list[str]) -> None:
    provenance = record.get("cache_provenance")
    if record.get("cache_identity_hash_kind") != "canonical_name_and_image_id_sha256" or not isinstance(provenance, dict):
        errors.append("run metadata must distinguish canonical cache identity hashes from data-contract names-only hashes")
        return
    cache_commands = [shlex.split(command) for command in record.get("commands", []) if "cache_nvol_features.py" in command]
    if len(cache_commands) != 1:
        errors.append("run metadata must record exactly one cache extraction command")
        return
    command = cache_commands[0]
    def option(name: str) -> str | None:
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError):
            return None
    checkpoint = record.get("visual_checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"resolved_path", "sha256"} or option("--checkpoint") != checkpoint.get("resolved_path"):
        errors.append("recorded cache command checkpoint does not match visual checkpoint provenance")
    elif not isinstance(checkpoint["resolved_path"], str) or not isinstance(checkpoint["sha256"], str) or not Path(checkpoint["resolved_path"]).is_file() or Path(checkpoint["resolved_path"]).is_symlink():
        errors.append("recorded visual checkpoint must exist as a regular file")
    elif _sha256(Path(checkpoint["resolved_path"])) != checkpoint["sha256"]:
        errors.append("recorded visual checkpoint SHA-256 mismatch")
    contract = _read_json(run / "data_contract.json", errors)
    contract_hashes = record.get("data_contract_ordered_names_sha256")
    if not isinstance(contract_hashes, dict) or any(
        contract_hashes.get(split) != contract.get(split, {}).get("ordered_names_sha256") for split in ("train", "test")
    ):
        errors.append("run metadata data-contract names-only hashes do not match the data contract")
    cudnn_disabled_in_command = "--disable-cudnn" in command
    for split in ("train", "test"):
        artifact = _read_json(run / "cache" / f"nvol_{split}_features.provenance.json", errors)
        value = provenance.get(split)
        if value != artifact or artifact.get("cache_identity_hash") != record.get(f"{split}_cache_hash"):
            errors.append(f"{split} cache provenance does not match cache identity hash")
        if not isinstance(artifact.get("disable_cudnn_requested"), bool) or not isinstance(artifact.get("cudnn_enabled_effective"), bool):
            errors.append(f"{split} cache provenance lacks requested/effective cuDNN state")
        elif artifact["disable_cudnn_requested"] != cudnn_disabled_in_command or (artifact["disable_cudnn_requested"] and artifact["cudnn_enabled_effective"]):
            errors.append(f"{split} cache provenance cuDNN state does not match recorded cache command")
        if artifact.get("source_checkpoint") != checkpoint:
            errors.append(f"{split} cache provenance checkpoint does not match recorded visual checkpoint")
        payload = _load_torch(run / "cache" / f"nvol_{split}_features.pt", errors)
        names = payload.get("names")
        if not isinstance(names, list) or _names_hash(names) != contract.get(split, {}).get("ordered_names_sha256"):
            errors.append(f"{split} cache names-only hash does not match data contract")


def _check_allowed_files(run: Path, errors: list[str]) -> None:
    for path in run.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlinked artifact is prohibited: {path.relative_to(run)}")
        if path.is_file() and path.relative_to(run).as_posix() not in ALLOWED_FILES:
            errors.append(f"unexpected artifact is prohibited: {path.relative_to(run)}")


def _check_required_files(run: Path, errors: list[str]) -> None:
    for relative in sorted(ALLOWED_FILES):
        path = run / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"required evidence artifact is missing or not a regular file: {relative}")


def verify_run(run_dir: str | Path) -> list[str]:
    """Return protocol violations; an empty list passes only for real-data runs."""
    run = Path(run_dir)
    errors: list[str] = []
    record = _read_json(run / "run_metadata.json", errors)
    train = _read_json(run / "train_metadata.json", errors)
    contract = _read_json(run / "data_contract.json", errors)
    evaluation = run / "evaluation"
    metrics = _read_json(evaluation / "metrics.json", errors)
    identities = _read_json(evaluation / "query_names.json", errors)
    _check_required_files(run, errors)
    _check_allowed_files(run, errors)
    if record.get("real_data") is not True:
        errors.append("synthetic smoke artifacts are not formal real-data evidence")
    if record.get("subject") != 8 or train.get("subject") != 8 or contract.get("subject") != "sub-08":
        errors.append("only subject=8 is valid")
    if contract.get("train", {}).get("official_array_shape") != [16540, 4, 63, 250] or contract.get("test", {}).get("official_array_shape") != [200, 80, 63, 250]:
        errors.append("data contract does not prove official sub-08 train/test shapes")
    contract_path = run / "data_contract.json"
    if not contract_path.is_file() or record.get("data_contract_sha256") != _sha256(contract_path):
        errors.append("data contract SHA-256 mismatch")
    if record.get("checkpoint_selection") != "final_epoch" or train.get("checkpoint_selection") != "final_epoch":
        errors.append("test-set checkpoint selection is prohibited; require final_epoch")
    if record.get("trial_aggregation") != AGGREGATION or metrics.get("trial_aggregation") != AGGREGATION:
        errors.append(f"trial aggregation must be {AGGREGATION}")
    if record.get("ground_truth") != "exact_image_identity":
        errors.append("ground truth must be exact_image_identity, not class-level labels")
    if not record.get("commands") or record.get("environment_export") != "environment.yml" or not record.get("device") or not isinstance(record.get("seed"), int):
        errors.append("run metadata must record commands, environment export, device, and integer seed")
    else:
        commands_path, environment_path = run / "commands.txt", run / "environment.yml"
        if commands_path.is_file() and commands_path.read_text(encoding="utf-8").splitlines() != record["commands"]:
            errors.append("commands.txt does not exactly match run metadata")
        if environment_path.is_file() and not environment_path.read_text(encoding="utf-8").strip():
            errors.append("environment export is empty")
    if not isinstance(record.get("metrics"), dict) or not set(METRICS) <= set(record["metrics"]):
        errors.append("run metadata must record raw and CSLS Top-1/Top-5 metrics")
    names, image_ids = identities.get("names"), identities.get("image_ids")
    if not isinstance(names, list) or not isinstance(image_ids, list) or len(names) != COUNT or len(image_ids) != COUNT:
        errors.append("evaluation requires exactly 200 ordered candidates")
    elif len(set(names)) != COUNT or len(set(image_ids)) != COUNT or any(not str(value).startswith("image_set/") for value in image_ids):
        errors.append("candidate identities must be 200 unique canonical exact-image IDs")
    else:
        observed = name_hash(names, image_ids)
        if identities.get("candidate_name_hash") != observed or metrics.get("candidate_name_hash") != observed or record.get("test_cache_hash") != observed:
            errors.append("candidate order/hash mismatch")
    _check_cache(run / "cache/nvol_train_features.pt", 16540, record.get("train_cache_hash"), record.get("train_cache_file_sha256"), errors)
    _check_cache(run / "cache/nvol_test_features.pt", COUNT, record.get("test_cache_hash"), record.get("test_cache_file_sha256"), errors)
    _check_cache_provenance(run, record, errors)
    if train.get("cache_hash") != record.get("train_cache_hash") or train.get("cache_file_sha256") != record.get("train_cache_file_sha256"):
        errors.append("training cache provenance does not match run metadata")
    _check_matrices(evaluation, metrics, record, errors)
    checkpoint = _load_torch(run / "final_epoch.pt", errors)
    checkpoint_metadata = checkpoint.get("metadata")
    if not {"atm", "fusion", "logit_scale", "metadata"} <= set(checkpoint) or not isinstance(checkpoint.get("atm"), dict) or not checkpoint["atm"] or not isinstance(checkpoint.get("fusion"), dict) or not checkpoint["fusion"]:
        errors.append("checkpoint lacks trained ATM/fusion state")
    elif not isinstance(checkpoint.get("logit_scale"), torch.Tensor) or checkpoint["logit_scale"].shape != torch.Size([]) or not torch.isfinite(checkpoint["logit_scale"]).item():
        errors.append("checkpoint has invalid logit_scale")
    elif checkpoint_metadata != train:
        errors.append("checkpoint metadata differs from train metadata")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = verify_run(args.run_dir)
    if errors:
        print("FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("PASS: final-only sub-08 exact-image fixed-200 retrieval evidence is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
