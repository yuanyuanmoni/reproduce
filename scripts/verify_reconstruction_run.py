#!/usr/bin/env python
"""Fail-closed verification for a complete sub-08 reconstruction evidence run.

The verifier never writes into ``--run-dir``.  A run is formal evidence only
when its immutable evidence manifest binds every persisted artifact to the
official 200-image test order supplied by ``--data-root``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch.torch_version import TorchVersion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_reconstructions import CONDITION_SOURCE, METRIC_METADATA, PAPER_TARGETS
from generate_reconstructions import assert_non_oracle_condition, load_prediction_bundle, tensor_sha256
from nvol_retrieval.data import canonical_image_ids, load_sub08_split
from nvol_retrieval.reconstruction_data import ReconstructionTargetCache, mean_80_raw_test_trials
from nvol_retrieval.vision import name_hash


COUNT = 200
RAW_80_AGGREGATION = "mean_80_raw_eeg_before_atm_once"
EVIDENCE_FILE = "reconstruction_evidence.json"
ARTIFACTS = (
    "retrieval_checkpoint", "decoder_checkpoint", "train_target_cache", "train_metadata",
    "prediction_bundle", "generation_manifest", "metrics_csv", "metrics_json", "comparison_grid",
)
METRICS = tuple(PAPER_TARGETS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid or missing {path.name}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return value


def _torch(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with torch.serialization.safe_globals([TorchVersion]):
            value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"invalid or missing {path.name}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a mapping")
        return {}
    return value


def _artifact_path(run: Path, spec: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        errors.append(f"evidence artifact {label} must contain only path and sha256")
        return None
    relative, expected_sha = spec["path"], spec["sha256"]
    if not isinstance(relative, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        errors.append(f"evidence artifact {label} has invalid path or SHA-256")
        return None
    path = (run / relative).resolve()
    try:
        path.relative_to(run.resolve())
    except ValueError:
        errors.append(f"evidence artifact {label} must be inside run-dir")
        return None
    if not path.is_file() or path.is_symlink():
        errors.append(f"evidence artifact {label} is missing or not a regular file")
        return None
    if _sha256(path) != expected_sha:
        errors.append(f"evidence artifact {label} SHA-256 mismatch")
    return path


def _official_test_identities(data_root: Path) -> tuple[list[str], list[str]]:
    test = load_sub08_split(data_root, "test", subject=8)
    # This validates the raw official [200,80,63,250] contract independently
    # from the persisted prediction label.  It intentionally discards values.
    if mean_80_raw_test_trials(test.eeg).shape != (COUNT, 63, 250):
        raise ValueError("official test EEG is not the expected raw-80 sub-08 array")
    image_ids = canonical_image_ids(data_root, "test")
    if len(test.names) != COUNT or len(image_ids) != COUNT:
        raise ValueError("official sub-08 test manifest must contain exactly 200 identities")
    return test.names, image_ids


def _validate_evidence(run: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, Path]]:
    evidence = _json(run / EVIDENCE_FILE, errors)
    if evidence.get("real_data") is not True or evidence.get("subject") != 8:
        errors.append("evidence must declare real_data=true and subject=8")
    if evidence.get("checkpoint_selection") != "final_epoch":
        errors.append("evidence must use final_epoch checkpoint selection")
    if evidence.get("trial_aggregation") != RAW_80_AGGREGATION:
        errors.append("evidence must record mean_80_raw_eeg_before_atm_once")
    if evidence.get("condition_source") != CONDITION_SOURCE:
        errors.append("evidence condition source must be the prediction bundle")
    specs = evidence.get("artifacts")
    if not isinstance(specs, dict) or set(specs) != set(ARTIFACTS):
        errors.append("evidence must list exactly the required immutable artifacts")
        return evidence, {}
    paths = {label: _artifact_path(run, specs[label], label, errors) for label in ARTIFACTS}
    resolved = [path for path in paths.values() if path is not None]
    if len(resolved) != len(set(resolved)):
        errors.append("evidence artifacts must use distinct paths; overwrite aliases are prohibited")
    return evidence, {label: path for label, path in paths.items() if path is not None}


def _validate_training(paths: dict[str, Path], errors: list[str]) -> None:
    required = {"retrieval_checkpoint", "decoder_checkpoint", "train_target_cache", "train_metadata"}
    if not required <= set(paths):
        return
    retrieval, decoder = _torch(paths["retrieval_checkpoint"], errors), _torch(paths["decoder_checkpoint"], errors)
    train = _json(paths["train_metadata"], errors)
    try:
        targets = ReconstructionTargetCache.load(paths["train_target_cache"], expected_split="train")
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"invalid train reconstruction target cache: {error}")
        return
    rmeta, dmeta = retrieval.get("metadata"), decoder.get("metadata")
    if not isinstance(rmeta, dict) or rmeta.get("subject") != 8 or rmeta.get("checkpoint_selection") != "final_epoch":
        errors.append("retrieval checkpoint must be a final-epoch sub-08 checkpoint")
    if not isinstance(dmeta, dict) or dmeta != train or dmeta.get("subject") != 8 or dmeta.get("checkpoint_selection") != "final_epoch":
        errors.append("decoder checkpoint metadata must exactly match final-epoch train metadata")
    elif (dmeta.get("retrieval_checkpoint_sha256") != _sha256(paths["retrieval_checkpoint"])
          or dmeta.get("train_target_cache_sha256") != _sha256(paths["train_target_cache"])
          or dmeta.get("train_target_identity_hash") != targets.name_hash):
        errors.append("decoder checkpoint does not bind its retrieval checkpoint and train target cache")
    if isinstance(rmeta, dict) and rmeta.get("cache_file_sha256") != targets.source_feature_cache_sha256:
        errors.append("train target cache source feature provenance does not match retrieval checkpoint")


def _validate_prediction_and_generation(paths: dict[str, Path], data_root: Path, errors: list[str]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    needed = {"prediction_bundle", "generation_manifest", "retrieval_checkpoint", "decoder_checkpoint"}
    if not needed <= set(paths):
        return [], [], []
    try:
        bundle = load_prediction_bundle(paths["prediction_bundle"])
        official_names, official_ids = _official_test_identities(data_root)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"invalid prediction/official test evidence: {error}")
        return [], [], []
    if bundle["names"] != official_names or bundle["image_ids"] != official_ids:
        errors.append("prediction bundle order is not the official canonical 200-image test order")
    metadata = bundle["metadata"]
    if (metadata.get("trial_aggregation") != RAW_80_AGGREGATION
            or metadata.get("retrieval_checkpoint_sha256") != _sha256(paths["retrieval_checkpoint"])
            or metadata.get("decoder_checkpoint_sha256") != _sha256(paths["decoder_checkpoint"])):
        errors.append("prediction bundle aggregation or checkpoint provenance mismatch")
    try:
        records = [json.loads(line) for line in paths["generation_manifest"].read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        errors.append(f"invalid generation manifest: {error}")
        return bundle["names"], bundle["image_ids"], []
    if len(records) != COUNT or not all(isinstance(record, dict) for record in records):
        errors.append("generation manifest must contain exactly 200 records")
        return bundle["names"], bundle["image_ids"], []
    expected_hash = name_hash(bundle["names"], bundle["image_ids"])
    generated_paths: set[Path] = set()
    for index, record in enumerate(records):
        if (record.get("index") != index or record.get("name") != bundle["names"][index]
                or record.get("image_id") != bundle["image_ids"][index]):
            errors.append("generation manifest order/identity differs from prediction bundle")
            break
        try:
            assert_non_oracle_condition(record)
        except ValueError as error:
            errors.append(str(error))
            break
        if (record.get("trial_aggregation") != RAW_80_AGGREGATION
                or record.get("prediction_identity_hash") != expected_hash
                or record.get("prediction_bundle_sha256") != _sha256(paths["prediction_bundle"])
                or record.get("retrieval_checkpoint_sha256") != metadata["retrieval_checkpoint_sha256"]
                or record.get("decoder_checkpoint_sha256") != metadata["decoder_checkpoint_sha256"]
                or record.get("condition_feature_sha256") != tensor_sha256(bundle["predicted_features"][index])):
            errors.append("generation manifest prediction condition provenance mismatch")
            break
        image = Path(record.get("generated_path", ""))
        if not image.is_file() or image.is_symlink():
            errors.append("generation manifest has a missing or symlinked generated image")
            break
        image = image.resolve()
        if image in generated_paths:
            errors.append("generation manifest reuses a generated image path; overwrite evidence is prohibited")
            break
        generated_paths.add(image)
    return bundle["names"], bundle["image_ids"], records


def _validate_metrics(paths: dict[str, Path], names: list[str], image_ids: list[str], records: list[dict[str, Any]], errors: list[str]) -> None:
    if not {"metrics_csv", "metrics_json", "generation_manifest"} <= set(paths):
        return
    metric_json = _json(paths["metrics_json"], errors)
    if (metric_json.get("count") != COUNT or metric_json.get("condition_source") != CONDITION_SOURCE
            or metric_json.get("paper_ts_diff_targets") != PAPER_TARGETS
            or metric_json.get("metric_metadata") != METRIC_METADATA
            or set(metric_json.get("summary", {})) != set(METRICS)):
        errors.append("metrics JSON must contain exactly the five paper metrics, targets, and directions")
    if metric_json.get("manifest") != str(paths["generation_manifest"].resolve()):
        errors.append("metrics JSON manifest provenance mismatch")
    try:
        with paths["metrics_csv"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        errors.append(f"invalid metrics CSV: {error}")
        return
    expected_fields = {"index", "name", "image_id", *METRICS}
    if len(rows) != COUNT or set(rows[0] if rows else ()) != expected_fields:
        errors.append("metrics CSV must have exactly 200 rows and the five required metric columns")
        return
    values = {metric: [] for metric in METRICS}
    for index, row in enumerate(rows):
        if row.get("index") != str(index) or row.get("name") != names[index] or row.get("image_id") != image_ids[index]:
            errors.append("metrics CSV identity/order differs from prediction bundle")
            return
        if records and (records[index].get("name"), records[index].get("image_id")) != (names[index], image_ids[index]):
            errors.append("metrics CSV identity/order differs from generation manifest")
            return
        try:
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(metric)
                values[metric].append(value)
        except (KeyError, ValueError):
            errors.append("metrics CSV contains non-finite metric values")
            return
    saved = metric_json.get("summary", {})
    for metric in METRICS:
        if not isinstance(saved.get(metric), (int, float)) or abs(float(saved[metric]) - sum(values[metric]) / COUNT) > 1e-9:
            errors.append(f"metrics JSON {metric} does not match the persisted per-image CSV")


def verify_reconstruction_run(run_dir: str | Path, data_root: str | Path) -> list[str]:
    """Return violations; no violations is a complete, real-data evidence bundle."""
    run, data = Path(run_dir), Path(data_root)
    errors: list[str] = []
    evidence, paths = _validate_evidence(run, errors)
    _validate_training(paths, errors)
    names, image_ids, records = _validate_prediction_and_generation(paths, data, errors)
    if len(names) == COUNT and len(image_ids) == COUNT:
        _validate_metrics(paths, names, image_ids, records, errors)
    identity_hash = name_hash(names, image_ids) if names and image_ids else None
    if evidence.get("test_identity_hash") != identity_hash:
        errors.append("evidence test_identity_hash does not match the official ordered test identities")
    return errors


def render_report(run_dir: str | Path, data_root: str | Path) -> str:
    errors = verify_reconstruction_run(run_dir, data_root)
    status = "VERIFIED" if not errors else "PENDING / FAILED"
    rows = "\n".join(f"- {error}" for error in errors) or "- All fail-closed evidence checks passed."
    return f"# Sub-08 Image Reconstruction Evidence\n\nStatus: **{status}**\n\n{rows}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", action="store_true", help="Print a status report; never writes artifacts.")
    args = parser.parse_args()
    errors = verify_reconstruction_run(args.run_dir, args.data_root)
    if args.report:
        print(render_report(args.run_dir, args.data_root), end="")
    elif errors:
        print("FAILED\n" + "\n".join(f"- {error}" for error in errors))
    else:
        print("PASS: complete, prediction-conditioned sub-08 reconstruction evidence is verified")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
