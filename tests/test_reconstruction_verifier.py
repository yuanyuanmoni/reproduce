"""Fail-closed evidence coverage for the formal reconstruction gate."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def _module():
    path = Path(__file__).parents[1] / "scripts" / "verify_reconstruction_run.py"
    spec = importlib.util.spec_from_file_location("verify_reconstruction_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(run: Path, monkeypatch) -> tuple[list[str], list[str]]:
    names = [f"test_{i:03d}" for i in range(200)]
    image_ids = [f"image_set/test/{i:03d}.jpg" for i in range(200)]
    monkeypatch.setattr(verify, "_official_test_identities", lambda _root: (names, image_ids))
    train_names = [f"train_{i:04d}" for i in range(1654)]
    train_ids = [f"image_set/train/{i:04d}.jpg" for i in range(1654)]
    source_sha = "a" * 64
    targets = verify.ReconstructionTargetCache(
        train_names, train_ids, torch.zeros(1654, 1280), torch.zeros(1654, 1024), "train",
        source_sha, "b" * 64, "/local/vision.bin",
    )
    targets_path = run / "targets/train_targets.pt"; targets.save(targets_path)
    retrieval = run / "checkpoints/retrieval.pt"; retrieval.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"atm": {"w": torch.ones(1)}, "fusion": {"w": torch.ones(1)},
                "metadata": {"subject": 8, "checkpoint_selection": "final_epoch", "cache_file_sha256": source_sha}}, retrieval)
    metadata = {"subject": 8, "checkpoint_selection": "final_epoch", "retrieval_checkpoint_sha256": _sha(retrieval),
                "train_target_cache_sha256": _sha(targets_path), "train_target_identity_hash": targets.name_hash}
    decoder = run / "checkpoints/decoder.pt"; decoder.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prior": {"w": torch.ones(1)}, "adapter": {"w": torch.ones(1)}, "metadata": metadata}, decoder)
    train_metadata = run / "reconstruction_train_metadata.json"; _write_json(train_metadata, metadata)
    prediction = run / "prediction_bundle.pt"
    features = torch.arange(200 * 1024, dtype=torch.float32).reshape(200, 1024)
    digest = verify.name_hash(names, image_ids)
    torch.save({"names": names, "image_ids": image_ids, "predicted_features": features,
                "metadata": {"subject": 8, "count": 200, "identity_hash": digest,
                             "trial_aggregation": verify.RAW_80_AGGREGATION,
                             "retrieval_checkpoint_sha256": _sha(retrieval), "decoder_checkpoint_sha256": _sha(decoder)}}, prediction)
    manifest = run / "generation/manifest.jsonl"; manifest.parent.mkdir(parents=True)
    rows = []
    with manifest.open("w", encoding="utf-8") as handle:
        for i in range(200):
            image = run / "generation/images" / f"{i:03d}.png"; image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (i % 255, 0, 0)).save(image)
            row = {"index": i, "name": names[i], "image_id": image_ids[i], "generated_path": str(image),
                   "condition_source": verify.CONDITION_SOURCE, "condition_feature_sha256": verify.tensor_sha256(features[i]),
                   "prediction_bundle_sha256": _sha(prediction), "prediction_identity_hash": digest,
                   "retrieval_checkpoint_sha256": _sha(retrieval), "decoder_checkpoint_sha256": _sha(decoder),
                   "trial_aggregation": verify.RAW_80_AGGREGATION}
            handle.write(json.dumps(row) + "\n"); rows.append(row)
    metrics_csv = run / "evaluation/metrics.csv"; metrics_csv.parent.mkdir(parents=True)
    fieldnames = ["index", "name", "image_id", *verify.METRICS]
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for i in range(200): writer.writerow({"index": i, "name": names[i], "image_id": image_ids[i], **{key: 0.5 for key in verify.METRICS}})
    metrics_json = run / "evaluation/metrics.json"
    _write_json(metrics_json, {"paper_ts_diff_targets": verify.PAPER_TARGETS, "metric_metadata": verify.METRIC_METADATA,
                               "summary": {key: 0.5 for key in verify.METRICS}, "count": 200,
                               "manifest": str(manifest.resolve()), "condition_source": verify.CONDITION_SOURCE})
    grid = run / "evaluation/comparison_grid.jpg"; Image.new("RGB", (2, 2)).save(grid)
    artifacts = {"retrieval_checkpoint": retrieval, "decoder_checkpoint": decoder, "train_target_cache": targets_path,
                 "train_metadata": train_metadata, "prediction_bundle": prediction, "generation_manifest": manifest,
                 "metrics_csv": metrics_csv, "metrics_json": metrics_json, "comparison_grid": grid}
    _write_json(run / verify.EVIDENCE_FILE, {"real_data": True, "subject": 8, "checkpoint_selection": "final_epoch",
                                             "trial_aggregation": verify.RAW_80_AGGREGATION, "condition_source": verify.CONDITION_SOURCE,
                                             "test_identity_hash": digest,
                                             "artifacts": {key: {"path": str(path.relative_to(run)), "sha256": _sha(path)} for key, path in artifacts.items()}})
    return names, image_ids


def test_verifier_accepts_complete_hash_bound_prediction_conditioned_evidence(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    assert verify.verify_reconstruction_run(tmp_path, tmp_path / "data") == []
    assert "VERIFIED" in verify.render_report(tmp_path, tmp_path / "data")


def test_verifier_fails_closed_for_missing_order_aggregation_and_oracle_drift(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "evaluation/comparison_grid.jpg").unlink()
    assert any("comparison_grid" in error for error in verify.verify_reconstruction_run(tmp_path, tmp_path / "data"))

    _fixture(tmp_path / "order", monkeypatch)
    manifest = tmp_path / "order/generation/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["name"], rows[1]["name"] = rows[1]["name"], rows[0]["name"]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert any("SHA-256 mismatch" in error or "order/identity" in error for error in verify.verify_reconstruction_run(tmp_path / "order", tmp_path / "data"))

    _fixture(tmp_path / "aggregation", monkeypatch)
    prediction = tmp_path / "aggregation/prediction_bundle.pt"
    payload = torch.load(prediction, weights_only=True); payload["metadata"]["trial_aggregation"] = "wrong"; torch.save(payload, prediction)
    assert any("prediction/official" in error or "aggregation" in error for error in verify.verify_reconstruction_run(tmp_path / "aggregation", tmp_path / "data"))

    _fixture(tmp_path / "oracle", monkeypatch)
    manifest = tmp_path / "oracle/generation/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]; rows[0]["oracle_feature_path"] = "/test.png"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert any("SHA-256 mismatch" in error or "oracle" in error for error in verify.verify_reconstruction_run(tmp_path / "oracle", tmp_path / "data"))


def test_verifier_recomputes_exactly_five_metric_summary_and_directions(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    metrics = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    metrics["metric_metadata"]["swav"]["direction"] = "higher"
    _write_json(tmp_path / "evaluation/metrics.json", metrics)
    errors = verify.verify_reconstruction_run(tmp_path, tmp_path / "data")
    assert any("SHA-256 mismatch" in error or "five paper metrics" in error for error in errors)
