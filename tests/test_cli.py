import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from verify_run import verify_run
from package_existing_run import package
from nvol_retrieval.vision import name_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names_sha(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _cache(path: Path, count: int, prefix: str) -> str:
    names = [f"{prefix}_{index}" for index in range(count)]
    ids = [f"image_set/{prefix}/{index}.jpg" for index in range(count)]
    digest = name_hash(names, ids)
    # Zero-stride keeps this protocol-shape fixture tiny while exercising shape checks.
    torch.save({"names": names, "image_ids": ids, "features": torch.empty_strided((count, 3, 1280), (0, 0, 0)), "name_hash": digest}, path)
    return digest


def _valid_run(tmp_path: Path) -> Path:
    run, cache, evaluation = tmp_path / "run", tmp_path / "run" / "cache", tmp_path / "run" / "evaluation"
    cache.mkdir(parents=True); evaluation.mkdir()
    checkpoint_path = tmp_path / "vit-h.bin"
    checkpoint_path.write_bytes(b"visual checkpoint")
    train_hash = _cache(cache / "nvol_train_features.pt", 16540, "train")
    test_hash = _cache(cache / "nvol_test_features.pt", 200, "test")
    names = [f"test_{index}" for index in range(200)]; ids = [f"image_set/test/{index}.jpg" for index in range(200)]
    digest = name_hash(names, ids)
    train = {"subject": 8, "checkpoint_selection": "final_epoch", "cache_hash": train_hash, "cache_file_sha256": _sha(cache / "nvol_train_features.pt"), "seed": 0, "device": "cpu", "disable_cudnn_requested": False, "epochs": 30, "batch_size": 100, "lr": 3e-4, "incomplete_batch": "keep"}
    score = torch.eye(200); order = score.argsort(1, descending=True); ranks = order.argsort(1).diagonal() + 1
    metric_values = {"raw_top1": 1.0, "raw_top5": 1.0, "csls_top1": 1.0, "csls_top5": 1.0}
    train_names = [f"train_{index}" for index in range(16540)]
    contract = {"subject": "sub-08", "train": {"official_array_shape": [16540, 4, 63, 250], "ordered_names_sha256": _names_sha(train_names)}, "test": {"official_array_shape": [200, 80, 63, 250], "ordered_names_sha256": _names_sha(names)}}
    (run / "data_contract.json").write_text(json.dumps(contract)); (run / "environment.yml").write_text("name: test\n"); (run / "commands.txt").write_text(f"python cache_nvol_features.py --checkpoint {checkpoint_path}\nevaluate\n")
    source_checkpoint = {"resolved_path": str(checkpoint_path), "sha256": _sha(checkpoint_path)}
    provenance = {split: {"split": split, "cache_identity_hash": digest, "disable_cudnn_requested": False, "cudnn_enabled_effective": True, "source_checkpoint": source_checkpoint} for split, digest in (("train", train_hash), ("test", test_hash))}
    for split, value in provenance.items():
        (cache / f"nvol_{split}_features.provenance.json").write_text(json.dumps(value))
    record = {"real_data": True, "subject": 8, "checkpoint_selection": "final_epoch", "trial_aggregation": "mean_80_raw_eeg_before_atm", "ground_truth": "exact_image_identity", "commands": [f"python cache_nvol_features.py --checkpoint {checkpoint_path}", "evaluate"], "environment_export": "environment.yml", "device": "cpu", "seed": 0, "data_contract_sha256": _sha(run / "data_contract.json"), "data_contract_ordered_names_sha256": {"train": contract["train"]["ordered_names_sha256"], "test": contract["test"]["ordered_names_sha256"]}, "cache_identity_hash_kind": "canonical_name_and_image_id_sha256", "visual_checkpoint": source_checkpoint, "train_cache_hash": train_hash, "test_cache_hash": test_hash, "train_cache_file_sha256": train["cache_file_sha256"], "test_cache_file_sha256": _sha(cache / "nvol_test_features.pt"), "cache_provenance": provenance, "metrics": metric_values}
    (run / "train_metadata.json").write_text(json.dumps(train)); (run / "run_metadata.json").write_text(json.dumps(record))
    (evaluation / "metrics.json").write_text(json.dumps({"candidate_name_hash": digest, "trial_aggregation": "mean_80_raw_eeg_before_atm", "csls_k": 10, **metric_values}))
    (evaluation / "query_names.json").write_text(json.dumps({"names": names, "image_ids": ids, "candidate_name_hash": digest}))
    torch.save({"raw_scores": score, "csls_scores": score}, evaluation / "scores.pt")
    torch.save({"raw_ranks": ranks, "csls_ranks": ranks, "raw_order": order, "csls_order": order}, evaluation / "rankings.pt")
    torch.save({"atm": {"weight": torch.ones(1)}, "fusion": {"weight": torch.ones(1)}, "logit_scale": torch.tensor(0.0), "metadata": train}, run / "final_epoch.pt")
    return run


def test_verifier_accepts_real_complete_run_and_rejects_aggregation_drift(tmp_path):
    run = _valid_run(tmp_path)
    assert verify_run(run) == []
    metrics = json.loads((run / "evaluation/metrics.json").read_text()); metrics["trial_aggregation"] = "wrong"
    (run / "evaluation/metrics.json").write_text(json.dumps(metrics))
    assert any("aggregation" in error for error in verify_run(run))


def test_verifier_loads_torch_version_from_trusted_train_metadata(tmp_path):
    run = _valid_run(tmp_path)
    train = json.loads((run / "train_metadata.json").read_text())
    train["torch_version"] = str(torch.__version__)
    (run / "train_metadata.json").write_text(json.dumps(train))
    checkpoint = torch.load(run / "final_epoch.pt", weights_only=True)
    checkpoint["metadata"] = train | {"torch_version": torch.__version__}
    torch.save(checkpoint, run / "final_epoch.pt")
    assert verify_run(run) == []


def test_verifier_recomputes_metrics_and_rejects_synthetic_or_unlisted_artifacts(tmp_path):
    run = _valid_run(tmp_path)
    scores = torch.load(run / "evaluation/scores.pt", weights_only=True)
    scores["raw_scores"][0, 0] = 0.0; scores["raw_scores"][0, 1] = 0.9; scores["csls_scores"][0, 0] = 5.0
    torch.save(scores, run / "evaluation/scores.pt")
    record = json.loads((run / "run_metadata.json").read_text()); record["real_data"] = False
    (run / "run_metadata.json").write_text(json.dumps(record)); (run / "raw_eeg.bin").write_bytes(b"forbidden")
    errors = verify_run(run)
    assert any("synthetic" in error for error in errors)
    assert any("theoretical" in error for error in errors)
    assert any("complete descending" in error for error in errors)
    assert any("unexpected artifact" in error for error in errors)


def test_verifier_allows_only_exact_score_tie_permutations(tmp_path):
    run = _valid_run(tmp_path)
    scores = torch.load(run / "evaluation/scores.pt", weights_only=True)
    rankings = torch.load(run / "evaluation/rankings.pt", weights_only=True)
    # Candidate 0 is correct for query 0.  A backend may put either exact tie first.
    for prefix in ("raw", "csls"):
        scores[f"{prefix}_scores"][0].zero_()
        scores[f"{prefix}_scores"][0, :2] = 0.8
        rankings[f"{prefix}_order"] = scores[f"{prefix}_scores"].argsort(1, descending=True)
        rankings[f"{prefix}_order"][0, :2] = torch.tensor([1, 0])
        rankings[f"{prefix}_ranks"] = rankings[f"{prefix}_order"].argsort(1).diagonal() + 1
    raw_top1 = (rankings["raw_ranks"] <= 1).float().mean().item()
    raw_top5 = (rankings["raw_ranks"] <= 5).float().mean().item()
    torch.save(scores, run / "evaluation/scores.pt")
    torch.save(rankings, run / "evaluation/rankings.pt")
    for path in (run / "evaluation/metrics.json", run / "run_metadata.json"):
        payload = json.loads(path.read_text())
        target = payload["metrics"] if path.name == "run_metadata.json" else payload
        target.update({"raw_top1": raw_top1, "raw_top5": raw_top5, "csls_top1": raw_top1, "csls_top5": raw_top5})
        path.write_text(json.dumps(payload))
    assert verify_run(run) == []

    # 5e-7 is representable in float32 at 0.8: it is close, but not an exact tie.
    scores["raw_scores"][0, 0] = 0.8000005
    torch.save(scores, run / "evaluation/scores.pt")
    assert any("exactly equal ties" in error for error in verify_run(run))


def test_packager_binds_existing_artifacts_without_claiming_real_data_from_a_supplied_contract(tmp_path):
    run = _valid_run(tmp_path)
    contract = json.loads((run / "data_contract.json").read_text())
    checkpoint = Path(json.loads((run / "run_metadata.json").read_text())["visual_checkpoint"]["resolved_path"])
    for name in ("data_contract.json", "environment.yml", "commands.txt", "run_metadata.json"):
        (run / name).unlink()
    package(run, contract, "name: fixture\n", data_root=None, checkpoint_path=checkpoint,
            device="cpu", seed=0, disable_cudnn=False)
    record = json.loads((run / "run_metadata.json").read_text())
    assert record["real_data"] is False
    assert record["train_cache_file_sha256"] == _sha(run / "cache/nvol_train_features.pt")
    assert any("synthetic" in error for error in verify_run(run))


def test_verifier_fails_closed_for_subject_count_class_labels_selection_and_hash(tmp_path):
    run = _valid_run(tmp_path)
    record = json.loads((run / "run_metadata.json").read_text())
    record.update({"subject": 7, "checkpoint_selection": "best_test_epoch", "ground_truth": "class_label"})
    (run / "run_metadata.json").write_text(json.dumps(record))
    identities = json.loads((run / "evaluation/query_names.json").read_text()); identities["names"] = identities["names"][:-1]; identities["candidate_name_hash"] = "bad"
    (run / "evaluation/query_names.json").write_text(json.dumps(identities))
    errors = verify_run(run)
    assert any("subject" in error for error in errors)
    assert any("checkpoint" in error for error in errors)
    assert any("class-level" in error for error in errors)
    assert any("200" in error for error in errors)


def test_verifier_requires_all_packaged_provenance_artifacts_and_matching_commands(tmp_path):
    run = _valid_run(tmp_path)
    (run / "environment.yml").unlink()
    assert any("required evidence artifact" in error for error in verify_run(run))
    (run / "environment.yml").write_text("name: test\n")
    (run / "commands.txt").write_text("different command\n")
    assert any("commands.txt" in error for error in verify_run(run))


def test_verifier_binds_cache_names_cudnn_command_and_visual_checkpoint(tmp_path):
    run = _valid_run(tmp_path)
    contract = json.loads((run / "data_contract.json").read_text())
    contract["train"]["ordered_names_sha256"] = "wrong"
    (run / "data_contract.json").write_text(json.dumps(contract))
    record = json.loads((run / "run_metadata.json").read_text())
    record["data_contract_sha256"] = _sha(run / "data_contract.json")
    record["data_contract_ordered_names_sha256"]["train"] = "wrong"
    for split in ("train", "test"):
        provenance_path = run / "cache" / f"nvol_{split}_features.provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["disable_cudnn_requested"] = True
        provenance["cudnn_enabled_effective"] = True
        provenance["source_checkpoint"] = {"resolved_path": "/models/other.bin", "sha256": "other"}
        provenance_path.write_text(json.dumps(provenance))
        record["cache_provenance"][split] = provenance
    (run / "run_metadata.json").write_text(json.dumps(record))
    errors = verify_run(run)
    assert any("names-only" in error for error in errors)
    assert any("cuDNN state" in error for error in errors)
    assert any("checkpoint" in error for error in errors)


def test_verifier_recomputes_visual_checkpoint_checksum_and_requires_file(tmp_path):
    run = _valid_run(tmp_path / "changed")
    checkpoint = Path(json.loads((run / "run_metadata.json").read_text())["visual_checkpoint"]["resolved_path"])
    checkpoint.write_bytes(b"altered")
    assert any("SHA-256" in error for error in verify_run(run))
    missing_run = _valid_run(tmp_path / "missing")
    missing = Path(json.loads((missing_run / "run_metadata.json").read_text())["visual_checkpoint"]["resolved_path"])
    missing.unlink()
    assert any("must exist" in error for error in verify_run(missing_run))
