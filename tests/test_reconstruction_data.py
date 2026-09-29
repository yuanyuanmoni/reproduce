import sys
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nvol_retrieval.reconstruction_data import ReconstructionTargetCache, mean_80_raw_test_trials
from nvol_retrieval.vision import FeatureCache


def _cache() -> ReconstructionTargetCache:
    feature_cache = FeatureCache(["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"], torch.arange(2 * 3 * 1280, dtype=torch.float32).reshape(2, 3, 1280))
    return ReconstructionTargetCache.from_feature_cache(
        feature_cache, torch.randn(2, 1024), split="train", source_feature_cache_sha256="a" * 64,
        visual_checkpoint_sha256="b" * 64, visual_checkpoint_path="/models/vit-h.pt",
    )


def test_target_cache_preserves_raw_layer12_and_rejects_order_mismatch(tmp_path):
    cache = _cache()
    assert cache.layer12_pooled.shape == (2, 1280)
    assert torch.equal(cache.layer12_pooled, torch.arange(2 * 3 * 1280, dtype=torch.float32).reshape(2, 3, 1280)[:, 1])
    path = tmp_path / "targets.pt"
    cache.save(path)
    assert ReconstructionTargetCache.load(path, expected_names=["a", "b"], expected_image_ids=cache.image_ids).name_hash == cache.name_hash
    with pytest.raises(ValueError, match="order"):
        ReconstructionTargetCache.load(path, expected_names=["b", "a"])
    with pytest.raises(ValueError, match="identities"):
        ReconstructionTargetCache.load(path, expected_image_ids=list(reversed(cache.image_ids)))
    with pytest.raises(ValueError, match="split"):
        ReconstructionTargetCache.load(path, expected_split="test")


def test_target_cache_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        ReconstructionTargetCache(
            ["a", "b"], ["image_set/a.jpg", "image_set/a.jpg"], torch.randn(2, 1280), torch.randn(2, 1024),
            "train", "a" * 64, "b" * 64, "/models/vit-h.pt",
        )


def test_target_cache_rejects_nonfinite_targets():
    with pytest.raises(ValueError, match="finite"):
        ReconstructionTargetCache(
            ["a"], ["image_set/a.jpg"], torch.ones(1, 1280), torch.full((1, 1024), float("nan")),
            "train", "a" * 64, "b" * 64, "/models/vit-h.pt",
        )


def test_mean_80_raw_trials_is_exact_and_rejects_other_trial_counts():
    eeg = np.arange(2 * 80 * 63 * 250, dtype=np.float64).reshape(2, 80, 63, 250)
    assert np.array_equal(mean_80_raw_test_trials(eeg), eeg.mean(axis=1))
    with pytest.raises(ValueError, match="80"):
        mean_80_raw_test_trials(eeg[:, :79])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_tool():
    path = Path(__file__).parents[1] / "tools" / "cache_reconstruction_targets.py"
    spec = importlib.util.spec_from_file_location("cache_reconstruction_targets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_target_extraction_requires_matching_adjacent_cache_provenance(tmp_path):
    feature_path, checkpoint = tmp_path / "nvol_train_features.pt", tmp_path / "vit-h.pt"
    checkpoint.write_bytes(b"local checkpoint")
    feature = FeatureCache(["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"], torch.randn(2, 3, 1280))
    feature.save(feature_path)
    tool = _target_tool()
    with pytest.raises(FileNotFoundError, match="adjacent"):
        tool._load_validated_feature_cache(feature_path, split="train", names=feature.names, image_ids=feature.image_ids, checkpoint=checkpoint)
    provenance_path = feature_path.with_name("nvol_train_features.provenance.json")
    provenance = {"split": "train", "cache_file": feature_path.name, "cache_file_sha256": _sha(feature_path),
                  "cache_identity_hash": feature.name_hash,
                  "source_checkpoint": {"resolved_path": str(checkpoint.resolve()), "sha256": "0" * 64}}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="visual checkpoint"):
        tool._load_validated_feature_cache(feature_path, split="train", names=feature.names, image_ids=feature.image_ids, checkpoint=checkpoint)
    provenance["source_checkpoint"]["sha256"] = _sha(checkpoint)
    provenance["cache_file_sha256"] = "f" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA256"):
        tool._load_validated_feature_cache(feature_path, split="train", names=feature.names, image_ids=feature.image_ids, checkpoint=checkpoint)
