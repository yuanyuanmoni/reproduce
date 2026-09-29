"""CPU smoke coverage for the train-only reconstruction CLI boundary."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nvol_retrieval.data import ThingsEEGSplit
from nvol_retrieval.reconstruction_data import ReconstructionTargetCache
from nvol_retrieval.vision import NVOLLayerFusion


def _module(name: str):
    path = Path(__file__).parents[1] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_reconstruction = _module("train_reconstruction.py")
export_reconstruction_features = _module("export_reconstruction_features.py")


class TinyATM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.feature = torch.nn.Parameter(torch.ones(1024))

    def forward(self, eeg):
        return torch.nn.functional.normalize(self.feature.expand(len(eeg), -1), dim=-1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieval_checkpoint(path: Path, fusion: NVOLLayerFusion, cache_sha: str, identity: str) -> None:
    torch.save({"atm": TinyATM().state_dict(), "fusion": fusion.state_dict(), "metadata": {
        "subject": 8, "checkpoint_selection": "final_epoch", "cache_file_sha256": cache_sha, "cache_hash": identity,
    }}, path)


def _targets(path: Path, names: list[str], image_ids: list[str], split: str, source_sha: str) -> ReconstructionTargetCache:
    cache = ReconstructionTargetCache(names, image_ids, torch.randn(len(names), 1280), torch.randn(len(names), 1024),
                                      split, source_sha, "d" * 64, "/local/vit-h.pt")
    cache.save(path)
    return cache


def test_train_reconstruction_rejects_test_targets_and_mismatched_atm_provenance(tmp_path, monkeypatch):
    names, ids = ["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"]
    train_split = ThingsEEGSplit(np.zeros((8, 63, 250), dtype=np.float32), ["a"] * 4 + ["b"] * 4,
                                 np.array([0] * 4 + [1] * 4), 4)
    monkeypatch.setattr(train_reconstruction, "load_sub08_split", lambda *_args, **_kwargs: train_split)
    monkeypatch.setattr(train_reconstruction, "canonical_image_ids", lambda *_args, **_kwargs: ids)
    monkeypatch.setattr(train_reconstruction, "ATMAlignmentModel", TinyATM)
    fusion, retrieval = NVOLLayerFusion(), tmp_path / "retrieval.pt"
    _retrieval_checkpoint(retrieval, fusion, "a" * 64, "x" * 64)
    test_targets = tmp_path / "test_targets.pt"; _targets(test_targets, names, ids, "test", "a" * 64)
    monkeypatch.setattr(sys, "argv", ["train_reconstruction.py", "--data-root", str(tmp_path), "--retrieval-checkpoint", str(retrieval), "--train-target-cache", str(test_targets), "--output-dir", str(tmp_path / "out"), "--device", "cpu"])
    with pytest.raises(ValueError, match="hard-rejects test"):
        train_reconstruction.main()
    test_targets.unlink(); good = _targets(test_targets, names, ids, "train", "b" * 64)
    with pytest.raises(ValueError, match="same source-cache provenance"):
        train_reconstruction.main()
    assert good.name_hash != "x" * 64


def test_train_and_export_reconstruction_cpu_smoke_with_raw_80_mean(tmp_path, monkeypatch):
    train_names, train_ids = ["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"]
    test_names = [f"test_{i}" for i in range(200)]; test_ids = [f"image_set/test/{i}.jpg" for i in range(200)]
    train_split = ThingsEEGSplit(np.zeros((8, 63, 250), dtype=np.float32), ["a"] * 4 + ["b"] * 4,
                                 np.array([0] * 4 + [1] * 4), 4)
    # Zero-stride fixture retains the official [200,80,63,250] layout without allocating ~2 GB.
    test_eeg = np.broadcast_to(np.zeros((1, 1, 1, 1), dtype=np.float64), (200, 80, 63, 250))
    test_split = ThingsEEGSplit(test_eeg, test_names, np.arange(200), 80)
    monkeypatch.setattr(train_reconstruction, "load_sub08_split", lambda *_args, **_kwargs: train_split)
    monkeypatch.setattr(train_reconstruction, "canonical_image_ids", lambda *_args, **_kwargs: train_ids)
    monkeypatch.setattr(train_reconstruction, "ATMAlignmentModel", TinyATM)
    fusion, retrieval = NVOLLayerFusion(), tmp_path / "retrieval.pt"
    targets_path = tmp_path / "train_targets.pt"; targets = _targets(targets_path, train_names, train_ids, "train", "a" * 64)
    _retrieval_checkpoint(retrieval, fusion, "a" * 64, targets.name_hash)
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_reconstruction.py", "--data-root", str(tmp_path), "--retrieval-checkpoint", str(retrieval), "--train-target-cache", str(targets_path), "--output-dir", str(run), "--epochs", "1", "--batch-size", "4", "--prior-hidden-dim", "16", "--prior-timesteps", "4", "--inference-steps", "2", "--adapter-bottleneck-dim", "16", "--max-train-batches", "1", "--device", "cpu"])
    assert train_reconstruction.main() == 0
    decoder = run / "final_epoch_reconstruction.pt"
    assert (run / "reconstruction_train_metadata.json").is_file() and decoder.is_file()
    monkeypatch.setattr(export_reconstruction_features, "load_sub08_split", lambda _root, split, **_kwargs: train_split if split == "train" else test_split)
    monkeypatch.setattr(export_reconstruction_features, "canonical_image_ids", lambda _root, split: train_ids if split == "train" else test_ids)
    monkeypatch.setattr(export_reconstruction_features, "ATMAlignmentModel", TinyATM)
    output = tmp_path / "predictions.pt"
    monkeypatch.setattr(sys, "argv", ["export_reconstruction_features.py", "--data-root", str(tmp_path), "--retrieval-checkpoint", str(retrieval), "--decoder-checkpoint", str(decoder), "--output", str(output), "--batch-size", "200", "--device", "cpu"])
    assert export_reconstruction_features.main() == 0
    bundle = torch.load(output, weights_only=True)
    assert bundle["names"] == test_names and bundle["image_ids"] == test_ids
    assert bundle["predicted_features"].shape == (200, 1024)
    assert bundle["metadata"]["trial_aggregation"] == "mean_80_raw_eeg_before_atm_once"
