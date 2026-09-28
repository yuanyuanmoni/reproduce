import torch
from torch.utils.data import DataLoader, TensorDataset
import importlib.util
import json
import numpy as np
from pathlib import Path
import sys

from nvol_retrieval.atm import ATMAlignmentModel
from nvol_retrieval.loss import symmetric_multi_positive_infonce
from nvol_retrieval.data import ThingsEEGSplit
from nvol_retrieval.vision import FeatureCache, NVOLLayerFusion
_train_spec = importlib.util.spec_from_file_location("nvol_train", Path(__file__).parents[1] / "train.py")
_train = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train)
GroupedTrialBatchSampler, build_optimizer = _train.GroupedTrialBatchSampler, _train.build_optimizer


def test_atm_stages_and_normalized_embedding():
    model = ATMAlignmentModel().eval()
    stages = model.stages(torch.randn(2, 63, 250))
    assert stages["attention"].shape == (2, 63, 250)
    assert stages["subject"].shape == (2, 63, 250)
    assert stages["patches"].shape == (2, 36, 40)
    assert stages["flattened"].shape == (2, 1440)
    assert model(torch.randn(2, 63, 250)).shape == (2, 1024)
    assert torch.allclose(model(torch.randn(2, 63, 250)).norm(dim=-1), torch.ones(2), atol=1e-5)


def test_same_image_ids_are_multi_positives_in_both_directions():
    eeg = torch.eye(3)
    image = torch.eye(3)
    ids = torch.tensor([0, 0, 1])
    scale = torch.nn.Parameter(torch.tensor(0.0))
    same = symmetric_multi_positive_infonce(eeg, image, ids, scale)
    distinct = symmetric_multi_positive_infonce(eeg, image, torch.arange(3), scale)
    assert torch.isfinite(same) and torch.isfinite(distinct)
    assert same < distinct
    same.backward()
    assert torch.isfinite(scale.grad)


def test_high_temperature_is_clamped_before_exp_and_has_finite_gradient():
    scale = torch.nn.Parameter(torch.tensor(1_000.0))
    loss = symmetric_multi_positive_infonce(torch.eye(2), torch.eye(2), torch.arange(2), scale)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scale.grad)


def test_grouped_sampler_never_separates_four_trials_and_optimizer_has_all_trainables():
    ids = torch.arange(30).repeat_interleave(4)
    sampler = GroupedTrialBatchSampler(ids, batch_size=100, generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TensorDataset(torch.zeros(len(ids), 63, 250), ids), batch_sampler=sampler)
    batches = list(loader)
    assert len(batches[0][1]) == 100 and torch.all(torch.unique(batches[0][1], return_counts=True)[1] == 4)
    assert len(batches[-1][1]) == 20 and torch.all(torch.unique(batches[-1][1], return_counts=True)[1] == 4)
    model, fusion, scale = ATMAlignmentModel(), NVOLLayerFusion(), torch.nn.Parameter(torch.zeros(()))
    optimizer = build_optimizer(model, fusion, scale, 3e-4)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert {id(parameter) for parameter in model.parameters()} <= optimized
    assert {id(parameter) for parameter in fusion.parameters()} <= optimized
    assert id(scale) in optimized


def test_disable_cudnn_is_process_local_and_recorded_for_cpu():
    original = torch.backends.cudnn.enabled
    try:
        _train._configure_cudnn(True)
        provenance = _train._runtime_provenance(torch.device("cpu"), True)
        assert torch.backends.cudnn.enabled is False
        assert provenance["disable_cudnn_requested"] is True
        assert provenance["cudnn_enabled_effective"] is False
        assert provenance["device"] == "cpu" and provenance["device_name"] is None
        assert "torch_version" in provenance and "torch_cuda_version" in provenance
    finally:
        torch.backends.cudnn.enabled = original
    assert torch.backends.cudnn.enabled is original


def test_max_train_batches_runs_one_complete_cpu_batch_and_records_metadata(tmp_path, monkeypatch):
    class TinyATM(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(63 * 250, 1024); self.batch_sizes = []

        def forward(self, eeg):
            self.batch_sizes.append(len(eeg))
            return torch.nn.functional.normalize(self.linear(eeg.flatten(1)), dim=-1)

    class TinyFusion(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(3 * 1280, 1024)

        def forward(self, features):
            return torch.nn.functional.normalize(self.linear(features.flatten(1)), dim=-1)

    names, image_ids = ["first", "second"], ["image_set/a.jpg", "image_set/b.jpg"]
    cache_path = tmp_path / "features.pt"
    FeatureCache(names, image_ids, torch.randn(2, 3, 1280)).save(cache_path)
    split = ThingsEEGSplit(np.random.randn(8, 63, 250).astype(np.float32), [name for name in names for _ in range(4)], np.arange(2).repeat(4), 4)
    model = TinyATM()
    monkeypatch.setattr(_train, "load_sub08_split", lambda *_args, **_kwargs: split)
    monkeypatch.setattr(_train, "canonical_image_ids", lambda *_args, **_kwargs: image_ids)
    monkeypatch.setattr(_train, "ATMAlignmentModel", lambda: model)
    monkeypatch.setattr(_train, "NVOLLayerFusion", TinyFusion)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train.py", "--data-root", str(tmp_path), "--visual-cache", str(cache_path), "--output-dir", str(output_dir), "--epochs", "1", "--batch-size", "4", "--device", "cpu", "--max-train-batches", "1"])

    assert _train.main() == 0
    metadata = json.loads((output_dir / "train_metadata.json").read_text())
    assert model.batch_sizes == [4]
    assert metadata["max_train_batches"] == 1 and len(metadata["losses"]) == 1
