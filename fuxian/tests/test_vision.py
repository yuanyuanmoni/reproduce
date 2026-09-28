import pytest
import torch
from torch import nn
import importlib.util
from pathlib import Path

from nvol_retrieval.vision import FeatureCache, FrozenNVOLTeacher, name_hash


class _Block(nn.Module):
    def forward(self, x):
        return x + 1


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_Block() for _ in range(14)])

    def forward(self, images):
        tokens = torch.zeros(3, images.shape[0], 1280, device=images.device)
        for block in self.transformer.resblocks:
            tokens = block(tokens)
        return tokens


def test_teacher_uses_requested_blocks_and_returns_unit_vectors():
    teacher = FrozenNVOLTeacher(_Visual())
    teacher.train()
    result = teacher.encode(torch.randn(2, 3, 4, 4))
    pooled = teacher.encode_pooled(torch.randn(2, 3, 4, 4))
    assert teacher.blocks == (10, 12, 14)
    assert result.shape == (2, 1024)
    assert pooled.shape == (2, 3, 1280)
    assert torch.allclose(result.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(teacher.fusion_weights.sum(), torch.tensor(1.0))
    assert all(not parameter.requires_grad for parameter in teacher.backbone.parameters())
    assert not teacher.backbone.training
    assert any(parameter.requires_grad for parameter in teacher.fusion.parameters())


def test_cache_rejects_name_order_mismatch(tmp_path):
    cache = FeatureCache(["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"], torch.randn(2, 3, 1280))
    path = tmp_path / "features.pt"
    cache.save(path)
    assert FeatureCache.load(path, expected_names=["a", "b"], expected_image_ids=["image_set/a.jpg", "image_set/b.jpg"]).name_hash == name_hash(["a", "b"], ["image_set/a.jpg", "image_set/b.jpg"])
    with pytest.raises(ValueError, match="order"):
        FeatureCache.load(path, expected_names=["b", "a"])
    with pytest.raises(ValueError, match="identities"):
        FeatureCache.load(path, expected_image_ids=["image_set/b.jpg", "image_set/a.jpg"])


def test_extraction_refuses_raw_output_or_invalid_batch(tmp_path):
    tool = Path(__file__).parents[1] / "tools" / "cache_nvol_features.py"
    spec = importlib.util.spec_from_file_location("cache_nvol_features", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="outside"):
        module._validate_extraction_args(tmp_path, tmp_path / "cache", 1)
    with pytest.raises(ValueError, match="positive"):
        module._validate_extraction_args(tmp_path, tmp_path.parent / "cache", 0)
    provenance = module._runtime_provenance(torch.device("cpu"), True, {"resolved_path": "/model.pt", "sha256": "x"})
    assert provenance["disable_cudnn_requested"] is True
    assert provenance["cudnn_enabled_effective"] == torch.backends.cudnn.enabled
    assert provenance["identity_hash_kind"] == "canonical_name_and_image_id_sha256"
    assert provenance["source_checkpoint"] == {"resolved_path": "/model.pt", "sha256": "x"}
