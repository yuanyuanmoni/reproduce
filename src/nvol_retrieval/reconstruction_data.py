"""Provenance-checked visual targets for sub-08 image reconstruction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .vision import FEATURE_DIM, HIDDEN_DIM, FeatureCache, name_hash


_SHA256 = re.compile(r"[0-9a-f]{64}")


def mean_80_raw_test_trials(eeg: np.ndarray) -> np.ndarray:
    """Average the official raw held-out trials once, before the ATM."""
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 4 or eeg.shape[1:] != (80, 63, 250):
        raise ValueError("Expected raw test EEG with shape [N,80,63,250]")
    return eeg.mean(axis=1)


@dataclass(frozen=True)
class ReconstructionTargetCache:
    """Ordered raw layer-12 and frozen final-semantic targets with provenance."""

    names: list[str]
    image_ids: list[str]
    layer12_pooled: Tensor
    final_semantic: Tensor
    split: str
    source_feature_cache_sha256: str
    visual_checkpoint_sha256: str
    visual_checkpoint_path: str
    name_hash: str = ""

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("Reconstruction target cache split must be 'train' or 'test'")
        if (not self.names or len(self.names) != len(self.image_ids)
                or len(set(self.names)) != len(self.names) or len(set(self.image_ids)) != len(self.image_ids)):
            raise ValueError("Reconstruction target cache requires unique ordered names and image IDs")
        if (self.layer12_pooled.shape != (len(self.names), HIDDEN_DIM)
                or self.final_semantic.shape != (len(self.names), FEATURE_DIM)):
            raise ValueError("Reconstruction targets require [N,1280] layer-12 and [N,1024] final semantic tensors")
        if (not torch.is_floating_point(self.layer12_pooled) or not torch.is_floating_point(self.final_semantic)
                or not torch.isfinite(self.layer12_pooled).all() or not torch.isfinite(self.final_semantic).all()):
            raise ValueError("Reconstruction targets must be finite floating-point tensors")
        if not all(_SHA256.fullmatch(value) for value in (self.source_feature_cache_sha256, self.visual_checkpoint_sha256)):
            raise ValueError("Reconstruction target cache requires SHA256 source provenance")
        digest = name_hash(self.names, self.image_ids)
        if self.name_hash and self.name_hash != digest:
            raise ValueError("Reconstruction target cache name hash does not match ordered identities")
        object.__setattr__(self, "name_hash", digest)

    @classmethod
    def from_feature_cache(cls, feature_cache: FeatureCache, final_semantic: Tensor, *, split: str,
                           source_feature_cache_sha256: str, visual_checkpoint_sha256: str,
                           visual_checkpoint_path: str) -> "ReconstructionTargetCache":
        return cls(
            names=feature_cache.names,
            image_ids=feature_cache.image_ids,
            layer12_pooled=feature_cache.features[:, 1].detach().cpu(),
            final_semantic=final_semantic.detach().cpu(),
            split=split,
            source_feature_cache_sha256=source_feature_cache_sha256,
            visual_checkpoint_sha256=visual_checkpoint_sha256,
            visual_checkpoint_path=visual_checkpoint_path,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable reconstruction target cache: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "names": self.names, "image_ids": self.image_ids,
            "layer12_pooled": self.layer12_pooled.cpu(), "final_semantic": self.final_semantic.cpu(),
            "split": self.split, "source_feature_cache_sha256": self.source_feature_cache_sha256,
            "visual_checkpoint_sha256": self.visual_checkpoint_sha256,
            "visual_checkpoint_path": self.visual_checkpoint_path, "name_hash": self.name_hash,
        }, path)

    @classmethod
    def load(cls, path: str | Path, *, expected_names: Sequence[str] | None = None,
             expected_image_ids: Sequence[str] | None = None, expected_split: str | None = None) -> "ReconstructionTargetCache":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
        fields = {"names", "image_ids", "layer12_pooled", "final_semantic", "split",
                  "source_feature_cache_sha256", "visual_checkpoint_sha256", "visual_checkpoint_path", "name_hash"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("Invalid reconstruction target cache payload")
        cache = cls(**payload)
        if expected_names is not None and cache.names != list(expected_names):
            raise ValueError("Reconstruction target cache image order does not match the EEG manifest")
        if expected_image_ids is not None and cache.image_ids != list(expected_image_ids):
            raise ValueError("Reconstruction target cache canonical image identities do not match the manifest")
        if expected_split is not None and cache.split != expected_split:
            raise ValueError("Reconstruction target cache split does not match the requested split")
        return cache
