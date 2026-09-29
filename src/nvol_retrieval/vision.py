"""Frozen OpenCLIP ViT-H/14 targets used by the retrieval experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BLOCKS = (10, 12, 14)
FEATURE_DIM = 1024
HIDDEN_DIM = 1280
OPENCLIP_REQUIREMENT = "open_clip_torch (import open_clip; create_model_and_transforms API)"


def name_hash(names: Sequence[str], image_ids: Sequence[str] | None = None) -> str:
    """Hash ordered names and canonical image identities without ambiguity."""
    image_ids = names if image_ids is None else image_ids
    if len(names) != len(image_ids):
        raise ValueError("Names and image identities must have the same length")
    digest = hashlib.sha256()
    for name, image_id in zip(names, image_ids):
        for value in (name, image_id):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureCache:
    names: list[str]
    image_ids: list[str]
    features: Tensor
    name_hash: str = ""

    def __post_init__(self) -> None:
        if (len(self.names) != len(self.image_ids) or len(self.names) != self.features.shape[0]
                or self.features.shape[1:] != (len(BLOCKS), HIDDEN_DIM)):
            raise ValueError("FeatureCache requires ordered [N,3,1280] pooled block features per image")
        digest = name_hash(self.names, self.image_ids)
        if self.name_hash and self.name_hash != digest:
            raise ValueError("FeatureCache name hash does not match ordered names")
        object.__setattr__(self, "name_hash", digest)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"names": self.names, "image_ids": self.image_ids, "features": self.features.cpu(), "name_hash": self.name_hash}, path)

    @classmethod
    def load(cls, path: str | Path, expected_names: Sequence[str] | None = None,
             expected_image_ids: Sequence[str] | None = None) -> "FeatureCache":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or set(payload) != {"names", "image_ids", "features", "name_hash"}:
            raise ValueError("Invalid feature cache payload")
        cache = cls(list(payload["names"]), list(payload["image_ids"]), payload["features"], payload["name_hash"])
        if expected_names is not None and cache.names != list(expected_names):
            raise ValueError("Feature cache image order does not match the EEG manifest")
        if expected_image_ids is not None and cache.image_ids != list(expected_image_ids):
            raise ValueError("Feature cache canonical image identities do not match the manifest")
        return cache


class NVOLLayerFusion(nn.Module):
    """Trainable 1280-to-1024 projections and softmax layer fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(HIDDEN_DIM, FEATURE_DIM) for _ in BLOCKS])
        self.norms = nn.ModuleList([nn.LayerNorm(FEATURE_DIM) for _ in BLOCKS])
        self.fusion_logits = nn.Parameter(torch.zeros(len(BLOCKS)))

    @property
    def fusion_weights(self) -> Tensor:
        return self.fusion_logits.softmax(dim=0)

    def forward(self, pooled: Tensor) -> Tensor:
        if pooled.ndim != 3 or pooled.shape[1:] != (len(BLOCKS), HIDDEN_DIM):
            raise ValueError("Expected frozen pooled features with shape [B,3,1280]")
        layers = [norm(projection(pooled[:, index])) for index, (projection, norm) in enumerate(zip(self.projections, self.norms))]
        return F.normalize(sum(weight * value for weight, value in zip(self.fusion_weights, layers)), dim=-1)


class FrozenNVOLTeacher(nn.Module):
    """Frozen vision backbone plus trainable paper-specified layer fusion."""

    blocks = BLOCKS

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.backbone = visual.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.fusion = NVOLLayerFusion()

    @property
    def fusion_weights(self) -> Tensor:
        return self.fusion.fusion_weights

    def train(self, mode: bool = True) -> "FrozenNVOLTeacher":
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _resblocks(visual: nn.Module) -> nn.ModuleList:
        try:
            blocks = visual.transformer.resblocks
        except AttributeError as error:
            raise ValueError("OpenCLIP visual transformer must expose transformer.resblocks") from error
        if len(blocks) < max(BLOCKS):
            raise ValueError("OpenCLIP vision backbone does not have blocks 10, 12, and 14")
        return blocks

    @staticmethod
    def _batch_tokens(tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [tokens,batch,1280] or [batch,tokens,1280], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != HIDDEN_DIM:
            raise ValueError(f"Expected ViT-H hidden size {HIDDEN_DIM}, got {tokens.shape[-1]}")
        return tokens if tokens.shape[0] == batch_size else tokens.transpose(0, 1) if tokens.shape[1] == batch_size else (_ for _ in ()).throw(ValueError("Token batch dimension does not match images"))

    def encode_pooled(self, images: Tensor) -> Tensor:
        captured: dict[int, Tensor] = {}
        hooks = [
            block.register_forward_hook(lambda _module, _args, output, layer=layer: captured.__setitem__(layer, output))
            for layer, block in zip(self.blocks, (self._resblocks(self.backbone)[index - 1] for index in self.blocks))
        ]
        try:
            with torch.no_grad():
                self.backbone(images)
        finally:
            for hook in hooks:
                hook.remove()
        if tuple(captured) != self.blocks:
            raise RuntimeError("Failed to capture all requested OpenCLIP blocks")
        return torch.stack([self._batch_tokens(captured[layer], images.shape[0]).mean(dim=1) for layer in self.blocks], dim=1)

    def encode(self, images: Tensor) -> Tensor:
        return self.fusion(self.encode_pooled(images))

    forward = encode


def preflight_openclip() -> object:
    """Fail before allocating/loading ViT-H when OpenCLIP is unavailable."""
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError(
            f"Feature extraction requires {OPENCLIP_REQUIREMENT}; it is not importable in this environment."
        ) from error
    if not hasattr(open_clip, "create_model_and_transforms"):
        raise RuntimeError("Installed open_clip lacks create_model_and_transforms; install a compatible open_clip_torch release.")
    print(f"OpenCLIP preflight passed: required={OPENCLIP_REQUIREMENT}; detected={getattr(open_clip, '__version__', 'unknown')}")
    return open_clip


def load_openclip_teacher(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[FrozenNVOLTeacher, object]:
    """Load only a local ViT-H/14 checkpoint; this function never downloads weights."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OpenCLIP ViT-H/14 checkpoint not found: {checkpoint}")
    open_clip = preflight_openclip()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=str(checkpoint), precision="fp32", device=device,
    )
    return FrozenNVOLTeacher(model.visual).to(device), preprocess
