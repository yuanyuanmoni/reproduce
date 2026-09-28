"""Multi-positive contrastive alignment loss."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _multi_positive_nll(logits: Tensor, positives: Tensor) -> Tensor:
    return -(torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1) - torch.logsumexp(logits, dim=1)).mean()


def symmetric_multi_positive_infonce(eeg: Tensor, image: Tensor, image_idx: Tensor, logit_scale: Tensor) -> Tensor:
    """Contrast every EEG/image pair, treating all matching image IDs as positives."""
    if eeg.ndim != 2 or image.shape != eeg.shape or image_idx.shape != (eeg.shape[0],):
        raise ValueError("Expected EEG/image [B,D] embeddings and image_idx [B]")
    if not torch.isfinite(logit_scale).all():
        raise ValueError("logit_scale must be finite")
    positives = image_idx[:, None].eq(image_idx[None, :])
    if not positives.any(dim=1).all():
        raise ValueError("Every query needs at least one positive")
    scale = logit_scale.clamp(max=math.log(100.0)).exp()
    logits = F.normalize(eeg, dim=-1) @ F.normalize(image, dim=-1).T * scale
    return (_multi_positive_nll(logits, positives) + _multi_positive_nll(logits.T, positives.T)) / 2
