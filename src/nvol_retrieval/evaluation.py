"""Fixed-bank exact-image retrieval metrics for the 200 THINGS-EEG2 tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .vision import name_hash


TEST_COUNT = 200


@dataclass(frozen=True)
class RetrievalResult:
    raw_scores: Tensor
    csls_scores: Tensor
    raw_ranks: Tensor
    csls_ranks: Tensor
    raw_top1: float
    raw_top5: float
    csls_top1: float
    csls_top5: float
    candidate_name_hash: str
    trial_aggregation: str
    csls_k: int


def _unit(x: Tensor, label: str) -> Tensor:
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f"{label} must be finite floating point")
    return F.normalize(x, dim=-1)


def _ranks(scores: Tensor) -> Tensor:
    order = scores.argsort(dim=1, descending=True)
    return order.argsort(dim=1).diagonal() + 1


def evaluate_exact_200(
    trials: Tensor, gallery: Tensor, csls_k: int = 10, *,
    candidate_names: Sequence[str] | None = None,
    candidate_image_ids: Sequence[str] | None = None,
    trial_aggregation: str = "mean_80_embeddings",
) -> RetrievalResult:
    """Rank the complete ordered 200-image bank; diagonal identity is truth.

    ``trials`` is normally [200,80,D].  A [200,D] tensor is accepted only for
    callers that averaged raw EEG before passing it through ATM.
    """
    if gallery.ndim != 2 or gallery.shape[0] != TEST_COUNT:
        raise ValueError("gallery must be [200,D] for the complete fixed test bank")
    if trials.ndim == 3:
        if trials.shape[:2] != (TEST_COUNT, 80):
            raise ValueError("trials must be [200,80,D]")
        query = trials.mean(dim=1)  # Aggregate before query L2 normalization/ranking.
    elif trials.ndim == 2 and trials.shape[0] == TEST_COUNT:
        query = trials
    else:
        raise ValueError("trials must be [200,80,D] or pre-aggregated [200,D]")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions must match")
    if not 1 <= csls_k <= TEST_COUNT:
        raise ValueError("csls_k must be in [1,200]")
    names = list(candidate_names or [str(index) for index in range(TEST_COUNT)])
    ids = list(candidate_image_ids or names)
    if len(names) != TEST_COUNT or len(ids) != TEST_COUNT:
        raise ValueError("exact retrieval requires 200 ordered candidate names and canonical image IDs")
    raw = _unit(query, "query") @ _unit(gallery, "gallery").T
    csls = 2 * raw - raw.topk(csls_k, dim=1).values.mean(1, keepdim=True) - raw.topk(csls_k, dim=0).values.mean(0, keepdim=True)
    raw_ranks, csls_ranks = _ranks(raw), _ranks(csls)
    return RetrievalResult(
        raw, csls, raw_ranks, csls_ranks,
        (raw_ranks <= 1).float().mean().item(), (raw_ranks <= 5).float().mean().item(),
        (csls_ranks <= 1).float().mean().item(), (csls_ranks <= 5).float().mean().item(),
        name_hash(names, ids), trial_aggregation, csls_k,
    )


def persist_result(result: RetrievalResult, output_dir: str | Path, candidate_names: Sequence[str],
                   candidate_image_ids: Sequence[str], runtime_metadata: dict[str, object] | None = None) -> None:
    """Write metrics and complete per-query ranking evidence without raw EEG."""
    names, image_ids = list(candidate_names), list(candidate_image_ids)
    if name_hash(names, image_ids) != result.candidate_name_hash:
        raise ValueError("Candidate names/canonical image IDs do not match the evaluated result hash")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics = {key: value for key, value in asdict(result).items() if not isinstance(value, Tensor)}
    if runtime_metadata:
        metrics.update(runtime_metadata)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    torch.save({"raw_ranks": result.raw_ranks.cpu(), "csls_ranks": result.csls_ranks.cpu(),
                "raw_order": result.raw_scores.argsort(dim=1, descending=True).cpu(),
                "csls_order": result.csls_scores.argsort(dim=1, descending=True).cpu()}, path / "rankings.pt")
    torch.save({"raw_scores": result.raw_scores.cpu(), "csls_scores": result.csls_scores.cpu()}, path / "scores.pt")
    (path / "query_names.json").write_text(json.dumps({"names": names, "image_ids": image_ids,
                                                           "candidate_name_hash": result.candidate_name_hash}, indent=2), encoding="utf-8")
