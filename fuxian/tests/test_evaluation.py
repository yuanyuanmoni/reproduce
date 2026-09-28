import json
import importlib.util
from pathlib import Path

import pytest
import torch

from nvol_retrieval.evaluation import evaluate_exact_200, persist_result

_evaluate_spec = importlib.util.spec_from_file_location("nvol_evaluate", Path(__file__).parents[1] / "evaluate.py")
_evaluate = importlib.util.module_from_spec(_evaluate_spec)
assert _evaluate_spec.loader is not None
_evaluate_spec.loader.exec_module(_evaluate)


def _identity_trials():
    gallery = torch.eye(200)
    return gallery[:, None, :].repeat(1, 80, 1), gallery


def test_identity_is_exactly_perfect_for_raw_and_csls():
    trials, gallery = _identity_trials()
    result = evaluate_exact_200(trials, gallery, csls_k=10)
    assert (result.raw_top1, result.raw_top5, result.csls_top1, result.csls_top5) == (1.0, 1.0, 1.0, 1.0)


def test_trials_are_meaned_before_l2_normalization():
    trials, gallery = _identity_trials()
    trials[0, 0] = -gallery[0]  # 79 positive trials and one negative: nonzero mean retains ID 0.
    result = evaluate_exact_200(trials, gallery)
    assert result.raw_ranks[0].item() == 1


def test_exact_diagonal_rejects_same_class_style_match_and_csls_is_full_matrix():
    trials, gallery = _identity_trials()
    trials[0] = gallery[1]  # Semantically similar is irrelevant; only candidate 0 is correct.
    result = evaluate_exact_200(trials, gallery, csls_k=200)
    assert result.raw_ranks[0].item() != 1 and result.csls_ranks[0].item() != 1
    expected = 2 * result.raw_scores - result.raw_scores.mean(1, keepdim=True) - result.raw_scores.mean(0, keepdim=True)
    assert torch.allclose(result.csls_scores, expected)


def test_requires_complete_bank_and_persists_auditable_artifacts(tmp_path):
    with pytest.raises(ValueError, match=r"\[200,80,D\]"):
        evaluate_exact_200(torch.randn(199, 80, 4), torch.randn(200, 4))
    trials, gallery = _identity_trials()
    result = evaluate_exact_200(trials, gallery, candidate_names=[f"n{i}" for i in range(200)], candidate_image_ids=[f"id/{i}" for i in range(200)])
    persist_result(result, tmp_path, [f"n{i}" for i in range(200)], [f"id/{i}" for i in range(200)],
                   {"disable_cudnn_requested": True, "cudnn_enabled_effective": False, "device": "cpu"})
    assert {"metrics.json", "rankings.pt", "scores.pt", "query_names.json"} <= {path.name for path in tmp_path.iterdir()}
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["candidate_name_hash"] == result.candidate_name_hash
    assert metrics["disable_cudnn_requested"] is True and metrics["cudnn_enabled_effective"] is False


def test_persistence_rejects_identity_order_that_differs_from_evaluation(tmp_path):
    trials, gallery = _identity_trials()
    names, ids = [f"n{i}" for i in range(200)], [f"id/{i}" for i in range(200)]
    result = evaluate_exact_200(trials, gallery, candidate_names=names, candidate_image_ids=ids)
    with pytest.raises(ValueError, match="result hash"):
        persist_result(result, tmp_path / "mismatch", names[::-1], ids[::-1])
    assert not (tmp_path / "mismatch").exists()


def test_evaluation_disable_cudnn_is_recorded_for_cpu():
    original = torch.backends.cudnn.enabled
    try:
        _evaluate._configure_cudnn(True)
        provenance = _evaluate._runtime_provenance(torch.device("cpu"), True)
        assert provenance["disable_cudnn_requested"] is True
        assert provenance["cudnn_enabled_effective"] is False
        assert provenance["device"] == "cpu" and provenance["device_name"] is None
    finally:
        torch.backends.cudnn.enabled = original


def test_checkpoint_loader_accepts_local_torch_version_metadata(tmp_path):
    checkpoint = tmp_path / "final_epoch.pt"
    torch.save({"atm": {}, "fusion": {}, "logit_scale": torch.tensor(0.0),
                "metadata": {"subject": 8, "checkpoint_selection": "final_epoch", "torch_version": torch.__version__}}, checkpoint)
    assert _evaluate._load_checkpoint(checkpoint, torch.device("cpu"))["metadata"]["torch_version"] == torch.__version__
