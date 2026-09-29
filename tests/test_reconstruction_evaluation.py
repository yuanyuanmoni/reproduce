"""Focused leakage and metric-definition coverage for Task 4."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


def _module(name: str):
    path = Path(__file__).parents[1] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _module("generate_reconstructions.py")
evaluate = _module("evaluate_reconstructions.py")


def _bundle():
    names = [f"n{i}" for i in range(200)]; ids = [f"image_set/test/{i}.jpg" for i in range(200)]
    digest = generate.name_hash(names, ids)
    return {"names": names, "image_ids": ids, "predicted_features": torch.ones(200, 1024), "metadata": {"subject": 8, "count": 200, "identity_hash": digest, "trial_aggregation": "mean_80_raw_eeg_before_atm_once", "retrieval_checkpoint_sha256": "a" * 64, "decoder_checkpoint_sha256": "b" * 64}}


def test_manifest_identity_and_prediction_condition_guard():
    bundle = _bundle()
    generate.validate_bundle_against_test_manifest(bundle, bundle["names"], bundle["image_ids"])
    with pytest.raises(ValueError, match="identities/order"):
        generate.validate_bundle_against_test_manifest(bundle, list(reversed(bundle["names"])), bundle["image_ids"])
    condition = generate.condition_from_prediction(bundle["predicted_features"][0])
    assert condition.shape == (1, 1, 1024)
    with pytest.raises(ValueError, match="oracle"):
        generate.assert_non_oracle_condition({"condition_source": generate.CONDITION_SOURCE, "oracle_feature_path": "/test.png"})
    with pytest.raises(ValueError, match="condition source"):
        generate.assert_non_oracle_condition({"condition_source": "original_pixels"})


def test_captureframe_compatible_pixcorr_and_ssim_are_one_for_identical_rgb():
    image = Image.new("RGB", (9, 7), color=(10, 70, 220))
    # A nonconstant patch avoids the documented zero-variance PixCorr convention.
    image.putpixel((0, 0), (100, 20, 30))
    assert evaluate.pixcorr(image, image) == pytest.approx(1.0, abs=1e-6)
    assert evaluate.ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_summary_has_five_paper_metrics_and_only_swav_is_lower_better():
    rows = [{key: 0.25 for key in evaluate.PAPER_TARGETS}, {key: 0.75 for key in evaluate.PAPER_TARGETS}]
    result = evaluate.summary(rows)
    assert set(result) == set(evaluate.PAPER_TARGETS)
    assert result["pixcorr"] == pytest.approx(0.5)
    assert evaluate.METRIC_METADATA["swav"]["direction"] == "lower"
    assert all(evaluate.METRIC_METADATA[key]["direction"] == "higher" for key in ("pixcorr", "ssim", "inception", "clip"))
