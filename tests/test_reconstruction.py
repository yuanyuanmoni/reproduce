import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nvol_retrieval.reconstruction import (
    ConditionalNVOLPrior, FrozenLayer12Projection, SemanticAdapter, adapter_loss, prior_loss,
)
from nvol_retrieval.vision import NVOLLayerFusion


def _prior():
    return ConditionalNVOLPrior(hidden_dim=32, timesteps=8)


def _adapter():
    return SemanticAdapter(FrozenLayer12Projection.from_retrieval_fusion(NVOLLayerFusion()), bottleneck_dim=16)


def test_prior_shapes_and_seeded_sampling_are_deterministic():
    prior, condition = _prior().eval(), torch.randn(3, 1024)
    assert prior(torch.randn(3, 1024), torch.tensor([0, 1, 2]), condition).shape == (3, 1024)
    first = prior.sample(condition, seed=7, inference_steps=4)
    second = prior.sample(condition, seed=7, inference_steps=4)
    assert first.shape == (3, 1024)
    assert torch.isfinite(first).all() and torch.equal(first, second)


def test_prior_loss_is_finite_and_does_not_backpropagate_into_fixed_inputs():
    prior = _prior()
    condition, target = torch.randn(2, 1024, requires_grad=True), torch.randn(2, 1024, requires_grad=True)
    loss, terms = prior_loss(prior, condition, target, timesteps=torch.tensor([1, 6]), noise=torch.randn(2, 1024))
    loss.backward()
    assert set(terms) == {"noise", "mse", "cosine", "geometry", "variance", "covariance", "total"}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    assert condition.grad is None and target.grad is None
    assert all(parameter.grad is not None for parameter in prior.parameters())


def test_prior_variance_penalty_is_squared_vicreg_hinge():
    prior = _prior()
    for parameter in prior.parameters():
        parameter.data.zero_()
    # With zero noise and a zero denoiser, x0 equals target exactly.  Each
    # feature has population std=.5, so Eq.19 is (1 - sqrt(.25 + 1e-4))^2.
    target = torch.tensor([[-0.5] * 1024, [0.5] * 1024])
    _, terms = prior_loss(prior, torch.zeros_like(target), target, timesteps=torch.tensor([2, 2]), noise=torch.zeros_like(target))
    expected = (1.0 - torch.sqrt(torch.tensor(0.25 + 1e-4))).square()
    assert torch.allclose(terms["variance"], expected, atol=1e-6)


def test_prior_rejects_invalid_timestep_or_sampling_step_count():
    prior, feature = _prior(), torch.randn(1, 1024)
    with pytest.raises(ValueError, match="timesteps"):
        prior.q_sample(feature, torch.tensor([8]), feature)
    with pytest.raises(ValueError, match="inference_steps"):
        prior.sample(feature, inference_steps=1.5)


def test_semantic_adapter_uses_frozen_l12_boundary_and_losses_are_finite():
    adapter = _adapter().train()
    assert not adapter.layer12_projection.training
    assert all(not parameter.requires_grad for parameter in adapter.layer12_projection.parameters())
    assert all(parameter.requires_grad for parameter in adapter.adapter.parameters())
    raw = torch.randn(3, 1280, requires_grad=True)
    generated, target = torch.randn(3, 1024, requires_grad=True), torch.randn(3, 1024, requires_grad=True)
    real_output, generated_output = adapter.from_real(raw), adapter.from_generated(generated)
    assert real_output.shape == generated_output.shape == (3, 1024)
    assert torch.allclose(real_output.norm(dim=-1), torch.ones(3), atol=1e-5)
    loss, terms = adapter_loss(adapter, raw, generated, target)
    loss.backward()
    assert torch.isfinite(loss) and all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    assert raw.grad is None and generated.grad is None and target.grad is None
    assert all(parameter.grad is None for parameter in adapter.layer12_projection.parameters())
    assert all(parameter.grad is not None for parameter in adapter.adapter.parameters())
