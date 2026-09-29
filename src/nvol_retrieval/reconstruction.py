"""Two-stage feature decoder used by the reconstruction branch.

The paper specifies loss functions but not the DDPM U-Net architecture.  This
module therefore keeps the prior deliberately small: an MLP denoiser over the
1024-dimensional feature vector.  Architecture and schedule are constructor
arguments so the training command can record them as implementation details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vision import BLOCKS, FEATURE_DIM, HIDDEN_DIM, NVOLLayerFusion


def _features(value: Tensor, *, name: str, dim: int = FEATURE_DIM) -> None:
    if value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(f"{name} must have shape [B,{dim}]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _time_embedding(timesteps: Tensor, dim: int) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [B]")
    half = dim // 2
    scale = -math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


@dataclass(frozen=True)
class PriorLossWeights:
    """Paper defaults for Eq. (21), with VICReg variance floor gamma=1."""

    mse: float = 0.4
    cosine: float = 3.0
    geometry: float = 0.8
    variance: float = 8e-4
    covariance: float = 3e-4
    variance_floor: float = 1.0


@dataclass(frozen=True)
class AdapterLossWeights:
    """Paper defaults for Eq. (26)."""

    mse: float = 1.0
    smooth_l1: float = 0.5
    cosine: float = 0.5
    contrastive: float = 0.2
    generated: float = 1.0
    domain: float = 0.1
    temperature: float = 0.07


class ConditionalNVOLPrior(nn.Module):
    """Conditional DDPM from a frozen ATM embedding to projected layer-12 NVOL."""

    def __init__(self, *, feature_dim: int = FEATURE_DIM, hidden_dim: int = 1024,
                 timesteps: int = 1_000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or timesteps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Invalid DDPM dimensions or beta schedule")
        self.feature_dim, self.hidden_dim, self.timesteps = feature_dim, hidden_dim, timesteps
        self.input = nn.Linear(feature_dim * 3, hidden_dim)
        self.denoiser = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, feature_dim),
        )
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alphas.cumprod(dim=0))

    def forward(self, noisy_target: Tensor, timesteps: Tensor, condition: Tensor) -> Tensor:
        _features(noisy_target, name="noisy_target", dim=self.feature_dim)
        _features(condition, name="condition", dim=self.feature_dim)
        if timesteps.shape != (len(noisy_target),) or timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError("timesteps must be integer [B] matching features")
        if (timesteps < 0).any() or (timesteps >= self.timesteps).any():
            raise ValueError("timesteps are outside the DDPM schedule")
        timestep_features = _time_embedding(timesteps, self.feature_dim).to(dtype=noisy_target.dtype)
        return self.denoiser(self.input(torch.cat((noisy_target, condition, timestep_features), dim=-1)))

    def q_sample(self, clean_target: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        _features(clean_target, name="clean_target", dim=self.feature_dim)
        _features(noise, name="noise", dim=self.feature_dim)
        if (noise.shape != clean_target.shape or timesteps.shape != (len(clean_target),)
                or timesteps.dtype not in (torch.int32, torch.int64)):
            raise ValueError("noise and timesteps must match clean_target batch")
        if (timesteps < 0).any() or (timesteps >= self.timesteps).any():
            raise ValueError("timesteps are outside the DDPM schedule")
        alpha_bar = self.alpha_bars[timesteps].to(dtype=clean_target.dtype).unsqueeze(1)
        return alpha_bar.sqrt() * clean_target + (1.0 - alpha_bar).sqrt() * noise

    def predict_clean(self, noisy_target: Tensor, timesteps: Tensor, predicted_noise: Tensor) -> Tensor:
        _features(noisy_target, name="noisy_target", dim=self.feature_dim)
        _features(predicted_noise, name="predicted_noise", dim=self.feature_dim)
        if (predicted_noise.shape != noisy_target.shape or timesteps.shape != (len(noisy_target),)
                or timesteps.dtype not in (torch.int32, torch.int64)):
            raise ValueError("predicted_noise and timesteps must match noisy_target batch")
        if (timesteps < 0).any() or (timesteps >= self.timesteps).any():
            raise ValueError("timesteps are outside the DDPM schedule")
        alpha_bar = self.alpha_bars[timesteps].to(dtype=noisy_target.dtype).unsqueeze(1)
        return (noisy_target - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

    @torch.no_grad()
    def sample(self, condition: Tensor, *, seed: int | None = None, generator: torch.Generator | None = None,
               inference_steps: int | None = None) -> Tensor:
        """Deterministic DDIM (eta=0) sampling when supplied a seed/generator."""
        _features(condition, name="condition", dim=self.feature_dim)
        if seed is not None and generator is not None:
            raise ValueError("Specify seed or generator, not both")
        if seed is not None:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
        steps = self.timesteps if inference_steps is None else inference_steps
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= self.timesteps:
            raise ValueError("inference_steps must be in [1, timesteps]")
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).round().long().unique_consecutive()
        latent = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        for position, timestep in enumerate(indices):
            batch_t = timestep.expand(len(condition))
            noise = self(latent, batch_t, condition)
            clean = self.predict_clean(latent, batch_t, noise)
            next_alpha = torch.ones((), device=condition.device, dtype=condition.dtype) if position + 1 == len(indices) else self.alpha_bars[indices[position + 1]].to(dtype=condition.dtype)
            latent = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * noise
        return latent


def prior_loss(prior: ConditionalNVOLPrior, condition: Tensor, target: Tensor, *, timesteps: Tensor | None = None,
               noise: Tensor | None = None, weights: PriorLossWeights = PriorLossWeights()) -> tuple[Tensor, dict[str, Tensor]]:
    """Eq. (21): DDPM noise loss plus point, angular, geometry and VICReg terms."""
    _features(condition, name="condition", dim=prior.feature_dim)
    _features(target, name="target", dim=prior.feature_dim)
    if condition.shape != target.shape:
        raise ValueError("condition and target must share [B,1024] shape")
    # The trained retrieval encoder/projection is a fixed boundary for this stage.
    condition, target = condition.detach(), target.detach()
    if timesteps is None:
        timesteps = torch.randint(prior.timesteps, (len(target),), device=target.device)
    if noise is None:
        noise = torch.randn_like(target)
    noisy = prior.q_sample(target, timesteps, noise)
    predicted_noise = prior(noisy, timesteps, condition)
    predicted_clean = prior.predict_clean(noisy, timesteps, predicted_noise)
    noise_loss = F.mse_loss(predicted_noise, noise)
    mse_loss = F.mse_loss(predicted_clean, target)
    cosine_loss = (1.0 - F.cosine_similarity(predicted_clean, target, dim=-1)).mean()
    predicted_normalized, target_normalized = F.normalize(predicted_clean, dim=-1), F.normalize(target, dim=-1)
    geometry_loss = ((predicted_normalized @ predicted_normalized.T - target_normalized @ target_normalized.T).square().sum()
                     / (len(target) ** 2))
    centered = predicted_clean - predicted_clean.mean(dim=0, keepdim=True)
    std = (centered.var(dim=0, unbiased=False) + 1e-4).sqrt()
    variance_loss = F.relu(weights.variance_floor - std).square().mean()
    covariance = centered.T @ centered / max(len(target) - 1, 1)
    covariance_loss = (covariance - torch.diag_embed(covariance.diag())).square().sum() / (prior.feature_dim * (prior.feature_dim - 1))
    total = (noise_loss + weights.mse * mse_loss + weights.cosine * cosine_loss + weights.geometry * geometry_loss
             + weights.variance * variance_loss + weights.covariance * covariance_loss)
    return total, {"noise": noise_loss, "mse": mse_loss, "cosine": cosine_loss, "geometry": geometry_loss,
                   "variance": variance_loss, "covariance": covariance_loss, "total": total}


class FrozenLayer12Projection(nn.Module):
    """The trained retrieval L12 linear projection/norm, frozen for reconstruction."""

    def __init__(self, projection: nn.Linear, norm: nn.LayerNorm) -> None:
        super().__init__()
        if projection.in_features != HIDDEN_DIM or projection.out_features != FEATURE_DIM or norm.normalized_shape != (FEATURE_DIM,):
            raise ValueError("Expected the ViT-H layer-12 1280-to-1024 retrieval projection")
        self.projection, self.norm = projection, norm
        self.freeze()

    @classmethod
    def from_retrieval_fusion(cls, fusion: NVOLLayerFusion) -> "FrozenLayer12Projection":
        index = BLOCKS.index(12)
        return cls(fusion.projections[index], fusion.norms[index])

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenLayer12Projection":
        return super().train(False)

    def forward(self, raw_layer12: Tensor) -> Tensor:
        if raw_layer12.ndim != 2 or raw_layer12.shape[1] != HIDDEN_DIM:
            raise ValueError("raw_layer12 must have shape [B,1280]")
        if not torch.is_floating_point(raw_layer12) or not torch.isfinite(raw_layer12).all():
            raise ValueError("raw_layer12 must be finite floating point")
        return F.normalize(self.norm(self.projection(raw_layer12)), dim=-1)


class SemanticAdapter(nn.Module):
    """Frozen L12 retrieval projection followed by a trainable semantic bottleneck."""

    def __init__(self, layer12_projection: FrozenLayer12Projection, *, bottleneck_dim: int = 512) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.layer12_projection = layer12_projection
        self.layer12_projection.freeze()
        self.adapter = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, FEATURE_DIM), nn.LayerNorm(FEATURE_DIM),
        )

    def train(self, mode: bool = True) -> "SemanticAdapter":
        super().train(mode)
        self.layer12_projection.eval()
        return self

    def _adapt(self, projected: Tensor) -> Tensor:
        _features(projected, name="projected_nvol")
        return F.normalize(self.adapter(projected), dim=-1)

    def from_real(self, raw_layer12: Tensor) -> Tensor:
        # Do not retain an autograd path through the frozen retrieval projection.
        with torch.no_grad():
            projected = self.layer12_projection(raw_layer12)
        return self._adapt(projected)

    def from_generated(self, generated_nvol: Tensor) -> Tensor:
        _features(generated_nvol, name="generated_nvol")
        return self._adapt(generated_nvol)

    forward = from_generated


def _alignment_loss(predicted: Tensor, target: Tensor, weights: AdapterLossWeights) -> tuple[Tensor, dict[str, Tensor]]:
    mse = F.mse_loss(predicted, target)
    smooth_l1 = F.smooth_l1_loss(predicted, target)
    cosine = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    logits = F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).T / weights.temperature
    labels = torch.arange(len(target), device=target.device)
    contrastive = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    total = weights.mse * mse + weights.smooth_l1 * smooth_l1 + weights.cosine * cosine + weights.contrastive * contrastive
    return total, {"mse": mse, "smooth_l1": smooth_l1, "cosine": cosine, "contrastive": contrastive}


def adapter_loss(adapter: SemanticAdapter, real_layer12: Tensor, generated_nvol: Tensor, final_target: Tensor, *,
                 weights: AdapterLossWeights = AdapterLossWeights()) -> tuple[Tensor, dict[str, Tensor]]:
    """Eq. (22)--(26), aligning real and generated NVOL paths to one target."""
    if real_layer12.ndim != 2 or real_layer12.shape[1] != HIDDEN_DIM:
        raise ValueError("real_layer12 must have shape [B,1280]")
    _features(generated_nvol, name="generated_nvol")
    _features(final_target, name="final_target")
    if len(real_layer12) != len(generated_nvol) or generated_nvol.shape != final_target.shape:
        raise ValueError("adapter inputs must share a batch size")
    target = final_target.detach()
    real, generated = adapter.from_real(real_layer12), adapter.from_generated(generated_nvol.detach())
    real_loss, real_terms = _alignment_loss(real, target, weights)
    generated_loss, generated_terms = _alignment_loss(generated, target, weights)
    domain = (1.0 - F.cosine_similarity(real, generated, dim=-1)).mean()
    total = real_loss + weights.generated * generated_loss + weights.domain * domain
    terms = {f"real_{name}": value for name, value in real_terms.items()}
    terms.update({f"generated_{name}": value for name, value in generated_terms.items()})
    terms.update({"real": real_loss, "generated": generated_loss, "domain": domain, "total": total})
    return total, terms
