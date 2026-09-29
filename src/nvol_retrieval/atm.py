"""The retrieval-only ATM EEG encoder transcribed from the approved Cogcap path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        values = torch.zeros(max_length, dim)
        values[:, 0::2] = torch.sin(position * exponent)
        values[:, 1::2] = torch.cos(position * exponent[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.values[: x.shape[0]].unsqueeze(1).to(dtype=x.dtype)


class EEGAttention(nn.Module):
    """One temporal TransformerEncoderLayer, matching Cogcap's batch layout."""

    def __init__(self, channels: int = 63) -> None:
        super().__init__()
        self.position = PositionalEncoding(channels)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=channels, nhead=1), num_layers=1
        )

    def forward(self, eeg: Tensor) -> Tensor:
        return self.encoder(self.position(eeg.permute(2, 0, 1))).permute(1, 2, 0)


class PatchEmbedding(nn.Module):
    """Cogcap's temporal/spatial convolutional PatchEmbedding for 63x250 EEG."""

    def __init__(self, emb_size: int = 40, channel_kernel: int = 63) -> None:
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)), nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40), nn.ELU(), nn.Conv2d(40, 40, (channel_kernel, 1)),
            nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.1), nn.Conv2d(40, emb_size, 1),
        )

    def forward(self, eeg: Tensor) -> Tensor:
        return self.tsconv(eeg.unsqueeze(1)).squeeze(2).transpose(1, 2)


class ProjEEG(nn.Module):
    def __init__(self, embedding_dim: int = 1440, output_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(output_dim, output_dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.norm(x + self.residual(x))


class ATMAlignmentModel(nn.Module):
    """ATM encoder for official sub-08 EEG, with no reconstruction/generation heads."""

    channels, sequence_length, feature_dim = 63, 250, 1024

    def __init__(self) -> None:
        super().__init__()
        self.attention_model = EEGAttention(self.channels)
        self.subject_wise_linear = nn.Linear(self.sequence_length, self.sequence_length)
        self.enc_eeg = PatchEmbedding(40, channel_kernel=self.channels)
        self.proj_eeg = ProjEEG(1440, self.feature_dim)

    def stages(self, eeg: Tensor) -> dict[str, Tensor]:
        if eeg.ndim != 3 or eeg.shape[1:] != (self.channels, self.sequence_length):
            raise ValueError("ATM requires EEG with shape [B,63,250]")
        attended = self.attention_model(eeg)
        subject = self.subject_wise_linear(attended)
        patches = self.enc_eeg(subject)
        flattened = patches.flatten(1)
        projected = self.proj_eeg(flattened)
        return {"attention": attended, "subject": subject, "patches": patches, "flattened": flattened, "projected": projected}

    def forward(self, eeg: Tensor) -> Tensor:
        return F.normalize(self.stages(eeg)["projected"], dim=-1)
