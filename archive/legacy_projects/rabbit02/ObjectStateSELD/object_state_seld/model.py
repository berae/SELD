from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LogmelIntensityFrontend(nn.Module):
    """Differentiable 4-channel log-mel plus 3-channel intensity frontend."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 600,
        n_mels: int = 256,
        fmin: float = 20.0,
        fmax: float = 12000.0,
    ) -> None:
        super().__init__()
        import librosa

        mel = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ).astype(np.float32)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mel", torch.from_numpy(mel), persistent=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 4:
            raise ValueError("waveform must have shape [batch, 4, samples]")
        batch, channels, samples = waveform.shape
        spectrum = torch.stft(
            waveform.reshape(batch * channels, samples),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).reshape(batch, channels, self.n_fft // 2 + 1, -1)
        power = spectrum.abs().square()
        logmel = torch.log(torch.einsum("mf,bcft->bcmt", self.mel, power).clamp_min(1e-10))

        reference = spectrum[:, 0]
        directional = spectrum[:, 1:4]
        intensity = (reference[:, None].conj() * directional).real
        intensity = intensity / intensity.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-10)
        intensity_mel = torch.einsum("mf,bcft->bcmt", self.mel, intensity)
        return torch.cat([logmel, intensity_mel], dim=1).transpose(-1, -2)


class ObjectStateSELD(nn.Module):
    """B0 current-state baseline with classification, activity and unit-DOA heads."""

    def __init__(
        self,
        num_classes: int = 14,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = LogmelIntensityFrontend()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(64, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),
        )
        self.temporal = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(hidden_size, num_classes)
        self.activity_head = nn.Linear(hidden_size, 1)
        self.doa_head = nn.Linear(hidden_size, 3)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.frontend(waveform)
        encoded = self.encoder(features).mean(dim=-1).transpose(1, 2)
        sequence, _ = self.temporal(encoded)
        latent = self.dropout(sequence[:, -1])
        doa_raw = self.doa_head(latent)
        return {
            "class_logits": self.class_head(latent),
            "activity_logits": self.activity_head(latent).squeeze(-1),
            "position_xyz": F.normalize(doa_raw, dim=-1, eps=1e-6),
            "latent": latent,
        }
