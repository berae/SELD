import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.utils.stft import STFT, LogmelFilterBank, intensityvector, spectrogram_STFTInput


class CausalLogmelIntensity_Extractor(nn.Module):
    """Right-aligned log-mel + intensity features with zero acoustic look-ahead."""

    def __init__(self, cfg):
        super().__init__()
        data = cfg["data"]
        self.n_fft = data["n_fft"]
        self.stft_extractor = STFT(
            n_fft=data["n_fft"], hop_length=data["hop_length"], win_length=data["n_fft"],
            window=data["window"], center=False, pad_mode="constant",
            freeze_parameters=data["feature_freeze"],
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=data["sample_rate"], n_fft=data["n_fft"], n_mels=data["n_mels"],
            fmin=data["fmin"], fmax=data["fmax"], ref=1.0, amin=1e-10,
            top_db=None, freeze_parameters=data["feature_freeze"],
        )

    def forward(self, x):
        # Frame k ends at original sample k * hop - 1. The all-zero first frame
        # preserves the author's 161 -> 40 temporal shape for a four-second clip.
        x = F.pad(x, (self.n_fft, 0), mode="constant", value=0.0)
        stft = self.stft_extractor(x)
        logmel = self.logmel_extractor(spectrogram_STFTInput(stft))
        iv = intensityvector(stft, self.logmel_extractor.melW)
        return torch.cat((logmel, iv), dim=1)

