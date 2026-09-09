import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.utils.model_utilities import init_layer


class ChannelLayerNorm2d(nn.Module):
    """Normalize channels independently at every (time, frequency) position."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__(in_channels, out_channels, kernel_size=(3, 3),
                         stride=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        # F.pad order: frequency left/right, time top/bottom.
        x = F.pad(x, (1, 1, 2, 0))
        return super().forward(x)


class CausalDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv2d(in_channels, out_channels),
            ChannelLayerNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CausalConv2d(out_channels, out_channels),
            ChannelLayerNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init_layer(layer)

    def forward(self, x):
        return self.net(x)


class CausalEINV2(nn.Module):
    """EINV2 topology with zero future receptive field at inference and training."""

    def __init__(self, cfg, dataset):
        super().__init__()
        self.sed_blocks = nn.ModuleList([
            nn.Sequential(CausalDoubleConv(4, 64), nn.AvgPool2d((2, 2))),
            nn.Sequential(CausalDoubleConv(64, 128), nn.AvgPool2d((2, 2))),
            nn.Sequential(CausalDoubleConv(128, 256), nn.AvgPool2d((1, 2))),
            nn.Sequential(CausalDoubleConv(256, 512), nn.AvgPool2d((1, 2))),
        ])
        self.doa_blocks = nn.ModuleList([
            nn.Sequential(CausalDoubleConv(7, 64), nn.AvgPool2d((2, 2))),
            nn.Sequential(CausalDoubleConv(64, 128), nn.AvgPool2d((2, 2))),
            nn.Sequential(CausalDoubleConv(128, 256), nn.AvgPool2d((1, 2))),
            nn.Sequential(CausalDoubleConv(256, 512), nn.AvgPool2d((1, 2))),
        ])
        self.stitch = nn.ParameterList([
            nn.Parameter(torch.empty(c, 2, 2).uniform_(0.1, 0.9)) for c in (64, 128, 256)
        ])
        enc = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024,
                                       dropout=0.2), num_layers=2)
        self.sed_transformers = nn.ModuleList([enc(), enc()])
        self.doa_transformers = nn.ModuleList([enc(), enc()])
        self.sed_heads = nn.ModuleList([nn.Linear(512, 14), nn.Linear(512, 14)])
        self.doa_heads = nn.ModuleList([nn.Linear(512, 3), nn.Linear(512, 3)])
        for head in self.sed_heads:
            init_layer(head)
        for head in self.doa_heads:
            init_layer(head)
            # The original scale drives the causal branch into tanh
            # saturation on its first optimizer step.  Preserve the same
            # head and activation, but start inside tanh's linear region.
            with torch.no_grad():
                head.weight.mul_(0.1)
                head.bias.zero_()

    @staticmethod
    def causal_mask(length, device, dtype):
        return torch.triu(torch.full((length, length), float("-inf"), device=device,
                                     dtype=dtype), diagonal=1)

    def forward(self, x):
        x_sed, x_doa = x[:, :4], x
        for idx, (sed_block, doa_block) in enumerate(zip(self.sed_blocks, self.doa_blocks)):
            x_sed = sed_block(x_sed)
            x_doa = doa_block(x_doa)
            if idx < 3:
                old_sed = x_sed
                # Preserve the author's sequential soft-stitch computation exactly.
                x_sed = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 0], old_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 1], x_doa)
                x_doa = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 0], x_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 1], x_doa)
        x_sed = x_sed.mean(dim=3).permute(2, 0, 1)
        x_doa = x_doa.mean(dim=3).permute(2, 0, 1)
        mask = self.causal_mask(x_sed.shape[0], x_sed.device, x_sed.dtype)
        sed_tracks = [head(trans(x_sed, mask=mask).transpose(0, 1))
                      for trans, head in zip(self.sed_transformers, self.sed_heads)]
        doa_tracks = [torch.tanh(head(trans(x_doa, mask=mask).transpose(0, 1)))
                      for trans, head in zip(self.doa_transformers, self.doa_heads)]
        return {"sed": torch.stack(sed_tracks, dim=2),
                "doa": torch.stack(doa_tracks, dim=2)}


class CausalEINV2JEPA(CausalEINV2):
    """Strict-causal EINV2 with object-track future-latent prediction."""

    def __init__(self, cfg, dataset):
        # Keep every common C0.1 initialization before adding C2-only modules.
        super().__init__(cfg, dataset)
        latent_dim = int(cfg['training']['jepa_latent_dim'])
        hidden_dim = int(cfg['training']['jepa_predictor_hidden_dim'])
        horizons = cfg['training']['jepa_horizons_frames']
        self.register_buffer('jepa_horizon_seconds',
                             torch.tensor(horizons, dtype=torch.float32) * 0.1)
        self.latent_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim))
            for _ in range(2)
        ])
        self.latent_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for module in self.latent_projectors:
            init_layer(module[0])
        init_layer(self.latent_predictor[0])
        init_layer(self.latent_predictor[2])

    def forward(self, x, latent_only=False):
        x_sed, x_doa = x[:, :4], x
        for idx, (sed_block, doa_block) in enumerate(zip(self.sed_blocks, self.doa_blocks)):
            x_sed = sed_block(x_sed)
            x_doa = doa_block(x_doa)
            if idx < 3:
                old_sed = x_sed
                x_sed = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 0], old_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 1], x_doa)
                x_doa = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 0], x_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 1], x_doa)
        x_sed = x_sed.mean(dim=3).permute(2, 0, 1)
        x_doa = x_doa.mean(dim=3).permute(2, 0, 1)
        mask = self.causal_mask(x_sed.shape[0], x_sed.device, x_sed.dtype)
        doa_features = [trans(x_doa, mask=mask).transpose(0, 1)
                        for trans in self.doa_transformers]
        latent_tracks = [projector(features)
                         for projector, features in zip(self.latent_projectors, doa_features)]
        latent = torch.stack(latent_tracks, dim=2)
        if latent_only:
            return {"latent": latent}

        sed_tracks = [head(trans(x_sed, mask=mask).transpose(0, 1))
                      for trans, head in zip(self.sed_transformers, self.sed_heads)]
        doa_tracks = [torch.tanh(head(features))
                      for features, head in zip(doa_features, self.doa_heads)]
        batch, time_steps, tracks, latent_dim = latent.shape
        horizon_count = self.jepa_horizon_seconds.numel()
        expanded_latent = latent.unsqueeze(3).expand(-1, -1, -1, horizon_count, -1)
        horizon = self.jepa_horizon_seconds.view(1, 1, 1, horizon_count, 1).expand(
            batch, time_steps, tracks, -1, -1)
        future_latent_pred = self.latent_predictor(
            torch.cat((expanded_latent, horizon), dim=-1))
        return {
            "sed": torch.stack(sed_tracks, dim=2),
            "doa": torch.stack(doa_tracks, dim=2),
            "latent": latent,
            "future_latent_pred": future_latent_pred,
        }


class CausalEINV2VelocityJEPA(CausalEINV2JEPA):
    """C2 strict-causal JEPA model with the C1 Cartesian velocity head."""

    def __init__(self, cfg, dataset):
        # Construct velocity heads after every C2 module so all C2-common
        # parameters are bit-identical under the same random seed.
        super().__init__(cfg, dataset)
        self.velocity_heads = nn.ModuleList([nn.Linear(512, 3), nn.Linear(512, 3)])
        for head in self.velocity_heads:
            init_layer(head)
            with torch.no_grad():
                head.weight.mul_(0.1)
                head.bias.zero_()

    def forward(self, x, latent_only=False):
        x_sed, x_doa = x[:, :4], x
        for idx, (sed_block, doa_block) in enumerate(zip(self.sed_blocks, self.doa_blocks)):
            x_sed = sed_block(x_sed)
            x_doa = doa_block(x_doa)
            if idx < 3:
                old_sed = x_sed
                x_sed = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 0], old_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 0, 1], x_doa)
                x_doa = torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 0], x_sed) + \
                        torch.einsum("c,nctf->nctf", self.stitch[idx][:, 1, 1], x_doa)
        x_sed = x_sed.mean(dim=3).permute(2, 0, 1)
        x_doa = x_doa.mean(dim=3).permute(2, 0, 1)
        mask = self.causal_mask(x_sed.shape[0], x_sed.device, x_sed.dtype)
        doa_features = [trans(x_doa, mask=mask).transpose(0, 1)
                        for trans in self.doa_transformers]
        latent_tracks = [projector(features)
                         for projector, features in zip(self.latent_projectors, doa_features)]
        latent = torch.stack(latent_tracks, dim=2)
        if latent_only:
            return {"latent": latent}

        sed_tracks = [head(trans(x_sed, mask=mask).transpose(0, 1))
                      for trans, head in zip(self.sed_transformers, self.sed_heads)]
        doa_tracks = [torch.tanh(head(features))
                      for features, head in zip(doa_features, self.doa_heads)]
        velocity_tracks = [head(features)
                           for features, head in zip(doa_features, self.velocity_heads)]
        batch, time_steps, tracks, latent_dim = latent.shape
        horizon_count = self.jepa_horizon_seconds.numel()
        expanded_latent = latent.unsqueeze(3).expand(-1, -1, -1, horizon_count, -1)
        horizon = self.jepa_horizon_seconds.view(1, 1, 1, horizon_count, 1).expand(
            batch, time_steps, tracks, -1, -1)
        future_latent_pred = self.latent_predictor(
            torch.cat((expanded_latent, horizon), dim=-1))
        return {
            "sed": torch.stack(sed_tracks, dim=2),
            "doa": torch.stack(doa_tracks, dim=2),
            "velocity": torch.stack(velocity_tracks, dim=2),
            "latent": latent,
            "future_latent_pred": future_latent_pred,
        }
