import torch
import torch.nn as nn

from methods.ein_seld.models.seld import EINV2
from methods.utils.model_utilities import init_layer


class OfflineEINV2JEPAConsistency(EINV2):
    """Bidirectional EINV2 with an EMA latent temporal-consistency objective.

    This intentionally retains E0's full-context Transformer. Consequently the
    auxiliary objective is a latent consistency regularizer, not a leakage-free
    causal future-prediction objective.
    """

    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset)
        latent_dim = int(cfg['training']['jepa_latent_dim'])
        hidden_dim = int(cfg['training']['jepa_predictor_hidden_dim'])
        horizons = cfg['training']['jepa_horizons_frames']
        self.register_buffer(
            'jepa_horizon_seconds',
            torch.tensor(horizons, dtype=torch.float32) * 0.1,
        )
        self.latent_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim)),
            nn.Sequential(nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim)),
        ])
        self.latent_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for projector in self.latent_projectors:
            init_layer(projector[0])
        init_layer(self.latent_predictor[0])
        init_layer(self.latent_predictor[2])

    def forward(self, x, latent_only=False):
        x_sed = x[:, :4]
        x_doa = x

        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block4(x_sed).mean(dim=3).permute(2, 0, 1)
        x_doa = self.doa_conv_block4(x_doa).mean(dim=3).permute(2, 0, 1)

        # No mask: this is the original E0 bidirectional receptive field.
        doa_features = [
            self.doa_trans_track1(x_doa).transpose(0, 1),
            self.doa_trans_track2(x_doa).transpose(0, 1),
        ]
        latent_tracks = [
            projector(features)
            for projector, features in zip(self.latent_projectors, doa_features)
        ]
        latent = torch.stack(latent_tracks, dim=2)
        if latent_only:
            return {'latent': latent}

        sed_features = [
            self.sed_trans_track1(x_sed).transpose(0, 1),
            self.sed_trans_track2(x_sed).transpose(0, 1),
        ]
        sed_tracks = [
            self.final_act_sed(self.fc_sed_track1(sed_features[0])),
            self.final_act_sed(self.fc_sed_track2(sed_features[1])),
        ]
        doa_tracks = [
            self.final_act_doa(self.fc_doa_track1(doa_features[0])),
            self.final_act_doa(self.fc_doa_track2(doa_features[1])),
        ]

        batch, time_steps, tracks, latent_dim = latent.shape
        horizon_count = self.jepa_horizon_seconds.numel()
        expanded_latent = latent.unsqueeze(3).expand(
            -1, -1, -1, horizon_count, -1
        )
        horizon = self.jepa_horizon_seconds.view(
            1, 1, 1, horizon_count, 1
        ).expand(batch, time_steps, tracks, -1, -1)
        future_latent_pred = self.latent_predictor(
            torch.cat((expanded_latent, horizon), dim=-1)
        )
        return {
            'sed': torch.stack(sed_tracks, dim=2),
            'doa': torch.stack(doa_tracks, dim=2),
            'latent': latent,
            'future_latent_pred': future_latent_pred,
        }
