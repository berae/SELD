import torch
import torch.nn as nn

from methods.ein_seld.models.seld import EINV2
from methods.utils.model_utilities import init_layer


class OfflineEINV2Velocity(EINV2):
    """Original bidirectional EINV2 with an auxiliary Cartesian velocity head."""

    def __init__(self, cfg, dataset):
        # Initialize every E0 parameter before adding E1-only modules. With the
        # same seed, all shared parameters therefore have identical initial values.
        super().__init__(cfg, dataset)
        self.velocity_heads = nn.ModuleList([
            nn.Linear(512, 3, bias=True),
            nn.Linear(512, 3, bias=True),
        ])
        for head in self.velocity_heads:
            init_layer(head)
            with torch.no_grad():
                head.weight.mul_(0.1)
                head.bias.zero_()

    def forward(self, x):
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

        # Intentionally no attention mask: retain E0's full bidirectional context.
        sed_features = [
            self.sed_trans_track1(x_sed).transpose(0, 1),
            self.sed_trans_track2(x_sed).transpose(0, 1),
        ]
        doa_features = [
            self.doa_trans_track1(x_doa).transpose(0, 1),
            self.doa_trans_track2(x_doa).transpose(0, 1),
        ]

        sed_tracks = [
            self.final_act_sed(self.fc_sed_track1(sed_features[0])),
            self.final_act_sed(self.fc_sed_track2(sed_features[1])),
        ]
        doa_tracks = [
            self.final_act_doa(self.fc_doa_track1(doa_features[0])),
            self.final_act_doa(self.fc_doa_track2(doa_features[1])),
        ]
        velocity_tracks = [
            head(features) for head, features in zip(self.velocity_heads, doa_features)
        ]
        return {
            'sed': torch.stack(sed_tracks, dim=2),
            'doa': torch.stack(doa_tracks, dim=2),
            'velocity': torch.stack(velocity_tracks, dim=2),
        }
