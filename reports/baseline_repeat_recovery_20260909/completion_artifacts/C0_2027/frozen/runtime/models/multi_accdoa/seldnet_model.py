# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed


class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def calculate_with_alignment(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        candidate_targets = torch.stack((
            target_A0A0A0 + pad4A,
            target_B0B0B1 + pad4B,
            target_B0B1B0 + pad4B,
            target_B0B1B1 + pad4B,
            target_B1B0B0 + pad4B,
            target_B1B0B1 + pad4B,
            target_B1B1B0 + pad4B,
            target_C0C1C2 + pad4C,
            target_C0C2C1 + pad4C,
            target_C1C0C2 + pad4C,
            target_C1C2C0 + pad4C,
            target_C2C0C1 + pad4C,
            target_C2C1C0 + pad4C,
        ), dim=0)
        candidate_losses = torch.stack((
            loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6,
            loss_7, loss_8, loss_9, loss_10, loss_11, loss_12,
        ), dim=0)
        loss_min = candidate_losses.argmin(dim=0)  # [B, T, C]
        chosen_loss = torch.gather(candidate_losses, 0, loss_min.unsqueeze(0)).squeeze(0)
        loss = chosen_loss.mean()

        # ADPIT chooses a permutation independently for every frame and class.
        # Return that exact target so all auxiliary losses use the same assignment.
        gather_index = loss_min.unsqueeze(0).unsqueeze(3).expand(
            1, output.shape[0], output.shape[1], output.shape[2], output.shape[3]
        )
        aligned = torch.gather(candidate_targets, 0, gather_index).squeeze(0)
        aligned = aligned.reshape(aligned.shape[0], aligned.shape[1], 3, 3, aligned.shape[-1])
        aligned = aligned.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,track,class,xyz]
        return loss, aligned, loss_min

    def __call__(self, output, target):
        return self.calculate_with_alignment(output, target)[0]


class DynamicMSELoss_ADPIT(object):
    """ADPIT localization loss plus PIT-aligned velocity and JEPA auxiliaries."""

    def __init__(self, params):
        self.params = params
        self.adpit = MSELoss_ADPIT()

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        count = mask.sum()
        if count.item() == 0:
            return values.sum() * 0.0, count
        return (values * mask).sum() / (count * values.shape[-1]), count

    def __call__(self, output, target, target_latent=None):
        doa_loss, aligned, assignment = self.adpit.calculate_with_alignment(output['doa'], target)
        activity = torch.linalg.vector_norm(aligned, dim=-1) > 0.5
        zero = doa_loss.new_zeros(())
        velocity_loss, velocity_count = zero, zero
        jepa_loss, jepa_count = zero, zero
        latent_std, latent_pair_cos = zero, zero

        if self.params.get('use_velocity', False):
            dt = float(self.params['velocity_dt'])
            velocity_target = (aligned[:, 1:] - aligned[:, :-1]) / dt
            stable = assignment[:, 1:] == assignment[:, :-1]
            velocity_mask = activity[:, 1:] & activity[:, :-1] & stable.unsqueeze(2)
            velocity_error = F.smooth_l1_loss(
                output['velocity'][:, 1:], velocity_target, reduction='none'
            )
            velocity_loss, velocity_count = self._masked_mean(velocity_error, velocity_mask)

        if self.params.get('use_jepa', False):
            if target_latent is None:
                raise ValueError('target_latent is required when JEPA is enabled')
            latent_std = target_latent.float().std(dim=(0, 1, 2, 3), unbiased=False).mean()
            flat_target = F.normalize(target_latent.reshape(-1, target_latent.shape[-1]), dim=-1)
            if flat_target.shape[0] > 1:
                latent_pair_cos = (flat_target[:-1] * flat_target[1:]).sum(dim=-1).mean()
            losses = []
            counts = []
            for horizon_index, horizon in enumerate(self.params['jepa_horizons_frames']):
                stable = assignment[:, horizon:] == assignment[:, :-horizon]
                mask = activity[:, horizon:] & activity[:, :-horizon] & stable.unsqueeze(2)
                pred = output['future_latent'][:, :-horizon, :, :, horizon_index]
                future = target_latent[:, horizon:].detach()
                cosine_error = 1.0 - F.cosine_similarity(pred, future, dim=-1)
                count = mask.to(cosine_error.dtype).sum()
                if count.item() > 0:
                    losses.append((cosine_error * mask).sum() / count)
                    counts.append(count)
            if losses:
                jepa_loss = torch.stack(losses).mean()
                jepa_count = torch.stack(counts).sum()

        total = doa_loss
        total = total + float(self.params['lambda_velocity']) * velocity_loss
        total = total + float(self.params['lambda_jepa']) * jepa_loss
        return {
            'total': total,
            'doa': doa_loss,
            'velocity': velocity_loss,
            'jepa': jepa_loss,
            'velocity_count': velocity_count,
            'jepa_count': jepa_count,
            'target_latent_std': latent_std,
            'target_latent_pair_cos': latent_pair_cos,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), causal=False):
        super().__init__()
        self.causal = causal
        self.time_left_padding = kernel_size[0] - 1 if causal else 0
        conv_padding = (0, padding[1]) if causal else padding
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=conv_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.causal:
            # Pad only the past side of the time axis. Frequency remains symmetric.
            x = F.pad(x, (0, 0, self.time_left_padding, 0))
        x = F.relu(self.bn(self.conv(x)))
        return x


class PositionalEmbedding(nn.Module):  # Not used in the baseline
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params
        self.causal = params.get('causal', False)
        self.use_velocity = params.get('use_velocity', False)
        self.use_jepa = params.get('use_jepa', False)
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt'], causal=self.causal))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=not self.causal)

        # self.pos_embedder = PositionalEmbedding(self.params['rnn_size'])

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=params['nb_heads'], dropout=params['dropout_rate'],  batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        # Always instantiate both auxiliary heads after the shared SELD backbone.
        # This keeps the shared initialization identical for C1/C2/C3.
        dynamic_width = 3 * self.nb_classes
        latent_dim = params.get('jepa_latent_dim', 128)
        predictor_hidden = params.get('jepa_predictor_hidden_dim', 256)
        self.velocity_head = nn.Linear(self.params['rnn_size'], dynamic_width * 3)
        self.latent_projection = nn.Linear(self.params['rnn_size'], dynamic_width * latent_dim)
        self.latent_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, latent_dim),
        )
        if not self.use_velocity:
            for parameter in self.velocity_head.parameters():
                parameter.requires_grad = False
        if not self.use_jepa:
            for module in (self.latent_projection, self.latent_predictor):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, x, latent_only=False):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        if not self.causal:
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        # pos_embedding = self.pos_embedder(x)
        # x = x + pos_embedding
        
        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x 
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
            x, _ = self.mhsa_block_list[mhsa_cnt](
                x_attn_in, x_attn_in, x_attn_in,
                attn_mask=attn_mask,
                need_weights=False,
            )
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        shared = x
        if latent_only:
            latent = self.latent_projection(shared).reshape(
                shared.shape[0], shared.shape[1], 3, self.nb_classes,
                self.params.get('jepa_latent_dim', 128),
            )
            return F.normalize(latent, dim=-1)

        head_input = shared
        for fnn_cnt in range(len(self.fnn_list) - 1):
            head_input = self.fnn_list[fnn_cnt](head_input)
        doa = torch.tanh(self.fnn_list[-1](head_input))
        if not (self.use_velocity or self.use_jepa):
            return doa

        output = {'doa': doa}
        if self.use_velocity:
            output['velocity'] = self.velocity_head(shared).reshape(
                shared.shape[0], shared.shape[1], 3, self.nb_classes, 3
            )
        if self.use_jepa:
            latent_dim = self.params.get('jepa_latent_dim', 128)
            latent = self.latent_projection(shared).reshape(
                shared.shape[0], shared.shape[1], 3, self.nb_classes, latent_dim
            )
            latent = F.normalize(latent, dim=-1)
            predictions = []
            for horizon in self.params['jepa_horizons_frames']:
                horizon_seconds = shared.new_full(
                    (*latent.shape[:-1], 1),
                    float(horizon) * float(self.params['label_hop_len_s'])
                )
                predictions.append(self.latent_predictor(torch.cat((latent, horizon_seconds), dim=-1)))
            output['latent'] = latent
            output['future_latent'] = torch.stack(predictions, dim=4)
        return output
