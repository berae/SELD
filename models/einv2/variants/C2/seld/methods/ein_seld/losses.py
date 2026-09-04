import numpy as np
import torch
import torch.nn.functional as F
from methods.utils.loss_utilities import BCEWithLogitsLoss, MSELoss


class Losses:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.beta = cfg['training']['loss_beta']
        self.lambda_jepa = float(cfg['training']['lambda_jepa'])
        self.jepa_horizons = [int(value) for value in cfg['training']['jepa_horizons_frames']]

        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]

        self.names = ['loss_all'] + [loss.name for loss in self.losses] + ['loss_jepa']
    
    def calculate(self, pred, target, target_latent, epoch_it=0):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            updated_target = target
            pit_original = torch.ones(pred['sed'].shape[:2], dtype=torch.bool,
                                      device=pred['sed'].device)
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa, updated_target, pit_original = self.tPIT(pred, target)

        canonical_prediction = self.canonicalize_tracks(
            pred['future_latent_pred'], pit_original)
        canonical_target = self.canonicalize_tracks(target_latent.detach(), pit_original)
        horizon_losses = []
        prediction_vectors, target_vectors = [], []
        for horizon_idx, frame_offset in enumerate(self.jepa_horizons):
            prediction = canonical_prediction[:, :-frame_offset, :, horizon_idx, :]
            future_target = canonical_target[:, frame_offset:, :, :]
            valid = target['jepa_valid_mask'][:, :-frame_offset, :, horizon_idx] > 0.5
            if valid.any():
                prediction_valid = prediction[valid]
                target_valid = future_target[valid]
                horizon_loss = (1.0 - F.cosine_similarity(
                    prediction_valid.float(), target_valid.float(), dim=-1, eps=1e-6)).mean()
                prediction_vectors.append(prediction_valid)
                target_vectors.append(target_valid)
            else:
                horizon_loss = pred['sed'].new_zeros(())
            horizon_losses.append(horizon_loss)
        loss_jepa = torch.stack(horizon_losses).mean()
        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa + \
            self.lambda_jepa * loss_jepa
        if prediction_vectors:
            all_prediction_vectors = torch.cat(prediction_vectors, dim=0)
            all_target_vectors = torch.cat(target_vectors, dim=0)
        else:
            latent_dim = pred['future_latent_pred'].shape[-1]
            all_prediction_vectors = pred['future_latent_pred'].reshape(-1, latent_dim)[:0]
            all_target_vectors = target_latent.reshape(-1, latent_dim)[:0]
        losses_dict = {
            'all': loss_all,
            'sed': loss_sed,
            'doa': loss_doa,
            'jepa': loss_jepa,
            'jepa_horizon_losses': horizon_losses,
            'jepa_prediction_vectors': all_prediction_vectors,
            'jepa_target_vectors': all_target_vectors,
            'updated_target': updated_target,
            'pit_original': pit_original,
        }
        return losses_dict

    @staticmethod
    def canonicalize_tracks(value, pit_original):
        selector_shape = list(pit_original.shape) + [1] * (value.dim() - 2)
        selector = pit_original.reshape(selector_shape)
        return torch.where(selector, value, value.flip(dims=[2]))

    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 2 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]            
            }
        Return:
            updated_target: updated target with the minimum loss frame-wisely
                {
                    'sed': [batch_size, T, num_tracks=2, num_classes], 
                    'doa': [batch_size, T, num_tracks=2, doas=3]            
                }
        """
        target_flipped = {
            'sed': target['sed'].flip(dims=[2]),
            'doa': target['doa'].flip(dims=[2])
        }

        loss_sed1 = self.losses_pit[0].calculate_loss(pred['sed'], target['sed'])
        loss_sed2 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped['sed'])
        loss_doa1 = self.losses_pit[1].calculate_loss(pred['doa'], target['doa'])
        loss_doa2 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped['doa'])

        loss1 = loss_sed1 + loss_doa1
        loss2 = loss_sed2 + loss_doa2

        loss_sed = (loss_sed1 * (loss1 <= loss2) + loss_sed2 * (loss1 > loss2)).mean()
        loss_doa = (loss_doa1 * (loss1 <= loss2) + loss_doa2 * (loss1 > loss2)).mean()
        updated_target_sed = target['sed'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['sed'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target_doa = target['doa'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['doa'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target = {
            'sed': updated_target_sed,
            'doa': updated_target_doa
        }
        return loss_sed, loss_doa, updated_target, (loss1 <= loss2)
