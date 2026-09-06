"""Versioned EINV2 rerun runtime; the historical variant sources remain frozen.

All four causal ablations instantiate the same network in the same order.
Only loss coefficients change. Unused heads receive no gradients.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'models/einv2/variants/C3/seld'))
sys.path.insert(0, str(ROOT / 'evaluation'))
from methods.ein_seld.models.seld_causal import CausalEINV2VelocityJEPA
from methods.feature_causal import CausalLogmelIntensity_Extractor

VERSION = 'einv2_rb05_weightprobe_v1'
WEIGHTS = {'C0': (0., 0.), 'C1': (.2, 0.), 'C2': (0., .2), 'C3': (.2, .2)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class AuditedEINV2(CausalEINV2VelocityJEPA):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset)
        self.alignment = cfg['audit']['alignment']
        if self.alignment != 'current_frame_75ms':
            raise ValueError(self.alignment)

    def forward(self, x, latent_only=False):
        # Keep historical pooling. Frame j uses feature <= 4*j+3, ending at
        # j*100+75 ms. No samples from the next 100 ms label frame are used.
        return super().forward(x, latent_only=latent_only)


class Frontend(nn.Module):
    """One identical frontend for training, validation and independent inference."""
    def __init__(self, cfg):
        super().__init__()
        self.extractor = CausalLogmelIntensity_Extractor(cfg)
        with h5py.File(cfg['causal_scalar_path'], 'r') as hf:
            if hf.attrs.get('channel_order') != 'logmel_WYZX_IV_XYZ':
                raise ValueError('Use the new train-only canonical-order scalar')
            if hf.attrs.get('folds') != '2,3,4,5,6':
                raise ValueError('Scaler must exclude validation fold 1')
            self.register_buffer('mean', torch.tensor(hf['mean'][:], dtype=torch.float32))
            self.register_buffer('std', torch.tensor(hf['std'][:], dtype=torch.float32))
        self.requires_grad_(False)

    def forward(self, x):
        return (self.extractor(x) - self.mean) / self.std


def assignment(pred, target, beta=.5):
    """Return whether prediction slots match canonical target slots at each frame."""
    costs = []
    for flipped in (False, True):
        sed = target['sed'].flip(2) if flipped else target['sed']
        doa = target['doa'].flip(2) if flipped else target['doa']
        costs.append(beta * F.binary_cross_entropy_with_logits(pred['sed'], sed, reduction='none').mean((-1, -2))
                     + (1 - beta) * (pred['doa'] - doa).square().mean((-1, -2)))
    return costs[0] <= costs[1]


def canonical(value, original):
    selector = original.reshape(*original.shape, *([1] * (value.ndim - 2)))
    return torch.where(selector, value, value.flip(2))


def loss_values(pred, target, teacher, cfg):
    beta = cfg['training']['loss_beta']
    original = assignment(pred, target, beta)
    sed = F.binary_cross_entropy_with_logits(pred['sed'], canonical(target['sed'], original))
    doa = F.mse_loss(pred['doa'], canonical(target['doa'], original))
    velocity = sed.new_zeros(())
    jepa = sed.new_zeros(())
    lv, lj = cfg['training']['lambda_velocity'], cfg['training']['lambda_jepa']
    if lv:
        mask = canonical(target['velocity_mask'], original)
        error = F.smooth_l1_loss(pred['velocity'], canonical(target['velocity'], original), reduction='none').mean(-1)
        velocity = (error * mask).sum() / mask.sum().clamp_min(1.)
    if lj:
        # Teacher output slots can differ from the online model: align independently.
        with torch.no_grad():
            teacher_original = assignment(teacher, target, beta)
            teacher_latent = canonical(teacher['latent'].detach(), teacher_original)
        prediction = canonical(pred['future_latent_pred'], original)
        losses = []
        for index, horizon in enumerate(cfg['training']['jepa_horizons_frames']):
            mask = target['jepa_valid_mask'][:, :-horizon, :, index] > .5
            if mask.any():
                losses.append((1 - F.cosine_similarity(prediction[:, :-horizon, :, index][mask].float(),
                              teacher_latent[:, horizon:][mask].float(), dim=-1, eps=1e-6)).mean())
            else:
                losses.append(sed.new_zeros(()))
        jepa = torch.stack(losses).mean()
    total = beta * sed + (1 - beta) * doa + lv * velocity + lj * jepa
    return {'all': total, 'sed': sed, 'doa': doa, 'velocity': velocity, 'jepa': jepa}


@torch.no_grad()
def update_teacher(teacher, online, momentum):
    for dst, src in zip(teacher.parameters(), online.parameters()):
        dst.mul_(momentum).add_(src, alpha=1 - momentum)
    for dst, src in zip(teacher.buffers(), online.buffers()):
        if dst.is_floating_point():
            dst.mul_(momentum).add_(src, alpha=1 - momentum)
        else:
            dst.copy_(src)


def make_teacher(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(project, scalar, variant, seed, alignment='current_frame_75ms', *, hdf5_dir=None, lambda_jepa=None):
    from ruamel.yaml import YAML
    cfg = YAML(typ='safe').load((ROOT / 'configs/einv2/C3.yaml').read_text())
    cfg.update(dataset_dir=str(project / 'EINV2/dataset_root'),
               hdf5_dir=str(project / 'EINV2/C0_CausalEINV2_seed2026/_hdf5'),
               causal_scalar_path=str(scalar),
               velocity_hdf5_dir=str(project / 'EINV2/C1_CausalEINV2_Velocity_seed2026/velocity_hdf5/dcase2020task3/velocity'),
               jepa_hdf5_dir=str(project / 'EINV2/C2_CausalEINV2_JEPA_seed2026/jepa_hdf5/dcase2020task3/jepa'))
    cfg['data']['foa_iv_acn_reorder'] = False  # New scaler already uses canonical XYZ.
    # Keep validation and independent inference numerically comparable as well.
    cfg['inference']['batch_size'] = cfg['training']['batch_size']
    cfg['training']['lambda_velocity'], cfg['training']['lambda_jepa'] = WEIGHTS[variant]
    if hdf5_dir is not None:
        cfg['hdf5_dir'] = str(Path(hdf5_dir).resolve())
    if lambda_jepa is not None:
        if not np.isfinite(lambda_jepa) or lambda_jepa < 0:
            raise ValueError('lambda_jepa must be finite and nonnegative')
        cfg['training']['lambda_jepa'] = float(lambda_jepa)
    cfg['audit'] = dict(version=VERSION, variant=variant, seed=seed, alignment=alignment,
                        metric='dcase2023_micro', checkpoint_selection='minimum_validation_SELD_LR',
                        scalar_sha256=sha256(scalar), resume='fresh_runs_only',
                        context='4 s independently reset chunks; no streaming cache',
                        auxiliary_heads='identically initialized in every variant; zero-weight heads receive no gradients')
    cfg['audit']['parent_version'] = 'einv2_reaudit_v1 / repository 0.2.0'
    return cfg
