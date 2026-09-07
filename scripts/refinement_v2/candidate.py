"""V2 reference components: frozen-feature heads and prediction-only corrections.

No dataset loader, optimizer, GT input, or training entry point is provided here.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motion_decomposition"))
from decoder import associate, labels


def prediction_history(probability, raw, frames, chunks, threshold=.5):
    """Immediate raw predecessor only; never a GT identity or corrected direction."""
    classes = probability.argmax(-1)
    active = probability.max(-1) > threshold
    mapping = np.full(active.shape, -1, dtype=np.int64)
    for t in range(1, len(raw)):
        if frames[t] == frames[t-1] + 1 and chunks[t] == chunks[t-1]:
            mapping[t] = associate(raw[t], raw[t-1], classes[t], classes[t-1],
                                   active[t], active[t-1], gate=45., ambiguity=1e-6)
    return mapping


def head_inputs(h, raw, mapping, explicit_history):
    """h is the detached 512-D pre-DOA-head causal Transformer representation."""
    h, raw = h.detach(), raw.detach()
    u = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    current = torch.cat((h, u), dim=-1)
    if not explicit_history:
        return current
    previous_h, previous_u = torch.zeros_like(h), torch.zeros_like(u)
    valid = torch.zeros_like(u[..., :1])
    for t in range(1, len(h)):
        for slot, prev in enumerate(mapping[t]):
            if prev >= 0:
                previous_h[t, slot] = h[t-1, int(prev)]
                previous_u[t, slot] = u[t-1, int(prev)]
                valid[t, slot] = 1.
    return torch.cat((current, previous_h, previous_u, valid), dim=-1)


class ResidualHead(nn.Module):
    def __init__(self, explicit_history=False, width=128):
        super().__init__()
        self.explicit_history = explicit_history
        self.net = nn.Sequential(nn.Linear(1031 if explicit_history else 515, width),
                                 nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, raw, mapping):
        return self.net(head_inputs(h, raw, mapping, self.explicit_history))


class DerivativeHead(nn.Module):
    """Predict unit-DOA displacement per 100 ms frame, NOT per-second velocity."""
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h.detach())


def correct_direction(raw, delta, active, rho_degrees=15.):
    """Sphere exponential residual, norm-preserving; exact zero-output identity.

    Written as raw + increment to avoid normalization/rounding drift at zero.
    The zero initialization still has nonzero tangent derivatives.
    """
    raw = raw.detach()
    norm = raw.norm(dim=-1, keepdim=True)
    u = raw / norm.clamp_min(1e-12)
    xi = delta - (delta*u).sum(-1, keepdim=True)*u
    magnitude = xi.norm(dim=-1, keepdim=True)
    rho = np.deg2rad(rho_degrees)
    xi = xi * (rho / magnitude.clamp_min(1e-12)).clamp(max=1.)
    theta = xi.norm(dim=-1, keepdim=True)
    change = norm * ((torch.cos(theta)-1)*u + torch.sinc(theta/torch.pi)*xi)
    valid = active[..., None] & (norm > 1e-12) & torch.isfinite(delta).all(-1, keepdim=True)
    return torch.where(valid, raw + change, raw)


def derivative_fusion(raw, displacement, mapping):
    """Causal adaptation of the prior update on UNIT directions, alpha=.5.

    F-EMA must use the same unit-vector convention. No dt is applied to d/frame.
    This differs from the archived raw-xyz C1 decoder and is labelled separately.
    """
    out = raw.copy()
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    u = raw / np.maximum(norm, 1e-12)
    for t in range(1, len(raw)):
        for slot, prev in enumerate(mapping[t]):
            if prev < 0:
                continue
            q = .5*u[t, slot] + .5*(u[t-1, prev] + displacement[t, slot])
            qnorm = np.linalg.norm(q)
            if np.isfinite(q).all() and qnorm > 1e-12:
                out[t, slot] = norm[t, slot] * q/qnorm
    return out


def fixed_records(probability, raw):
    """IDs preserve slot/multiplicity. Scores/probabilities remain externally frozen."""
    lab = labels(probability, raw)
    return [(frame, row[0], row[1], row[2], row[3])
            for frame, rows in sorted(lab.items()) for row in rows]


def replace_record_directions(records, corrected):
    """No re-decoding, reassociation, or deduplication after correction."""
    result = []
    for frame, cls, rid, _, _ in records:
        x, y, z = corrected[frame, rid]
        if not np.isfinite([x,y,z]).all():
            raise ValueError("Non-finite corrected direction")
        az = int(np.rint(np.rad2deg(np.arctan2(y, x))))
        el = int(np.rint(np.rad2deg(np.arctan2(z, np.sqrt(x*x+y*y)))))
        result.append((frame, cls, rid, az, el))
    assert [r[:3] for r in records] == [r[:3] for r in result]
    return result
