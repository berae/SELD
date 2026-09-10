"""Frozen decoded-event direction adapters. No optimizer or training entry point."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from raw_association import associate, angles


def decoded_events(output, params, native):
    """Preserve native full-vector addition order, event order, and members."""
    c = params['unique_classes']
    tracks = native['get_tracks'](output, c, .5)
    records = []
    for t in range(tracks[0][0].shape[0]):
        for cls in range(c):
            flags = [native['determine_similar_location'](
                tracks[a][0][t, cls], tracks[b][0][t, cls],
                tracks[a][1][t], tracks[b][1][t], cls, params['thresh_unify'], c)
                for a, b in ((0, 1), (1, 2), (2, 0))]
            if sum(flags) == 0:
                groups = [(i,) for i in range(3) if tracks[i][0][t, cls]]
            elif sum(flags) == 1:
                a, b, other = ((0, 1, 2), (1, 2, 0), (2, 0, 1))[flags.index(1)]
                groups = ([(other,)] if tracks[other][0][t, cls] else []) + [(a, b)]
            else:
                groups = [(0, 1, 2)]
            for ordinal, members in enumerate(groups):
                vector = tracks[members[0]][1][t]
                if len(members) > 1:
                    vector = vector + tracks[members[1]][1][t]
                    if len(members) == 3:
                        vector = vector + tracks[members[2]][1][t]
                    vector = vector / len(members)
                xyz = vector[[cls, cls+c, cls+2*c]].copy()
                records.append(dict(frame=t, cls=cls, ordinal=ordinal,
                                    chunk=t//params['label_sequence_length'],
                                    members=list(members), xyz=xyz.tolist()))
    return records


def as_native(records, xyz=None):
    out = {}
    for i, r in enumerate(records):
        q = r['xyz'] if xyz is None else xyz[i]
        out.setdefault(r['frame'], []).append([r['cls'], *q])
    return out


def history(records):
    mapping = np.full(len(records), -1, np.int64)
    frames = {}
    for i, r in enumerate(records):
        frames.setdefault(r['frame'], []).append(i)
    raw = np.asarray([r['xyz'] for r in records], np.float32).reshape(-1, 3)
    cls = np.asarray([r['cls'] for r in records])
    for t, ids in frames.items():
        prev = frames.get(t-1, [])
        if not prev or records[ids[0]]['chunk'] != records[prev[0]]['chunk']:
            continue
        m = associate(raw[ids], raw[prev], cls[ids], cls[prev],
                      np.ones(len(ids), bool), np.ones(len(prev), bool))
        for i, j in zip(ids, m):
            if j >= 0:
                mapping[i] = prev[j]
    return mapping


def previous(x, mapping):
    index = torch.as_tensor(mapping, dtype=torch.long, device=x.device)
    if not len(index):
        return torch.zeros_like(x)
    return x[index.clamp_min(0)] * (index >= 0).to(x.dtype).unsqueeze(-1)


class DirectionHead(nn.Module):
    def __init__(self, use_history=False):
        super().__init__()
        self.use_history = use_history
        self.net = nn.Sequential(nn.Linear(263 if use_history else 131, 128),
                                 nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, raw, mapping):
        h, raw = h.detach(), raw.detach()
        u = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        current = torch.cat((h, u), -1)
        if self.use_history:
            valid = torch.as_tensor(mapping >= 0, dtype=h.dtype, device=h.device)[:, None]
            current = torch.cat((current, previous(h, mapping), previous(u, mapping), valid), -1)
        return self.net(current)


def correct(raw, delta):
    raw = raw.detach()
    norm = raw.norm(dim=-1, keepdim=True)
    u = raw / norm.clamp_min(1e-12)
    xi = delta - (delta*u).sum(-1, keepdim=True)*u
    magnitude = xi.norm(dim=-1, keepdim=True)
    xi = xi * (np.deg2rad(15.) / magnitude.clamp_min(1e-12)).clamp(max=1.)
    theta = xi.norm(dim=-1, keepdim=True)
    change = norm*((torch.cos(theta)-1)*u + torch.sinc(theta/torch.pi)*xi)
    valid = (norm > 1e-12) & torch.isfinite(delta).all(-1, keepdim=True)
    return torch.where(valid, raw+change, raw)


def targets(records, mapping, gt):
    n = len(records)
    raw = np.asarray([r['xyz'] for r in records], np.float32).reshape(-1, 3)
    target = np.zeros_like(raw)
    identity = np.full((n, 2), -1, np.int64)
    matched = np.zeros(n, bool)
    groups = {}
    for i, r in enumerate(records):
        if np.linalg.norm(raw[i]) > 1e-12:
            groups.setdefault((r['frame'], r['cls']), []).append(i)
    for (t, c), ids in groups.items():
        rows = [r for r in gt.get(t, []) if r[0] == c]
        if not rows:
            continue
        units = []
        for _, _, az, el in rows:
            a, e = np.deg2rad([az, el])
            units.append([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        units = np.asarray(units)
        ii, jj = linear_sum_assignment(angles(raw[ids], units))
        for i, j in zip(ii, jj):
            k = ids[i]
            target[k] = units[j]
            identity[k] = rows[j][:2]
            matched[k] = True
    valid = np.zeros(n, bool)
    displacement = np.zeros_like(raw)
    for i, j in enumerate(mapping):
        if j >= 0 and matched[i] and matched[j] and np.array_equal(identity[i], identity[j]):
            valid[i] = True
            displacement[i] = target[i]-target[j]
    return dict(target=target, matched=matched, motion_valid=valid, displacement=displacement,
                identity=identity)


def masked_mean(x, mask):
    return (x*mask.to(x.dtype)).sum()/mask.sum().clamp_min(1)


def loss(head, h, raw, mapping, target, motion):
    q = correct(raw, head(h, raw, mapping))
    u = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    orig = raw/raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    d = masked_mean(1-(u*target['target']).sum(-1), target['matched'])
    reg = masked_mean((u-orig).square().sum(-1), ~target['matched'])
    m = masked_mean((u-previous(u, mapping)-target['displacement']).square().mean(-1), target['motion_valid'])
    return d+.1*reg+(m if motion else 0), dict(direction=d, regularizer=reg, motion=m)
