"""Registered v2 adapters. Prediction association never consumes GT targets."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from candidate import (ResidualHead, DerivativeHead, prediction_history,
                       correct_direction, derivative_fusion, fixed_records,
                       replace_record_directions)
from decoder import angles, unit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def csv_bytes(records):
    s = io.StringIO(newline='')
    w = csv.writer(s)
    for frame, cls, slot, az, el in records:
        w.writerow([frame, cls, az, el])
    return s.getvalue().encode()


def record_labels(records):
    d = {}
    for frame, cls, slot, az, el in records:
        d.setdefault(frame, []).append([cls, slot, az, el])
    return d


def polar(az, el):
    az, el = np.deg2rad([az, el])
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


def read_gt(path):
    labels = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            assert len(row) == 5
            t, c, identity = map(int, row[:3])
            az, el = map(float, row[3:])
            assert 0 <= t < 600 and 0 <= c < 14 and np.isfinite([az, el]).all()
            labels.setdefault(t, []).append([c, identity, az, el])
    return labels


def fixed_targets(prob, raw, mapping, chunks, gt):
    """Supervision is a separate sidecar; fixed raw matching, no 20-deg pruning."""
    active = prob.max(-1) > .5
    cls = prob.argmax(-1)
    target = np.zeros_like(raw, dtype=np.float32)
    identity = np.full((*active.shape, 2), -1, np.int64)
    matched = np.zeros_like(active)
    zero = np.linalg.norm(raw, axis=-1) <= 1e-12
    ties = 0
    for t, rows in sorted(gt.items()):
        keys = [(r[0], r[1]) for r in rows]
        assert len(set(keys)) == len(keys), 'Duplicate GT identity within frame'
        for c in sorted({r[0] for r in rows}):
            pi = np.flatnonzero(active[t] & (cls[t] == c) & ~zero[t])
            gr = [r for r in rows if r[0] == c]
            if not len(pi):
                continue
            gu = np.array([polar(r[2], r[3]) for r in gr])
            cost = angles(raw[t, pi], gu)
            for axis in (0, 1):
                ordered = np.sort(cost, axis=axis)
                if ordered.shape[axis] > 1:
                    ties += int(np.sum(np.diff(ordered, axis=axis).take(0, axis=axis) <= 1e-6))
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                slot = pi[i]
                matched[t, slot] = True
                identity[t, slot] = c, gr[j][1]
                target[t, slot] = gu[j]
    valid = np.zeros_like(active)
    displacement = np.zeros_like(raw, dtype=np.float32)
    counts = dict(active_detections=int(active.sum()), matched=int(matched.sum()),
                  unmatched=int((active & ~matched).sum()), gt_events=sum(map(len, gt.values())),
                  zero_raw_active=int((zero & active).sum()), gt_assignment_tie_signals=ties,
                  predicted_pairs=int((mapping >= 0).sum()), pair_gt_inconsistent=0,
                  pair_unmatched_endpoint=0, identity_continuous_gt_pairs=0)
    counts['FN_at_fixed_frame_class_matching'] = counts['gt_events']-counts['matched']
    for t in range(1, len(raw)):
        if chunks[t] != chunks[t-1]:
            assert (mapping[t] == -1).all()
            continue
        prev_keys = {(r[0], r[1]) for r in gt.get(t-1, [])}
        counts['identity_continuous_gt_pairs'] += sum((r[0], r[1]) in prev_keys for r in gt.get(t, []))
        for s, prev in enumerate(mapping[t]):
            if prev < 0:
                continue
            if not (matched[t, s] and matched[t-1, prev]):
                counts['pair_unmatched_endpoint'] += 1
            elif not np.array_equal(identity[t, s], identity[t-1, prev]):
                counts['pair_gt_inconsistent'] += 1
            else:
                valid[t, s] = True
                displacement[t, s] = target[t, s]-target[t-1, prev]
    counts['valid_motion_pairs'] = int(valid.sum())
    counts['static_pairs'] = int((valid & (np.linalg.norm(displacement, axis=-1) <= 1e-7)).sum())
    counts['moving_pairs'] = counts['valid_motion_pairs']-counts['static_pairs']
    assert counts['predicted_pairs'] == counts['valid_motion_pairs']+counts['pair_unmatched_endpoint']+counts['pair_gt_inconsistent']
    return dict(target=target, gt_identity=identity, matched=matched,
                motion_valid=valid, displacement=displacement), counts


def kalman(raw, mapping):
    """Fixed unit-DOA constant-displacement KF; state transported by raw mapping."""
    out = raw.copy()
    norms = np.linalg.norm(raw, axis=-1)
    u = unit(raw)
    f = np.eye(6); f[:3, 3:] = np.eye(3)
    h = np.zeros((3, 6)); h[:, :3] = np.eye(3)
    p0 = np.diag([1., 1., 1., .1, .1, .1])
    q = np.eye(6)*.001; r = np.eye(3)*.01
    states, covs = {}, {}
    for t in range(len(raw)):
        new_states, new_covs = {}, {}
        for s, prev in enumerate(mapping[t]):
            if norms[t, s] <= 1e-12 or not np.isfinite(raw[t, s]).all():
                continue
            state = np.r_[u[t, s], np.zeros(3)]
            cov = p0.copy()
            if prev >= 0 and int(prev) in states:
                state = f @ states[int(prev)]
                cov = f @ covs[int(prev)] @ f.T + q
                k = np.linalg.solve(h @ cov @ h.T+r, h @ cov).T
                state = state + k @ (u[t, s]-h @ state)
                a = np.eye(6)-k @ h
                cov = a @ cov @ a.T+k @ r @ k.T  # Joseph form
                n = np.linalg.norm(state[:3])
                if np.isfinite(state).all() and n > 1e-12:
                    out[t, s] = norms[t, s]*state[:3]/n
                else:
                    state, cov = np.r_[u[t, s], np.zeros(3)], p0.copy()
            new_states[s], new_covs[s] = state, cov
        states, covs = new_states, new_covs
    return out


def previous_tensor(x, mapping):
    """Batched [B,T,S,D]; first frame forcibly zero (no cross-chunk state)."""
    shifted = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    index = mapping.clamp_min(0)[..., None].expand(*mapping.shape, x.shape[-1])
    y = torch.gather(shifted, 2, index)
    valid = mapping >= 0
    valid = valid.clone(); valid[:, 0] = False
    return y * valid[..., None]


def head_forward(head, h, raw, mapping, condition):
    if condition == 'F-Deriv':
        return head(h)
    h, raw = h.detach(), raw.detach()
    u = torch.nn.functional.normalize(raw, dim=-1, eps=1e-12)
    x = torch.cat((h, u), dim=-1)
    if condition in ('R1', 'R2'):
        valid = (mapping >= 0).clone(); valid[:, 0] = False
        x = torch.cat((x, previous_tensor(h, mapping), previous_tensor(u, mapping), valid[..., None]), dim=-1)
    return head.net(x)


def make_head(condition):
    return DerivativeHead() if condition == 'F-Deriv' else ResidualHead(condition in ('R1', 'R2'))


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def loss_terms(head, batch, condition):
    raw = batch['doa']; active = batch['probability'].amax(-1) > .5
    delta = head_forward(head, batch['doa_features'], raw, batch['mapping'], condition)
    matched, valid = batch['matched'], batch['motion_valid']
    unmatched = active & ~matched
    if condition == 'F-Deriv':
        fit = torch.nn.functional.smooth_l1_loss(delta, batch['displacement'], beta=.1, reduction='none').mean(-1)
        direction = masked_mean(fit, valid)
        regularizer = masked_mean(delta.square().mean(-1), unmatched)
        motion = delta.sum()*0
    else:
        corrected = correct_direction(raw, delta, active)
        u = torch.nn.functional.normalize(corrected, dim=-1, eps=1e-12)
        orig = torch.nn.functional.normalize(raw, dim=-1, eps=1e-12)
        direction = masked_mean(1-(u*batch['target']).sum(-1), matched)
        regularizer = masked_mean((u-orig).square().sum(-1), unmatched)
        diff = u-previous_tensor(u, batch['mapping'])
        motion = masked_mean((diff-batch['displacement']).square().mean(-1), valid)
    total = direction + .1*regularizer + (motion if condition == 'R2' else 0)
    return total, dict(direction=direction.detach(), motion=motion.detach(), regularizer=regularizer.detach(),
                       matched=matched.sum(), motion_pairs=valid.sum(), unmatched=unmatched.sum())
