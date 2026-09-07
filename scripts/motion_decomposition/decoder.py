"""Prediction-only, nonrecursive two-frame controls. No GT inputs."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def angles(a, b):
    return np.degrees(np.arccos(np.clip(unit(a) @ unit(b).T, -1, 1)))


def associate(current, previous, current_class, previous_class, current_active,
              previous_active, gate=45., ambiguity=1e-6):
    """Maximum-cardinality gated Hungarian; reject ambiguous row/column edges.

    Arrays remain in original slot order. Invalid directions cannot associate.
    Costs outside the gate use a large penalty before assignment (not after).
    Equal-nearest row/column costs are rejected, never broken by GT or slot ID.
    """
    result = np.full(len(current), -1, dtype=int)
    good_c = current_active & np.isfinite(current).all(-1) & (np.linalg.norm(current, axis=-1) > 1e-12)
    good_p = previous_active & np.isfinite(previous).all(-1) & (np.linalg.norm(previous, axis=-1) > 1e-12)
    for cls in sorted(set(current_class[good_c]) & set(previous_class[good_p])):
        ci = np.flatnonzero(good_c & (current_class == cls))
        pi = np.flatnonzero(good_p & (previous_class == cls))
        cost = angles(current[ci], previous[pi])
        allowed = cost <= gate
        # Ambiguous nearest-neighbour evidence is conservatively excluded.
        for i in range(len(ci)):
            valid = np.sort(cost[i, allowed[i]])
            if len(valid) > 1 and valid[1] - valid[0] <= ambiguity:
                allowed[i] = False
        for j in range(len(pi)):
            valid = np.sort(cost[allowed[:, j], j])
            if len(valid) > 1 and valid[1] - valid[0] <= ambiguity:
                allowed[:, j] = False
        rows, cols = linear_sum_assignment(np.where(allowed, cost, 1e6))
        for i, j in zip(rows, cols):
            if allowed[i, j]:
                result[ci[i]] = pi[j]
    return result


def decode(probability, raw_xyz, velocity, frame_index, chunk_index, mode,
           alpha=.5, dt=.1, gate=45., ambiguity=1e-6, threshold=.5):
    if mode not in ('raw', 'smoothing', 'fusion'):
        raise ValueError(mode)
    p = np.asarray(raw_xyz)
    q = p.copy()
    classes = probability.argmax(-1)
    active = probability.max(-1) > threshold
    mapping = np.full(active.shape, -1, dtype=int)
    applied = np.zeros(active.shape, dtype=bool)
    if mode == 'raw':
        return q, mapping, applied
    if velocity.shape != p.shape:
        raise ValueError('velocity shape mismatch')
    for t in range(1, len(p)):
        if frame_index[t] != frame_index[t-1] + 1 or chunk_index[t] != chunk_index[t-1]:
            continue
        m = associate(p[t], p[t-1], classes[t], classes[t-1], active[t], active[t-1], gate, ambiguity)
        mapping[t] = m
        for slot in np.flatnonzero(m >= 0):
            prev = p[t-1, m[slot]]
            smooth = alpha*p[t, slot] + (1-alpha)*prev
            fused = alpha*p[t, slot] + (1-alpha)*(prev + dt*velocity[t, slot])
            # Shared numerical mask makes the two controls exactly comparable.
            if all(np.isfinite(x).all() and np.linalg.norm(x) > 1e-12 for x in (smooth, fused)):
                q[t, slot] = smooth if mode == 'smoothing' else fused
                applied[t, slot] = True
    return q, mapping, applied


def labels(probability, xyz, threshold=.5):
    """Frozen runner's polar quantization and slot row order, including empty files."""
    if not np.isfinite(probability).all() or not np.isfinite(xyz).all():
        raise FloatingPointError('Cannot serialize non-finite raw output')
    classes = probability.argmax(-1)
    active = probability.max(-1) > threshold
    az = np.rint(np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))).astype(int)
    el = np.rint(np.rad2deg(np.arctan2(xyz[..., 2], np.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2)))).astype(int)
    out = {}
    for frame, slot in np.argwhere(active):
        out.setdefault(int(frame), []).append([int(classes[frame, slot]), int(slot), int(az[frame, slot]), int(el[frame, slot])])
    return out
