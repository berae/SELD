#!/usr/bin/env python3
"""Build strict-causal direction-velocity labels aligned to EINV2 internal tracks."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def spherical_to_cartesian(azimuth_deg, elevation_deg):
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return np.asarray([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ], dtype=np.float32)


def build_one(csv_path, meta_path, output_path, frame_step):
    rows = np.loadtxt(csv_path, delimiter=',', dtype=np.float64, ndmin=2)
    with h5py.File(meta_path, 'r') as hf:
        sed = hf['sed_label'][:]
        doa = hf['doa_label'][:]

    frames, tracks = sed.shape[:2]
    identities = np.full((frames, tracks, 2), -1, dtype=np.int32)
    assigned = np.zeros((frames, tracks), dtype=bool)
    max_alignment_error = 0.0

    for row in rows:
        frame, event, track_number, azimuth, elevation = int(row[0]), int(row[1]), int(row[2]), row[3], row[4]
        position = spherical_to_cartesian(azimuth, elevation)
        candidates = [idx for idx in range(tracks)
                      if sed[frame, idx, event] > 0.5 and not assigned[frame, idx]]
        if not candidates:
            raise RuntimeError(f'No internal-track match: {csv_path.name}, frame={frame}, event={event}')
        errors = [float(np.linalg.norm(doa[frame, idx] - position)) for idx in candidates]
        slot = candidates[int(np.argmin(errors))]
        max_alignment_error = max(max_alignment_error, min(errors))
        identities[frame, slot] = (event, track_number)
        assigned[frame, slot] = True

    if max_alignment_error > 1e-4:
        raise RuntimeError(f'DOA/meta mismatch in {csv_path.name}: {max_alignment_error}')

    velocity = np.zeros_like(doa, dtype=np.float32)
    mask = np.zeros((frames, tracks), dtype=np.float32)
    for frame in range(1, frames):
        for slot in range(tracks):
            current_id = identities[frame, slot]
            if current_id[0] >= 0 and np.array_equal(current_id, identities[frame - 1, slot]):
                velocity[frame, slot] = (doa[frame, slot] - doa[frame - 1, slot]) / frame_step
                mask[frame, slot] = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('velocity_label', data=velocity, dtype=np.float32)
        hf.create_dataset('velocity_mask', data=mask, dtype=np.float32)
        hf.attrs['definition'] = '(p_t - p_t_minus_1) / 0.1s'
        hf.attrs['identity'] = '(event_class, original_track_number)'
    return velocity[mask > 0.5], int(mask.sum()), max_alignment_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--meta-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frame-step', type=float, default=0.1)
    args = parser.parse_args()

    all_velocity = []
    total_valid = 0
    worst_alignment = 0.0
    csv_paths = sorted(path for path in args.csv_dir.glob('*.csv')
                       if not path.name.startswith('.'))
    if not csv_paths:
        raise RuntimeError(f'No CSV files found in {args.csv_dir}')
    for idx, csv_path in enumerate(csv_paths, start=1):
        meta_path = args.meta_dir / f'{csv_path.stem}.h5'
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        values, valid, alignment = build_one(
            csv_path, meta_path, args.output_dir / f'{csv_path.stem}.h5', args.frame_step)
        all_velocity.append(values)
        total_valid += valid
        worst_alignment = max(worst_alignment, alignment)
        if idx % 100 == 0 or idx == len(csv_paths):
            print(f'processed={idx}/{len(csv_paths)} valid={total_valid}', flush=True)

    values = np.concatenate(all_velocity, axis=0)
    norms = np.linalg.norm(values, axis=-1)
    print(f'files={len(csv_paths)}')
    print(f'valid_vectors={total_valid}')
    print(f'worst_alignment_error={worst_alignment:.9g}')
    print('speed_norm_percentiles=' + ','.join(
        f'p{p}={np.percentile(norms, p):.6f}' for p in (0, 25, 50, 75, 95, 99, 100)))
    print(f'all_finite={bool(np.isfinite(values).all())}')


if __name__ == '__main__':
    main()
