#!/usr/bin/env python3
"""Build same-object continuity masks for 100/300/500 ms JEPA targets."""

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


def build_one(csv_path, meta_path, output_path, horizons):
    rows = np.loadtxt(csv_path, delimiter=',', dtype=np.float64, ndmin=2)
    with h5py.File(meta_path, 'r') as hf:
        sed = hf['sed_label'][:]
        doa = hf['doa_label'][:]
    frames, tracks = sed.shape[:2]
    identities = np.full((frames, tracks, 2), -1, dtype=np.int32)
    assigned = np.zeros((frames, tracks), dtype=bool)
    max_alignment_error = 0.0
    for row in rows:
        frame, event, track_number = int(row[0]), int(row[1]), int(row[2])
        position = spherical_to_cartesian(row[3], row[4])
        candidates = [idx for idx in range(tracks)
                      if sed[frame, idx, event] > 0.5 and not assigned[frame, idx]]
        if not candidates:
            raise RuntimeError(f'No slot match: {csv_path.name}, frame={frame}, event={event}')
        errors = [float(np.linalg.norm(doa[frame, idx] - position)) for idx in candidates]
        slot = candidates[int(np.argmin(errors))]
        max_alignment_error = max(max_alignment_error, min(errors))
        identities[frame, slot] = (event, track_number)
        assigned[frame, slot] = True
    if max_alignment_error > 1e-4:
        raise RuntimeError(f'DOA/meta mismatch in {csv_path.name}: {max_alignment_error}')

    valid_mask = np.zeros((frames, tracks, len(horizons)), dtype=np.float32)
    for horizon_idx, horizon in enumerate(horizons):
        for frame in range(frames - horizon):
            for slot in range(tracks):
                identity = identities[frame, slot]
                if identity[0] < 0:
                    continue
                if np.all(identities[frame:frame + horizon + 1, slot] == identity):
                    valid_mask[frame, slot, horizon_idx] = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('identity_label', data=identities, dtype=np.int32)
        hf.create_dataset('jepa_valid_mask', data=valid_mask, dtype=np.float32)
        hf.attrs['horizons_frames'] = np.asarray(horizons, dtype=np.int32)
        hf.attrs['identity'] = '(event_class, original_track_number), continuous through target frame'
    return valid_mask.sum(axis=(0, 1)).astype(np.int64), max_alignment_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--meta-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5])
    args = parser.parse_args()
    paths = sorted(path for path in args.csv_dir.glob('*.csv')
                   if not path.name.startswith('.'))
    totals = np.zeros(len(args.horizons), dtype=np.int64)
    worst_alignment = 0.0
    for index, csv_path in enumerate(paths, start=1):
        counts, alignment = build_one(
            csv_path, args.meta_dir / f'{csv_path.stem}.h5',
            args.output_dir / f'{csv_path.stem}.h5', args.horizons)
        totals += counts
        worst_alignment = max(worst_alignment, alignment)
        if index % 100 == 0 or index == len(paths):
            print(f'processed={index}/{len(paths)} counts={totals.tolist()}', flush=True)
    print(f'files={len(paths)}')
    print(f'horizons_frames={args.horizons}')
    print(f'valid_counts={totals.tolist()}')
    print(f'worst_alignment_error={worst_alignment:.9g}')
    print('all_finite=True')


if __name__ == '__main__':
    main()
