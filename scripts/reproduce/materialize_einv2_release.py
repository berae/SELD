"""Restore hash-exact runtime files for an existing audited EINV2 checkpoint.

No datasets/checkpoints are copied. Existing destinations are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = ROOT / 'provenance/runtime_snapshots'


def resolve_files(version):
    snapshot = SNAPSHOTS / version
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    files = {}
    for relative, expected in manifest['files'].items():
        candidate = snapshot / relative
        if not candidate.is_file():
            candidate = ROOT / relative
        content = candidate.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Source hash mismatch: ' + relative)
        files[relative] = content
    return files


def materialize(version, output):
    files = resolve_files(version)
    output.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, choices=sorted(p.name for p in SNAPSHOTS.iterdir() if p.is_dir()))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    files = resolve_files(args.version)
    if args.check_only:
        print(json.dumps({'version': args.version, 'verified_files': len(files), 'status': 'PASS'}))
    elif args.output:
        count = materialize(args.version, args.output)
        print(json.dumps({'version': args.version, 'files': count, 'output': str(args.output.resolve())}))
    else:
        parser.error('--output or --check-only is required')


if __name__ == '__main__':
    main()
