"""Named EINV2 recipes, separate from host paths and evaluation. One seed per run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def build_command(recipe, paths, seed, output, workers=4):
    # Loss weights are resolved by the versioned audited runtime, not duplicated here.
    command = [sys.executable, str(ROOT/'scripts/train/einv2_audited.py'),
               '--variant', recipe['variant'], '--seed', str(seed),
               '--project-root', paths['project_root'], '--scalar', paths['scalar'],
               '--hdf5-dir', paths['hdf5_dir'], '--output-root', str(output),
               '--workers', str(workers), '--run-id', f"{recipe['name']}_seed{seed}_v030"]
    if 'lambda_jepa' in recipe:
        command += ['--lambda-jepa', str(recipe['lambda_jepa'])]
    if 'velocity_min_norm' in recipe:
        command += ['--velocity-min-norm', str(recipe['velocity_min_norm'])]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    paths = json.loads(args.paths.read_text(encoding='utf-8'))
    command = build_command(recipe, paths, args.seed, args.output_root, args.workers)
    print(json.dumps({'recipe': recipe, 'command': command}, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        for key in ('project_root', 'scalar', 'hdf5_dir'):
            if not Path(paths[key]).exists():
                raise FileNotFoundError(key + ': ' + paths[key])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
