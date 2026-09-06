"""Shared training and evaluation implementation for the audited causal matrix."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

from runtime import (ROOT, VERSION, WEIGHTS, AuditedEINV2, Frontend, configuration,
                     loss_values, make_teacher, seed_all, sha256, update_teacher)
import numpy as np
import torch
from aligned_metrics import AlignedMetrics, UPSTREAM, load_csv
from utils.config import get_dataset, get_generator


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def source_manifest():
    paths = [*sorted((ROOT / 'models/einv2/audited').glob('*.py')),
             *sorted((ROOT / 'models/einv2/variants/C3/seld').rglob('*.py')),
             *sorted((ROOT / 'evaluation').rglob('*.py')),
             ROOT / 'scripts/train/einv2_audited.py', ROOT / 'scripts/eval/einv2_audited.py',
             ROOT / 'scripts/data/build_einv2_train_scalar.py', ROOT / 'configs/einv2/C3.yaml',
             ROOT / 'scripts/train/einv2_weightprobe_queue.py', ROOT / 'docs/RB05_WEIGHTPROBE_PLAN.md']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def data_args(seed, workers):
    return SimpleNamespace(seed=seed, num_workers=workers, read_into_mem=False, cuda=torch.cuda.is_available(), no_cuda=False)


def targets(batch, device):
    return {key: batch[value].to(device, non_blocking=True) for key, value in {
        'sed': 'sed_label', 'doa': 'doa_label', 'velocity': 'velocity_label',
        'velocity_mask': 'velocity_mask', 'jepa_valid_mask': 'jepa_valid_mask'}.items()}


@torch.no_grad()
def predict(model, frontend, generator, device):
    model.eval()
    frontend.eval()
    chunks = {}
    for batch in generator:
        pred = model(frontend(batch['waveform'].to(device, non_blocking=True)))
        sed, doa = pred['sed'].sigmoid().cpu().numpy(), pred['doa'].cpu().numpy()
        if not np.isfinite(sed).all() or not np.isfinite(doa).all():
            raise FloatingPointError('Non-finite inference output')
        for index, (filename, segment) in enumerate(zip(batch['filename'], batch['n_segment'])):
            segments = chunks.setdefault(filename, {})
            if segment in segments:
                raise ValueError('Duplicate segment: ' + filename)
            segments[segment] = (sed[index], doa[index])
    output = {}
    for filename, segments in sorted(chunks.items()):
        assert set(segments) == set(range(15)), (filename, sorted(segments))
        sed = np.concatenate([segments[s][0] for s in range(15)])
        doa = np.concatenate([segments[s][1] for s in range(15)])
        assert sed.shape == (600, 2, 14) and doa.shape == (600, 2, 3)
        classes, active = sed.argmax(-1), sed.max(-1) > .5
        az = np.rint(np.rad2deg(np.arctan2(doa[..., 1], doa[..., 0]))).astype(int)
        el = np.rint(np.rad2deg(np.arctan2(doa[..., 2], np.sqrt(doa[..., 0] ** 2 + doa[..., 1] ** 2)))).astype(int)
        labels = {}
        for frame, track in np.argwhere(active):
            labels.setdefault(int(frame), []).append([int(classes[frame, track]), int(track), int(az[frame, track]), int(el[frame, track])])
        output[filename + '.csv'] = labels
    return output


def score(predictions, cfg, split):
    reference = Path(cfg['dataset_dir']) / ('metadata_dev' if split == 'validation' else 'metadata_eval')
    names = {p.name for p in reference.glob('fold1*.csv' if split == 'validation' else '*.csv') if not p.name.startswith('.')}
    if set(predictions) != names:
        raise ValueError('Prediction/reference file coverage mismatch')
    metric = AlignedMetrics(classes=14, threshold=20, frames=600, frames_per_second=10)
    for filename in sorted(names):
        metric.update(predictions[filename], load_csv(reference / filename, 'polar5'))
    return {'files': len(names), 'scores': metric.scores(), 'upstream': UPSTREAM}


def save_checkpoint(path, model, teacher, frontend, optimizer, scheduler, cfg, metrics, epoch, best):
    temp = path.with_suffix('.pth.tmp')
    torch.save(dict(version=VERSION, epoch=epoch, model=model.state_dict(),
                    teacher=teacher.state_dict() if teacher is not None else None,
                    frontend=frontend.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), config=cfg, metrics=metrics, best=best,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    np_rng=np.random.get_state(), random_rng=random.getstate(),
                    resume_supported=False), temp)
    temp.replace(path)


def train_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scalar', type=Path, required=True)
    parser.add_argument('--variant', choices=WEIGHTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke-batches', type=int, default=0)
    parser.add_argument('--hdf5-dir', type=Path)
    parser.add_argument('--lambda-jepa', type=float)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Full training requires an explicitly allocated CUDA device')
    device = torch.device('cuda:0')
    cfg = configuration(args.project_root.resolve(), args.scalar.resolve(), args.variant, args.seed,
                        hdf5_dir=args.hdf5_dir, lambda_jepa=args.lambda_jepa)
    run_id = ('smoke_' if args.smoke_batches else '') + '{}_seed{}_framecausal_v020'.format(args.variant, args.seed)
    if args.run_id:
        if not all(c.isalnum() or c in '_-.' for c in args.run_id) or args.run_id in ('.', '..'):
            raise ValueError('run-id must be a single safe directory name')
        run_id = ('smoke_' if args.smoke_batches else '') + args.run_id
    elif args.lambda_jepa is not None:
        raise ValueError('An explicit run-id is required for an override')
    run = args.output_root.resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    cfg['workspace_dir'] = str(run)
    cfg['training'].update(train_id=run_id, remark=run_id)
    if args.smoke_batches:
        cfg['training']['max_epoch'] = 1
    cfg['audit']['smoke_batches'] = args.smoke_batches
    cfg['audit']['source_hashes'] = source_manifest()
    write_json(run / 'config.json', cfg)
    write_json(run / 'environment.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), command=sys.argv,
               deterministic_algorithms=True, cudnn_benchmark=False))
    state = dict(status='initializing', run_id=run_id, pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
    write_json(run / 'status.json', state)
    try:
        dataset = get_dataset(cfg['dataset'], cfg['dataset_dir'])
        ns = data_args(args.seed, args.workers)
        train_set, train_generator, sampler = get_generator(ns, cfg, dataset, 'train')
        valid_set, valid_generator, _ = get_generator(ns, cfg, dataset, 'valid')
        assert len(train_set) == 7500 and len(valid_set) == 1500, (len(train_set), len(valid_set))
        train_files = sorted({str(p).split('%')[0] for p in train_set.paths_list})
        valid_files = sorted({str(p).split('%')[0] for p in valid_set.paths_list})
        assert set(train_files).isdisjoint(valid_files)
        write_json(run / 'split_manifest.json', dict(train=train_files, validation=valid_files, test_used_in_training=False))
        frontend = Frontend(cfg).to(device).eval()
        model = AuditedEINV2(cfg, dataset).to(device)
        # Auxiliary modules have identical initialization in every ablation.
        teacher = make_teacher(model) if cfg['training']['lambda_jepa'] else None
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'], amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['lr_step_size'], gamma=cfg['training']['lr_gamma'])
        # Keep training/dropout RNG independent of constructor and loader setup.
        seed_all(args.seed + 100000)
        iterator = iter(train_generator)
        best = {'SELD_LR': float('inf'), 'epoch': None}
        steps = args.smoke_batches or len(sampler)
        state.update(status='training', batches_per_epoch=steps, total_epochs=cfg['training']['max_epoch'])
        for epoch in range(1, cfg['training']['max_epoch'] + 1):
            start = time.monotonic()
            model.train()
            sums = {key: 0. for key in ('all', 'sed', 'doa', 'velocity', 'jepa')}
            lr = optimizer.param_groups[0]['lr']
            for step in range(steps):
                batch = next(iterator)
                target = targets(batch, device)
                with torch.no_grad():
                    features = frontend(batch['waveform'].to(device, non_blocking=True))
                    teacher_pred = teacher(features) if teacher is not None else None
                pred = model(features)
                losses = loss_values(pred, target, teacher_pred, cfg)
                if not all(torch.isfinite(v) for v in losses.values()):
                    raise FloatingPointError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True)
                losses['all'].backward()
                optimizer.step()
                if teacher is not None:
                    update_teacher(teacher, model, cfg['training']['jepa_ema_momentum'])
                for key, value in losses.items():
                    sums[key] += value.detach().item()
                if step % 25 == 0 or step == steps - 1:
                    state.update(epoch=epoch, batch=step + 1, losses={key: value / (step + 1) for key, value in sums.items()},
                                 updated_at=datetime.now(timezone.utc).isoformat())
                    write_json(run / 'status.json', state)
                    print(json.dumps(state), flush=True)
            scheduler.step()  # decay after 80 complete epochs, before epoch 81
            state.update(status='validating')
            write_json(run / 'status.json', state)
            prediction = predict(model, frontend, valid_generator, device)
            result = score(prediction, cfg, 'validation')
            primary = result['scores']['dcase2023_micro']
            improved = primary['SELD_LR'] < best['SELD_LR']
            if improved:
                best = dict(SELD_LR=primary['SELD_LR'], epoch=epoch)
            metrics = dict(epoch=epoch, lr=lr, train={key: val / steps for key, val in sums.items()},
                           validation=result, elapsed_seconds=time.monotonic() - start, best=best)
            with (run / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            save_checkpoint(run / 'latest.pth', model, teacher, frontend, optimizer, scheduler, cfg, metrics, epoch, best)
            if improved:
                save_checkpoint(run / 'best.pth', model, teacher, frontend, optimizer, scheduler, cfg, metrics, epoch, best)
            state.update(status='training', best=best, validation=primary, epoch_seconds=metrics['elapsed_seconds'])
            write_json(run / 'status.json', state)
            print(json.dumps(metrics), flush=True)
        state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(run / 'best.pth'))
        write_json(run / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(run / 'status.json', state)
        raise


def eval_main():
    parser = argparse.ArgumentParser(description='Independent checkpoint inference; no optimizer or teacher.')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'evaluation'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.run / 'best.pth', map_location='cpu')
    cfg = checkpoint['config']
    if cfg['audit']['source_hashes'] != source_manifest():
        raise RuntimeError('Runtime source differs from checkpoint provenance')
    seed_all(cfg['audit']['seed'])
    if sha256(cfg['causal_scalar_path']) != cfg['audit']['scalar_sha256']:
        raise RuntimeError('Scaler changed since training')
    output = args.run / args.split
    output.mkdir(exist_ok=False)
    cfg['inference'].update(testset_type='dev' if args.split == 'validation' else 'eval', test_fold='1' if args.split == 'validation' else 'None')
    dataset = get_dataset(cfg['dataset'], cfg['dataset_dir'])
    _, generator, _ = get_generator(data_args(cfg['audit']['seed'], args.workers), cfg, dataset, 'test')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, frontend = AuditedEINV2(cfg, dataset).to(device), Frontend(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    frontend.load_state_dict(checkpoint['frontend'])
    del checkpoint
    predictions = predict(model, frontend, generator, device)
    for filename, labels in predictions.items():
        with (output / filename).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            for frame, values in sorted(labels.items()):
                for cls, track, az, el in values:
                    writer.writerow([frame, cls, az, el])
    result = score(predictions, cfg, args.split)
    # Re-read written CSV files through the standalone scorer path.
    reread = score({name: load_csv(output / name, 'polar4') for name in predictions}, cfg, args.split)
    assert result['scores'] == reread['scores'], 'In-memory/CSV scorer mismatch'
    if args.split == 'validation':
        best = torch.load(args.run / 'best.pth', map_location='cpu')['metrics']['validation']['scores']['dcase2023_micro']
        delta = max(abs(best[k] - result['scores']['dcase2023_micro'][k]) for k in best)
        result['training_validation_max_abs_diff'] = delta
        if delta > 1e-6:
            raise AssertionError('Training/independent validation mismatch: ' + str(delta))
    result.update(checkpoint=str(args.run / 'best.pth'), split=args.split, version=VERSION, status='completed')
    write_json(output / 'metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)
