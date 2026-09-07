"""Export immutable original-forward C0 features plus separate GT supervision."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from candidate import prediction_history, fixed_records
from experiment_core import digest, save_json, csv_bytes, read_gt, fixed_targets
from probe_frozen_c0 import state_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation'], required=True)
    p.add_argument('--relocation', type=Path, help='Only storage-path relocation; original config and hashes remain authoritative')
    args = p.parse_args()
    spec = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ('float', 'targets', 'raw', 'gt'):
        (args.output/sub).mkdir()
    started = time.time(); torch.set_num_threads(4)
    relocation = json.loads(args.relocation.read_text()) if args.relocation else {}
    assert set(relocation) <= {'run','runtime','causal_scalar_path','dataset_dir','hdf5_dir','original_validation'}
    run = Path(relocation.get('run', spec['baseline_run']))
    runtime = Path(relocation.get('runtime', spec['runtime']))
    sys.path.insert(0, str(runtime/'models/einv2/audited'))
    import runner
    cfg_disk = json.loads((run/'config.json').read_text())
    assert runner.source_manifest() == cfg_disk['audit']['source_hashes']
    assert digest(run/'best.pth') == spec['baseline_checkpoint_sha256']
    scalar = relocation.get('causal_scalar_path', cfg_disk['causal_scalar_path'])
    assert digest(scalar) == cfg_disk['audit']['scalar_sha256']
    ck = torch.load(run/'best.pth', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg_disk
    for key in ('optimizer', 'scheduler', 'rng', 'cuda_rng', 'np_rng', 'random_rng'):
        ck.pop(key, None)
    runner.seed_all(2026)
    cfg = copy.deepcopy(cfg_disk)
    for key in ('causal_scalar_path','dataset_dir','hdf5_dir'):
        if key in relocation:
            cfg[key] = relocation[key]
    assert cfg['inference']['batch_size'] == 32
    cfg['inference'].update(testset_type='dev', test_fold='1' if args.split == 'validation' else '2,3,4,5,6')
    dataset = runner.get_dataset(cfg['dataset'], cfg['dataset_dir'])
    ds, generator, _ = runner.get_generator(runner.data_args(2026, 4), cfg, dataset, 'test')
    expected = 100 if args.split == 'validation' else 500
    assert len(ds) == expected*15
    model = runner.AuditedEINV2(cfg, dataset).cuda().eval().requires_grad_(False)
    frontend = runner.Frontend(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck['model']); frontend.load_state_dict(ck['frontend']); del ck
    before = state_digest(model), state_digest(frontend)
    saved = {}
    def hook(i):
        def capture(module, inputs):
            saved[i] = inputs[0].detach().clone()
        return capture
    handles = [head.register_forward_pre_hook(hook(i)) for i, head in enumerate(model.doa_heads)]
    references = Path(cfg['dataset_dir'])/'metadata_dev'
    gt_paths = sorted(p for p in references.glob('*.csv') if p.name.startswith(tuple('fold'+str(i) for i in ([1] if args.split == 'validation' else [2,3,4,5,6]))))
    assert len(gt_paths) == expected
    gt_hashes = {p.name: digest(p) for p in gt_paths}
    manifest = dict(split=args.split, run=str(run), runtime=str(runtime), config=cfg_disk,
                    checkpoint_sha256=spec['baseline_checkpoint_sha256'], source_hashes=runner.source_manifest(),
                    scalar_sha256=digest(scalar), gt_files=gt_hashes, storage_relocation=relocation,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, command=sys.argv,
                    original_forward_predictions=True, optimizer_updates=0,
                    feature_shape=[600,2,512], script_sha256=digest(__file__))
    save_json(args.output/'input_manifest.json', manifest)
    chunks = {}; hashes = {}; counts = {}; probes = []; predictions = {}
    historical = Path(relocation.get('original_validation', str(Path(spec['existing_export'])/'validation')))
    if args.split == 'validation':
        assert gt_hashes == json.loads((historical/'input_manifest.json').read_text())['gt_files']
    with torch.no_grad():
        for bi, batch in enumerate(generator):
            wave = batch['waveform'].cuda(non_blocking=True)
            pred = model(frontend(wave))
            hidden = torch.stack([saved[i] for i in range(2)], dim=2)
            values = {k: pred[k].cpu().numpy() for k in ('sed', 'doa')}
            values['probability'] = pred['sed'].sigmoid().cpu().numpy()
            values['doa_features'] = hidden.cpu().numpy()
            assert all(np.isfinite(v).all() for v in values.values())
            if bi in (0, len(generator)-1):
                # Real-waveform prefix test through the actual export path.
                changed = wave.clone(); changed[..., 20*2400:] = 0
                perturbed = model(frontend(changed))
                h2 = torch.stack([saved[i] for i in range(2)], dim=2)
                delta = max((pred[k][:,:20]-perturbed[k][:,:20]).abs().max().item() for k in ('sed', 'doa'))
                hd = (hidden[:,:20]-h2[:,:20]).abs().max().item()
                assert delta <= 1e-6 and hd <= 1e-6
                probes.append(dict(batch=bi, raw_prefix_max_delta=delta, feature_prefix_max_delta=hd))
            for i, (filename, segment) in enumerate(zip(batch['filename'], batch['n_segment'])):
                filename = str(filename); segment = int(segment)
                record = chunks.setdefault(filename, {})
                assert segment not in record
                record[segment] = {k:v[i] for k,v in values.items()}
                if len(record) != 15:
                    continue
                assert set(record) == set(range(15))
                out = {k:np.concatenate([record[s][k] for s in range(15)]) for k in values}
                assert out['doa_features'].shape == (600,2,512)
                out.update(frame_index=np.arange(600), chunk_index=np.repeat(np.arange(15),40),
                           raw_slot=np.tile(np.arange(2),(600,1)), filename=filename)
                records = fixed_records(out['probability'], out['doa'])
                # occurrence index is explicit even though this decoder has one occurrence/slot.
                out['records'] = np.array([(*r[:3],0,*r[3:]) for r in records], dtype=np.int64).reshape(-1,6)
                out['mapping'] = prediction_history(out['probability'], out['doa'], out['frame_index'], out['chunk_index'])
                raw_bytes = csv_bytes(records)
                if args.split == 'validation':
                    with np.load(historical/'float'/(filename+'.npz')) as old:
                        for field in ('sed','probability','doa'):
                            assert np.array_equal(out[field], old[field]), (filename,field,'original raw mismatch')
                    assert raw_bytes == (historical/'raw'/(filename+'.csv')).read_bytes()
                gt_path = references/(filename+'.csv')
                assert digest(gt_path) == gt_hashes[gt_path.name]
                shutil.copyfile(gt_path, args.output/'gt'/gt_path.name)
                targets, stats = fixed_targets(out['probability'], out['doa'], out['mapping'], out['chunk_index'], read_gt(gt_path))
                for key, value in stats.items():
                    counts[key] = counts.get(key,0)+value
                np.savez_compressed(args.output/'float'/(filename+'.npz'), **out)
                np.savez_compressed(args.output/'targets'/(filename+'.npz'), **targets)
                with (args.output/'raw'/(filename+'.csv')).open('xb') as f:
                    f.write(raw_bytes)
                for sub, ext in [('float','.npz'),('targets','.npz'),('raw','.csv'),('gt','.csv')]:
                    rel = sub+'/'+filename+ext
                    hashes[rel] = digest(args.output/rel)
                predictions[filename+'.csv'] = runner.load_csv(args.output/'raw'/(filename+'.csv'), 'polar4')
                del chunks[filename]
            print(json.dumps(dict(batch=bi+1,batches=len(generator),recordings=len(predictions),elapsed_seconds=time.time()-started)), flush=True)
    for handle in handles:
        handle.remove()
    assert not chunks and set(predictions) == set(gt_hashes)
    assert before == (state_digest(model),state_digest(frontend))
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    scores = None
    if args.split == 'validation':
        scores = runner.score(predictions, cfg, 'validation')['scores']
        assert scores == json.loads((historical/'raw_regression.json').read_text())['scores']
    save_json(args.output/'file_manifest.json',hashes)
    save_json(args.output/'target_audit.json',counts)
    save_json(args.output/'COMPLETED.json',dict(status='PASS',files=expected,elapsed_seconds=time.time()-started,
              exact_validation_float_csv_score=args.split=='validation',scores=scores,
              parameters_buffers_unchanged=True,prefix_probes=probes,
              causality_scope='first and last export batches, prefix20; not universal proof',
              in_sample_training_cache=args.split=='train',training_updates=0))


if __name__ == '__main__':
    main()
