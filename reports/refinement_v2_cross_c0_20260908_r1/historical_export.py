"""Read-only fixed checkpoint inference using its exact frozen runtime."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'runtime', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', choices=['validation','evaluation'], required=True)
    parser.add_argument('--latent', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    sys.path.insert(0, str(args.runtime/'models/einv2/audited'))
    import torch
    import numpy as np
    import runner
    from decoder import labels
    torch.set_num_threads(4)
    cfg_disk = json.loads((args.run/'config.json').read_text())
    assert cfg_disk['audit']['source_hashes'] == runner.source_manifest(), 'Runtime source mismatch'
    assert digest(cfg_disk['causal_scalar_path']) == cfg_disk['audit']['scalar_sha256'], 'Scaler mismatch'
    # Container has historical optimizer tensors, which are discarded immediately.
    # No optimizer object, load_state_dict, gradients, backward, or update exists here.
    ck = torch.load(args.run/'best.pth', map_location='cpu', weights_only=False)
    for key in ('optimizer','scheduler','rng','cuda_rng','np_rng','random_rng'): ck.pop(key, None)
    assert ck['config'] == cfg_disk, 'Checkpoint/disk config mismatch'
    cfg = copy.deepcopy(ck['config'])
    assert cfg['audit']['version'] == runner.VERSION
    assert cfg['inference']['threshold_sed'] == cfg['training']['threshold_sed'] == .5
    runner.seed_all(cfg['audit']['seed'])
    cfg['inference'].update(testset_type='dev' if args.split=='validation' else 'eval', test_fold='1' if args.split=='validation' else 'None')
    dataset = runner.get_dataset(cfg['dataset'], cfg['dataset_dir'])
    ds, generator, _ = runner.get_generator(runner.data_args(cfg['audit']['seed'], 4), cfg, dataset, 'test')
    assert len(ds) == (1500 if args.split=='validation' else 3000)
    device = torch.device('cuda:0')
    model = runner.AuditedEINV2(cfg,dataset).to(device).eval().requires_grad_(False)
    frontend = runner.Frontend(cfg).to(device).eval()
    model.load_state_dict(ck['model']); frontend.load_state_dict(ck['frontend'])
    teacher = None
    if args.latent and ck.get('teacher') is not None:
        teacher = runner.AuditedEINV2(cfg,dataset).to(device).eval().requires_grad_(False)
        teacher.load_state_dict(ck['teacher'])
    checkpoint_validation = ck['metrics']['validation']['scores']
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        run=str(args.run), runtime=str(args.runtime), config=cfg_disk,
        checkpoint_sha256=digest(args.run/'best.pth'), config_sha256=digest(args.run/'config.json'),
        scalar_sha256=digest(cfg['causal_scalar_path']), source_hashes=runner.source_manifest(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), split=args.split,
        checkpoint_epoch=ck['epoch'], optimizer_state_restored=False,
        checkpoint_container_deserialized=True, teacher_exported=teacher is not None,
        raw_xyz_normalized=False, frame_count=600, chunks=15)
    del ck
    reference = Path(cfg['dataset_dir'])/('metadata_dev' if args.split=='validation' else 'metadata_eval')
    gt_paths=sorted(p for p in reference.glob('fold1*.csv' if args.split=='validation' else '*.csv') if not p.name.startswith('.'))
    manifest['gt_files']={p.name:digest(p) for p in gt_paths}
    save(args.output/'input_manifest.json',manifest)
    (args.output/'float').mkdir(); (args.output/'raw').mkdir()
    chunks={}; predictions={}; float_hashes={}; historical_equal=True
    with torch.inference_mode():
        for batch_i,batch in enumerate(generator):
            features=frontend(batch['waveform'].to(device,non_blocking=True))
            pred=model(features)
            values={k:pred[k].cpu().numpy() for k in ('sed','doa','velocity')}
            values['probability']=pred['sed'].sigmoid().cpu().numpy()
            if args.latent:
                for k in ('latent','future_latent_pred'): values[k]=pred[k].cpu().numpy()
                if teacher is not None:
                    tp=teacher(features)
                    for k in ('sed','doa','latent'): values['teacher_'+k]=tp[k].cpu().numpy()
            assert all(np.isfinite(x).all() for x in values.values())
            for i,(filename,segment) in enumerate(zip(batch['filename'],batch['n_segment'])):
                segment=int(segment)
                record=chunks.setdefault(filename,{})
                assert segment not in record
                record[segment]={k:v[i] for k,v in values.items()}
                if len(record)!=15: continue
                assert set(record)==set(range(15))
                out={k:np.concatenate([record[s][k] for s in range(15)]) for k in values}
                assert out['probability'].shape==(600,2,14) and out['doa'].shape==(600,2,3)
                out.update(frame_index=np.arange(600),chunk_index=np.repeat(np.arange(15),40),slot=np.tile(np.arange(2),(600,1)),filename=filename)
                path=args.output/'float'/(filename+'.npz')
                np.savez_compressed(path,**out); float_hashes[path.name]=digest(path)
                lab=labels(out['probability'],out['doa']); predictions[filename+'.csv']=lab
                with (args.output/'raw'/(filename+'.csv')).open('x',newline='') as f:
                    w=csv.writer(f)
                    for frame, vals in sorted(lab.items()):
                        for cls,slot,az,el in vals: w.writerow([frame,cls,az,el])
                old=args.run/args.split/(filename+'.csv')
                if old.exists():
                    # Source slot is absent in polar4; preserve row multiplicity/order.
                    historical_equal &= old.read_bytes()==(args.output/'raw'/old.name).read_bytes()
                del chunks[filename]
            print(json.dumps(dict(batch=batch_i+1,files=len(predictions),elapsed=time.time()-started)),flush=True)
    assert not chunks
    result=runner.score(predictions,cfg,args.split)
    reread=runner.score({n:runner.load_csv(args.output/'raw'/n,'polar4') for n in predictions},cfg,args.split)
    assert result['scores']==reread['scores']
    old_metric=args.run/args.split/'metrics.json'
    expected=json.loads(old_metric.read_text())['scores'] if old_metric.exists() else checkpoint_validation if args.split=='validation' else None
    delta=max(abs(result['scores'][p][k]-expected[p][k]) for p in expected for k in expected[p]) if expected else None
    save(args.output/'raw_regression.json',dict(expected_path=str(old_metric),max_abs_delta=delta,
        historical_csv_bytes_equal=historical_equal,validation_checkpoint_scores=checkpoint_validation if args.split=='validation' else None,
        csv_reread_equal=True,files=len(predictions),scores=result['scores']))
    if expected is not None: assert delta<=1e-6 and historical_equal, 'Raw regression failed; stop decoder comparison'
    save(args.output/'float_manifest.json',float_hashes)
    save(args.output/'COMPLETED.json',dict(elapsed_seconds=time.time()-started,files=len(predictions),scores=result['scores']))


if __name__=='__main__':
    main()
