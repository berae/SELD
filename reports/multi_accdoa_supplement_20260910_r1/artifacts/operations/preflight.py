"""Exactly one bounded, two-recording preflight. No optimizer is constructed."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R/'source/runtime'))
sys.path.insert(0, str(R/'operations'))
import numpy as np
import torch
import cls_data_generator
import cls_feature_class
import seldnet_model
import train_seldnet
from cls_compute_seld_results import reshape_3Dto2D
import event_adapter as adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)


def check(name, condition, detail=None):
    row = dict(check=name, passed=bool(condition), detail=detail)
    with (R/'checks.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)
    assert condition, name


def native_decoder():
    tree = ast.parse((R/'source/historical_threshold_decoder.py').read_text())
    names = ('get_tracks', 'append_track', 'decode')
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = dict(np=np, reshape_3Dto2D=reshape_3Dto2D,
                 determine_similar_location=train_seldnet.determine_similar_location)
    exec(compile(selected, 'sealed_historical_decoder', 'exec'), scope)
    return scope


def write_csv(path, records, feature_class, xyz=None):
    assert not path.exists()
    feature_class.write_output_format_file(str(path), adapter.as_native(records, xyz))


def synthetic(native, params):
    # All native merge branches, a merged norm below threshold, and empty frame.
    vectors = np.zeros((7, 3, 3, 14), np.float32)
    def direction(a, norm=.9):
        return np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), 0])*norm
    cases = [(0, 90, 180), (0, 5, 100), (100, 0, 5), (5, 100, 0), (0, 5, 10)]
    for t, values in enumerate(cases):
        for track, angle in enumerate(values):
            vectors[t, track, :, 0] = direction(angle)
    vectors[5, 0, :, 0] = direction(-7, .5001)
    vectors[5, 1, :, 0] = direction(7, .5001)
    out = vectors.reshape(1, 7, 126)
    records = adapter.decoded_events(out, params, native)
    check('synthetic_native_merge_identity', adapter.as_native(records) == native['decode'](out, params, .5))
    q = [r for r in records if r['frame'] == 5]
    check('merged_below_threshold_retained', len(q) == 1 and np.linalg.norm(q[0]['xyz']) < .5)
    check('empty_frame_no_event', not any(r['frame'] == 6 for r in records))
    def row(t, x, ordinal=0, chunk=0):
        return dict(frame=t, cls=0, ordinal=ordinal, chunk=chunk, members=[ordinal], xyz=x)
    seq = [row(0, [1,0,0]), row(0, [0,1,0], 1),
           row(1, [0,1,0]), row(1, [1,0,0], 1), row(2, [1,0,0], chunk=1),
           row(3, [0,0,0], chunk=1)]
    mapping = adapter.history(seq)
    check('track_switch_chunk_reset_zero_norm', mapping.tolist() == [-1,-1,1,0,-1,-1], mapping.tolist())
    tie = [row(0,[1,0,0]), row(0,[1,0,0],1), row(1,[1,0,0])]
    check('near_tie_rejected', adapter.history(tie).tolist() == [-1,-1,-1])
    raw = torch.tensor([[.4,.3,.1], [0.,0.,0.]], device='cuda')
    delta = torch.tensor([[0.,.2,0.], [.2,.2,.2]], device='cuda', requires_grad=True)
    corrected = adapter.correct(raw, delta)
    check('zero_norm_fallback', torch.equal(corrected[1], raw[1]))
    check('norm_preservation', torch.allclose(corrected.norm(dim=-1), raw.norm(dim=-1), atol=1e-7, rtol=1e-6))
    empty = torch.zeros(2, dtype=torch.bool, device='cuda')
    term = adapter.masked_mean(delta.square().mean(-1), empty)
    term.backward()
    check('no_motion_pair_zero_loss_gradient', term.item() == 0 and torch.count_nonzero(delta.grad).item() == 0)


def main():
    start = time.monotonic()
    seal = json.loads((R/'SOURCE.json').read_text())
    for rel, expected in seal['files'].items():
        check('source:'+rel, sha(R/rel) == expected)
    params = json.loads((R/'source/manifest.json').read_text())['params']
    check('baseline_scope', params['causal'] and params['causal_frontend'] and not params['use_velocity'] and not params['use_jepa'])
    check('native_runtime', torch.__version__ == '1.10.0+cu111', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda))
    np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    native = native_decoder()
    synthetic(native, params)
    outputs = R/'preflight'
    outputs.mkdir(exist_ok=False)
    timings = {}
    model = None
    original_state = None
    real_listdir = os.listdir
    for split, paths in seal['records'].items():
        t0 = time.monotonic()
        fname = Path(paths['features']).name
        feature_dir = Path(paths['features']).parent
        def list_one(path):
            return [fname] if Path(path).resolve() == feature_dir.resolve() else real_listdir(path)
        with patch('cls_data_generator.os.listdir', side_effect=list_one):
            generator = cls_data_generator.DataGenerator(params, split=[2 if split=='train' else 1], shuffle=False, per_file=True)
        batch, labels = next(generator.generate())
        x = torch.as_tensor(batch, dtype=torch.float32, device='cuda')
        if model is None:
            model = seldnet_model.SeldModel(*generator.get_data_sizes(), params).cuda()
            model.load_state_dict(torch.load(str(R/'source/C0.h5'), map_location='cuda'), strict=True)
            model.eval()
            original_state = state_hash(model)
        # Preserve native requires_grad flags for no-hook baseline, no_grad only.
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        hooked = []
        handle = model.fnn_list[-1].register_forward_pre_hook(lambda module, args: hooked.append(args[0].detach().clone()))
        with torch.no_grad():
            y_hook = model(x)
        handle.remove()
        h = hooked[0]
        check(split+':hook_output_bitwise', torch.equal(y, y_hook))
        check(split+':shapes', tuple(h.shape) == (12,50,128) and tuple(y.shape) == (12,50,126), dict(input=list(x.shape), hidden=list(h.shape), output=list(y.shape)))
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            frozen = model(x)
        check(split+':frozen_flags_output_bitwise', torch.equal(y, frozen))
        # Same input shape, future changed only after feature frame 124 inclusive prefix.
        future = x.clone(); future[:, :, 125:, :] = .123
        with torch.no_grad():
            changed = model(future)
        prefix_diff = (changed[:, :25]-frozen[:, :25]).abs().max().item()
        check(split+':future_feature_prefix_bitwise', prefix_diff == 0, dict(max_abs=prefix_diff, prefix_labels=25, latest_available_seconds=2.5))
        # Change other native chunks without changing batch layout.
        other = x.clone(); other[1:] = .234
        with torch.no_grad():
            reset = model(other)
        check(split+':chunk_independence_bitwise', torch.equal(reset[0], frozen[0]))
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        raw_np = y.detach().cpu().numpy()
        records = adapter.decoded_events(raw_np, params, native)
        check(split+':native_decode_exact', adapter.as_native(records) == native['decode'](raw_np, params, .5))
        mapping = adapter.history(records)
        split_dir = outputs/split; split_dir.mkdir()
        feat_class = cls_feature_class.FeatureClass(params)
        write_csv(split_dir/'raw.csv', records, feat_class)
        # Execute original training decoder on this already computed output.
        class Replay(torch.nn.Module):
            def forward(self, unused):
                return y
        native_dir = split_dir/'native'; native_dir.mkdir()
        train_seldnet.test_epoch(generator, Replay(), seldnet_model.MSELoss_ADPIT(), str(native_dir), params, torch.device('cuda'))
        check(split+':native_train_decoder_csv_bytes', (split_dir/'raw.csv').read_bytes() == (native_dir/fname.replace('.npy','.csv')).read_bytes())
        np.savez_compressed(split_dir/'native_outputs.npz', raw=raw_np, hidden=h.cpu().numpy(), mapping=mapping)
        save(split_dir/'events.json', records)
        if 'historical_csv' in paths:
            equal = (split_dir/'raw.csv').read_bytes() == Path(paths['historical_csv']).read_bytes()
            check(split+':historical_best_csv_bytes', equal, dict(actual=sha(split_dir/'raw.csv'), historical=sha(paths['historical_csv'])))
        gt = {}
        for row in csv.reader(Path(paths['gt']).open()):
            if row:
                assert len(row) == 5
                frame, cls, identity = map(int, row[:3])
                gt.setdefault(frame, []).append([cls, identity, float(row[3]), float(row[4])])
        targets = adapter.targets(records, mapping, gt)
        raw = torch.tensor([r['xyz'] for r in records], dtype=torch.float32, device='cuda')
        frame_ids = torch.tensor([r['frame'] for r in records], dtype=torch.long, device='cuda')
        event_h = h.reshape(-1,128)[frame_ids].detach()
        target_tensors = {k:torch.as_tensor(v, device='cuda') for k,v in targets.items()}
        for name, use_history, motion in [('direction',False,False), ('history',True,False), ('motion',True,True)]:
            torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            head = adapter.DirectionHead(use_history).cuda()
            head_before = state_hash(head)
            delta = head(event_h, raw, mapping)
            q = adapter.correct(raw, delta)
            check(split+':'+name+':zero_identity', torch.equal(q, raw))
            write_csv(split_dir/(name+'_zero.csv'), records, feat_class, q.detach().cpu().numpy())
            check(split+':'+name+':zero_csv_bytes', (split_dir/(name+'_zero.csv')).read_bytes() == (split_dir/'raw.csv').read_bytes())
            total, terms = adapter.loss(head, event_h, raw, mapping, target_tensors, motion)
            total.backward()
            gradients = [p.grad for p in head.parameters()]
            total_grad = sum(g.abs().sum().item() for g in gradients if g is not None)
            check(split+':'+name+':finite_nonzero_head_gradient', all(g is not None and torch.isfinite(g).all().item() for g in gradients) and total_grad > 0,
                  dict(total_loss=total.item(), gradient_abs_sum=total_grad, terms={k:v.item() for k,v in terms.items()}))
            check(split+':'+name+':no_parameter_update', state_hash(head) == head_before)
        check(split+':backbone_buffers_unchanged', state_hash(model) == original_state and all(p.grad is None for p in model.parameters()))
        # Prefix test of historical raw waveform frontend; no audio is exported.
        audio, fs = feat_class._load_audio(paths['audio'])
        cut = int(fs*2.5)
        altered = audio.copy(); altered[cut:] = .0123
        spec = feat_class._spectrogram(audio, 3000)
        spec_future = feat_class._spectrogram(altered, 3000)
        check(split+':waveform_frontend_prefix_bitwise', np.array_equal(spec[:125], spec_future[:125]),
              dict(cut_samples=cut, prefix_feature_frames=125, nfft=feat_class._nfft, left_pad=feat_class._nfft-feat_class._hop_len))
        counts = dict(events=len(records), matched=int(targets['matched'].sum()), valid_motion_pairs=int(targets['motion_valid'].sum()),
                      nonzero_motion_coordinates=int(np.count_nonzero(targets['displacement'])),
                      fixed_unmatched=int((~targets['matched']).sum()), gt_events=sum(map(len,gt.values())),
                      predicted_pairs=int((mapping>=0).sum()))
        save(split_dir/'COUNTS.json', counts)
        timings[split] = time.monotonic()-t0
        print(json.dumps(dict(split=split, seconds=timings[split], counts=counts)), flush=True)
    for path, expected in seal['inputs'].items():
        check('input_unchanged:'+path, sha(path) == expected)
    save(R/'PREFLIGHT.json', dict(status='PASS', scope='two recordings only', seconds=time.monotonic()-start,
          timings=timings, max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
          source_sha256=sha(R/'SOURCE.json'), optimizer_created=False, training_started=False,
          full_validation_scoring='NOT_RUN', evaluation_accessed=False,
          remaining=['full validation regression and official scoring','all-cache checks','trained-head causal prefix regression',
                     'full padding tests beyond native full-length samples','full-cache training and evaluation authorization']))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(R/'FAILURE.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        traceback.print_exc()
        sys.exit(1)
