"""Authorized fixture-only recovery; frozen model/loss/training code is imported unchanged."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--C0-seed', type=int, choices=(2027, 2028), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); seed = a.C0_seed
    code = root/'code/refinement_v2'
    sys.path.insert(0, str(code)); sys.path.insert(0, str(root/'code/motion_decomposition'))
    from experiment_core import digest, save_json, make_head, loss_terms
    from train_heads import CachedSplit, predict, evaluate, initialization_gate
    torch.set_num_threads(2); torch.manual_seed(2026); np.random.seed(2026)
    prepared = json.loads((root/'PREPARED.json').read_text())
    for rel, h in prepared['code_hashes'].items():
        assert digest(root/'code'/rel) == h, rel
    for name, h in prepared['config_hashes'].items():
        assert digest(root/'configs'/name) == h, name
    auth = root/'PREFLIGHT_RECOVERY_AUTHORIZATION.md'
    assert auth.is_file()
    cohort = root/f'C0_{seed}'; train = cohort/'cache/train'; val = cohort/'cache/validation'
    assert not (cohort/'PREFLIGHT.json').exists() and not (cohort/'heads').exists()
    a.output.mkdir(parents=True, exist_ok=False)
    save_json(a.output/'STARTED.json', dict(command=sys.argv, script_sha256=digest(__file__),
        authorization_sha256=digest(auth), prepared_sha256=digest(root/'PREPARED.json'),
        fixture_seed=2026, backbone_loaded=False, optimizer_updates=0))
    sys.path.insert(0, str(root/'scorer'))
    config = json.loads((root/f'configs/pilot_C0_{seed}.json').read_text())
    for path, expected in ((train, 500), (val, 100)):
        done = json.loads((path/'COMPLETED.json').read_text())
        assert done['status'] == 'PASS' and done['files'] == expected
    data = CachedSplit(val); tm = json.loads((train/'input_manifest.json').read_text())
    assert tm['checkpoint_sha256'] == data.manifest['checkpoint_sha256'] == config['baseline_checkpoint_sha256']
    assert not set(tm['gt_files']) & set(data.names)
    f0 = None
    for c in ('F0', 'F-EMA', 'F-KF'):
        result = evaluate(data, predict(None, data, c, 'cpu'), a.output/'validation_controls'/c)
        old = json.loads((cohort/'validation_controls'/c/'metrics.json').read_text())
        assert result == old
        for name in data.names:
            assert (a.output/'validation_controls'/c/name).read_bytes() == (cohort/'validation_controls'/c/name).read_bytes()
        if c == 'F0':
            f0 = result
            assert result['scores'] == json.loads((val/'COMPLETED.json').read_text())['scores']
        assert result['sed'] == f0['sed']
    gates = {c: initialization_gate(data, 'cpu', c) for c in ('F-Deriv', 'R0', 'R1', 'R2')}
    stem = Path(sorted(tm['gt_files'])[0]).stem
    assert stem == 'fold2_room1_mix001_ov1'
    hashes = json.loads((train/'file_manifest.json').read_text())
    for sub in ('float', 'targets'):
        assert digest(train/sub/(stem+'.npz')) == hashes[sub+'/'+stem+'.npz']
    with np.load(train/'float'/(stem+'.npz')) as z, np.load(train/'targets'/(stem+'.npz')) as t:
        b = {}
        for k in ('doa_features', 'doa', 'mapping', 'probability', 'target', 'matched', 'motion_valid', 'displacement'):
            x = z[k] if k in z else t[k]
            b[k] = torch.from_numpy(x.reshape(15, 40, *x.shape[1:]).copy())
    assert (b['mapping'][:, 0] == -1).all() and not b['motion_valid'][:, 0].any()
    static = {k: v[:2] for k, v in b.items()}
    assert static['motion_valid'].sum() == 59
    assert not torch.count_nonzero(static['displacement'][static['motion_valid']])
    torch.manual_seed(2026)
    head = make_head('F-Deriv'); loss, _ = loss_terms(head, static, 'F-Deriv'); loss.backward()
    assert loss.item() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() == 0 for p in head.parameters())
    static_check = dict(chunks=2, matched=int(static['matched'].sum()), valid_pairs=59, loss=0.0,
                        all_gradients_finite_and_zero=True)
    assert torch.count_nonzero(b['displacement'][b['motion_valid']]) > 0
    gradient_checks = {}
    for c in gates:
        torch.manual_seed(2026)
        head = make_head(c)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        loss, terms = loss_terms(head, b, c); loss.backward()
        finite = bool(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        grad = float(head.net[-1].weight.grad.abs().sum())
        row = dict(loss=float(loss.detach()), final_weight_gradient_abs_sum=grad, all_gradients_finite=finite,
                   chunks=15, matched=int(b['matched'].sum()), valid_pairs=int(b['motion_valid'].sum()),
                   nonzero_target_coordinates=int(torch.count_nonzero(b['displacement'][b['motion_valid']])))
        save_json(a.output/(c+'_gradient.json'), row)
        assert finite and grad > 0
        assert all(torch.equal(v, head.state_dict()[k]) for k, v in before.items())
        gradient_checks[c] = row
    for rel, h in prepared['code_hashes'].items():
        assert digest(root/'code'/rel) == h, rel
    receipt = dict(status='PASS', C0_seed=seed, C0_sha256=config['baseline_checkpoint_sha256'],
        split_disjoint=True, validation_original_float_csv_scores_exact=True, real_head_gates=gates,
        real_train_gradient_finite_nonzero=True, optimizer_updates=0, baselines_pure_sed_exact=True,
        static_zero_target_check=static_check, gradient_checks=gradient_checks, fixture_seed=2026,
        fixture_recording=stem, no_training_data_change=True, code_snapshot_unchanged=True,
        script_sha256=digest(__file__), authorization_sha256=digest(auth), recovery_output=str(a.output),
        manifests_sha256={s:digest(cohort/'cache'/s/'file_manifest.json') for s in ('train','validation')})
    save_json(a.output/'COMPLETED.json', receipt)
    save_json(cohort/'PREFLIGHT.json', receipt)
    print(json.dumps(dict(status='PASS', C0_seed=seed, optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    main()
