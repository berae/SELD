"""Validate each C0's own caches and frozen controls, with zero optimizer updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from experiment_core import digest,save_json,make_head,loss_terms
from train_heads import CachedSplit,predict,evaluate,initialization_gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--C0-seed',type=int,choices=[2027,2028],required=True);a=p.parse_args()
    torch.set_num_threads(2);seed=a.C0_seed;cohort=a.root/('C0_%d'%seed)
    sys.path.insert(0,str(a.root/'scorer'))
    config=json.loads((a.root/'configs'/('pilot_C0_%d.json'%seed)).read_text())
    train=cohort/'cache/train';val=cohort/'cache/validation'
    for path,expected in ((train,500),(val,100)):
        assert json.loads((path/'COMPLETED.json').read_text())['files']==expected
    data=CachedSplit(val);tm=json.loads((train/'input_manifest.json').read_text())
    assert tm['checkpoint_sha256']==data.manifest['checkpoint_sha256']==config['baseline_checkpoint_sha256']
    assert not set(tm['gt_files'])&set(data.names)
    f0=evaluate(data,predict(None,data,'F0','cpu'),cohort/'validation_controls/F0')
    assert f0['scores']==json.loads((val/'COMPLETED.json').read_text())['scores']
    for c in ('F-EMA','F-KF'):
        r=evaluate(data,predict(None,data,c,'cpu'),cohort/'validation_controls'/c);assert r['sed']==f0['sed']
    gates={c:initialization_gate(data,'cpu',c) for c in ('F-Deriv','R0','R1','R2')}
    stem=Path(sorted(tm['gt_files'])[0]).stem
    hashes=json.loads((train/'file_manifest.json').read_text())
    for sub in ('float','targets'):assert digest(train/sub/(stem+'.npz'))==hashes[sub+'/'+stem+'.npz']
    with np.load(train/'float'/(stem+'.npz')) as z,np.load(train/'targets'/(stem+'.npz')) as t:
        b={}
        for k in ('doa_features','doa','mapping','probability','target','matched','motion_valid','displacement'):
            x=z[k] if k in z else t[k];b[k]=torch.from_numpy(x.reshape(15,40,*x.shape[1:])[:2].copy())
    for c in gates:
        head=make_head(c);loss,_=loss_terms(head,b,c);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert head.net[-1].weight.grad.abs().sum()>0
    save_json(cohort/'PREFLIGHT.json',dict(status='PASS',C0_seed=seed,C0_sha256=config['baseline_checkpoint_sha256'],
        split_disjoint=True,validation_original_float_csv_scores_exact=True,real_head_gates=gates,
        real_train_gradient_finite_nonzero=True,optimizer_updates=0,baselines_pure_sed_exact=True,
        manifests_sha256={s:digest(cohort/'cache'/s/'file_manifest.json') for s in ('train','validation')}))
    print(json.dumps(dict(status='PASS',C0_seed=seed,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
