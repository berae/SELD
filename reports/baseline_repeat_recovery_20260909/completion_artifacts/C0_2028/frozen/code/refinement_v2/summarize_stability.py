"""All registered head seeds, paired descriptive comparisons; no seed selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiment_core import save_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--status',action='store_true');a=p.parse_args()
    cs=['F-Deriv','R0','R1','R2'];rows=[]
    for seed in (2026,2027,2028):
        for c in cs:
            path=a.base/'pilot'/c if seed==2026 else a.root/('seed%d'%seed)/c
            if not (path/'COMPLETED.json').exists():
                hist=path/'epochs.jsonl';h=[json.loads(x) for x in hist.read_text().splitlines()] if hist.exists() else []
                rows.append(dict(seed=seed,condition=c,status='RUNNING_OR_NOT_STARTED',epoch=h[-1]['epoch'] if h else 0,
                    best_epoch=h[-1]['best_epoch'] if h else None));continue
            r=json.loads((path/'COMPLETED.json').read_text());metrics=r['result']['scores']['dcase2023_micro']
            rows.append(dict(seed=seed,condition=c,status=r['status'],epochs=r['epochs'],best_epoch=r['best_epoch'],
                converged=r['convergence_confirmed'],**metrics))
    if a.status:print(json.dumps(rows));return
    assert json.loads((a.root/'COMPLETED.json').read_text())['status']=='PASS'
    assert len(rows)==12 and all(r['status']=='PASS' for r in rows)
    for seed in (2027,2028):
        config=a.root/('execution_seed%d.json'%seed);assert json.loads(config.read_text())['seed']==seed
        for c in cs:
            path=a.root/('seed%d'%seed)/c;start=json.loads((path/'RUNNING.json').read_text())
            assert start['execution_sha256']==digest(config)
            guard=json.loads((a.root/'guards'/('seed%d'%seed)/c/'EXIT.json').read_text());assert guard['exit_code']==0 and not guard['timed_out']
            h=[json.loads(line) for line in (path/'epochs.jsonl').read_text().splitlines()]
            done=json.loads((path/'COMPLETED.json').read_text())
            assert digest(path/'best.pth')==done['best_checkpoint_sha256']
            checkpoint=torch.load(path/'best.pth',map_location='cpu',weights_only=False)
            assert checkpoint['seed']==seed and checkpoint['condition']==c
            assert checkpoint['checkpoint_sha256']==json.loads((a.base/'configs/pilot.json').read_text())['baseline_checkpoint_sha256']
            original=json.loads((a.base/'pilot'/c/'RUNNING.json').read_text())
            for source in ('candidate.py','experiment_core.py'):
                assert start['code_hashes'][source]==original['code_hashes'][source]
            assert min(h,key=lambda x:(x['scores']['dcase2023_micro']['SELD_LR'],x['epoch']))['epoch']==done['best_epoch']
            ref=json.loads((a.base/'pilot'/c/'COMPLETED.json').read_text())
            assert done['result']['sed']==ref['result']['sed']
    f0=json.loads((a.base/'baselines/F0/COMPLETED.json').read_text())['result']['scores']['dcase2023_micro']
    summary={};paired={}
    for c in cs:
        summary[c]={k:dict(mean=float(np.mean([r[k] for r in rows if r['condition']==c])),
                sample_sd=float(np.std([r[k] for r in rows if r['condition']==c],ddof=1))) for k in ('LE_CD','SELD_LR','F20','LR_CD')}
    for c,b in [('R0','F0'),('R0','F-Deriv'),('R1','R0'),('R2','R1')]:
        pair={}
        for k in ('LE_CD','SELD_LR'):
            differences=[]
            for seed in (2026,2027,2028):
                left=next(r for r in rows if r['seed']==seed and r['condition']==c)
                right=f0 if b=='F0' else next(r for r in rows if r['seed']==seed and r['condition']==b)
                differences.append(left[k]-right[k])
            pair[k]=dict(by_seed=differences,mean=float(np.mean(differences)),sample_sd=float(np.std(differences,ddof=1)),improved_seeds=sum(x<0 for x in differences))
        paired[c+'-'+b]=pair
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'summary.json',dict(rows=rows,summary=summary,paired=paired,C0_seeds=[2026],head_seeds=[2026,2027,2028],
        limits='Head RNG sensitivity under one fixed C0 and validation-selected checkpoints; no independent-C0 or test-set confirmation, no p-values; sample SD over only three heads.'))
    save_json(a.output/'COMPLETED.json',dict(status='PASS',verified_runs=12,new_runs=8,all_detection_preserved=True,
                all_new_guard_exit_zero=True,all_new_best_epoch_selection_checked=True,script_sha256=digest(__file__)))
    print(json.dumps(dict(summary=summary,paired=paired)))


if __name__=='__main__':main()
