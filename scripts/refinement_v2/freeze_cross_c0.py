"""Freeze all eight validation-selected heads before cross-C0 evaluation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import torch
from experiment_core import digest,save_json
from freeze_results import copy_verified


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    prepared=json.loads((root/'PREPARED.json').read_text());assert prepared['status']=='PASS'
    # Do not open any new evaluation cache or score; require both training waves first.
    for seed in (2027,2028):
        assert json.loads((root/'dispatch'/('train_C0_%d'%seed)/'COMPLETED.json').read_text())['status']=='PASS'
    all_runs=[];seals=[]
    for seed in (2027,2028):
        cohort=root/('C0_%d'%seed);out=cohort/'frozen';out.mkdir(exist_ok=False)
        config=json.loads((root/'configs'/('pilot_C0_%d.json'%seed)).read_text())
        copy_verified(root/'source'/('C0_%d'%seed)/'best.pth',out/'C0/best.pth',config['baseline_checkpoint_sha256'])
        copy_verified(root/'source'/('C0_%d'%seed)/'config.json',out/'C0/config.json')
        copy_verified(root/'scalar_trainfolds2-6.h5',out/'scalar_trainfolds2-6.h5')
        for name in ('AUTHORIZATION.md','CROSS_C0_PREREGISTRATION.md'):copy_verified(root/name,out/name)
        copy_verified(root/'configs'/('pilot_C0_%d.json'%seed),out/'configs/pilot.json')
        copy_verified(root/'configs/execution_cross_c0.json',out/'configs/execution_r1.json')
        relocation=json.loads((root/'configs'/('relocation_C0_%d.json'%seed)).read_text())
        relocation.update(run=str(out/'C0'),runtime=str(out/'runtime'),causal_scalar_path=str(out/'scalar_trainfolds2-6.h5'))
        save_json(out/'configs/evaluation_relocation.json',relocation)
        for sub in ('runtime','scorer','code'):shutil.copytree(root/sub,out/sub,ignore=shutil.ignore_patterns('__pycache__'))
        runs=[]
        for c in ('F-Deriv','R0','R1','R2'):
            source=cohort/'heads'/c;done=json.loads((source/'COMPLETED.json').read_text());assert done['status']=='PASS'
            hist=[json.loads(x) for x in (source/'epochs.jsonl').read_text().splitlines()]
            best=min(hist,key=lambda x:(x['scores']['dcase2023_micro']['SELD_LR'],x['epoch']))
            assert best['epoch']==done['best_epoch'] and hist[-1]['epoch']==done['epochs']
            assert done['epochs']>=40 and done['epochs']<=200
            if done['convergence_confirmed']:
                assert done['stopping_reason']=='BOTH_PLATEAUS'
                assert hist[-1]['train_plateau_relative_change']<=.01 and hist[-1]['validation_epochs_without_progress']>=20
            else:assert done['stopping_reason']=='MAX_EPOCHS' and done['epochs']==200
            ck=torch.load(source/'best.pth',map_location='cpu',weights_only=False)
            assert ck['seed']==2026 and ck['condition']==c and ck['epoch']==best['epoch']
            assert ck['config']==config and ck['checkpoint_sha256']==config['baseline_checkpoint_sha256']
            target=out/'heads'/('head2026_'+c)
            h=copy_verified(source/'best.pth',target/'best.pth',done['best_checkpoint_sha256'])
            for name in ('COMPLETED.json','RUNNING.json','epochs.jsonl','preflight.json'):copy_verified(source/name,target/name)
            copy_verified(root/'guards'/('train_C0_%d'%seed)/('%d_%s'%(seed,c))/'EXIT.json',target/'GUARD_EXIT.json')
            save_json(target/'ACTUAL_EXECUTION.json',ck['execution'])
            row=dict(C0_seed=seed,head_seed=2026,condition=c,source=str(source),weight_path=str((target/'best.pth').relative_to(out)),
                weight_sha256=h,stopped_epoch=done['epochs'],selected_epoch=best['epoch'],convergence_confirmed=done['convergence_confirmed'],
                validation_best_scores=done['result']['scores'],stopping_record=hist[-1],validation_selection_verified=True)
            runs.append(row);all_runs.append(row)
        files={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        save_json(out/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_EVALUATION',created_utc=datetime.now(timezone.utc).isoformat(),
            C0_seed=seed,C0_sha256=config['baseline_checkpoint_sha256'],runs=runs,files_sha256=files,expected_learned_runs=4,
            fixed_controls=['F0','F-EMA','F-KF'],preregistration_sha256=prepared['preregistration_sha256'],
            authorization_sha256=prepared['authorization_sha256'],evaluation_selection_prohibited=True,historical_evaluation_seen=True))
        seals.append(dict(C0_seed=seed,manifest_sha256=digest(out/'FROZEN.json')))
    save_json(root/'ALL_EIGHT_FROZEN.json',dict(status='PASS',seals=seals,runs=all_runs,head_seed=2026,
        new_formal_runs=8,old_head_runs_retrained=0,bootstrap_repeated=False,all_new_evaluation_still_unopened=True))
    print(json.dumps(dict(status='PASS',frozen_runs=len(all_runs))))


if __name__=='__main__':main()
