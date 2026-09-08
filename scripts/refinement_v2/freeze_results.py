"""Seal all existing validation-best artifacts before evaluation, no training."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import torch
from experiment_core import digest,save_json


def copy_verified(src,dst,expected=None):
    value=digest(src)
    if expected is not None:assert value==expected,str(src)
    dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists()
    shutil.copy2(src,dst);assert digest(dst)==value
    dst.chmod(0o444)
    return value


def main():
    p=argparse.ArgumentParser()
    for key in ('base','stability','preregistration','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((a.base/'configs/pilot.json').read_text())
    registry_hash=copy_verified(a.preregistration,a.output/'CROSS_C0_PREREGISTRATION.md')
    for name in ('pilot.json','execution_r1.json'):copy_verified(a.base/'configs'/name,a.output/'configs'/name)
    copy_verified(a.base/'source/C0/best.pth',a.output/'C0/best.pth',spec['baseline_checkpoint_sha256'])
    copy_verified(a.base/'source/C0/config.json',a.output/'C0/config.json')
    cfg=json.loads((a.output/'C0/config.json').read_text())
    copy_verified(a.base/'source/scalar_trainfolds2-6.h5',a.output/'scalar_trainfolds2-6.h5',cfg['audit']['scalar_sha256'])
    # Runtime is already hash-pinned to the C0 checkpoint; independent snapshot.
    shutil.copytree(a.base/'source/runtime',a.output/'runtime')
    shutil.copytree(a.base/'scorer',a.output/'scorer')
    shutil.copytree(Path(__file__).parent,a.output/'code/refinement_v2',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(Path(__file__).parent.parent/'motion_decomposition',a.output/'code/motion_decomposition',ignore=shutil.ignore_patterns('__pycache__'))
    runs=[]
    for seed in (2026,2027,2028):
        for c in ('F-Deriv','R0','R1','R2'):
            source=a.base/'pilot'/c if seed==2026 else a.stability/('seed%d'%seed)/c
            done=json.loads((source/'COMPLETED.json').read_text());assert done['status']=='PASS'
            hist=[json.loads(x) for x in (source/'epochs.jsonl').read_text().splitlines()]
            best=min(hist,key=lambda x:(x['scores']['dcase2023_micro']['SELD_LR'],x['epoch']))
            assert best['epoch']==done['best_epoch'] and hist[-1]['epoch']==done['epochs']
            assert done['convergence_confirmed'] and done['stopping_reason']=='BOTH_PLATEAUS'
            assert hist[-1]['train_plateau_relative_change']<=.01 and hist[-1]['validation_epochs_without_progress']>=20
            assert done['epochs']>=40
            target=a.output/'heads'/('head%d_%s'%(seed,c))
            h=copy_verified(source/'best.pth',target/'best.pth',done['best_checkpoint_sha256'])
            ck=torch.load(target/'best.pth',map_location='cpu',weights_only=False)
            assert ck['seed']==seed and ck['condition']==c and ck['checkpoint_sha256']==spec['baseline_checkpoint_sha256']
            assert ck['epoch']==best['epoch']
            for name in ('COMPLETED.json','RUNNING.json','epochs.jsonl','preflight.json'):
                copy_verified(source/name,target/name)
            save_json(target/'ACTUAL_EXECUTION.json',ck['execution'])
            runs.append(dict(condition=c,head_seed=seed,C0_seed=2026,source=str(source),
                weight_path=str((target/'best.pth').relative_to(a.output)),weight_sha256=h,
                stopped_epoch=done['epochs'],selected_epoch=best['epoch'],
                validation_best_scores=done['result']['scores'],validation_pure_sed=done['result']['sed'],
                stopping_record=hist[-1],validation_selection_verified=True))
    files={str(p.relative_to(a.output)):digest(p) for p in a.output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    save_json(a.output/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_EVALUATION',created_utc=datetime.now(timezone.utc).isoformat(),
        C0_seed=2026,C0_sha256=spec['baseline_checkpoint_sha256'],runs=runs,files_sha256=files,
        fixed_controls=['F0','F-EMA','F-KF'],preregistration_sha256=registry_hash,
        evaluation_selection_prohibited=True,historical_evaluation_seen=True,new_training_runs=0,bootstrap_repeated=False,
        cross_C0_training_authorization='PENDING_SEPARATE_CURRENT_USER_CONFIRMATION'))
    print(json.dumps(dict(status='FROZEN',learned_runs=len(runs),files=len(files),manifest_sha256=digest(a.output/'FROZEN.json'))))


if __name__=='__main__':main()
