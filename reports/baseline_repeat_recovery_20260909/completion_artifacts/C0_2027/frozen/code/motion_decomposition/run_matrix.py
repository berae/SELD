"""Sequential single-GPU inference/scoring, validation gate before evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from export import save,digest

root=Path('/home/zhanghc/SELD'); out=root/'reports/motion_decomposition_20260907'; code=out/'code'
py=sys.executable; repo=root/'code_releases/SELD_v0.3.0_20260907'
runs=json.loads((out/'live_inputs/input_manifest.json').read_text())['runs']
primary=[]; appendix=[]
for r in runs:
    cfg=r['config']; name=Path(r['run']).name; seed=cfg['audit']['seed']
    if name.startswith('B0_'):variant='C0'
    elif name.startswith('B1_'):variant='C1'
    elif name.startswith('C3_jepa005'):variant='C3_j005'
    elif name.startswith('C3_jepa020'):variant='C3_j020'
    elif name.startswith('D1_'):variant='D1'
    elif name.startswith('D3_'):variant='D3'
    else:continue
    item=(variant,seed,r)
    (primary if variant in ('C0','C1','C3_j005') else appendix).append(item)
assert len(primary)==9 and len(appendix)==5
primary.sort(key=lambda x:(x[1],x[0]));appendix.sort()


def execute(cmd,log):
    with (out/'commands.sh').open('a') as f:
        import shlex
        f.write('CUDA_VISIBLE_DEVICES=0 '+shlex.join(cmd)+'\n')
    with log.open('x') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},timeout=3600)
    if p.returncode:
        with (out/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(command=cmd,returncode=p.returncode,log=str(log)))+'\n')
        raise RuntimeError(str(log))


def one(variant,seed,r,split,raw_only=False):
    name=f'{variant}_{seed}'; exp=out/'exports'/name/split; ana=out/'analysis'/name/split
    if exp.exists() and not (exp/'COMPLETED.json').exists():exp=out/'exports_v2'/name/split
    if not exp.exists():
        # Verify GPU 0 UUID is absent from compute process list before every run.
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i','0'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert uuid not in processes, 'GPU 0 occupied; stop'
        exporter=code/'export_v2.py' if (code/'export_v2.py').exists() else code/'export.py'
        cmd=[py,str(exporter),'--run',r['run'],'--runtime',r['exact_runtime_candidates'][0],'--output',str(exp),'--split',split]
        if variant.startswith('C3') or variant=='D3':cmd+=['--latent']
        execute(cmd,out/(name+'_'+split+'_export.log'))
    assert (exp/'COMPLETED.json').exists(), 'Incomplete export: '+str(exp)
    if not ana.exists():
        table=root/('DynamicCausalMultiACCDOA_TAU2020/runs/analysis/weight_pilot_v2/motion_source_frames_validation_fold1.csv' if split=='validation' else 'experiment_completion_20260904/assets/motion_source_frames_evaluation.csv')
        cmd=[py,str(code/'analyze.py'),'--export',str(exp),'--repository',str(repo),'--output',str(ana),'--variant',variant,'--motion-table',str(table)]
        if raw_only:cmd+=['--raw-only']
        execute(cmd,out/(name+'_'+split+'_analyze.log'))
    assert (ana/'COMPLETED.json').exists(), 'Incomplete analysis: '+str(ana)
    print(json.dumps(dict(variant=variant,seed=seed,split=split,status='COMPLETED')),flush=True)


for variant,seed,r in primary:one(variant,seed,r,'validation')
protocol=json.loads((code/'protocol.json').read_text())
protocol.update(status='FROZEN_AFTER_VALIDATION_QA',frozen_utc=datetime.now(timezone.utc).isoformat(),
                decoder_sha256=digest(code/'decoder.py'),analysis_sha256=digest(code/'analyze.py'),
                validation_runs_completed=9,threshold_search_performed=False)
save(out/'protocol.json',protocol)
for variant,seed,r in primary:one(variant,seed,r,'evaluation')
for variant,seed,r in appendix:one(variant,seed,r,'evaluation',True)
save(out/'MATRIX_COMPLETED.json',dict(primary_run_splits=18,appendix_run_splits=5))
