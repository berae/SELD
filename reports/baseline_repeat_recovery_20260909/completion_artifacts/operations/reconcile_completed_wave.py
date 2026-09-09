"""Recover only a missing launcher receipt after transport loss; never launches computation."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--C0-seed',type=int,choices=(2027,2028),required=True)
    a=p.parse_args(); root=a.root.resolve()
    sys.path.insert(0,str(root/'code/refinement_v2')); sys.path.insert(0,str(root/'code/motion_decomposition'))
    from experiment_core import digest,save_json
    dispatch=root/f'dispatch/train_C0_{a.C0_seed}'
    assert not (dispatch/'EXIT.json').exists() and not (dispatch/'COMPLETED.json').exists()
    jobs=json.loads((dispatch/'DISPATCHED.json').read_text())
    assert len(jobs)==4 and {j['item'] for j in jobs}=={'F-Deriv','R0','R1','R2'}
    evidence=[]
    for job in jobs:
        assert job['C0_seed']==a.C0_seed and job['head_seed']==2026
        guard=Path(job['guard']); result_path=root/f'C0_{a.C0_seed}/heads'/job['item']
        ex=json.loads((guard/'EXIT.json').read_text()); done=json.loads((result_path/'COMPLETED.json').read_text())
        assert ex['exit_code']==0 and not ex['timed_out'] and not ex['automatic_retry']
        assert done['status']=='PASS' and 40<=done['epochs']<=200 and done['training_updates']>0
        assert digest(result_path/'best.pth')==done['best_checkpoint_sha256']
        evidence.append(dict(condition=job['item'],guard_exit_sha256=digest(guard/'EXIT.json'),
            completed_sha256=digest(result_path/'COMPLETED.json'),checkpoint_sha256=done['best_checkpoint_sha256']))
    save_json(dispatch/'TRANSPORT_RECOVERY.json',dict(status='VERIFIED_COMPLETED_JOBS',created_unix=time.time(),
        reason='SSH stdout connection lost; launcher no longer listed; original independent guards and training processes continued',
        new_processes_launched=0,training_retried=False,script_sha256=digest(__file__),evidence=evidence))
    save_json(dispatch/'EXIT.json',dict(exit_codes=[0,0,0,0],automatic_retry=False,
        source='Recovered from four original guarded job EXIT records, not launcher wait returncodes'))
    save_json(dispatch/'COMPLETED.json',dict(status='PASS',jobs=4,stage='train',
        recovered_after_transport_loss=True,evidence_sha256=digest(dispatch/'TRANSPORT_RECOVERY.json')))
    print(json.dumps(dict(status='PASS',new_processes_launched=0,training_retried=False)))


if __name__=='__main__':main()
