"""Read-only cross-check of frozen head configs and original guardian exits."""
import argparse
import json
from pathlib import Path
import torch
from experiment_core import digest, save_json


def main():
    p=argparse.ArgumentParser()
    for k in ('frozen','base','stability','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();seal=json.loads((a.frozen/'FROZEN.json').read_text())
    config=json.loads((a.frozen/'configs/pilot.json').read_text())
    original=json.loads((a.frozen/'configs/execution_r1.json').read_text());rows=[]
    for run in seal['runs']:
        seed=run['head_seed'];condition=run['condition']
        ck=torch.load(a.frozen/run['weight_path'],map_location='cpu',weights_only=False)
        assert ck['config']==config
        execution=ck['execution'];assert execution['stopping']==original['stopping']
        logged=json.loads((Path(run['source'])/'RUNNING.json').read_text())
        core_hashes={name:digest(a.frozen/'code/refinement_v2'/name) for name in ('candidate.py','experiment_core.py')}
        assert all(logged['code_hashes'][name]==value for name,value in core_hashes.items())
        exit_path=(a.base/'guards_pilot'/condition/'EXIT.json' if seed==2026 else
                   a.stability/'guards'/('seed%d'%seed)/condition/'EXIT.json')
        exit_record=json.loads(exit_path.read_text())
        assert exit_record['exit_code']==0 and not exit_record['timed_out']
        rows.append(dict(condition=condition,head_seed=seed,config_exact=True,stopping_policy_exact=True,original_core_hashes_exact=core_hashes,
            source_weight_sha256=digest(Path(run['source'])/'best.pth'),frozen_weight_sha256=digest(a.frozen/run['weight_path']),
            original_guard_exit=exit_record,original_guard_exit_path=str(exit_path),original_guard_exit_sha256=digest(exit_path)))
        assert rows[-1]['source_weight_sha256']==rows[-1]['frozen_weight_sha256']==run['weight_sha256']
    save_json(a.output,dict(status='PASS',runs=rows,freeze_sha256=digest(a.frozen/'FROZEN.json'),
        audit_script_sha256=digest(__file__),scope='read-only source/config/guardian audit; no inference or training'))
    print(json.dumps(dict(status='PASS',runs=len(rows))))


if __name__=='__main__':main()
