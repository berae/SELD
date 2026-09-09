"""Bind separately authorized C0 sources to the unchanged registered head recipe."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import torch
from experiment_core import digest,save_json

EXPECTED={2027:'1f924b811f8cc8ed3cfde60d9727f54bb83e950b47404cfa09bcbc694c8aa08c',
          2028:'5d128ad298b110d3afff8ae95fc37b08d25e38fb9d1f595e8d51d99708c76683'}


def normalized(config):
    x=copy.deepcopy(config);x.pop('workspace_dir')
    x['training'].pop('train_id');x['training'].pop('remark');x['audit'].pop('seed')
    return x


def main():
    p=argparse.ArgumentParser()
    for k in ('root','previous'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preserve-source-model-flags',action='store_true')
    a=p.parse_args();a.root=a.root.resolve();frozen=a.previous/'frozen'
    registry=json.loads((frozen/'FROZEN.json').read_text())
    assert digest(a.root/'CROSS_C0_PREREGISTRATION.md')==registry['preregistration_sha256']
    assert (a.root/'AUTHORIZATION.md').is_file()
    spec=json.loads((frozen/'configs/pilot.json').read_text())
    execution=json.loads((frozen/'configs/execution_r1.json').read_text())
    old=json.loads((frozen/'C0/config.json').read_text())
    for name in ('candidate.py','experiment_core.py','train_heads.py'):
        assert digest(Path(__file__).parent/name)==digest(frozen/'code/refinement_v2'/name)
    shutil.copytree(frozen/'runtime',a.root/'runtime')
    shutil.copytree(frozen/'scorer',a.root/'scorer')
    shutil.copy2(frozen/'scalar_trainfolds2-6.h5',a.root/'scalar_trainfolds2-6.h5')
    (a.root/'configs').mkdir(exist_ok=False);sources=[]
    for seed,expected in EXPECTED.items():
        source=a.root/'source'/('C0_%d'%seed)
        assert digest(source/'best.pth')==expected
        cfg=json.loads((source/'config.json').read_text())
        ck=torch.load(source/'best.pth',map_location='cpu',weights_only=False)
        assert ck['config']==cfg;del ck
        assert normalized(cfg)==normalized(old),'Same-cohort configuration mismatch'
        assert cfg['audit']['seed']==seed and cfg['audit']['variant']=='C0'
        assert digest(a.root/'scalar_trainfolds2-6.h5')==cfg['audit']['scalar_sha256']
        for relative,h in cfg['audit']['source_hashes'].items():assert digest(a.root/'runtime'/relative)==h,relative
        historical=source/'validation'
        hmanifest=json.loads((historical/'input_manifest.json').read_text())
        assert hmanifest['checkpoint_sha256']==expected
        assert len(list((historical/'float').glob('*.npz')))==100
        assert len(list((historical/'raw').glob('*.csv')))==100
        updated=copy.deepcopy(spec)
        updated.update(baseline_seed=seed,baseline_checkpoint_sha256=expected,baseline_run=str(source),
            runtime=str(a.root/'runtime'),existing_export=str(source),
            proposed_feature_cache=str(a.root/('C0_%d'%seed)/'cache'),
            proposed_pilot_output=str(a.root/('C0_%d'%seed)/'heads'))
        delta={k for k in spec if spec[k]!=updated[k]}
        assert delta=={'baseline_seed','baseline_checkpoint_sha256','baseline_run','runtime','existing_export','proposed_feature_cache','proposed_pilot_output'}
        save_json(a.root/'configs'/('pilot_C0_%d.json'%seed),updated)
        relocation=dict(run=str(source),runtime=str(a.root/'runtime'),causal_scalar_path=str(a.root/'scalar_trainfolds2-6.h5'),
            dataset_dir='/work/zhanghc/Myllm/SELD/EINV2/dataset_root',
            hdf5_dir='/work/zhanghc/Myllm/SELD/EINV2/C0_CausalEINV2_seed2026/_hdf5',original_validation=str(historical))
        save_json(a.root/'configs'/('relocation_C0_%d.json'%seed),relocation)
        sources.append(dict(C0_seed=seed,C0_sha256=expected,config_sha256=digest(source/'config.json'),
            files_sha256={str(f.relative_to(source)):digest(f) for f in source.rglob('*') if f.is_file() and '.partial.' not in f.name and '.stream.' not in f.name}))
    execution.update(seed=2026,user_authorization='Explicit user confirmation; see AUTHORIZATION.md',
        authorized_scope='preregistered same-cohort C0 seeds2027/2028, fixed head2026, maximum8 new formal heads',
        smoke_batches_per_condition=0,no_automatic_three_seed_expansion=False)
    save_json(a.root/'configs/execution_cross_c0.json',execution)
    save_json(a.root/'PREPARED.json',dict(status='PASS',created_unix=time.time(),C0_seeds=[2027,2028],head_seed=2026,
        conditions=['F-Deriv','R0','R1','R2'],maximum_new_training_runs=8,authorization_sha256=digest(a.root/'AUTHORIZATION.md'),
        preregistration_sha256=digest(a.root/'CROSS_C0_PREREGISTRATION.md'),sources=sources,
        code_hashes={str(f.relative_to(a.root/'code')):digest(f) for f in (a.root/'code').rglob('*.py')},
        config_hashes={f.name:digest(f) for f in (a.root/'configs').glob('*.json')},
        preserve_source_model_flags=a.preserve_source_model_flags,
        unchanged_model_loss_and_training_core=True,evaluation_selection_prohibited=True,bootstrap_repeated=False))
    print(json.dumps(dict(status='PASS',C0_seeds=[2027,2028],maximum_new_training_runs=8)))


if __name__=='__main__':main()
