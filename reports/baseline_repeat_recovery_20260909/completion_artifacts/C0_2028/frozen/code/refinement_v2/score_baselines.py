"""Score all registered fixed controls directly on authoritative raw caches (CPU)."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from candidate import fixed_records, prediction_history
from experiment_core import digest, save_json, csv_bytes, read_gt
from train_heads import predict, evaluate


def main():
    p=argparse.ArgumentParser()
    for name in ('export','metadata','scorer','output','config','execution'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();torch.set_num_threads(2)
    manifest=json.loads((a.export/'input_manifest.json').read_text())
    spec=json.loads(a.config.read_text())
    assert manifest['split']=='validation' and manifest['checkpoint_sha256']==spec['baseline_checkpoint_sha256']
    for rel,h in manifest['source_hashes'].items():
        if rel.startswith('evaluation/'):
            assert digest(a.scorer/Path(rel).relative_to('evaluation'))==h
    sys.path.insert(0,str(a.scorer))
    hashes=json.loads((a.export/'float_manifest.json').read_text())
    data=SimpleNamespace(names=sorted(manifest['gt_files']),gt=[],records=[],raw_csv=[],data={})
    assert len(data.names)==100
    raw_values=[];maps=[]
    for name in data.names:
        fn=Path(name).stem+'.npz';path=a.export/'float'/fn
        assert digest(path)==hashes[fn]
        assert digest(a.metadata/name)==manifest['gt_files'][name]
        with np.load(path,allow_pickle=False) as z:
            raw,prob=z['doa'],z['probability']
            assert raw.shape==(600,2,3) and prob.shape==(600,2,14)
            rec=fixed_records(prob,raw)
            assert csv_bytes(rec)==(a.export/'raw'/name).read_bytes()
            raw_values.append(raw.copy())
            maps.append(prediction_history(prob,raw,z['frame_index'],z['chunk_index']))
            data.records.append(rec);data.raw_csv.append(csv_bytes(rec));data.gt.append(read_gt(a.metadata/name))
    data.data['doa']=torch.from_numpy(np.stack(raw_values).reshape(-1,40,2,3))
    data.data['mapping']=torch.from_numpy(np.stack(maps).reshape(-1,40,2))
    save_json(a.output/'input_receipt.json',dict(checkpoint_sha256=spec['baseline_checkpoint_sha256'],
              raw_manifest_sha256=digest(a.export/'float_manifest.json'),config_sha256=digest(a.config),
              execution_sha256=digest(a.execution),script_sha256=digest(__file__),
              source='original RB05 C0 prediction cache, transported without changes',
              scoring_host='rabbit02',training_updates=0,selection_split='validation'))
    results={}
    for condition in ('F0','F-EMA','F-KF'):
        out=predict(None,data,condition,'cpu')
        result=evaluate(data,out,a.output/condition/'validation')
        if condition=='F0':
            assert result['scores']==json.loads((a.export/'raw_regression.json').read_text())['scores']
        else:
            assert result['sed']==results['F0']['sed']
        results[condition]=result
        save_json(a.output/condition/'COMPLETED.json',dict(status='PASS',condition=condition,result=result,training_updates=0))
        print(json.dumps(dict(condition=condition,scores=result['scores']['dcase2023_micro'],elapsed_seconds=time.time()-start)),flush=True)
    save_json(a.output/'COMPLETED.json',dict(status='PASS',conditions=list(results),files=100,
              elapsed_seconds=time.time()-start,training_updates=0,pure_sed_identity=True))


if __name__=='__main__':main()
