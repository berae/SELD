"""Descriptive shared-target diagnostics on final quantized CSVs, no re-matching."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from candidate import fixed_records
from experiment_core import digest, save_json, polar


def errors(path, records, targets):
    with path.open(newline='') as f:rows=list(csv.reader(f))
    assert len(rows)==len(records)
    out=np.full((600,2),np.nan)
    for row,record in zip(rows,records):
        frame,cls,slot=record[:3]
        assert len(row)==4 and [int(row[0]),int(row[1])]==[frame,cls]
        u=polar(float(row[2]),float(row[3]))
        if targets['matched'][frame,slot]:
            gt=targets['target'][frame,slot].astype(np.float64)
            gt/=np.linalg.norm(gt)
            out[frame,slot]=np.rad2deg(np.arccos(np.clip(u@gt,-1,1)))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cache=a.root/'cache/validation'
    manifest=json.loads((cache/'input_manifest.json').read_text())
    file_hashes=json.loads((cache/'file_manifest.json').read_text())
    methods={c:a.root/'baselines'/c/'validation' for c in ('F0','F-EMA','F-KF')}
    methods.update({c:a.root/'pilot'/c/'validation' for c in ('F-Deriv','R0','R1','R2')})
    for path in methods.values():
        assert json.loads((path.parent/'COMPLETED.json').read_text())['status']=='PASS'
    groups=('all_fixed_matched','static_pair','moving_pair','matched_without_valid_motion_pair')
    pools={(c,g):[] for c in methods for g in groups};raw_pools={g:[] for g in groups}
    by_record=[];prediction_hashes={c:{} for c in methods}
    for name in sorted(manifest['gt_files']):
        stem=Path(name).stem
        for sub in ('float','targets'):
            assert digest(cache/sub/(stem+'.npz'))==file_hashes[sub+'/'+stem+'.npz']
        with np.load(cache/'float'/(stem+'.npz')) as z,np.load(cache/'targets'/(stem+'.npz')) as target:
            records=fixed_records(z['probability'],z['doa'])
            valid=target['motion_valid'];matched=target['matched']
            moving=np.linalg.norm(target['displacement'],axis=-1)>1e-7
            masks=dict(all_fixed_matched=matched,static_pair=valid&~moving,moving_pair=valid&moving,
                       matched_without_valid_motion_pair=matched&~valid)
            assert sum(masks[g].sum() for g in groups[1:])==matched.sum()
            raw_errors=errors(methods['F0']/name,records,target)
            for group,mask in masks.items():raw_pools[group].extend(raw_errors[mask].tolist())
            for condition,path in methods.items():
                prediction_hashes[condition][name]=digest(path/name)
                error=errors(path/name,records,target)
                for group,mask in masks.items():
                    vals=error[mask];assert np.isfinite(vals).all()
                    pools[condition,group].extend(vals.tolist())
                    by_record.append(dict(condition=condition,filename=name,stratum=group,count=len(vals),
                                          mean_quantized_fixed_error_deg=float(vals.mean()) if len(vals) else None))
    table=[]
    for (condition,group),values in pools.items():
        value=np.asarray(values);base=np.asarray(raw_pools[group])
        assert len(value)==len(base)
        table.append(dict(condition=condition,stratum=group,count=len(value),
             mean_quantized_fixed_error_deg=float(value.mean()),median_error_deg=float(np.median(value)),
             p90_error_deg=float(np.quantile(value,.9)),mean_delta_vs_F0_deg=float((value-base).mean()),
             newly_within20=int(((base>20)&(value<=20)).sum()),newly_outside20=int(((base<=20)&(value>20)).sum())))
    save_json(a.output/'summary.json',table);save_json(a.output/'per_recording.json',by_record)
    save_json(a.output/'provenance.json',dict(prediction_hashes=prediction_hashes,cache_manifest_sha256=digest(cache/'file_manifest.json'),
              scope='validation; fixed raw-C0 class Hungarian mapping and final quantized CSV directions',
              pair_strata='frame displacement masks, not whole-source static/dynamic categories',
              claim_limits='Descriptive single-seed validation diagnostics after validation selection; not official LE_CD or causal attribution; no independent confirmatory uncertainty',
              script_sha256=digest(__file__)))
    save_json(a.output/'COMPLETED.json',dict(status='PASS',conditions=list(methods),strata=list(groups),recordings=100))
    print(json.dumps(table))


if __name__=='__main__':main()
