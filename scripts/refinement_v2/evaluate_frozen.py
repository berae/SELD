"""Evaluate every preregistered frozen head; no optimizer or selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiment_core import digest, save_json, make_head, head_forward
from train_heads import CachedSplit, predict, evaluate
from diagnose_fixed_targets import errors


def verify(root, manifest):
    for relative, expected in manifest['files_sha256'].items():
        assert digest(root/relative)==expected, relative


def diagnostics(data, methods, output, motion_source):
    spec=importlib.util.spec_from_file_location('frozen_motion_coverage',motion_source)
    motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
    counts,by_class,rows=motion.audit_split('evaluation',[data.path/'gt'/n for n in data.names],1e-6)
    assert counts['duplicate_source_frames']==counts['malformed_rows']==0
    labels={(r['metadata_file'],r['frame'],r['class_id'],r['source_id']):r['motion_group'] for r in rows}
    groups=('all_fixed_matched','static_pair','moving_pair','matched_without_valid_motion_pair',
            'source_static','source_dynamic','source_unobservable')
    pools={(c,g):[] for c in methods for g in groups};per_record=[];denominators=[]
    for i,name in enumerate(data.names):
        with np.load(data.path/'targets'/(Path(name).stem+'.npz')) as targets:
            matched=targets['matched'];valid=targets['motion_valid']
            moving=np.linalg.norm(targets['displacement'],axis=-1)>1e-7
            masks=dict(all_fixed_matched=matched,static_pair=valid&~moving,moving_pair=valid&moving,
                       matched_without_valid_motion_pair=matched&~valid)
            for group in ('static','dynamic','unobservable'):
                mask=np.zeros_like(matched)
                for t,s in zip(*np.where(matched)):
                    cls,identity=targets['gt_identity'][t,s]
                    mask[t,s]=labels[name,int(t),int(cls),int(identity)]==group
                masks['source_'+group]=mask
            assert sum(masks['source_'+g].sum() for g in ('static','dynamic','unobservable'))==matched.sum()
            denominators.append(dict(filename=name,**{g:int(m.sum()) for g,m in masks.items()}))
            for condition,path in methods.items():
                err=errors(path/name,data.records[i],targets)
                for group,mask in masks.items():
                    values=err[mask];assert np.isfinite(values).all()
                    pools[condition,group].extend(values.tolist())
                    per_record.append(dict(condition=condition,filename=name,stratum=group,count=len(values),
                        mean_error_deg=float(values.mean()) if len(values) else None))
    table=[]
    for (condition,group),values in pools.items():
        val=np.asarray(values);base=np.asarray(pools['F0',group]);assert len(val)==len(base)
        table.append(dict(condition=condition,stratum=group,count=len(val),
            mean_error_deg=float(val.mean()) if len(val) else None,
            mean_delta_vs_F0_deg=float((val-base).mean()) if len(val) else None,
            newly_within20=int(((base>20)&(val<=20)).sum()),newly_outside20=int(((base<=20)&(val>20)).sum())))
    save_json(output/'fixed_matching_diagnostics.json',dict(summary=table,per_recording=per_record,
        denominators=denominators,all_GT_source_coverage=counts,GT_by_class=by_class,
        fixed_target_audit=json.loads((data.path/'target_audit.json').read_text()),
        scope='Final quantized CSV; raw-C0 fixed assignment; source strata are whole contiguous GT segments; pair strata are separate frame-pair diagnostics; no re-matching or bootstrap',
        motion_source_sha256=digest(motion_source)))


def main():
    p=argparse.ArgumentParser()
    for k in ('frozen','cache','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();start=time.time();torch.set_num_threads(2)
    seal=json.loads((a.frozen/'FROZEN.json').read_text());verify(a.frozen,seal)
    assert seal['evaluation_selection_prohibited'] and len(seal['runs'])==12
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'STARTED.json',dict(start_unix=start,freeze_sha256=digest(a.frozen/'FROZEN.json'),
        registry_sha256=seal['preregistration_sha256'],new_training_runs=0,historical_evaluation_seen=True))
    sys.path.insert(0,str(a.frozen/'scorer'))
    data=CachedSplit(a.cache);assert data.manifest['split']=='evaluation'
    assert data.manifest['checkpoint_sha256']==seal['C0_sha256']
    for relative,expected in data.manifest['source_hashes'].items():
        if relative.startswith('evaluation/'):
            assert digest(a.frozen/'scorer'/Path(relative).relative_to('evaluation'))==expected
    unchanged={k:v.clone() for k,v in data.data.items() if k in ('doa','probability','mapping')}
    conditions=[dict(condition=c,head_seed=None) for c in seal['fixed_controls']]+seal['runs']
    results=[];methods={};reference=None
    for entry in conditions:
        c=entry['condition'];seed=entry['head_seed'];key=c if seed is None else 'head%d_%s'%(seed,c)
        head=None;prefix=None
        if seed is not None:
            ck=torch.load(a.frozen/entry['weight_path'],map_location='cpu',weights_only=False)
            assert ck['seed']==seed and ck['epoch']==entry['selected_epoch'] and ck['condition']==c
            assert ck['checkpoint_sha256']==seal['C0_sha256']
            head=make_head(c);head.load_state_dict(ck['head']);head.eval().requires_grad_(False)
            with torch.no_grad():
                h=data.data['doa_features'][:2];raw=data.data['doa'][:2];maps=data.data['mapping'][:2]
                original=head_forward(head,h,raw,maps,c)
                h2=h.clone();h2[:,20:]+=10;raw2=raw.clone();raw2[:,20:]*=-1
                changed=head_forward(head,h2,raw2,maps,c)
                prefix=float((original[:,:20]-changed[:,:20]).abs().max());assert prefix==0
        prediction=predict(head,data,c,'cpu')
        path=a.output/key
        result=evaluate(data,prediction,path)
        if reference is None:reference=result['sed']
        assert result['sed']==reference
        for field,value in unchanged.items():assert torch.equal(data.data[field],value),field
        if c=='F0':
            assert all((path/n).read_bytes()==data.raw_csv[i] for i,n in enumerate(data.names))
        row=dict(id=key,condition=c,head_seed=seed,C0_seed=2026,weight_sha256=entry.get('weight_sha256'),
            selected_validation_epoch=entry.get('selected_epoch'),**result,
            detection_preservation=dict(ordered_frame_class_slot_records_exact=True,pure_SED_counts_exact=True,
                probability_raw_doa_mapping_unchanged=True),head_prefix_max_delta=prefix,
            prediction_sha256={n:digest(path/n) for n in data.names})
        results.append(row);methods[key]=path;save_json(path/'PROVENANCE.json',row)
        print(json.dumps(dict(stage='evaluated',id=key,completed=len(results),total=15)),flush=True)
    diagnostics(data,methods,a.output,a.frozen/'code/motion_decomposition/analyze_motion_coverage.py')
    verify(a.frozen,seal)
    save_json(a.output/'SUMMARY.json',dict(results=results,freeze_sha256=digest(a.frozen/'FROZEN.json'),
        cache_manifest_sha256=digest(a.cache/'file_manifest.json'),input_manifest_sha256=digest(a.cache/'input_manifest.json'),
        historical_evaluation_seen=True,new_blind_test=False,selection_on_evaluation=False,
        new_training_runs=0,bootstrap_repeated=False,scope='fixed C0 seed2026, three head seeds, not cross-C0 stability'))
    save_json(a.output/'COMPLETED.json',dict(status='PASS',conditions=len(results),files=len(data.names),
        elapsed_seconds=time.time()-start,all_frozen_hashes_unchanged=True,all_detection_checks_pass=True))
    print(json.dumps(dict(status='PASS',conditions=len(results),seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
