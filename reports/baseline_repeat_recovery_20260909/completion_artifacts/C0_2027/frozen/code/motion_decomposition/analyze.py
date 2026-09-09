"""CPU scoring/diagnostics. GT never enters decoder.py."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from decoder import decode, labels, angles, unit
from export import save, digest


def write_csv(path, rows, fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def polar(az,el):
    a,e=np.deg2rad([az,el]); return np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])


def summarize(errors):
    e=np.asarray(errors,float); ok=np.isfinite(e); v=e[ok]
    return dict(gt_denominator=len(e),matched=int(ok.sum()),within20=int((v<=20).sum()),
                matched_recall=float(ok.mean()) if len(e) else None,
                recall20=float((v<=20).sum()/len(e)) if len(e) else None,
                frame_LE=float(v.mean()) if len(v) else None,
                error_p50=float(np.median(v)) if len(v) else None,error_p90=float(np.quantile(v,.9)) if len(v) else None)


def prepare_gt(reference, names, repository, existing_table):
    path=repository/'models/multi_accdoa/analysis/analyze_motion_coverage.py'
    spec=importlib.util.spec_from_file_location('coverage',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    counts,_,rows=mod.audit_split('analysis',[reference/n for n in names],1e-6)
    by_key={}; by_file=defaultdict(list); segments=defaultdict(list)
    for row in rows:
        key=(row['metadata_file'],int(row['frame']),int(row['class_id']),int(row['source_id']))
        assert key not in by_key
        row['key']=key; row['xyz']=polar(row['azimuth_deg'],row['elevation_deg']); by_key[key]=row
        by_file[key[0]].append(row); segments[(key[0],row['segment_id'])].append(row)
    if existing_table:
        with Path(existing_table).open() as f:
            old={(Path(r['metadata_file']).name,int(r['frame']),int(r['class_id']),int(r['source_id'])):r for r in csv.DictReader(f)}
        assert set(old)==set(by_key), 'Existing motion table coverage mismatch'
        for key,row in by_key.items():
            r=old[key]
            assert r['motion_group']==row['motion_group']
            assert np.allclose([float(r['azimuth_deg']),float(r['elevation_deg'])],[row['azimuth_deg'],row['elevation_deg']])
    for rs in segments.values():
        rs.sort(key=lambda r:r['frame'])
        for i,r in enumerate(rs):
            r['boundary']=i<3 or i>=len(rs)-3
            r['turn']=False; r['jump']=r['angular_step_deg']>=10
            if i>=2:
                a=rs[i-1]['xyz']-rs[i-2]['xyz']; b=r['xyz']-rs[i-1]['xyz']
                if min(np.linalg.norm(a),np.linalg.norm(b))>1e-6:
                    r['turn']=angles(a[None],b[None])[0,0]>=45
    for filename,rs in by_file.items():
        groups=defaultdict(list)
        for r in rs: groups[r['frame']].append(r)
        for frame,items in groups.items():
            for r in items:
                r['overlap']='same_class' if sum(x['class_id']==r['class_id'] for x in items)>1 else 'different_class' if len(items)>1 else 'single'
    return by_file,counts


def matched_rows(gt, lab):
    grouped=defaultdict(list); assigned={}
    for i,r in enumerate(gt): grouped[(r['frame'],r['class_id'])].append(i)
    for (frame,cls),indices in grouped.items():
        pred=[v for v in lab.get(frame,[]) if v[0]==cls]
        if pred:
            cost=angles(np.array([gt[i]['xyz'] for i in indices]),np.array([polar(v[2],v[3]) for v in pred]))
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj): assigned[indices[i]]=(float(cost[i,j]),int(pred[j][1]))
    return np.array([assigned.get(i,(np.nan,-1))[0] for i in range(len(gt))]),np.array([assigned.get(i,(np.nan,-1))[1] for i in range(len(gt))])


def sed_counts(gt,lab):
    g=np.zeros((600,14),bool); p=g.copy()
    for r in gt:g[r['frame'],r['class_id']]=True
    for f,vs in lab.items():
        for v in vs:p[f,v[0]]=True
    return np.stack([(g&p).sum(0),(~g&p).sum(0),(g&~p).sum(0)]),p


def analyze(args):
    args.output.mkdir(parents=True,exist_ok=False)
    assert (args.export/'COMPLETED.json').exists(), 'Incomplete raw export'
    manifest=json.loads((args.export/'input_manifest.json').read_text()); cfg=manifest['config']
    sys.path.insert(0,str(args.repository/'evaluation'))
    from aligned_metrics import AlignedMetrics,load_csv
    regression=json.loads((args.export/'raw_regression.json').read_text())
    assert regression['max_abs_delta'] is None or regression['max_abs_delta']<=1e-6
    names=sorted(manifest['gt_files'])
    reference=Path(cfg['dataset_dir'])/('metadata_dev' if manifest['split']=='validation' else 'metadata_eval')
    gt,counts=prepare_gt(reference,names,args.repository,args.motion_table)
    save(args.output/'gt_qa.json',dict(counts=counts,motion_table_sha256=digest(args.motion_table) if args.motion_table else None,regenerated_and_verified=True))
    modes=['raw'] if args.raw_only else ['raw','smoothing'] if cfg['training']['lambda_velocity']==0 else ['raw','smoothing','fusion']
    metrics={m:AlignedMetrics() for m in modes}; all_errors={m:[] for m in modes}; sed={m:np.zeros((3,14),int) for m in modes}
    strata={m:defaultdict(list) for m in modes}; records=[]; recording=[]; applied_counts=defaultdict(int)
    for filename in names:
        z=np.load(args.export/'float'/(Path(filename).stem+'.npz'))
        prob=z['probability']; xyz=z['doa']; velocity=z['velocity'] if cfg['training']['lambda_velocity'] else np.zeros_like(xyz)
        raw_sed=None; shared=None; rows=gt[filename]
        frame_results={}
        for mode in modes:
            q,assoc,mask=decode(prob,xyz,velocity,z['frame_index'],z['chunk_index'],mode)
            if mode!='raw':
                if shared is not None:np.testing.assert_array_equal(shared,mask)
                shared=mask
            applied_counts[mode]+=int(mask.sum())
            lab=labels(prob,q)
            output_dir=args.output/mode; output_dir.mkdir(exist_ok=True)
            with (output_dir/filename).open('x',newline='') as f:
                w=csv.writer(f)
                for frame,vs in lab.items():
                    for cls,slot,az,el in vs:w.writerow([frame,cls,az,el])
            metrics[mode].update(lab,load_csv(reference/filename,'polar5'))
            sc,activity=sed_counts(rows,lab);sed[mode]+=sc
            if raw_sed is None:raw_sed=activity
            else:np.testing.assert_array_equal(activity,raw_sed)
            errors,slots=matched_rows(rows,lab);frame_results[mode]=errors;all_errors[mode].extend(errors)
            recording.append(dict(filename=filename,decoder=mode,**summarize(errors)))
            for i,row in enumerate(rows):
                groups=['all',row['motion_group'],row['overlap']]
                for flag in ('boundary','turn','jump'):
                    if row[flag]:groups.append(flag)
                for group in groups:strata[mode][group].append(errors[i])
        for i,r in enumerate(rows):
            records.append(dict(filename=filename,frame=r['frame'],class_id=r['class_id'],source_id=r['source_id'],
                segment_id=r['segment_id'],motion_group=r['motion_group'],overlap=r['overlap'],boundary=r['boundary'],turn=r['turn'],jump=r['jump'],
                **{m:float(frame_results[m][i]) if np.isfinite(frame_results[m][i]) else '' for m in modes}))
    results=[]; stratum=[]
    for mode in modes:
        scores=metrics[mode].scores()
        if mode=='raw':
            assert max(abs(scores[p][k]-regression['scores'][p][k]) for p in scores for k in scores[p])<=1e-6
        tp,fp,fn=sed[mode];f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(14,float),where=(2*tp+fp+fn)>0)
        base=dict(variant=args.variant,seed=cfg['audit']['seed'],split=manifest['split'],decoder=mode,files=len(names),
            sed_micro_F1=float(2*tp.sum()/(2*tp+fp+fn).sum()),sed_macro_F1=float(f1.mean()),sed_TP=int(tp.sum()),sed_FP=int(fp.sum()),sed_FN=int(fn.sum()),
            applied_frames=applied_counts[mode],**summarize(all_errors[mode]))
        for profile in ('dcase2023_micro','dcase2023_macro'):results.append(dict(**base,profile=profile,**scores[profile]))
        for group,e in strata[mode].items():stratum.append(dict(variant=args.variant,seed=cfg['audit']['seed'],split=manifest['split'],decoder=mode,stratum=group,**summarize(e)))
        save(args.output/(mode+'_official_counters.json'),metrics[mode].counters())
    write_csv(args.output/'metrics_per_run.csv',results);write_csv(args.output/'motion_strata.csv',stratum)
    write_csv(args.output/'matches.csv',records);write_csv(args.output/'recording_metrics.csv',recording)
    save(args.output/'COMPLETED.json',dict(sed_identical=True,raw_scoring_regression=True,modes=modes,records=len(records)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('export','repository','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--motion-table',type=Path);p.add_argument('--variant',required=True);p.add_argument('--raw-only',action='store_true')
    analyze(p.parse_args())
