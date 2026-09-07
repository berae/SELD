"""Read-only source analysis; independently aggregate counters, paired resampling."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from experiment_core import save_json,digest


def scalar_counts(rows):
    keys=['_S','_D','_I','_Nref','_TP','_FP_spatial','_FP','_FN','_total_DE','_DE_TP','_DE_FN']
    return np.array([[np.asarray(r['counters']['2023'][k]).sum() for k in keys] for r in rows],float)


def score(x):
    x=np.asarray(x);s,d,i,n,tp,fps,fp,fn,de,dtp,dfn=np.moveaxis(x,-1,0)
    eps=np.finfo(float).eps
    er=(s+d+i)/(n+eps);f=tp/(eps+tp+fps+.5*(fp+fn))
    le=np.divide(de,dtp+eps,out=np.full_like(de,180.),where=dtp>0);lr=dtp/(eps+dtp+dfn)
    return np.stack((er,f,le,lr,(er+1-f+le/180+1-lr)/4),axis=-1)


METRICS=['ER20','F20','LE_CD','LR_CD','SELD_LR']
PAIRS=[('R0','F0'),('R0','F-Deriv'),('R1','R0'),('R2','R1')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    paths={c:a.root/'baselines'/c for c in ('F0','F-EMA','F-KF')}
    paths.update({c:a.root/'pilot'/c for c in ('F-Deriv','R0','R1','R2')})
    sources={};data={};complete={};curves={}
    for c,path in paths.items():
        for f in ('COMPLETED.json','validation/per_recording.json'):
            sources[str(path/f)]=digest(path/f)
        complete[c]=json.loads((path/'COMPLETED.json').read_text());assert complete[c]['status']=='PASS'
        data[c]=sorted(json.loads((path/'validation/per_recording.json').read_text()),key=lambda x:x['filename'])
        assert len(data[c])==100 and len({x['filename'] for x in data[c]})==100
        if c in ('F-Deriv','R0','R1','R2'):
            h=[json.loads(line) for line in (path/'epochs.jsonl').read_text().splitlines()]
            sources[str(path/'epochs.jsonl')]=digest(path/'epochs.jsonl')
            best=min(h,key=lambda x:(x['scores']['dcase2023_micro']['SELD_LR'],x['epoch']))
            assert best['epoch']==complete[c]['best_epoch'] and len(h)==complete[c]['epochs']
            curves[c]=dict(best=best,last=h[-1],first=h[0],epochs=len(h),
                at_common_epochs={str(e):h[e-1] for e in (20,40,60) if len(h)>=e},
                loss_change_best_to_last=h[-1]['loss']/best['loss']-1,
                validation_SELD_change_best_to_last=h[-1]['scores']['dcase2023_micro']['SELD_LR']-best['scores']['dcase2023_micro']['SELD_LR'])
    names=[r['filename'] for r in data['F0']]
    arrays={c:scalar_counts(rows) for c,rows in data.items()}
    totals={};groups={};classwise={};paired=[]
    rng=np.random.default_rng(91082026);weights=rng.multinomial(100,np.full(100,.01),size=20000)
    boot={c:score(weights@arr) for c,arr in arrays.items()}
    for c,arr in arrays.items():
        assert names==[r['filename'] for r in data[c]]
        assert [r['records'] for r in data[c]]==[r['records'] for r in data['F0']]
        assert complete[c]['result']['sed']==complete['F0']['result']['sed']
        total=score(arr.sum(0));expected=complete[c]['result']['scores']['dcase2023_micro']
        assert np.allclose(total,[expected[k] for k in METRICS],atol=1e-12,rtol=0),(c,total,expected)
        totals[c]=dict(zip(METRICS,total.tolist()))
        for axis,pattern in [('overlap',r'_ov(\d+)'),('room',r'_room(\d+)')]:
            labels=[re.search(pattern,n).group(1) for n in names]
            for label in sorted(set(labels)):
                mask=np.array(labels)==label
                groups.setdefault(axis+label,{})[c]=dict(recordings=int(mask.sum()),scores=dict(zip(METRICS,score(arr[mask].sum(0)).tolist())))
        counts=complete[c]['result']['counters']['2023'];dtp=np.asarray(counts['_DE_TP']);de=np.asarray(counts['_total_DE'])
        classwise[c]=[dict(class_id=k,DE_TP=int(dtp[k]),LE_CD=float(de[k]/dtp[k]) if dtp[k]>0 else None) for k in range(14)]
    for c,b in PAIRS:
        point=np.array(list(totals[c].values()))-np.array(list(totals[b].values()))
        delta=boot[c]-boot[b];ci=np.quantile(delta,[.025,.975],axis=0)
        per=score(arrays[c])-score(arrays[b])
        paired.append(dict(comparison=c+'-'+b,point=dict(zip(METRICS,point.tolist())),
            conditional_recording_resampling_interval95={k:ci[:,i].tolist() for i,k in enumerate(METRICS)},
            recording_LE_improved=int((per[:,2]<-1e-10).sum()),recording_LE_worse=int((per[:,2]>1e-10).sum()),
            recording_SELD_improved=int((per[:,4]<-1e-10).sum()),recording_SELD_worse=int((per[:,4]>1e-10).sum()),
            mean_recording_LE_difference=float(per[:,2].mean())))
    fixed=json.loads((a.root/'fixed_target_diagnostics/per_recording.json').read_text())
    fixed_stats=[]
    for c,b in PAIRS:
        for group in ('all_fixed_matched','static_pair','moving_pair','matched_without_valid_motion_pair'):
            left=sorted([r for r in fixed if r['condition']==c and r['stratum']==group],key=lambda r:r['filename'])
            right=sorted([r for r in fixed if r['condition']==b and r['stratum']==group],key=lambda r:r['filename'])
            assert [r['filename'] for r in left]==names and [r['filename'] for r in right]==names
            assert [r['count'] for r in left]==[r['count'] for r in right]
            cnt=np.array([r['count'] for r in left]);diff=np.array([(l['mean_quantized_fixed_error_deg'] or 0)-(r['mean_quantized_fixed_error_deg'] or 0) for l,r in zip(left,right)])
            draws=(weights@(diff*cnt))/(weights@cnt)
            fixed_stats.append(dict(comparison=c+'-'+b,stratum=group,count=int(cnt.sum()),
                 weighted_mean_difference=float((diff*cnt).sum()/cnt.sum()),
                 interval95=np.quantile(draws,[.025,.975]).tolist(),
                 improved_recordings=int(((diff<-1e-10)&(cnt>0)).sum()),worse_recordings=int(((diff>1e-10)&(cnt>0)).sum())))
    save_json(a.output/'analysis.json',dict(totals=totals,curves=curves,groups=groups,classwise=classwise,paired=paired,fixed_target_paired=fixed_stats))
    save_json(a.output/'provenance.json',dict(sources=sources,script_sha256=digest(__file__),resampling_seed=91082026,resamples=20000,
        limits='Post-selection descriptive paired recording resampling, conditional on fitted models; ignores training/selection uncertainty and possible shared room/source dependence; not independent test, no p-values, no multiplicity correction, no confirmatory significance.'))
    save_json(a.output/'COMPLETED.json',dict(status='PASS',recordings=100,counter_scores_reconstructed=True,all_detection_records_equal=True))
    print(json.dumps(dict(paired=paired,groups=groups)))


if __name__=='__main__':main()
