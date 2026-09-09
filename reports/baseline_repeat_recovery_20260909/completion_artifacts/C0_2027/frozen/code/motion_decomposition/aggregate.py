"""Paired seeds and common-target decomposition; blank means unavailable."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from analyze import write_csv,summarize
from export import save


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def key(r):return r['filename'],r['frame'],r['class_id'],r['source_id']
def num(x):return float(x) if x not in ('',None) else np.nan


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    metrics=[];strata=[];match={};diag=[];latent=[];boundary=[];dispersion=[]
    for path in sorted((a.root/'analysis').glob('*/*/metrics_per_run.csv')):
        rs=read(path);metrics.extend(rs);strata.extend(read(path.parent/'motion_strata.csv'))
        first=rs[0];match[(first['variant'],first['seed'],first['split'])]={key(r):r for r in read(path.parent/'matches.csv')}
    for path in sorted((a.root/'diagnostics').glob('*/*/velocity_diagnostics.csv')):
        diag.extend(read(path));latent.extend(read(path.parent/'latent_diagnostics.csv'))
        boundary.append(json.loads((path.parent/'chunk_boundary.json').read_text()))
        dispersion.append(dict(run=path.parent.parent.name,split=path.parent.name,**json.loads((path.parent/'latent_dispersion.json').read_text())))
    write_csv(a.output/'metrics_per_run.csv',metrics);write_csv(a.output/'motion_strata.csv',strata)
    write_csv(a.output/'velocity_diagnostics.csv',diag);write_csv(a.output/'latent_diagnostics.csv',latent)
    save(a.output/'chunk_boundaries.json',boundary);save(a.output/'latent_dispersion.json',dispersion)
    pairs=[('C0','raw','C1','raw'),('C0','raw','C0','smoothing'),('C1','raw','C1','smoothing'),
        ('C1','smoothing','C1','fusion'),('C1','raw','C1','fusion'),('C1','raw','C3_j005','raw'),
        ('C1','smoothing','C3_j005','smoothing'),('C1','fusion','C3_j005','fusion'),
        ('C3_j005','smoothing','C3_j005','fusion')]
    metric_index={(r['variant'],r['seed'],r['split'],r['decoder'],r['profile']):r for r in metrics}
    common=[];deltas=[]
    for bv,bd,v,d in pairs:
        for seed in ('2026','2027','2028'):
            for split in ('validation','evaluation'):
                left=match.get((bv,seed,split));right=match.get((v,seed,split))
                if left is None or right is None:continue
                assert left.keys()==right.keys()
                if bd not in next(iter(left.values())) or d not in next(iter(right.values())):continue
                keys=sorted(left);x=np.array([num(left[k][bd]) for k in keys]);y=np.array([num(right[k][d]) for k in keys])
                base=dict(baseline=bv+'/'+bd,variant=v+'/'+d,seed=seed,split=split)
                for group in ('all','static','dynamic','same_class','different_class','boundary','turn','jump'):
                    mask=np.array([group=='all' or left[k]['motion_group']==group or left[k]['overlap']==group or left[k].get(group)=='True' for k in keys])
                    xx=x[mask]; yy=y[mask];both=np.isfinite(xx)&np.isfinite(yy)
                    lost=np.isfinite(xx)&~np.isfinite(yy);new=~np.isfinite(xx)&np.isfinite(yy);neither=~np.isfinite(xx)&~np.isfinite(yy)
                    def mean(z):return float(np.mean(z)) if len(z) else None
                    common.append(dict(**base,stratum=group,gt_denominator=len(xx),both_matched=int(both.sum()),baseline_only=int(lost.sum()),variant_only=int(new.sum()),neither=int(neither.sum()),
                        baseline_matched=int(np.isfinite(xx).sum()),variant_matched=int(np.isfinite(yy).sum()),
                        baseline_recall=float(np.isfinite(xx).mean()) if len(xx) else None,variant_recall=float(np.isfinite(yy).mean()) if len(xx) else None,
                        common_baseline_LE=mean(xx[both]),common_variant_LE=mean(yy[both]),common_delta_LE=mean(yy[both]-xx[both]),
                        lost_baseline_LE=mean(xx[lost]),lost_baseline_p90=float(np.quantile(xx[lost],.9)) if lost.any() else None,
                        new_variant_LE=mean(yy[new]),new_variant_p90=float(np.quantile(yy[new],.9)) if new.any() else None))
                for profile in ('dcase2023_micro','dcase2023_macro'):
                    m0=metric_index[(bv,seed,split,bd,profile)];m1=metric_index[(v,seed,split,d,profile)]
                    deltas.append(dict(**base,profile=profile,**{'delta_'+k:num(m1[k])-num(m0[k]) for k in ('LE_CD','LR_CD','F20','ER20','SELD_LR','sed_micro_F1','sed_macro_F1','frame_LE','matched_recall','recall20')}))
    write_csv(a.output/'common_match_breakdown.csv',common);write_csv(a.output/'paired_deltas.csv',deltas)
    aggregate=[]
    for source,rows,groups,fields in [('metrics',metrics,['variant','split','decoder','profile'],['LE_CD','LR_CD','F20','ER20','SELD_LR','frame_LE','matched_recall','recall20']),
        ('paired',deltas,['baseline','variant','split','profile'],[k for k in deltas[0] if k.startswith('delta_')])]:
        buckets=defaultdict(list)
        for r in rows:buckets[tuple(r[k] for k in groups)].append(r)
        for k,rs in buckets.items():
            for field in fields:
                vals=np.array([num(r[field]) for r in rs]);vals=vals[np.isfinite(vals)]
                aggregate.append(dict(table=source,**dict(zip(groups,k)),metric=field,n_seeds=len(vals),mean=float(vals.mean()) if len(vals) else None,sample_SD=float(vals.std(ddof=1)) if len(vals)>1 else None))
    write_csv(a.output/'seed_summary.csv',aggregate)
    save(a.output/'coverage.json',dict(primary_metrics_rows=sum(r['variant'] in ('C0','C1','C3_j005') for r in metrics),all_metrics_rows=len(metrics),common_match_rows=len(common),diagnostic_rows=len(diag)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
