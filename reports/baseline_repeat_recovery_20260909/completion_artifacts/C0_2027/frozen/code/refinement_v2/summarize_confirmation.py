"""Descriptive head-seed summaries and counter QA, without any resampling."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment_core import digest,save_json
from review_results import scalar_counts,score,METRICS,PAIRS


def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(values=x.tolist(),mean=float(x.mean()),sample_SD=float(x.std(ddof=1)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert json.loads((a.evaluation/'COMPLETED.json').read_text())['status']=='PASS'
    summary=json.loads((a.evaluation/'SUMMARY.json').read_text());rows=summary['results']
    by_id={r['id']:r for r in rows};assert len(by_id)==15
    for row in rows:
        records=json.loads((a.evaluation/row['id']/'per_recording.json').read_text())
        assert len(records)==200
        recomputed=score(scalar_counts(records).sum(0))
        assert np.allclose(recomputed,[row['scores']['dcase2023_micro'][m] for m in METRICS],atol=1e-12,rtol=0)
    groups={c:{m:stats([by_id['head%d_%s'%(s,c)]['scores']['dcase2023_micro'][m] for s in (2026,2027,2028)])
               for m in METRICS} for c in ('F-Deriv','R0','R1','R2')}
    pairs={}
    for left,right in PAIRS:
        pairs[left+'-'+right]={}
        for m in METRICS:
            deltas=[]
            for seed in (2026,2027,2028):
                l=by_id['head%d_%s'%(seed,left)]['scores']['dcase2023_micro'][m]
                r=by_id['F0' if right=='F0' else 'head%d_%s'%(seed,right)]['scores']['dcase2023_micro'][m]
                deltas.append(l-r)
            pairs[left+'-'+right][m]=stats(deltas)
    result=dict(status='PASS',counter_reaggregation_all15=True,head_seed_groups=groups,paired_differences=pairs,
        controls={c:by_id[c]['scores']['dcase2023_micro'] for c in ('F0','F-EMA','F-KF')},
        scope='one C0, head-seed descriptive sample SD only; no bootstrap, p-values, CI or cross-C0 uncertainty',
        evaluation_summary_sha256=digest(a.evaluation/'SUMMARY.json'),script_sha256=digest(__file__))
    save_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
