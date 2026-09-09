"""Predeclared diagnostic examples plus explicitly selected failure examples."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aggregate import read,num
from analyze import write_csv

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);chosen=[]
for seed in ('2026','2027','2028'):
    rows=read(a.root/'analysis'/('C1_'+seed)/'evaluation/matches.csv')
    segments=defaultdict(list)
    for r in rows:segments[(r['filename'],r['segment_id'])].append(r)
    selected=[]
    if seed=='2026':
        for group in ('static','dynamic','same_class','turn'):
            eligible=[k for k,rs in segments.items() if len(rs)>=20 and any(r['motion_group']==group or r['overlap']==group or r.get(group)=='True' for r in rs)]
            if eligible:selected.append((group,min(eligible),'lexicographic predeclared eligibility'))
    rec=read(a.root/'analysis'/('C1_'+seed)/'evaluation/recording_metrics.csv');record=defaultdict(dict)
    for r in rec:record[r['filename']][r['decoder']]=num(r['frame_LE'])
    worst=max(record,key=lambda f:record[f]['fusion']-record[f]['smoothing'])
    eligible=[k for k,rs in segments.items() if k[0]==worst and len(rs)>=20]
    if eligible:
        def segment_delta(k):
            values=[num(r['fusion'])-num(r['smoothing']) for r in segments[k]]
            finite=[v for v in values if np.isfinite(v)]
            return float(np.mean(finite)) if finite else -float('inf')
        selected.append(('failure',max(sorted(eligible),key=segment_delta),'largest recording mean LE fusion-minus-smoothing; worst eligible source within recording, not representative'))
    for group,k,rule in selected:
        rs=sorted(segments[k],key=lambda r:int(r['frame']));t=np.array([int(r['frame']) for r in rs])*.1
        fig,ax=plt.subplots(figsize=(9,3.3))
        for mode,color in [('raw','#64748b'),('smoothing','#0284c7'),('fusion','#d97706')]:
            ax.plot(t,[num(r[mode]) for r in rs],label=mode,color=color,linewidth=1.3)
        for r in rs:
            if r['turn']=='True':ax.axvline(int(r['frame'])*.1,color='#dc2626',alpha=.18,linewidth=.8)
        ax.set(xlabel='Time (s)',ylabel='Matched angular error (deg)',title=f'C1 seed {seed} | {group} | {k[0]} | source {rs[0]["source_id"]}')
        ax.grid(alpha=.2);ax.legend(ncol=3,loc='upper right');fig.tight_layout()
        name=f'C1_{seed}_{group}.png';fig.savefig(a.output/name,dpi=160);plt.close(fig)
        chosen.append(dict(seed=seed,category=group,filename=k[0],segment=k[1],selection_rule=rule,figure=name,frames=len(rs),matched_fusion=sum(bool(r['fusion']) for r in rs)))
write_csv(a.output/'case_selection.csv',chosen)
