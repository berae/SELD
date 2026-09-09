"""GT/PIT is used only for offline diagnostic scoring, never main decoding."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.optimize import linear_sum_assignment
from analyze import write_csv,polar
from export import save,digest


def canonical_numpy(z,sed,doa,beta=.5,prefix=''):
    logits=z[prefix+'sed']; pred=z[prefix+'doa']
    costs=[]
    for flip in (False,True):
        s=sed[:,::-1] if flip else sed;d=doa[:,::-1] if flip else doa
        bce=(np.maximum(logits,0)-logits*s+np.log1p(np.exp(-np.abs(logits)))).mean((-1,-2))
        costs.append(beta*bce+(1-beta)*((pred-d)**2).mean((-1,-2)))
    original=costs[0]<=costs[1]
    def canon(x):
        return np.where(original.reshape((-1,)+(1,)*(x.ndim-1)),x,x[:,::-1])
    return canon


def identities(path,sed,doa):
    ids=np.full((600,2,2),-1,int)
    frames=defaultdict(list)
    for row in np.loadtxt(path,delimiter=',',ndmin=2):frames[int(row[0])].append(row)
    for frame,rs in frames.items():
        costs=np.array([[np.linalg.norm(doa[frame,s]-polar(r[3],r[4])) if sed[frame,s,int(r[1])]>.5 else 1e6 for s in range(2)] for r in rs])
        ii,jj=linear_sum_assignment(costs)
        assert len(ii)==len(rs) and max(costs[ii,jj],default=0)<1e-4
        for i,j in zip(ii,jj):ids[frame,j]=int(rs[i][1]),int(rs[i][2])
    return ids


def quant(x,p):return float(np.quantile(x,p)) if len(x) else None


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.export/'input_manifest.json').read_text()); cfg=manifest['config'];split=manifest['split']
    base=dict(variant=a.variant,seed=cfg['audit']['seed'],split=split)
    meta=Path(cfg['hdf5_dir'])/cfg['dataset']/'meta'/('dev' if split=='validation' else 'eval')
    ref=Path(cfg['dataset_dir'])/('metadata_dev' if split=='validation' else 'metadata_eval')
    vel=defaultdict(list);vnorm=defaultdict(list);gtnorm=defaultdict(list);pairs=0;active_count=0;identity_adjacent=0
    boundary_valid=0;boundary_moving=0;boundary_possible=0;boundary_errors=[];mask_checked=0
    le=defaultdict(list); latents=[]; online_latents=[]; cos_current=[]; input_hashes={}
    for name in sorted(manifest['gt_files']):
        stem=Path(name).stem; z=np.load(a.export/'float'/(stem+'.npz'))
        path=meta/(stem+'.h5');input_hashes[str(path)]=digest(path)
        with h5py.File(path,'r') as f:sed=f['sed_label'][:];doa=f['doa_label'][:]
        ids=identities(ref/name,sed,doa);active=ids[:,:,0]>=0
        canon=canonical_numpy(z,sed,doa,cfg['training']['loss_beta']);v=canon(z['velocity']);p=canon(z['doa'])
        valid=np.zeros((600,2),bool);valid[1:]=active[1:] & (ids[1:]==ids[:-1]).all(-1)
        tv=np.zeros_like(doa);tv[1:]=(doa[1:]-doa[:-1])/.1
        active_count+=int(active.sum());identity_adjacent+=int(valid.sum())
        cached=Path(cfg['velocity_hdf5_dir'])/'dev'/(stem+'.h5')
        if split=='validation' and cached.exists():
            with h5py.File(cached,'r') as f:
                np.testing.assert_array_equal(f['velocity_mask'][:]> .5,valid)
                np.testing.assert_allclose(f['velocity_label'][:][valid],tv[valid],atol=1e-5)
            mask_checked+=int(valid.sum());input_hashes[str(cached)]=digest(cached)
        boundary=np.zeros_like(valid);boundary[40::40]=True
        boundary_possible+=int((boundary&active).sum());boundary_valid+=int((boundary&valid).sum())
        boundary_moving+=int((boundary&valid&(np.linalg.norm(tv,axis=-1)>1e-6)).sum())
        boundary_errors.extend(np.linalg.norm(v-tv,axis=-1)[boundary&valid].tolist())
        # Main velocity comparison excludes cross-chunk pairs unavailable to model.
        use=valid&~boundary; pairs+=int(use.sum())
        derivative=np.zeros_like(p);derivative[1:]=(p[1:]-p[:-1])/.1
        speed=np.linalg.norm(tv,axis=-1)
        for group,mask in [('all',use),('static_pair',use&(speed<=1e-6)),('moving_pair',use&(speed>1e-6))]:
            for label,value in [('predicted',v),('zero',np.zeros_like(v)),('raw_doa_difference',derivative)]:
                vel[(group,label)].extend(np.linalg.norm(value-tv,axis=-1)[mask].tolist())
                vnorm[(group,label)].extend(np.linalg.norm(value,axis=-1)[mask].tolist())
            gtnorm[group].extend(speed[mask].tolist())
        if 'teacher_latent' not in z:continue
        tc=canonical_numpy(z,sed,doa,cfg['training']['loss_beta'],'teacher_')
        t=tc(z['teacher_latent']);on=canon(z['latent']);fp=canon(z['future_latent_pred'])
        latents.append(t[active]);online_latents.append(on[active])
        def cosine(x,y):return np.sum(x*y,axis=-1)/(np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1),1e-12))
        cos_current.extend(cosine(on[active],t[active]).tolist())
        for hi,h in enumerate(cfg['training']['jepa_horizons_frames']):
            mask=active[:-h].copy()
            for step in range(1,h+1):mask &= (ids[:-h]==ids[step:600-h+step]).all(-1)
            mask &= (np.arange(600-h)//40 == np.arange(h,600)//40)[:,None]
            target=t[h:][mask]
            le[(h,'predictor')].extend((1-cosine(fp[:-h,:,hi][mask],target)).tolist())
            le[(h,'teacher_persistence')].extend((1-cosine(t[:-h][mask],target)).tolist())
            le[(h,'online_persistence')].extend((1-cosine(on[:-h][mask],target)).tolist())
    rows=[]
    for (group,model),errors in vel.items():
        norms=vnorm[(group,model)]
        rows.append(dict(**base,stratum=group,estimate=model,valid_pairs=len(errors),all_active_gt=active_count,
            identity_adjacent_pairs=identity_adjacent,within_chunk_pairs=pairs,pair_coverage=pairs/active_count,
            mean_vector_l2_error=float(np.mean(errors)) if errors else None,error_p50=quant(errors,.5),error_p90=quant(errors,.9),
            norm_p50=quant(norms,.5),norm_p90=quant(norms,.9),norm_p99=quant(norms,.99),gt_norm_p50=quant(gtnorm[group],.5),gt_norm_p99=quant(gtnorm[group],.99)))
    write_csv(a.output/'velocity_diagnostics.csv',rows)
    save(a.output/'chunk_boundary.json',dict(**base,boundary_active_targets=boundary_possible,boundary_valid_targets=boundary_valid,
        boundary_moving_targets=boundary_moving,boundary_mean_vector_error=float(np.mean(boundary_errors)) if boundary_errors else None,
        cached_velocity_mask_verified_pairs=mask_checked,interpretation='Targets at chunk first frame can reference previous chunk; observed property, not demonstrated causal failure. No supervision changed.'))
    latent_rows=[];stats={}
    for layer,values in [('teacher',latents),('online',online_latents)]:
        if not values:continue
        x=np.concatenate(values).astype(np.float64)
        cov=np.cov(x,rowvar=False);eig=np.maximum(np.linalg.eigvalsh(cov),0);weights=eig/max(eig.sum(),1e-30)
        # Deterministic nonadjacent pseudo-pairing across the concatenated recordings.
        paired=np.sum(x*np.roll(x,len(x)//2,axis=0),axis=-1)/(np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(np.roll(x,len(x)//2,axis=0),axis=-1),1e-12))
        stats[layer]=dict(active_samples=len(x),mean_feature_variance=float(np.trace(cov)/len(eig)),effective_rank=float(np.exp(-np.sum(weights*np.log(np.maximum(weights,1e-30))))),
            cross_recording_shift_paired_cosine=float(paired.mean()),online_teacher_current_cosine=float(np.mean(cos_current)))
    for (h,estimate),errors in le.items():latent_rows.append(dict(**base,horizon_ms=h*100,estimate=estimate,valid_pairs=len(errors),cosine_error=float(np.mean(errors)),error_p90=quant(errors,.9)))
    write_csv(a.output/'latent_diagnostics.csv',latent_rows,['variant','seed','split','horizon_ms','estimate','valid_pairs','cosine_error','error_p90'])
    save(a.output/'latent_dispersion.json',stats or {'status':'NOT_APPLICABLE_NO_TRAINED_TEACHER_EXPORT'})
    save(a.output/'diagnostic_inputs.json',input_hashes)
    save(a.output/'COMPLETED.json',dict(velocity_pairs=pairs,latent_available=bool(latent_rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',required=True)
    main(p.parse_args())
