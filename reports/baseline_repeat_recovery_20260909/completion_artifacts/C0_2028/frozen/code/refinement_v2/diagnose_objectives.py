"""Read-only best-head train/validation objectives and local gradient alignment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_heads import CachedSplit,predict
from candidate import correct_direction
from experiment_core import make_head,head_forward,previous_tensor,masked_mean,loss_terms,digest,save_json


def components(head,b,c):
    delta=head_forward(head,b['doa_features'],b['doa'],b['mapping'],c)
    u=torch.nn.functional.normalize(correct_direction(b['doa'],delta,b['probability'].amax(-1)>.5),dim=-1,eps=1e-12)
    direction=masked_mean(1-(u*b['target']).sum(-1),b['matched'])
    motion=masked_mean(((u-previous_tensor(u,b['mapping']))-b['displacement']).square().mean(-1),b['motion_valid'])
    return direction,motion


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['both','train','validation'],default='both')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    results=[];gradients=[]
    for split in (('train','validation') if a.split=='both' else (a.split,)):
        data=CachedSplit(a.root/'cache'/split)
        for c in ('F0','F-EMA','F-KF','F-Deriv','R0','R1','R2'):
            learned=c in ('F-Deriv','R0','R1','R2')
            head=make_head(c if learned else 'R0')
            if learned:
                checkpoint=a.root/'pilot'/c/'best.pth';done=json.loads((checkpoint.parent/'COMPLETED.json').read_text())
                assert digest(checkpoint)==done['best_checkpoint_sha256']
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False);head.load_state_dict(saved['head'])
            head.eval();sums={};nb=0
            with torch.no_grad():
                for start in range(0,data.n,32):
                    b=data.batch(torch.arange(start,min(start+32,data.n)),'cpu')
                    loss,terms=loss_terms(head,b,c if learned else 'R0')
                    for key,val in dict(loss=loss,**terms).items():sums[key]=sums.get(key,0)+float(val)
                    nb+=1
                out=torch.from_numpy(predict(head if learned else None,data,c,'cpu').reshape(data.n,40,2,3)).double()
                u=torch.nn.functional.normalize(out,dim=-1,eps=1e-12)
                raw=torch.nn.functional.normalize(data.data['doa'].double(),dim=-1,eps=1e-12)
                target=torch.nn.functional.normalize(data.data['target'].double(),dim=-1,eps=1e-12)
                # atan2 is stable at identical directions; float32 acos(dot) gave
                # a spurious ~0.008-degree correction even for the F0 identity.
                err=torch.rad2deg(torch.atan2(torch.linalg.cross(u,target).norm(dim=-1),(u*target).sum(-1)))
                correction=torch.rad2deg(torch.atan2(torch.linalg.cross(u,raw).norm(dim=-1),(u*raw).sum(-1)))
                if c=='F0':assert float(correction.abs().max())<1e-10
                motion=((u-previous_tensor(u,data.data['mapping']))-data.data['displacement']).square().mean(-1)
                valid=data.data['motion_valid'];moving=data.data['displacement'].norm(dim=-1)>1e-7
                masks=dict(matched=data.data['matched'],static_pair=valid&~moving,moving_pair=valid&moving,
                           unmatched=(data.data['probability'].amax(-1)>.5)&~data.data['matched'])
                stats={name:dict(count=int(mask.sum()),float_angular_error_deg=float(err[mask].mean()) if name!='unmatched' else None,
                     mean_correction_deg=float(correction[mask].mean()),p95_correction_deg=float(torch.quantile(correction[mask],.95)),
                     correction_above14deg_fraction=float((correction[mask]>14).float().mean()),
                     displacement_MSE=float(motion[mask].mean()) if name in ('static_pair','moving_pair') else None) for name,mask in masks.items()}
            results.append(dict(split=split,condition=c,unshuffled_batch_mean=None if c in ('F-EMA','F-KF') else {k:v/nb for k,v in sums.items() if k not in ('matched','motion_pairs','unmatched')},strata=stats))
            if c=='R2':
                rng=np.random.default_rng(91082026);ids=torch.from_numpy(rng.choice(data.n,64,replace=False));b=data.batch(ids,'cpu')
                d,m=components(head,b,c);gd=torch.autograd.grad(d,head.parameters(),retain_graph=True);gm=torch.autograd.grad(m,head.parameters())
                gd=torch.cat([x.flatten() for x in gd]);gm=torch.cat([x.flatten() for x in gm])
                gradients.append(dict(split=split,condition=c,chunks=64,sampling_seed=91082026,
                    direction_loss=float(d.detach()),motion_loss=float(m.detach()),direction_grad_norm=float(gd.norm()),motion_grad_norm=float(gm.norm()),
                    grad_cosine=float((gd@gm)/(gd.norm()*gm.norm()).clamp_min(1e-12))))
            print(json.dumps(dict(split=split,condition=c,status='completed')),flush=True)
    save_json(a.output/'objectives.json',results);save_json(a.output/'gradients.json',gradients)
    save_json(a.output/'COMPLETED.json',dict(status='PASS',optimizer_updates=0,backbone_loaded=False,
        scope='Original head seed2026 selected best checkpoints only; float directions separate from quantized official metrics; gradients are local descriptive diagnostics, not proof of optimization cause',script_sha256=digest(__file__)))


if __name__=='__main__':main()
