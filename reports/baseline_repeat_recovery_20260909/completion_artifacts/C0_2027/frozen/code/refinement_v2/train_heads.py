"""Convergence-first same-C0 cached-head experiments; never loads a backbone."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from candidate import correct_direction, derivative_fusion, fixed_records, replace_record_directions
from experiment_core import (digest, save_json, csv_bytes, read_gt, record_labels,
                             make_head, head_forward, loss_terms, kalman)


class CachedSplit:
    def __init__(self, path):
        self.path=path
        completed=json.loads((path/'COMPLETED.json').read_text())
        assert completed['status']=='PASS'
        self.manifest=json.loads((path/'input_manifest.json').read_text())
        self.names=sorted(self.manifest['gt_files'])
        assert len(self.names)==completed['files']
        hashes=json.loads((path/'file_manifest.json').read_text())
        self.data={};self.gt=[];self.records=[];self.raw_csv=[]
        for i,name in enumerate(self.names):
            stem=Path(name).stem
            for sub,ext in [('float','.npz'),('targets','.npz'),('raw','.csv'),('gt','.csv')]:
                rel=sub+'/'+stem+ext
                assert digest(path/rel)==hashes[rel],rel
            assert digest(path/'gt'/name)==self.manifest['gt_files'][name]
            with np.load(path/'float'/(stem+'.npz'),allow_pickle=False) as z, np.load(path/'targets'/(stem+'.npz'),allow_pickle=False) as targets:
                for key in ('doa_features','doa','probability','mapping','target','matched','motion_valid','displacement'):
                    arr=z[key] if key in z else targets[key]
                    assert np.isfinite(arr).all()
                    if key not in self.data:
                        self.data[key]=torch.empty((len(self.names)*15,40,*arr.shape[1:]),dtype=torch.from_numpy(arr).dtype)
                    self.data[key][i*15:(i+1)*15]=torch.from_numpy(arr.reshape(15,40,*arr.shape[1:]))
                raw,prob=z['doa'],z['probability']
                rec=fixed_records(prob,raw)
                expected=np.array([(*r[:3],0,*r[3:]) for r in rec],dtype=np.int64).reshape(-1,6)
                assert np.array_equal(z['records'],expected)
                self.records.append(rec)
                self.raw_csv.append((path/'raw'/name).read_bytes())
                assert csv_bytes(rec)==self.raw_csv[-1]
                assert (z['mapping'][::40]==-1).all()
                assert not targets['motion_valid'][::40].any()
            self.gt.append(read_gt(path/'gt'/name))
            if (i+1)%50==0:
                print(json.dumps(dict(stage='cache_load',split=self.manifest['split'],files=i+1)),flush=True)
        self.n=len(self.names)*15

    def batch(self, indices, device):
        return {k:v[indices].to(device) for k,v in self.data.items()}


def predict(head, data, condition, device):
    raw=data.data['doa'].numpy().reshape(-1,600,2,3)
    maps=data.data['mapping'].numpy().reshape(-1,600,2)
    if condition=='F0':
        return raw.copy()
    if condition in ('F-EMA','F-KF'):
        return np.stack([kalman(r,m) if condition=='F-KF' else derivative_fusion(r,np.zeros_like(r),m) for r,m in zip(raw,maps)])
    head.eval();predictions=[]
    with torch.no_grad():
        for start in range(0,data.n,32):
            # Only inference fields are handed to the model; no GT enters decoding.
            b={k:data.data[k][start:start+32].to(device) for k in ('doa_features','doa','mapping','probability')}
            delta=head_forward(head,b['doa_features'],b['doa'],b['mapping'],condition)
            if condition!='F-Deriv':
                delta=correct_direction(b['doa'],delta,b['probability'].amax(-1)>.5)
            predictions.append(delta.cpu().numpy())
    out=np.concatenate(predictions).reshape(-1,600,2,3)
    if condition=='F-Deriv':
        out=np.stack([derivative_fusion(r,d,m) for r,d,m in zip(raw,out,maps)])
    assert np.isfinite(out).all()
    return out


def evaluate(data, output, destination=None):
    from aligned_metrics import AlignedMetrics
    metric=AlignedMetrics();counts=np.zeros((3,14),np.int64);per_file=[]
    if destination is not None:
        destination.mkdir(parents=True,exist_ok=False)
    for i,name in enumerate(data.names):
        rec=replace_record_directions(data.records[i],output[i])
        assert [r[:3] for r in rec]==[r[:3] for r in data.records[i]]
        pl=record_labels(rec);metric.update(pl,data.gt[i])
        pa=np.zeros((600,14),bool);ga=np.zeros((600,14),bool)
        for t,c,*_ in rec:pa[t,c]=True
        for t,rs in data.gt[i].items():
            for r in rs:ga[t,r[0]]=True
        cnt=np.stack([(pa&ga).sum(0),(pa&~ga).sum(0),(~pa&ga).sum(0)])
        counts+=cnt
        if destination is not None:
            with (destination/name).open('xb') as f:f.write(csv_bytes(rec))
            one=AlignedMetrics();one.update(pl,data.gt[i])
            per_file.append(dict(filename=name,records=len(rec),scores=one.scores(),counters=one.counters()))
    tp,fp,fn=counts
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(14),where=(2*tp+fp+fn)>0)
    result=dict(scores=metric.scores(),counters=metric.counters(),files=len(data.names),
                sed=dict(TP=tp.tolist(),FP=fp.tolist(),FN=fn.tolist(),micro_F1=float(2*tp.sum()/max(1,(2*tp+fp+fn).sum())),macro_F1=float(f.mean())))
    if destination is not None:
        save_json(destination/'metrics.json',result);save_json(destination/'per_recording.json',per_file)
    return result


def initialization_gate(data, device, condition):
    head=make_head(condition).to(device)
    if condition!='F-Deriv':
        q=predict(head,data,condition,device)
        raw=data.data['doa'].numpy().reshape(-1,600,2,3)
        assert np.array_equal(q,raw)
        for i,rec in enumerate(data.records):
            assert csv_bytes(replace_record_directions(rec,q[i]))==data.raw_csv[i]
    # Check nonzero trained-path forward on real first validation chunks.
    with torch.no_grad():
        torch.nn.init.normal_(head.net[-1].weight,std=.01)
        b=data.batch(torch.arange(2),device)
        first=head_forward(head,b['doa_features'],b['doa'],b['mapping'],condition)
        changed=b['doa_features'].clone();changed[:,20:]+=10
        later=head_forward(head,changed,b['doa'],b['mapping'],condition)
        assert torch.equal(first[:,:20],later[:,:20])
    return dict(zero_float_and_csv_exact=condition!='F-Deriv',
                derivative_zero_is_smoothing=condition=='F-Deriv',real_hidden_prefix=True)


def stopping_decision(history, epoch, last_progress, policy):
    w=policy['train_loss_windows']
    if epoch<policy['minimum_epochs'] or len(history)<2*w:
        return False,None
    old=np.mean([r['loss'] for r in history[-2*w:-w]])
    new=np.mean([r['loss'] for r in history[-w:]])
    relative=float(abs(new-old)/max(abs(old),1e-8))
    stop=epoch-last_progress>=policy['validation_patience'] and relative<=policy['relative_train_loss_plateau']
    return stop,relative


def main():
    p=argparse.ArgumentParser()
    for key in ('cache','config','execution','scorer','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--condition',choices=['F0','F-EMA','F-KF','F-Deriv','R0','R1','R2'],required=True)
    p.add_argument('--smoke-batches',type=int,default=0)
    a=p.parse_args()
    started=time.time();torch.set_num_threads(2)
    spec=json.loads(a.config.read_text());execution=json.loads(a.execution.read_text())
    seed=execution.get('seed',2026)
    assert isinstance(seed,int) and 0 <= seed < 2**32
    sys.path.insert(0,str(a.scorer))
    a.output.mkdir(parents=True,exist_ok=False)
    condition=a.condition;learned=condition in ('F-Deriv','R0','R1','R2')
    device='cuda:0' if learned else 'cpu'
    save_json(a.output/'RUNNING.json',dict(pid=os.getpid(),command=sys.argv,condition=condition,
              config_sha256=digest(a.config),execution_sha256=digest(a.execution),
              code_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
              torch=torch.__version__,gpu=torch.cuda.get_device_name() if learned else None,
              smoke_batches=a.smoke_batches,backbone_loaded=False,start_unix=started))
    validation=CachedSplit(a.cache/'validation')
    assert validation.manifest['checkpoint_sha256']==spec['baseline_checkpoint_sha256']
    for rel, expected_hash in validation.manifest['source_hashes'].items():
        if rel.startswith('evaluation/'):
            assert digest(a.scorer/Path(rel).relative_to('evaluation'))==expected_hash, 'Scorer source mismatch: '+rel
    raw=predict(None,validation,'F0','cpu');reference=evaluate(validation,raw)
    assert reference['scores']==json.loads((a.cache/'validation'/'COMPLETED.json').read_text())['scores']
    if not learned:
        out=predict(None,validation,condition,'cpu')
        result=evaluate(validation,out,a.output/'validation')
        assert result['sed']==reference['sed']
        save_json(a.output/'COMPLETED.json',dict(status='PASS',condition=condition,training_updates=0,
                  elapsed_seconds=time.time()-started,result=result))
        return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    gate=initialization_gate(validation,device,condition)
    save_json(a.output/'preflight.json',dict(status='PASS',**gate,raw_scores_exact=True))
    train=CachedSplit(a.cache/'train')
    assert not set(train.names)&set(validation.names)
    assert train.manifest['checkpoint_sha256']==validation.manifest['checkpoint_sha256']
    assert train.manifest['source_hashes']==validation.manifest['source_hashes']
    # Gates consume RNG; reset before formal head and optimizer construction.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=make_head(condition).to(device)
    proposal=spec['training_proposal']
    optimizer=torch.optim.AdamW(head.parameters(),lr=proposal['learning_rate'],weight_decay=proposal['weight_decay'])
    expected=next(c['parameters'] for c in spec['conditions'] if c['id']==condition)
    assert sum(p.numel() for p in head.parameters())==expected
    generator=torch.Generator().manual_seed(seed)
    best=float('inf');progress_best=float('inf');last_progress=0;best_epoch=0;history=[];updates=0
    policy=execution['stopping'];stop=False
    epochs=1 if a.smoke_batches else policy['maximum_epochs']
    before_first=[p.detach().cpu().clone() for p in head.parameters()]
    for epoch in range(1,epochs+1):
        head.train();order=torch.randperm(train.n,generator=generator)
        sums={};batches=0
        epoch_start=time.time()
        for start in range(0,train.n,proposal['batch_chunks']):
            ids=order[start:start+proposal['batch_chunks']]
            b=train.batch(ids,device)
            optimizer.zero_grad(set_to_none=True)
            loss,terms=loss_terms(head,b,condition)
            assert torch.isfinite(loss),'nonfinite loss'
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()),'nonfinite gradient'
            grad=torch.nn.utils.clip_grad_norm_(head.parameters(),proposal['gradient_clip_norm'])
            optimizer.step();updates+=1;batches+=1
            assert all(torch.isfinite(p).all() for p in head.parameters())
            for k,v in dict(loss=loss.detach(),gradient_norm=grad,**terms).items():sums[k]=sums.get(k,0)+float(v)
            if a.smoke_batches and batches>=a.smoke_batches:break
        result=evaluate(validation,predict(head,validation,condition,device))
        assert result['sed']==reference['sed'],'detection preservation failure'
        value=result['scores']['dcase2023_micro']['SELD_LR']
        if value<best:
            best=value;best_epoch=epoch
            # This file is owned by this new run. Atomic best replacement only.
            temp=a.output/'best.tmp.pth'
            torch.save(dict(head=head.state_dict(),epoch=epoch,condition=condition,seed=seed,
                            config=spec,execution=execution,checkpoint_sha256=spec['baseline_checkpoint_sha256']),temp)
            temp.replace(a.output/'best.pth')
        if value<progress_best-policy['validation_min_delta']:
            progress_best=value;last_progress=epoch
        row=dict(epoch=epoch,updates=updates,loss=sums['loss']/batches,
                 terms={k:v/batches for k,v in sums.items() if k not in ('loss','matched','motion_pairs','unmatched')},
                 denominators={k:int(sums[k]) for k in ('matched','motion_pairs','unmatched')},
                 scores=result['scores'],best_epoch=best_epoch,epoch_seconds=time.time()-epoch_start)
        history.append(row);stop,relative=stopping_decision(history,epoch,last_progress,policy)
        row['train_plateau_relative_change']=relative
        row['validation_epochs_without_progress']=epoch-last_progress
        with (a.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(condition=condition,epoch=epoch,loss=row['loss'],SELD_LR=value,best_epoch=best_epoch,
                              plateau=stop,elapsed_seconds=time.time()-started)),flush=True)
        if stop:break
    assert any(not torch.equal(old,new.detach().cpu()) for old,new in zip(before_first,head.parameters()))
    saved=torch.load(a.output/'best.pth',map_location=device,weights_only=False)
    head.load_state_dict(saved['head'])
    result=evaluate(validation,predict(head,validation,condition,device),a.output/'validation')
    assert result['sed']==reference['sed']
    assert result['scores']['dcase2023_micro']['SELD_LR']==best
    save_json(a.output/'COMPLETED.json',dict(status='SMOKE_PASS' if a.smoke_batches else 'PASS',
              condition=condition,epochs=len(history),best_epoch=best_epoch,training_updates=updates,
              stopping_reason='SMOKE_LIMIT' if a.smoke_batches else 'BOTH_PLATEAUS' if stop else 'MAX_EPOCHS',
              convergence_confirmed=bool(stop and not a.smoke_batches),
              no_backbone_optimizer=True,parameters=expected,elapsed_seconds=time.time()-started,
              result=result,best_checkpoint_sha256=digest(a.output/'best.pth')))


if __name__=='__main__':
    main()
