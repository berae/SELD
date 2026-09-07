"""CPU regression for the newly implemented execution path; no experiments."""
import argparse
from pathlib import Path
import numpy as np
import torch
from candidate import head_inputs, ResidualHead, prediction_history, correct_direction
from experiment_core import (fixed_targets, kalman, head_forward, loss_terms,
                             make_head, save_json, previous_tensor)


def run():
    torch.set_num_threads(2); torch.manual_seed(2026)
    results = []
    def check(name, fn):
        fn(); results.append(dict(test=name,status='PASS'))
    prob = np.zeros((80,2,14),np.float32); prob[:,:,0] = .9
    raw = np.tile(np.array([[1.,0.,0.],[0.,1.,0.]],np.float32),(80,1,1))
    mapping = prediction_history(prob,raw,np.arange(80),np.arange(80)//40)
    ht = torch.randn(2,40,2,512)
    rt = torch.from_numpy(raw.reshape(2,40,2,3))
    mt = torch.from_numpy(mapping.reshape(2,40,2))
    def equivalence():
        for cond in ('R0','R1','R2'):
            head = make_head(cond)
            torch.nn.init.normal_(head.net[-1].weight,std=.01)
            batch = head_forward(head,ht,rt,mt,cond)
            ref = head(ht.reshape(80,2,512),rt.reshape(80,2,3),mapping)
            torch.testing.assert_close(batch.reshape(80,2,3),ref,rtol=0,atol=1e-6)
    check('vectorized_same_inputs_as_reference_head',equivalence)
    def prefix():
        for cond in ('R0','R1','R2','F-Deriv'):
            head=make_head(cond);torch.nn.init.normal_(head.net[-1].weight,std=.01)
            h2=ht.clone();h2[:,20:]+=100
            a=head_forward(head,ht,rt,mt,cond);b=head_forward(head,h2,rt,mt,cond)
            assert torch.equal(a[:,:20],b[:,:20])
        assert not previous_tensor(ht,mt)[:,0].any()
    check('batched_future_prefix_and_chunk_reset',prefix)
    gt={t:[[0,0,0.,0.],[0,1,90.,0.]] for t in range(80)}
    targets,stats=fixed_targets(prob,raw,mapping,np.arange(80)//40,gt)
    def supervision():
        assert stats['matched']==160 and stats['valid_motion_pairs']==156
        assert stats['static_pairs']==156 and stats['moving_pairs']==0
        assert not targets['motion_valid'][::40].any()
        bad={t:[row.copy() for row in rs] for t,rs in gt.items()}
        bad[10][0][1]=3
        _,other=fixed_targets(prob,raw,mapping,np.arange(80)//40,bad)
        assert other['valid_motion_pairs']==154 and other['pair_gt_inconsistent']==2
        assert np.array_equal(mapping,prediction_history(prob,raw,np.arange(80),np.arange(80)//40))
        far={0:[[0,4,179.,0.]]}
        f,s=fixed_targets(prob[:1,:1],raw[:1,:1],mapping[:1,:1],np.array([0]),far)
        assert s['matched']==1, 'No 20-degree training gate'
    check('fixed_GT_masks_identity_switch_no_angle_gate',supervision)
    def kf():
        out=kalman(raw,mapping)
        np.testing.assert_allclose(out,raw,atol=1e-7)
        changed=raw.copy();changed[20:,:,2]=.2
        later=kalman(changed,mapping)
        assert np.array_equal(out[:20],later[:20])
        assert np.array_equal(later[40],changed[40])
        swapped=raw.copy();swapped[1:]=swapped[1:,::-1]
        mp=prediction_history(prob,swapped,np.arange(80),np.arange(80)//40)
        np.testing.assert_allclose(kalman(swapped,mp),swapped,atol=1e-7)
    check('KF_constant_direction_slot_swap_prefix_chunk_reset',kf)
    def differentiability():
        batch=dict(doa_features=ht,doa=rt,mapping=mt,probability=torch.from_numpy(prob.reshape(2,40,2,14)))
        for k in ('target','matched','motion_valid','displacement'):
            v=targets[k];batch[k]=torch.from_numpy(v.reshape(2,40,*v.shape[1:]))
        batch['target']=torch.nn.functional.normalize(batch['target']+.1,dim=-1)
        batch['displacement']=batch['displacement']+.01
        for cond in ('R0','R1','R2','F-Deriv'):
            head=make_head(cond)
            total,_=loss_terms(head,batch,cond);total.backward()
            assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert head.net[-1].weight.grad.abs().sum()>0
        empty={**batch,'matched':torch.zeros_like(batch['matched']),'motion_valid':torch.zeros_like(batch['motion_valid'])}
        for cond in ('R0','R1','R2','F-Deriv'):
            total,_=loss_terms(make_head(cond),empty,cond)
            assert torch.isfinite(total);total.backward()
    check('nonzero_learnable_gradient_and_empty_mask_losses',differentiability)
    def exact_zero():
        active=torch.from_numpy(prob.reshape(2,40,2,14).max(-1)>.5)
        for cond in ('R0','R1','R2'):
            out=correct_direction(rt,head_forward(make_head(cond),ht,rt,mt,cond),active)
            assert torch.equal(out,rt)
    check('batched_zero_residual_exact_identity',exact_zero)
    return dict(status='PASS',tests=results,scope='synthetic CPU execution adapters; backward only, no optimizer updates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run();save_json(a.output,r);print(r)
