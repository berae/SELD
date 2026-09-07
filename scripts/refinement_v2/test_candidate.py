"""CPU contract tests; synthetic hidden features do not certify the C0 frontend."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from candidate import (ResidualHead, DerivativeHead, prediction_history, head_inputs,
                       correct_direction, derivative_fusion, fixed_records, replace_record_directions)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    results = []
    def check(name, fn):
        fn()
        results.append({"test": name, "status": "PASS"})
    p = np.zeros((8,2,14), np.float32); p[:,:,0] = .9
    raw = np.tile(np.array([[1.,.2,0.], [-1.,.2,0.]], np.float32), (8,1,1))
    frames=np.arange(8); chunks=np.array([0,0,0,0,1,1,1,1])
    mapping=prediction_history(p,raw,frames,chunks)
    h=torch.randn(8,2,512,requires_grad=True); r=torch.tensor(raw)
    active=torch.tensor(p.max(-1)>.5)

    def zero_identity():
        for history in (False,True):
            q=correct_direction(r,ResidualHead(history)(h,r,mapping),active)
            assert torch.equal(q,r)
            records=fixed_records(p,raw)
            assert records==replace_record_directions(records,q.detach().numpy())
    check("zero_head_exact_float_and_quantized_record_identity", zero_identity)

    def gradient():
        model=ResidualHead(False)
        q=correct_direction(r,model(h,r,mapping),active)
        q[...,2].sum().backward()
        assert h.grad is None
        assert torch.isfinite(model.net[-1].weight.grad).all()
        assert model.net[-1].weight.grad.abs().sum()>0
    check("zero_initialization_has_learning_gradient_features_detached",gradient)

    def frozen():
        backbone=torch.nn.Sequential(torch.nn.Linear(4,512),torch.nn.BatchNorm1d(512))
        backbone.eval().requires_grad_(False)
        before={k:v.clone() for k,v in backbone.state_dict().items()}
        head=ResidualHead()
        z=backbone(torch.randn(16,4)).reshape(8,2,512)
        head(z,r,mapping).sum().backward()
        assert all(torch.equal(v,before[k]) for k,v in backbone.state_dict().items())
        assert all(v.grad is None and not v.requires_grad for v in backbone.parameters())
    check("toy_frozen_parameters_buffers_unchanged_not_real_C0_certificate",frozen)

    def causality():
        model=ResidualHead(True)
        torch.nn.init.normal_(model.net[-1].weight,std=.01)
        a=model(h,r,mapping)
        h2=h.detach().clone(); h2[3:]+=100
        r2=r.clone(); r2[3:]=-r2[3:]
        p2=p.copy(); p2[3:]=np.roll(p2[3:],1,axis=-1)
        m2=prediction_history(p2,r2.numpy(),frames,chunks)
        b=model(h2,r2,m2)
        assert torch.equal(a[:3],b[:3])
        m3=mapping.copy(); m3[4:]=-1
        assert torch.equal(head_inputs(h,r,m3,True)[4,:,:515],
                           head_inputs(h,r,mapping,False)[4])
        assert mapping[4].tolist()==[-1,-1]
    check("nonzero_head_future_perturbation_and_chunk_reset",causality)

    def association():
        x=raw.copy(); x[1]=x[1,::-1]
        m=prediction_history(p,x,frames,chunks)
        assert m[1].tolist()==[1,0]
        boundary=np.array([[np.cos(np.deg2rad(179)),np.sin(np.deg2rad(179)),0.],
                           [np.cos(np.deg2rad(-179)),np.sin(np.deg2rad(-179)),0.]])[:,None,:]
        pb=np.zeros((2,1,14)); pb[:,:,0]=.9
        assert prediction_history(pb,boundary,np.arange(2),np.zeros(2))[1,0]==0
        p2=p.copy();p2[1]=0
        assert (prediction_history(p2,raw,frames,chunks)[2]==-1).all()
        p3=p.copy();p3[1]=np.roll(p3[1],1,axis=-1)
        assert (prediction_history(p3,raw,frames,chunks)[1]==-1).all()
        fr=frames.copy();fr[1:]+=1
        assert (prediction_history(p,raw,fr,chunks)[1]==-1).all()
        duplicate=raw.copy();duplicate[0,1]=duplicate[0,0]
        assert prediction_history(p,duplicate,frames,chunks)[1,0]==-1
    check("slot_swap_wrap_gap_class_change_ambiguity",association)

    def preserve():
        records=fixed_records(p,raw); records.append(records[0])
        changed=raw.copy();changed[...,1]+=.3
        new=replace_record_directions(records,changed)
        assert len(new)==len(records)
        assert [x[:3] for x in new]==[x[:3] for x in records]
        assert new[-1]==new[0]
        assert replace_record_directions([],changed)==[]
    check("record_multiplicity_ids_empty_and_no_post_dedup",preserve)

    def numerical():
        out=correct_direction(r,torch.ones_like(r)*100,active)
        cosine=(out*r).sum(-1)/(out.norm(dim=-1)*r.norm(dim=-1))
        angle=torch.rad2deg(torch.acos(cosine.clamp(-1,1)))
        assert angle.max()<15.001
        zero=torch.zeros_like(r)
        assert torch.equal(correct_direction(zero,torch.ones_like(r),active),zero)
        assert torch.equal(correct_direction(r,torch.ones_like(r),~active),r)
        assert torch.equal(correct_direction(r,torch.full_like(r,float("nan")),active),r)
    check("rho_degrees_zero_norm_inactive_nonfinite_residual_fallback",numerical)

    def units():
        d=np.zeros_like(raw);d[...,1]=.02
        velocity=d/.1
        a=derivative_fusion(raw,d,mapping)
        b=derivative_fusion(raw,.1*velocity,mapping)
        assert np.allclose(a,b)
        smoothing=derivative_fusion(raw,np.zeros_like(raw),mapping)
        assert np.array_equal(smoothing[0],raw[0])
        assert np.array_equal(smoothing[4],raw[4])
    check("displacement_per_frame_vs_velocity_dt_and_fusion_reset",units)
    params={"F-Deriv":sum(x.numel() for x in DerivativeHead().parameters()),
            "R0":sum(x.numel() for x in ResidualHead().parameters()),
            "R1":sum(x.numel() for x in ResidualHead(True).parameters())}
    params["R2"]=params["R1"]
    return {"scope":"CPU synthetic module contracts; no training, real h_t or waveform test",
            "torch":torch.__version__,"tests":results,"parameter_counts":params}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); started=time.perf_counter()
    result=run();result["elapsed_seconds"]=time.perf_counter()-started
    result["code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),Path(__file__).with_name("candidate.py")]}
    with args.output.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
