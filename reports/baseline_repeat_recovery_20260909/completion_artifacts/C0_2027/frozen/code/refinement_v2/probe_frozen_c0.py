"""One-batch, no-update C0 feature-hook/causality probe with source verification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from candidate import ResidualHead, prediction_history, correct_direction


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def state_digest(module):
    h=hashlib.sha256()
    for key,value in module.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    m=json.loads(args.manifest.read_text());run=Path(m["run"])
    sys.path.insert(0,str(Path(m["runtime"])/"models/einv2/audited"))
    import runner
    cfg=json.loads((run/"config.json").read_text())
    assert cfg["audit"]["source_hashes"]==runner.source_manifest()
    assert digest(run/"best.pth")==m["checkpoint_sha256"]
    assert digest(Path(cfg["causal_scalar_path"]))==m["scalar_sha256"]
    ck=torch.load(run/"best.pth",map_location="cpu",weights_only=False)
    for key in ("optimizer","scheduler","rng","cuda_rng","np_rng","random_rng"):ck.pop(key,None)
    assert ck["config"]==cfg
    runner.seed_all(cfg["audit"]["seed"])
    dataset=runner.get_dataset(cfg["dataset"],cfg["dataset_dir"])
    model=runner.AuditedEINV2(cfg,dataset).eval().requires_grad_(False).cuda()
    frontend=runner.Frontend(cfg).eval().requires_grad_(False).cuda()
    model.load_state_dict(ck["model"]);frontend.load_state_dict(ck["frontend"]);del ck
    before=(state_digest(model),state_digest(frontend))
    cfg=copy.deepcopy(cfg)
    cfg["inference"].update(testset_type="dev",test_fold="1")
    ds,generator,_=runner.get_generator(runner.data_args(cfg["audit"]["seed"],0),cfg,dataset,"test")
    batch=next(iter(generator));wave=batch["waveform"].cuda()
    saved={}
    def hook(index):
        def capture(module,inputs):saved[index]=inputs[0].detach().clone()
        return capture
    handles=[head.register_forward_pre_hook(hook(i)) for i,head in enumerate(model.doa_heads)]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        start=time.perf_counter()
        pred=model(frontend(wave));torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        captured_inputs=[saved[i] for i in range(2)]
        features=torch.stack(captured_inputs,dim=2)
        assert tuple(features.shape[1:])==(40,2,512)
        packed_doa=torch.stack([torch.tanh(model.doa_heads[i](features[:,:,i])) for i in range(2)],dim=2)
        hook_doa=torch.stack([torch.tanh(model.doa_heads[i](captured_inputs[i])) for i in range(2)],dim=2)
        packed_delta=(packed_doa-pred["doa"]).abs().max().item()
        original_layout_delta=(hook_doa-pred["doa"]).abs().max().item()
        print(json.dumps({"diagnostic":"head_input_layout","packed_max_delta":packed_delta,
                          "preserved_layout_max_delta":original_layout_delta,
                          "original_stride":list(captured_inputs[0].stride()),
                          "packed_slice_stride":list(features[:,:,0].stride())}),flush=True)
        assert torch.allclose(hook_doa,pred["doa"],atol=1e-7,rtol=0)
        # No hook-induced change to baseline forward outputs.
        for handle in handles:handle.remove()
        plain=model(frontend(wave));torch.cuda.synchronize()
        hook_plain_delta=max((pred[k]-plain[k]).abs().max().item() for k in ("sed","doa"))
        assert hook_plain_delta<=1e-7
        # Perturb samples from the next label frame onward, within the same chunk.
        changed=wave.clone();changed[...,20*2400:]=torch.randn_like(changed[...,20*2400:])
        handles=[head.register_forward_pre_hook(hook(i)) for i,head in enumerate(model.doa_heads)]
        future=model(frontend(changed))
        future_h=torch.stack([saved[i] for i in range(2)],dim=2)
        for handle in handles:handle.remove()
        prefix_delta=max((pred[k][:,:20]-future[k][:,:20]).abs().max().item() for k in ("sed","doa"))
        feature_delta=(features[:,:20]-future_h[:,:20]).abs().max().item()
        assert prefix_delta<=1e-6 and feature_delta<=1e-6
        h=features[0];raw=pred["doa"][0];prob=pred["sed"][0].sigmoid()
        mapping=prediction_history(prob.cpu().numpy(),raw.cpu().numpy(),np.arange(40),np.zeros(40))
        new=ResidualHead(True).cuda()
        zero=correct_direction(raw,new(h,raw,mapping),prob.max(-1).values>.5)
        assert torch.equal(raw,zero)
        torch.nn.init.normal_(new.net[-1].weight,std=.01)
        q=correct_direction(raw,new(h,raw,mapping),prob.max(-1).values>.5)
        future_prob=future["sed"][0].sigmoid();future_raw=future["doa"][0]
        future_mapping=prediction_history(future_prob.cpu().numpy(),future_raw.cpu().numpy(),np.arange(40),np.zeros(40))
        q_future=correct_direction(future_raw,new(future_h[0],future_raw,future_mapping),future_prob.max(-1).values>.5)
        assert torch.allclose(q[:20],q_future[:20],atol=1e-6,rtol=0)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(20):new(h,raw,mapping)
        torch.cuda.synchronize();head_ms=(time.perf_counter()-start)*1000/20
    assert before==(state_digest(model),state_digest(frontend))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    result={"status":"PASS","scope":"one original validation batch, two seconds prefix; not full cache/full causality proof",
            "checkpoint_sha256":m["checkpoint_sha256"],"runtime":m["runtime"],
            "batch_size":len(wave),"feature_shape":list(features.shape),
            "feature_layer":"model.doa_heads[k] forward_pre_hook; post-causal-Transformer, pre-Linear/tanh",
            "source_hashes_valid":True,"scaler_hash_valid":True,"parameters_and_buffers_unchanged":True,
            "hook_vs_plain_max_delta":hook_plain_delta,"future_prefix_max_delta":prefix_delta,
            "preserved_layout_head_max_delta":original_layout_delta,
            "packed_layout_head_max_delta":packed_delta,
            "r1_failure_note":"r1 recomputed the baseline head on a newly packed tensor slice. r2 diagnoses both layouts; authoritative raw DOA is never recomputed from packed cache.",
            "future_hidden_prefix_max_delta":feature_delta,"zero_residual_exact":True,
            "nonzero_R1_future_prefix_regression":True,"optimizer_created":False,"training_updates":0,
            "gpu":torch.cuda.get_device_name(),"visible_gpu":"0",
            "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
            "one_cold_baseline_batch_seconds":elapsed,"R1_40frame_head_mean_ms_20_iterations":head_ms,
            "timing_scope":"probe only, excludes disk/cache export, not a training throughput estimate",
            "recording":str(batch["filename"][0]),"segment":int(batch["n_segment"][0])}
    with args.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=="__main__":main()
