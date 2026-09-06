# Object-State SELD v0

Project root on `rabbit02`:

```text
/work/zhanghc/Myllm/SELD/ObjectStateSELD
```

## Environment

```bash
conda activate /work/zhanghc/Myllm/SELD/.venvs/object-state-seld-v0
cd /work/zhanghc/Myllm/SELD/ObjectStateSELD
python scripts/check_seld_env.py
```

The frozen environment and exact package lock are recorded in
`docs/environment_object_state_seld_v0.yml` and
`docs/requirements_object_state_seld_v0.lock.txt`.

## Data validation and manifests

```bash
python scripts/audit_raw_data_readiness.py
python scripts/build_object_state_manifest.py \
  --config configs/data_v0.yaml --require-audio
python scripts/audit_tracks.py \
  --config configs/data_v0.yaml --require-audio
```

The current v0 Motion manifests are written to
`data_intermediate/manifests/`. They use a 2.0 s context, 10 Hz labels,
0.1/0.3/0.5 s future targets, continuous tracks of at least 2.5 s, and
scene polyphony exactly one.

## B0 real-audio smoke test

This exercises real WAV loading, the log-mel plus intensity frontend,
classification/activity/DOA heads, loss computation, and backpropagation.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/smoke_test_b0.py \
  --device cpu --batch-size 2
```

The current implementation is an engineering B0 baseline and a pipeline
smoke test. Do not treat it as the final experiment implementation until the
training loop, train-only feature normalization, official current/future/gap
metrics, Oracle P0/P1, Activity auxiliary sampling, and B1-B3 objectives have
been implemented and reviewed.

See `docs/readiness_report_2026-08-03.md` for the audited readiness decision.
