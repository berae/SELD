# Object-State SELD v0 readiness report

Date: 2026-08-03  
Host: `rabbit02`  
Project: `/work/zhanghc/Myllm/SELD/ObjectStateSELD`

## Decision

- **Data preparation:** PASS for the DCASE2020 FOA development and released
  evaluation data.
- **Motion-manifest engineering:** PASS for the frozen v0 rules.
- **Dedicated runtime:** PASS, including a real CUDA allocation and a real-audio
  B0 CPU forward/backward test.
- **B0 engineering smoke experiment:** READY.
- **Full B0 scientific training/evaluation:** NOT YET COMPLETE.
- **B1-B3 ablation matrix:** NOT YET IMPLEMENTED.

The remaining limitations are code/experimental-design limitations, not raw
data corruption or environment failure.

## Audited raw data

All six DCASE2020 v1.2 archives match their official Zenodo MD5 values. The
split development archive also passes a complete ZIP CRC test.

| Split | FOA WAV | Metadata CSV | Rows | Classes | Pairing | Audio format |
|---|---:|---:|---:|---|---|---|
| Development | 600 | 600 | 354,298 | 0-13 | complete | all 4 ch, 24 kHz, 60 s |
| Released evaluation | 200 | 200 | 117,220 | 0-13 | complete | all 4 ch, 24 kHz, 60 s |

There are no invalid metadata rows, bad audio headers, missing audio files, or
missing metadata files in either split.

Additional candidates have been registered under `data_raw/` and their source
records archived under `docs/source_records/`:

- TAU-NIGENS 2021 metadata is present; FOA audio is not yet downloaded.
- STARSS22, STARSS23, and DCASE2024 synthetic source records are present; their
  large audio payloads are not part of the current P0 dataset.

## Frozen v0 Motion subset

Rules: folds 3-6 train, fold 2 validation, fold 1 test; 2.0 s context; 10 Hz
labels; 0.1/0.3/0.5 s future horizons; continuous track at least 2.5 s; hop one
label frame; scene polyphony exactly one.

| Split | Samples | Recordings | Tracks | Static | Slow | Medium | Fast |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 63,569 | 388 | 963 | 33,725 | 10,614 | 11,200 | 8,030 |
| Validation | 15,024 | 98 | 252 | 7,542 | 2,763 | 2,860 | 1,859 |
| Test | 14,765 | 99 | 231 | 7,653 | 3,039 | 2,459 | 1,614 |

Audit results:

- no recording overlap among train/validation/test;
- all audio paths exist and all excerpts contain exactly 48,000 samples;
- 7,143 continuous track segments, 2,498 at least 2.5 s;
- unit-position and future-position norm error below `8e-9`;
- tangent-velocity radial dot-product error below `1.3e-8`.

Important scientific limitation: strict Motion filtering removes classes 6 and
11 from every split. Class 9 has only 31 training samples and none in validation
or test; test also lacks class 3. Therefore this Motion manifest is suitable for
the dynamic `[h, p, v]` state study, but is not a valid stand-alone 14-class
activity/SELD benchmark. The Activity auxiliary subset proposed in the design
document must be added before making full-class detection claims.

## Runtime and model parameters

Environment:

```text
/work/zhanghc/Myllm/SELD/.venvs/object-state-seld-v0
Python 3.10.19
PyTorch 2.9.1+cu128
numpy 1.23.5, scipy 1.10.1, pandas 1.5.3, librosa 0.9.2
```

`pip check`, EIN-SELD imports, DCASE2022 imports, WAV reading, CUDA discovery,
and a real CUDA tensor allocation pass. B0-B3 frozen configuration files include
the requested 128-dimensional hidden/latent state, K=1, Adam `5e-4`, batch 32,
90 epochs, three seeds (2026-2028), StepLR at epoch 80, dropout 0.1, and the
0.1/0.3/0.5 s future horizons.

No external pretrained checkpoint is present. This is not a blocker for
training from scratch, but it means pretrained-model comparisons cannot yet be
run.

The new B0 engineering baseline reads real manifest excerpts, computes a
7-channel log-mel plus intensity representation, and predicts class, activity,
and unit Cartesian DOA. A batch of two real excerpts passed forward, combined
loss, and backward propagation on CPU:

```text
waveform: [2, 4, 48000]
class logits: [2, 14]
DOA: [2, 3]
trainable parameter count: 229,458
gradient-bearing tensors: 28
status: PASS
```

## What remains before the paper-grade run

1. Build the Activity auxiliary subset around event starts, ends, and background,
   then define the documented Motion:Activity sampling ratio (target 3:1).
2. Implement the reviewed trainer and train-only feature normalization; the
   current B0 code is a verified model/data smoke path, not the final trainer.
3. Implement B1 velocity, B2 future regression, and B3 velocity+future+JEPA,
   including EMA and loss masks.
4. Implement Oracle P0/P1 plus current/future/gap metrics and per-motion-bin
   reporting.
5. Run three seeds only after a GPU is available. At audit time all five RTX
   3090 cards were 81-98% utilized, with 20-24 GB allocated on each card.

Conclusion: the DCASE2020 data, manifests, frozen parameters, and environment
are ready and reproducible. An engineering B0 step can run now on CPU. A
paper-grade experiment should wait for the five implementation items above and
for a GPU allocation window.
