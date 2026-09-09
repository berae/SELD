# RB05 dynamic-pair velocity pilot

## Hypothesis

Static zero-velocity pairs may dilute the localization-oriented auxiliary signal.
Restricting the existing velocity loss to truly moving adjacent label-frame pairs
may improve dynamic-source localization without changing inference.

## Controlled comparison

Both new runs reuse the audited frame-causal EINV2 pipeline, TAU2020 FOA data,
train folds 2–6, validation fold 1, seed 2026, batch size 32, Adam 5e-4,
90 epochs, and minimum validation `SELD_LR` checkpoint selection.

| Run | Variant | lambda_velocity | lambda_jepa | velocity_min_norm |
|---|---|---:|---:|---:|
| `D1_motionpairs_v020_seed2026_rb05_v1` | velocity-only | 0.2 | 0.0 | 1e-6 |
| `D3_motionpairs_v020_jepa005_seed2026_rb05_v1` | velocity + JEPA | 0.2 | 0.05 | 1e-6 |

The filter is applied after track assignment. A pair contributes to the velocity
loss only when its original validity mask is active and the aligned Cartesian
target velocity norm exceeds `1e-6`. No velocity or JEPA output is used during
inference.

## Decision gate

The pilot is compared with same-runtime seed-2026 controls. Promotion to seeds
2027 and 2028 requires a meaningful dynamic `LE` improvement without a material
loss of dynamic `Recall@20`, overall `LR`, or `F20`. Validation is used for this
gate; evaluation remains untouched during selection.
