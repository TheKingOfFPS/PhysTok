# Frozen PhysTok and its scalar forecasting adapters

This document describes the Full model used in the V2 benchmark and the V11
manuscript. It does not replace Full with any later control.

## Input and output contract

The public training interface is:

```python
from phystok.models import build_model
model = build_model(name, candidate, layout, bridge, eval_seed=20261017)
prediction_z = model(batch)  # shape [batch_size]
```

`batch['x']` contains the FIT-imputed/scaled 72-hour, 72-channel native input
view. PhysTok selects 54 history channels and 68 ordered issue-time state
features according to the versioned data contract. It uses the latest history
row to assemble the state view; the views can share underlying feature channels.
The scalar output is target-standardized PM2.5 at exactly `t+24`, not an average
or a sequence over the next day. `model.bridge.to_raw(prediction_z)` restores
physical units using the selected fold's target statistics.

The layout contains `feature_order`, `trajectory_indices`, `state_indices`,
`trajectory_family_ids`, and `state_family_positions`. Its membership/order is
fixed, not learned or selected on TEST. No city ID or city graph enters PhysTok.
All batch timestamps and features are governed by the separate data contract.

## Full architecture

- 54 historical variables each produce a trajectory token.
- Shared linear maps embed the last 12, 24 and 72 observations into width 512.
  A sample-and-variable-specific softmax gate mixes these scales. A
  512→128→512 bottleneck adds a residual to the 72-hour base. Its final map
  starts at zero and its learned residual scale starts at one.
- Six state families contain 15/4/15/8/10/16 features (68 total): PM2.5 history,
  calendar, transport and dispersion, moisture and stability context,
  pollution composition, precipitation and radiation. Each family uses its
  own `|family|→128→512→LayerNorm` projector.
- Learned variable, family and view identities preserve the token schema.
  History family counts are 1/4/15/8/10/16 (54 total).
- All 60 tokens pass through four shared post-normalized Transformer encoder
  layers, with width 512, 16 heads, feedforward width 2048, GELU and dropout .1.
- Only the 54 encoded historical rows are flattened into the prediction head:
  `54×512→512→1`, with GELU and dropout .1. The six state outputs are not
  concatenated directly into this Full head.
- Full has **27,403,141 trainable parameters**, including its joint readout.
  `phystok/models/vendor/phystok/strong_vtt_model.py` is the exact frozen
  architecture snapshot. No architecture math was rewritten for publication.

`from phystok.models import PhysTok` exposes that same `StrongVTT` class for
users constructing the core directly. The scalar adapter is recommended because
it enforces the recorded channel order and target-scale bridge.

The training runner must seed **CPU RNG before constructing the model**. A CUDA
seed applied only after construction does not protect CPU-initialized weights.
The public runner follows that order; models do not silently choose a seed.

## Baseline task adaptations

DLinear is a local 146-parameter scalar adapter with a 25-hour moving-average
trend and separate history-to-one seasonal/trend linear maps. It uses PM2.5
history. PatchTST, iTransformer, TimeXer and XLinear use their pinned source
cores with the frozen scalar output adapter. Their predictions in input-PM
standardized units pass through `ScaleBridge` to target-standardized units.
TimeXer/XLinear receive the endogenous PM channel last. Informer uses 36
observed decoder steps and one zero-value `t+24` query with the corresponding
calendar mark; its learned output directly represents target-standardized units.
Informer evaluation isolates its official CPU sampling RNG with a fixed seed
and does not consume the training CPU RNG state.

The model-specific input and optimization choices are disclosed in the frozen
training recipe. They should not be described as an identical architecture,
identical information representation, or identical optimizer across all models.
See `baselines/README.md` for license-dependent source availability.

## Matched representation controls

`build_control_model(control, candidate, layout, bridge, eval_seed)` supports
only the three recorded controls:

| Control | Changed design | Parameters |
|---|---|---:|
| `delayed_xview` | The first three encoder layers forbid history↔state attention; the fourth is fully open. Same weights, identities and history-only readout. | 27,403,141 |
| `strict_late_fusion` | All four layers forbid history↔state attention; the first cross-view fusion is a resized head reading all 60 outputs, hidden width 461. | 27,409,183 |
| `plain_state_flatten` | Six independent 68→128→512→LayerNorm state slots each see the whole state vector; generic slot identities replace state-family identities. History identity, full attention and original history-only head remain. | 27,449,733 |

The original Full constructor runs before a control replaces its relevant
modules, preserving common-tensor initialization order. Controls do not imply
that all comparisons establish independent benefits in every population.
Delayed's Joint RMSE difference was unresolved; the corrected Spatial
comparisons support Full against Plain-State and Strict A under their stated
families, as detailed in the results documentation.

## Capacity controls

`build_capacity_model(variant, candidate, layout, bridge, eval_seed)` enforces
the recorded widths and checks parameter counts:

| Variant | Width | Parameters |
|---|---:|---:|
| `phystok_small` | 256 | 7,021,701 |
| `itransformer_scaled` | 1400 | 27,284,497 |
| `timexer_scaled` | 1072 | 27,316,513 |

Small keeps four layers, width/32 heads, feedforward width 4×width and head
hidden width equal to the model width. `build_model('phystok', ...)` still
requires width 512, so Small cannot silently enter the main benchmark. T8 uses
one seed, 20260826, ten folds and five populations. Full and original-width
baseline references reuse their corresponding main-benchmark checkpoints.
