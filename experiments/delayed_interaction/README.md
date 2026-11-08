# Delayed cross-view interaction

Frozen variant: `delayed_xview`; three seeds 20260826/27/28, ten folds and five
scenarios. The first three layers have a block-diagonal history/state attention
mask; only the fourth layer permits cross-view interaction. Parameter count
27,403,141 and history-only readout equal Full. No other opening times are part
of this formal experiment.

Use the common `scripts/run_ablation.py` plan/fit/evaluate interface and the
exact recipe in `configs/ablations.json`. Joint pooled RMSE is the prespecified
primary endpoint; the recorded analysis uses a 10,000-draw pointwise city-cluster
interval (seed 20261021). Other endpoints are secondary/descriptive. The
observed Joint difference near 0.002 does not establish a clear independent
advantage for repeated early cross-view interaction and is not equivalence.
