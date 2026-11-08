# Frozen experiment definitions

The reference is the verified manuscript version dated November 8, 2026 (V11). Scientific identity: `phystok_v2_frozen_recipe_20261017_r1`. The main recipe was developed on Fold0 and then frozen; it is not nested model selection. Earlier research inspected TEST. The published 210 runs use FIT/VAL selection only, with no HPO or reused historical checkpoints.

## Main benchmark

Seven trainable models × seeds 20260826/20260827/20260828 × folds 0–9 = 210 fits. Within each of five regions, each fold assigns 8 FIT, 1 VAL and 1 TEST cities. Every development city has 8/1/1 roles across folds. Ten external cities never enter FIT or VAL.

| Model | Batch | Optimizer | Initial LR | Minimum updates / FIT passes |
|---|---:|---|---:|---:|
| DLinear | 8192 | Adam | 0.0003 | 26000 / 310 |
| Informer | 512 | AdamW | 0.0003 | 36000 / 26 |
| PatchTST | 128 | AdamW | 0.0003 | 36000 / 6 |
| iTransformer | 128 | AdamW | 0.0001 | 36000 / 6 |
| TimeXer | 128 | AdamW | 0.0001 | 36000 / 6 |
| XLinear | 128 | AdamW | 0.0003 | 36000 / 6 |
| PhysTok | 512 | AdamW | 0.0001 | 36000 / 26 |

`configs/benchmark_v2.json` preserves the actual scientific recipe, including legacy proposal-status fields. Those historical fields were superseded by the research freeze and release evidence; they are not active public server authorization tokens. No server approvals, device allocation policies or credentials are included.

Samples are drawn by a CPU-owned random permutation without replacement; the tail batch is retained. A pass counts sample presentations divided by FIT size. Both update and pass minima apply. FP32, disabled AMP/TF32, weight decay 0.0001 and gradient clipping 1.0 are used. Full-VAL physical-unit RMSE selects the best checkpoint. The scheduler halves LR after two plateau checks (relative threshold 0.0001; minimum LR 0.000001). Qualified stopping requires the minima, at least one actual LR reduction, two recovery checks and six non-improving checks. Initial targets and hard caps are separate from minima; a cap is reported as a cap, not relabeled convergence.

Complete resume checkpoints retain model, optimizer, scheduler, scaler, sampler permutation/position/generator, Python/NumPy/Torch RNGs, counters, budget history, best reference and runtime identity. Restoring a best-only checkpoint cannot resume a run. Exact resume requires unchanged runtime, device, recipe and supports. CPU initialization is seeded before constructing the model; validation's Informer sampling is isolated from training RNG.

## Five deployment populations

| Key | City role | Target year | Exposures per physical target and seed |
|---|---|---|---:|
| Spatial | TEST development cities | 2023–2024 | 1 |
| Future (`Future-Year` in text) | FIT development cities | 2025 | 8 |
| Joint | TEST development cities | 2025 | 1 |
| External-Spatial | Ten external cities | 2023–2024 | 10 |
| External-Joint | Ten external cities | 2025 | 10 |

The same best-VAL checkpoint and FIT preprocessing are used in all five populations for each model/seed/fold. Labels are not pooled to refit preprocessing, select a checkpoint or adjust a model after the freeze. Repeated checkpoint exposures contribute their squared/absolute errors, not an ensemble prediction. Internal selectors require issue and target in the same local year; external selectors use target-year membership and permit eligible cross-year issues.

For each seed, pool all ten-fold SSE, SAE and n before calculating RMSE or MAE. Then calculate mean and sample SD across the three seed metrics. Persistence is evaluated once per physical target and has no training seed. Its prediction is the current PM2.5 value or the last finite value in the frozen past-only fallback window.

## Frozen controls

`configs/ablations.json` plans **140 new fits**, with Full and baseline references reused from main selected runs:

- **Capacity / T8:** PhysTok-Small, enlarged iTransformer and enlarged TimeXer; seed 20260826, ten folds and five populations. Three new variants, 30 fits. Widths and optimizer policies are the actual frozen settings, not a new parameter search.
- **ABC:** A is the reused main Full model. B uses FIT-built climatology with refitted FIT preprocessing; C uses the frozen retrospective donor episode mapping. B and C each use seed 20260826, ten folds, Spatial only (20 fits). Both trajectories and state are changed for the selected meteorological channels. Donor selection does not use prediction errors or target values.
- **Delayed:** the first three encoder layers block cross-view attention, and the last opens it; three seeds × ten folds, five populations (30 fits). It has the same parameters and trajectory readout as Full. Joint RMSE is primary; other displayed contrasts are descriptive pointwise comparisons.
- **Strict Late Fusion A:** views remain separated through all encoder layers and the resized head reads all encoded tokens; three seeds × ten folds and five populations (30 fits). This changes both interaction and readout.
- **Plain-State:** six generic state slots each project the full 68-dimensional state instead of family-specific subsets; three seeds × ten folds and five populations (30 fits).

See `experiments/` for concrete plan/prepare/fit/evaluate commands. Reference variants refuse a new fit: provide the appropriate main seed826 selected checkpoint. The controls' original comparison families remain distinct: main12, Plain10, Strict20; Delayed and ABC have their documented pointwise intervals. No inference family is selected according to significance.
