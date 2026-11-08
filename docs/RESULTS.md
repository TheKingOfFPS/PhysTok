# Frozen results for the PhysTok manuscript

Reference manuscript: **Population-Frozen Evaluation and Schema-Organized Tokens for City-Level Forecasting**, V11, November 8, 2026. The project has not claimed conference acceptance.

The main benchmark has seven trainable models, three seeds (20260826/27/28), ten city folds, and five deployment populations: 210 accepted fits. Fifty development cities and ten external cities cover targets in 2023–2025. December 2022 supplies history only. `Future` is the stored name for **Future-Year**.

## Main benchmark

Values are mean ± sample SD across three complete seed pools, in µg m⁻³. Within each seed, sum all legal fold-exposure SSE/SAE/n before taking the square root or ratio. Persistence is deterministic and has no artificial training-seed SD.

### RMSE

| Model | Spatial | Future-Year | Joint | External-Spatial | External-Joint |
|---|---:|---:|---:|---:|---:|
| phystok | 19.459 ± 0.031 | 24.873 ± 0.020 | 25.071 ± 0.011 | 21.478 ± 0.070 | 21.937 ± 0.106 |
| informer | 20.911 ± 0.113 | 26.223 ± 0.055 | 26.422 ± 0.037 | 23.092 ± 0.127 | 24.836 ± 0.102 |
| timexer | 24.262 ± 0.038 | 25.161 ± 0.016 | 25.164 ± 0.056 | 21.737 ± 0.027 | 22.055 ± 0.050 |
| itransformer | 24.379 ± 0.042 | 25.578 ± 0.060 | 25.523 ± 0.067 | 22.054 ± 0.089 | 22.337 ± 0.087 |
| patchtst | 24.826 ± 0.013 | 25.413 ± 0.019 | 25.421 ± 0.049 | 21.765 ± 0.008 | 22.559 ± 0.042 |
| xlinear | 23.561 ± 0.111 | 25.845 ± 0.131 | 25.903 ± 0.101 | 22.116 ± 0.089 | 22.158 ± 0.140 |
| dlinear | 24.771 ± 0.001 | 25.220 ± 0.002 | 25.225 ± 0.004 | 21.864 ± 0.005 | 22.453 ± 0.003 |
| persistence | 28.529 | 29.427 | 29.427 | 23.644 | 24.902 |

### MAE

| Model | Spatial | Future-Year | Joint | External-Spatial | External-Joint |
|---|---:|---:|---:|---:|---:|
| phystok | 11.289 ± 0.028 | 13.806 ± 0.023 | 13.991 ± 0.005 | 13.195 ± 0.049 | 13.470 ± 0.071 |
| informer | 12.154 ± 0.076 | 14.403 ± 0.030 | 14.584 ± 0.093 | 13.971 ± 0.125 | 14.846 ± 0.052 |
| timexer | 14.015 ± 0.012 | 13.777 ± 0.011 | 13.771 ± 0.008 | 13.259 ± 0.019 | 13.211 ± 0.023 |
| itransformer | 14.103 ± 0.023 | 13.975 ± 0.035 | 13.979 ± 0.038 | 13.435 ± 0.032 | 13.387 ± 0.052 |
| patchtst | 14.408 ± 0.011 | 14.117 ± 0.020 | 14.117 ± 0.038 | 13.409 ± 0.007 | 13.547 ± 0.022 |
| xlinear | 13.756 ± 0.035 | 14.043 ± 0.034 | 14.095 ± 0.038 | 13.520 ± 0.057 | 13.286 ± 0.063 |
| dlinear | 14.765 ± 0.005 | 14.468 ± 0.005 | 14.477 ± 0.007 | 13.606 ± 0.005 | 13.839 ± 0.006 |
| persistence | 15.695 | 15.297 | 15.297 | 14.208 | 14.146 |

## Paired city uncertainty

The formal main family has 12 endpoints: PhysTok–TimeXer in all five populations × two metrics, plus PhysTok–Informer on Spatial × two metrics. The strongest observed baseline is Informer on Spatial and TimeXer in the other four populations. These comparators were identified from earlier benchmark ranks; the correction is not a selection adjustment over all six baselines.

Delta is comparator minus PhysTok; positive favors PhysTok. The percentage denominator is the comparator’s observed mean pooled metric. The following table uses RMSE and the same comparator choices as manuscript Table V.

| Population | Comparator | Delta ± paired seed SD | Gain (%) | Bonferroni family interval |
|---|---|---:|---:|---:|
| Spatial | informer | 1.451 ± 0.082 | 6.94 | [1.201, 1.726] |
| Future-Year | timexer | 0.288 ± 0.024 | 1.15 | [-0.025, 0.619] |
| Joint | timexer | 0.093 ± 0.047 | 0.37 | [-0.285, 0.486] |
| External-Spatial | timexer | 0.259 ± 0.043 | 1.19 | [-0.497, 0.929] |
| External-Joint | timexer | 0.118 ± 0.077 | 0.53 | [-0.488, 0.771] |

PhysTok has the lowest observed mean pooled RMSE among the main-benchmark models in all five populations. The Spatial contrast versus Informer is supported by the specified adjusted city-cluster interval: 6.94% lower RMSE, with delta 1.451 and family interval [1.201, 1.726]. Spatial comparisons against TimeXer also favor PhysTok in RMSE and MAE. The adjusted TimeXer intervals in all four shifted populations include zero; the analysis does not establish a clear difference there. Crossing zero is not equivalence. PhysTok does not lead MAE in every population.

City heterogeneity matters: versus TimeXer, PhysTok wins 50/50/50 of 50 cities on Spatial, 31/30/32 on Future-Year, 24/22/24 on Joint, 8/8/8 of 10 on External-Spatial, and 5/5/4 on External-Joint. Versus Informer it wins 49/49/49 on Spatial. Counts are descriptive; each city has one vote, while pooled errors weight legal prediction records. External-Joint leave-one-city-out deltas can change sign. No city is deleted from the main score.

## Controlled architectures

`ablation_model_means.csv` contains all complete three-seed metric pools. `ablation_results.csv` contains every reported contrast, its sign, bootstrap seed/replicates, and family definition.

- **Plain-State:** six generic state slots receive the complete 68-feature vector. Plain − Full Spatial RMSE is +0.201, with family-10 interval [0.089, 0.342]. Its other four RMSE intervals include zero; Plain has lower adjusted MAE on Future-Year and both external populations. The control jointly changes grouping, projection fanout, and state identity.
- **Strict Late Fusion A:** all-layer view separation and a 60-token readout. A − Full Spatial RMSE is +0.244, with family-20 interval [0.133, 0.357]. Other RMSE intervals versus Full include zero. This control changes both communication and readout, so it cannot isolate one mechanism.
- **Delayed:** unchanged parameter count and trajectory-only head; early cross-view attention is blocked and opened late. Its prespecified Joint RMSE delta is +0.001532, with pointwise interval [−0.057599, 0.056186]. It does not establish a clear Joint difference. The ten pointwise endpoints must not be relabeled as a simultaneous family.

These controls are additional specified alternatives, not members of the main-benchmark ranking. Their deployment-dependent directions are preserved. None demonstrates that every individual representation component has an independent effect.

## Single-seed capacity and weather checks

T8 and ABC use **one seed, 20260826, with all ten folds**. There is no multi-seed SD for these experiments. T8 includes reused Full/TimeXer/iTransformer operating points and three new capacity configurations. `capacity_results.csv` preserves all 30 configuration × population rows.

| Configuration | Parameters | Spatial RMSE | Spatial MAE |
|---|---:|---:|---:|
| phystok_small | 7,021,701 | 19.719 | 11.615 |
| itransformer_scaled | 27,284,497 | 24.727 | 14.259 |
| timexer_scaled | 27,316,513 | 24.613 | 14.186 |
| timexer_original | 8,461,313 | 24.277 | 14.018 |
| itransformer_original | 2,649,601 | 24.381 | 14.107 |
| phystok_full | 27,403,141 | 19.423 | 11.257 |

PhysTok-Small remains below the enlarged TimeXer/iTransformer Spatial errors. Capacity changes have different effects under external deployment; these data do not establish a universal capacity law.

ABC is a Spatial meteorological intervention, not an online deployment test: A reuses the aligned Full checkpoint, B retrains with FIT-derived meteorological climatology in both views, and C retrains with same-city/month/hour donor weather 14–16 days away. Recipient pollution, calendar, and target are retained.

| ABC configuration | RMSE | MAE |
|---|---:|---:|
| abc_b | 22.171 | 12.912 |
| abc_c | 19.331 | 11.320 |
| A | 19.423 | 11.257 |

B − A RMSE is +2.747; C − A is −0.093 with a pointwise interval crossing zero; B − C is +2.840. `meteorological_paired.csv` contains all six pointwise contrasts. City-cluster intervals condition on this single training seed.

## Traceability and release scope

Every distributed result file is listed in `results/PROVENANCE.json` with its SHA-256, source artifact hash/path, experiment family, seeds, fold scope, and aggregation definition. Source paths are historical **relative artifact identifiers**, not runtime dependencies. Sanitized run registries retain model/seed/fold and identity hashes, without server paths, GPU identifiers, checkpoints, or prediction records.

The sufficient-statistic files contain group counts and sums of prediction errors; they contain no timestamps, true values, predicted values, feature arrays, or data samples. Their smallest city groups contain 7,092 legal records (16,003 for ABC). They support aggregate arithmetic and conditional city resampling, but do not prove prediction correctness, label pairing, absence of leakage, checkpoint contents, or data-access rights. Persistence’s displayed frozen values are retained; this release does not provide its raw-error reconstruction.

These are local publication candidates pending the authors’ ownership/permission review. Inclusion is not a redistribution license for CNEMC, QuotSoft, ERA5, or other source data. See the repository release audit and data-source/license documentation.
