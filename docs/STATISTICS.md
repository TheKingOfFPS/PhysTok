# Results Verification and city-paired analysis

This module needs only Python and NumPy. It does not import PyTorch, initialize CUDA, train a predictor, access a server, or read original data. Released CSV files contain aggregate forecasting-error sufficient statistics, not observations or predictions.

## Commands

From the repository root:

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_tables.py --mode bootstrap
python scripts/reproduce_tables.py --mode bootstrap-all
python -m unittest discover -s tests -p test_statistics.py -v
```

- **verify:** SHA-256 inventory plus pooled arithmetic, three-seed sample SD, paired means/SD, legal aggregate exposure multiplicities, city wins/contributions, region pools, external leave-one-city-out sensitivity, single-seed capacity/weather, and reused Full reference checks. Frozen interval endpoints are read but are not independently recomputed in this mode.
- **bootstrap:** additionally regenerate all main-comparison intervals with the frozen 100,000 draws, seed 20261025, and 12-endpoint family.
- **bootstrap-all:** additionally regenerate Plain-State, Strict A, Delayed, and ABC intervals under their original distinct protocols. No common family is invented after the experiments.

The default receipt is `outputs/results_verification.json`; choose a new output path with `--output`. Inputs under `results/` are read-only. A mismatch causes a nonzero exit status. No row intersection, model reselection, or outcome-dependent exclusion repairs a failed comparison. Runtime depends on the user's CPU/BLAS setup; the full verification is CPU analysis, not a GPU experiment.

## Estimand and seed arithmetic

For model (m), seed (s), and population (e), let legal checkpoint exposures contribute count (n), sum of squared errors SSE, and sum of absolute errors SAE:


a. Sum all legal fold exposures within a seed.
b. Compute (\mathrm{RMSE}_{mse}=\sqrt{\sum\mathrm{SSE}/\sum n}\) and (\mathrm{MAE}_{mse}=\sum\mathrm{SAE}/\sum n\).
c. Average the three seed metrics and calculate sample SD with `ddof=1`.

This is neither a fold-average RMSE nor an ensemble prediction score. Fold SD, city SD, and bootstrap intervals are different quantities. Persistence is deterministic and is not assigned fictitious training seeds.

Main comparisons use Δ = comparator − PhysTok. Controls use variant − the declared comparator (usually Full). Positive favors the subtracted baseline. The CSV records every delta sign. The main percentage denominator is the comparator's observed mean pooled metric; architecture-control percentages use Full or the named comparator's observed mean.

## City clusters and repeated exposures

The frozen legal per-target exposure multiplicities are Spatial 1, Future-Year 8, Joint 1, External-Spatial 10, and External-Joint 10. The repeated fold/checkpoint exposures are retained in pooled errors; they are not independent cities. The sufficient statistics are already summed over each city's legal exposures per training seed.

One bootstrap draw gives each city one integer weight, **shared across models, all timestamps, all checkpoint exposures, and all three trained seeds**. For internal populations, sample cities with replacement within the five frozen geographic regions, preserving each region's count. External populations use their actual ten cities in one stratum. They do not inherit fabricated internal-region labels.

For each draw, weighted SSE, SAE, and n are re-pooled separately within each model and seed, the paired metric differences are computed, and those seed differences are averaged. Neither folds nor seeds are independently resampled. Percentile quantiles use NumPy's linear interpolation.

## Distinct frozen uncertainty families

| Experiment | Replicates / RNG seed | Intervals | Family |
|---|---|---|---|
| Main benchmark | 100,000 / 20261025 | Pointwise 0.025/0.975 and Bonferroni 0.05/(2×12), 1−0.05/(2×12) | 12: TimeXer×5 populations×2 metrics plus Informer×Spatial×2 |
| Plain-State | 100,000 / 20261024 | Pointwise and Bonferroni family | 10 endpoints |
| Strict Late Fusion A | 100,000 / 20261023 | Pointwise and Bonferroni family | 20: A versus Full and Delayed, five populations, two metrics |
| Delayed | 10,000 / 20261021 | Pointwise only; Joint RMSE prespecified primary | No simultaneous-family claim |
| ABC weather | 10,000 / 20260926 | Pointwise only; B−A, B−C, C−A for RMSE/MAE | One trained seed; no seed SD |
| T8 capacity | No bootstrap interval published here | Descriptive pooled metrics | One trained seed; no seed SD |

The main comparators were selected from previously seen benchmark rankings. The 12-endpoint correction is not complete selection adjustment across all six baselines. Coverage is approximate and conditional on the observed trained seeds, city panel, and frozen configuration. City resampling does not model all shared-weather correlations. The external ten-city panel especially limits geographic generalization. An interval crossing zero does not establish equivalence. No p-value or post hoc significance-selected population is used.

## City and region descriptions

`city_diagnostics` recomputes per-city seed metrics, win counts, mean paired changes, signed SSE/SAE contributions, and each leave-one-city-out pooled difference. It also pools regional sufficient statistics within each seed before computing regional descriptive metrics. Cities are never deleted from the main estimates. City-win rates, regional point estimates, and leave-one-city-out reversals are descriptive; they are not binomial significance tests or substitutes for pooled error.

## What this verification cannot prove

The release provides no actual input matrix, target observation, raw prediction record, or checkpoint. Aggregate verification therefore cannot independently check the truth/key pairing that produced each error, authenticate checkpoint tensors, establish no leakage in a private historical run, or reconstruct data permissions. SHA checks detect local drift relative to the included manifest; a self-contained manifest is not an external authenticity signature. Full Reproduction requires legally acquired data and the prediction-level pairing/time/exposure audits in the rest of the repository.

The included `VERIFICATION_RECEIPT.json` documents the tested snapshot and code hashes. Re-run the commands in your own environment. No publication right for third-party datasets is conveyed by these statistics.
