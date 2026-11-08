# Frozen data and task contract

The formal loader is `phystok.data.CityData(data_root, fold, contract=...)`.
`configs/data_contract.json` is a data-free schema template. It has no empirical
statistics or data hashes. `prepare_data.py bind` produces a local contract with
SHA-256 bindings for the feature schema, fold manifest, selectors and fitted
transforms, plus an `input_files` manifest for every file in the four internal/
external feature/target stores and both task Parquet files. Thus the dataset
identity includes actual input/target content, not only transform metadata.
The default template intentionally fails on unbound real data.

Formal fit/evaluation entry points call `verify_local_inputs(data_root, contract)`.
This checks the exact file membership, sizes and SHA-256 values in the six fixed
input roots, as well as bound metadata. Missing stores, missing/new chunks,
changed content and symlinked inputs are rejected. It does not scan other
project folders. Each formal call reads the bound bytes again; this costs one
local data-package read, rather than relying on filesystem timestamp caches.
The latter can miss same-size writes on coarse-timestamp filesystems. Keep the
user data workspace immutable and free of concurrent writers during a run.

## Continuous arrays and identities

Reconstruction retains the source project-relative layout under the selected
`data_root`; there is no dependency on the original private repository. The
internal continuous feature store is
`04_derived/issue_time_features/national_pm25_observed_feature_schema_v1/issue_time_feature_bank.zarr`.
Each of its 72 named feature arrays has axes `(time, city)`. The merged target
store is `04_derived/observed_era5_merged/observed_era5_50cities_v1.zarr`; the label
is `pm25_city_mean[target_time_index, city_index]`. On the original source
snapshot, the UTC intersection has 27,040 hourly rows and 50 cities. Arrays are
stored continuously, not as expanded training windows. External namespaces
replace the internal panel by ten cities under
`04_derived/external_validation/central_china_external_holdout_v1/`.

Internal city indices are published in `data/metadata/internal_cities.csv`;
external indices are in `external_cities.csv`. The internal canonical order is
region then English city name, and is distinct from the acquisition manifest's
priority order. The merger aligns city identities explicitly before combining
air-quality and meteorological records.

The task Parquet is generated locally at
`06_processed/task_index/pm25_24h_task_index_v1.parquet`. Its required fields are:

| Fields | Meaning |
|---|---|
| `sample_id`, `city_index`, `city_en`, `region` | Physical city/target identity and frozen city axis |
| `issue_time_index`, `target_time_index` | Integer positions, separated by exactly 24 hours |
| `issue_time_utc`, `target_time_utc` | Naive timestamps explicitly interpreted as UTC |
| `issue_time_local`, `target_time_local`, `target_local_year` | Asia/Shanghai civil-time coordinates |
| `forecast_horizon_hours` | 24; this is a scalar lead, not an output length of 24 |
| `target_valid_benchmark` | Frozen monthly-Q95/HJ-style station-support validity |

External tasks additionally contain `external_task_row` (row identity) and
`external_scenario`. These schemas are descriptive; no actual task rows are
shipped. Targets and source magnitudes are never embedded in public examples.

## Windows, transformations and permissions

`ScopedView.batch(indices)` returns `x` with shape `(batch,72,72)`, `y` and
`y_raw` with shape `(batch,)`, historical time marks `(batch,72,4)` and target
marks `(batch,1,4)`, plus sample/task identities. It gathers issue times `t-71`
through `t`; targets are at `t+24`. Derived lags in the baseline feature bank
can reach earlier raw history (down to `t-95`). Do not describe the native
baseline and PhysTok input schemas as an information-matched mechanism control.

Full PhysTok selects the 54 trajectory positions and the issue-time 68-feature
state view defined in `configs/data_contract.json`. The six state-family
positions and trajectory family identities are in that file; PM10/pressure
channels stay in the 72-channel native baseline adapter but do not enter Full
PhysTok's selected views.

For each fold, `preprocess_stats.npz` is rebuilt from **formal FIT task issue
centers only**, not from all windows: per-feature finite median, median
imputation, then population mean/std on the imputed values. Continuous fields
use those centers/scales. Calendar bounded features and rain indicators retain
the prescribed no-scaling treatment; a zero-variance scale uses 1. Target
mean/std are fitted on the same fold's legal FIT labels. Neither future years,
TEST cities, VAL labels nor the external panel fit these transformations.

The internal sample selector requires a valid target and equal issue/target
local years. The external selector uses the target year and retains legal
December-31 to January-1 forecasts. Internal FIT/VAL/Spatial use target years
2023–2024; Future/Joint use 2025. Each city is FIT eight folds, VAL once and TEST
once. Future uses its eight FIT-role checkpoints; external evaluation uses all
ten frozen checkpoints. Errors are pooled over legal exposures, without
averaging predictions into an ensemble.

The training API exposes only FIT and VAL. Evaluation needs an explicit frozen
fold identity with matching `transform_sha256` and `phase='final'`. Feature
caching is optional, continuous, and restricted to the requested cities/time
span/purpose; labels remain selected by task row. The original bounded loader
access diagnostics and support hashes are retained.
