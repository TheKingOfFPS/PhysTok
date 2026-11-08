# Reproducibility and actual verification status

## Two modes

**Results Verification** uses released aggregate CSVs, city/seed sufficient statistics and checkpoint/prediction hashes. It re-pools SSE/SAE/n, computes three-seed sample SD and paired differences, reconstructs city-cluster bootstrap intervals, and checks city/region/leave-one-city-out summaries. This does not read or validate original observations, raw predictions or checkpoint tensors.

**Full Reproduction** requires the researcher to obtain the original data and station registry lawfully, reconstruct the input banks and fold transforms, install the two externally licensed baseline cores where authorized, perform training and generate frozen predictions. Original inputs and the original GPU runtime are not distributed. Exact historical raw snapshots and institutional permissions remain prerequisites.

## Local input reconstruction

Follow `docs/DATA_SOURCES.md` for the precise user-supplied files, source availability and usage terms. Run these steps against a new data workspace, never the code repository:

```bash
python scripts/prepare_data.py --data-root ../local_data init
# Place legally obtained CNEMC files and the station registry in the documented paths.
python scripts/prepare_data.py --data-root ../local_data cnemc
python scripts/prepare_data.py --data-root ../local_data era5 --plan-only
# Download only when YOU have accepted CDS terms and configured your own account.
python scripts/prepare_data.py --data-root ../local_data era5
python scripts/prepare_data.py --data-root ../local_data internal
python scripts/prepare_data.py --data-root ../local_data external
python scripts/prepare_data.py --data-root ../local_data audit
python scripts/prepare_data.py --data-root ../local_data bind
```

`../local_data` is a sibling workspace outside the code repository. The resulting `../local_data/data_contract.json` binds the actual internal/external feature and target Zarr files, task Parquet files and fold metadata by relative file names, byte counts and SHA-256. Formal fit, evaluation and Persistence entry points re-read and hash the bound files before use. A changed, added, missing or symlinked input is rejected. Full content verification adds I/O proportional to the city-level input package; it avoids relying on file timestamps as proof of content identity.

The numerical preprocessing code was retained from the research pipeline. Legacy operational receipts are not distributed. Fresh metadata and ERA5 support checks are generated from the user's own inputs, with their limited scope stated in those receipts. The old CAMS acquisition gate is not part of this CNEMC/ERA5 benchmark. No historical internal PASS file is copied to certify a new dataset.

## Verified during candidate preparation

The independent environment was created without `--system-site-packages`. CPU PyTorch and all runtime dependencies were installed into it; project imports resolved to this candidate, not the private research package. Tests use generated data and temporary directories.

- Full 27,403,141-parameter PhysTok import, scalar forward/backward and an optimizer update.
- Baseline/control parameter counts and output shapes; Full rejects a silent Small-width replacement.
- A complete training interruption/resume reproduces the uninterrupted model state, optimizer accounting, sampler order and validation history exactly on a generated CPU problem.
- The real local loader reads runtime-generated Zarr/Parquet, applies fold transforms, constructs windows/labels and enforces access roles.
- Input membership/content tampering and same-path task-table rebuilding are tested.
- The preprocessing scripts compile/import inside the clean environment; local metadata initialization and ERA5 plan-only output match the original scientific request columns (555 internal and 111 external requests). No data download is performed by these checks.
- Pooled metrics, invalid supports, city-weight sharing, seed SD, uncertainty families, prediction pairing, stale result/cohort rejection and complete-run requirements are tested.
- Frozen main and ablation intervals are recomputed from public sufficient statistics; original observation/prediction correctness is outside that mode.
- The TimeXer/XLinear local installer and scalar forwards, including enlarged TimeXer, were checked separately on generated CPU tensors in a temporary authorized-source installation. Their source is absent from the public candidate.

Executable checks:

```bash
python -m pytest
python scripts/smoke.py
python scripts/evaluate.py --synthetic-smoke
python scripts/reproduce_tables.py --mode verify
python scripts/reproduce_tables.py --mode bootstrap-all
python scripts/audit_public_release.py
```

Exact check totals and runtime versions are recorded in `audits/release_validation/`. Source-equivalence checks against the frozen original were performed separately from clean-environment functional checks, and their runtimes are stated. Neither is described as a new scientific experiment.

Git attributes preserve file bytes rather than converting line endings. Some original source and aggregate CSV files use CRLF; automatic conversion would invalidate their recorded SHA-256 values even when parsed values look identical.

## Not executed here

No original raw-data-to-final rebuild, full benchmark fit, full ablation training, GPU evaluation, or comparison of regenerated predictions with historical predictions was executed in preparing this candidate. There is no claim that new hardware, alternate source snapshots or a newer dependency release will reproduce every bit of the original GPU results. Researchers must retain local source manifests and report deviations rather than substituting a new dataset or recipe under the old identity.

## Exact resume and new attempts

Use a complete resume checkpoint plus its SHA-256, in the original output directory and the same numerical runtime/device. A best-only checkpoint is insufficient. Do not discard failed attempts based on TEST quality. Cached results are accepted only when their selected checkpoint, transform, source, data contract and current training identity agree. All accepted main fits must satisfy the frozen qualified stopping rule before a complete cohort can be evaluated.
