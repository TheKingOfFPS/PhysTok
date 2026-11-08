# PhysTok public release candidate audit

**Status: LOCAL CANDIDATE COMPLETED; PUBLICATION AWAITS AUTHOR APPROVAL.**

Prepared on November 8, 2026. The controlling manuscript is **PhysTok_ICDE2027_v11_author_stars_20261108.pdf**, titled *Population-Frozen Evaluation and Schema-Organized Tokens for City-Level Forecasting*. The earlier V8 reference in the initial task was superseded by the author's explicit V11 instruction. The PDF itself is not included in this repository.

## What this candidate contains

An independent source repository containing the actual Full PhysTok, scalar adapters, frozen main recipe, data/schema/split reconstruction, 210-fit planning and execution, frozen evaluation, five classes of control experiments, aggregate results, statistical verification, and targeted audits. It was assembled from selected reviewed sources; it is not a copy of the research repository or its Git history.

Full PhysTok retains 54 trajectories, shared 12/24/72-hour projections, six state families spanning 68 issue-time features, variable/family/view identities, four encoder layers, and trajectory-only readout. Its parameter count is 27,403,141. The tests reject silently substituting the Small variant. Frozen models and controls were also compared against their actual research definitions, with source and numerical equivalence recorded separately from the clean-environment tests.

All main scientific recipe fields match the frozen V2 configuration. Optimizer, sampler, stopping, full-VAL selection and complete resume logic were retained. Private server authorization, allocation and supervision infrastructure was removed. Equivalent scientific integrity checks remain portable: actual input byte binding, checkpoint/transform identity, complete cohorts, stale-cache rejection and exclusive artifact publication.

## Excluded material

No raw CNEMC/ERA5 data, cleaned time series, input matrices, feature banks, NetCDF/GRIB, preprocessing NPZ files, trained weights, optimizer snapshots, prediction Parquet, server logs, SSH configuration, credentials or original Git history is included. Third-party station registry data must be acquired by the researcher. The full manuscript and its figures are also excluded pending any separate publication decision.

TimeXer and XLinear source redistribution could not be established from their pinned upstream versions. Their full source is excluded. The candidate includes verified commit/file hashes and an installer that requires an authorized local checkout and explicit confirmation of use rights. It does not download or mirror the code. The other vendored cores retain their licenses and modification notices.

## Data acquisition and reconstruction

`docs/DATA_SOURCES.md` provides verified QuotSoft/CNEMC and official CDS single-level/pressure-level/API entry points, required input coverage, layout, units, access conditions and license limitations. No third-party mirror is created. QuotSoft/CNEMC historical redistribution rights are not inferred from public access; exact old snapshots and the specified station registry may impose practical access limits.

The source pipeline reconstructs station-derived unrounded targets, ERA5 physical/city inputs, features, selectors and FIT-only fold transforms. A local contract binds the actual banks, targets, task indices and metadata. Hash verification reads actual bytes at formal entry points and rejects changed, added, missing and symlinked files. Task metadata caching is keyed by actual content to prevent cross-rebuild contamination.

No full real-data rebuild was executed in this release preparation. Synthetic tests and plan checks verify code and contracts, not the original observations. ERA5 planning reproduced 555 internal and 111 external scientific requests without sending downloads. These boundaries are stated in the README and reproducibility guide.

## Clean-environment execution

A new Python 3.11 virtual environment was created without inherited system packages. It used PyTorch 2.7.1 CPU, NumPy 2.2.6 and the recorded dependencies. Imports, model execution, tests and aggregate statistics were actually run. No GPU training or server connection was used.

| Check | Result | Evidence |
|---|---|---|
| Full model import/forward/backward/update | PASS; generated CPU tensors; output `[2]`; 27,403,141 parameters | `audits/release_validation/SYNTHETIC_TRAINING.json` |
| Evaluation CLI | PASS; generated CPU inputs, RMSE/MAE/SSE/SAE/n | `audits/release_validation/SYNTHETIC_EVALUATION.json` |
| Targeted test suite | 42 passed | `audits/release_validation/TEST_RESULTS.json` |
| Complete resume equivalence | PASS; sampler, optimizer, model state and validation history replay | Training tests |
| Loader, fold and input integrity | PASS; actual runtime-generated Zarr/Parquet; content tampering and stale metadata rejected | Data tests and `DATA_REBUILD_CHECKS.json` |
| Model/control equivalence to frozen sources | PASS for the eight compared definitions | `MODEL_EQUIVALENCE.json` (runtime stated) |
| Authorized external model installation/forward | PASS in isolated temporary installation, including enlarged TimeXer | `EXTERNAL_CORE_INSTALLER.json` |
| Aggregate arithmetic and interval reconstruction | 5,880/5,880 PASS, including main and control bootstrap | `STATISTICAL_RECOMPUTATION.json` |
| Independent runner/config review | 25 checks and eight stale-cache/cohort counterexamples passed | `INDEPENDENT_RUNNER_REVIEW.json` |
| Dependency consistency | PASS | `ENVIRONMENT.json` |
| Staged and committed content safety | PASS; no configured remote | `SECURITY_SCAN.json` and repeatable scanner |

The original GPU runtime was Python 3.12 / PyTorch 2.7.0a0 / CUDA 12.8. The CPU environment is a functional verification environment; it is not claimed to reproduce historical GPU tensors bit for bit. Exact resume rejects runtime/device drift.

## Results that can be checked without datasets

The released aggregates cover the 40-row main table, 105 seed-level pooled records, model/seed/fold provenance hashes, the twelve main paired endpoints, city/region/external sensitivity summaries, and all five experiment classes. Plain, Strict A, Delayed, ABC and T8 retain their original seed counts and uncertainty families. Each result file has a source/aggregation provenance entry.

The verifier re-pools sufficiently detailed SSE/SAE/n, computes seed means and sample SD (`ddof=1`), paired deltas, rank changes, city wins, regional summaries and leave-one-city-out results. Its full mode regenerates the original main 100,000-draw bootstrap and each control's frozen bootstrap. Cities share one weight across hours, checkpoint exposures and seeds; internal regions retain their strata, external cities do not acquire invented strata.

These are aggregate consistency checks. They cannot establish the correctness of absent source labels, original prediction rows, preprocessing outputs or model tensors. Hash registries identify absent artifacts; the hashes do not substitute for inspecting them.

The paper's conclusions are preserved: Spatial RMSE improves by 6.94% against Informer with a family interval excluding zero. The four shifted TimeXer RMSE comparisons are observed mean leads whose family intervals include zero. City gains vary; MAE can favor TimeXer. Delayed's primary Joint comparison does not establish a clear difference. ABC and capacity remain single-seed supporting checks.

## Full reproduction status

| Component | Runnable implementation | Actually executed here |
|---|---|---|
| Raw acquisition/preprocessing | Included; requires source access, registry and CDS account | Metadata init, ERA5 plans, source imports and synthetic numerical/loader tests; no real-data rebuild |
| Main 210 fits and five-population evaluation | Included | Synthetic CPU training/evaluation and guard tests; no formal fits |
| Capacity, ABC, Delayed, Strict A, Plain | Frozen variants/configs/entry points included | Generated CPU component tests and model equivalence; no formal ablation retraining |
| TimeXer/XLinear | Adapters and authorized-install instructions included; core redistribution unresolved | Local temporary installation and forward checks only |
| Aggregate verification/bootstrap | Included and data-free | Full released evidence recomputed successfully |

## Ownership, approval and licensing

No final license was selected on the authors' behalf. `LICENSE` states that no first-party license grant has yet been made. `licenses/candidates/` contains MIT and Apache-2.0 candidates; `docs/LICENSE_REVIEW.md` recommends Apache-2.0 for consideration and explains the alternative and existing third-party obligations.

Before public release, the author must approve first-party/institutional ownership, the actual code license, aggregate/metadata release and reuse terms, and the manuscript's permitted public status. The unresolved TimeXer/XLinear redistribution issue must remain explicitly documented unless rights are obtained. Neither candidate code license authorizes third-party dataset redistribution.

## File and Git review

`PUBLIC_FILE_MANIFEST.csv` lists the actual staged public files, byte counts and SHA-256 values. Its own row is explicitly self-excluded from hashing. Git attributes preserve exact source/CSV line endings so that a clean checkout does not silently invalidate frozen file hashes.

`scripts/audit_public_release.py` reads the actual index or a specified commit, checks private paths, credential patterns, internal network addresses, unexpected binaries, data/prediction columns, unconfirmed-license cores and remotes, and supplements the manual source/result/license review. This does not rely on `.gitignore` alone. The final independent repository has a new local Git history; inspect it with `git status --short`, `git log --oneline` and `git remote -v`.

## Publication decision

The local candidate is technically prepared and has passed the stated execution and content checks. It is **not yet cleared for publication**: the author retains the final decision on rights, license, aggregate reuse and release. No GitHub repository, push, external upload or public Release has been performed. This task stops after delivery of the local candidate and its audit evidence.
