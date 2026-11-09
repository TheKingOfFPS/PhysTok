# PhysTok

Research code for **Population-Frozen Evaluation and Schema-Organized Tokens for City-Level Forecasting** (manuscript; author and repository-link revision, November 9, 2026).

This private staging repository accompanies the authors' verified PhysTok manuscript. V12 updates the author layout and repository link; its scientific content is unchanged from V11. It is not a claim of conference acceptance. A publication DOI and a public paper URL have not been assigned in this repository. The manuscript PDF is not distributed here.

## Overview

PhysTok forecasts one city's PM2.5 concentration at a 24-hour lead. It represents historical evolution with 54 trajectory tokens and groups 68 issue-time descriptors into six domain-state tokens. Shared encoding and a trajectory-only readout combine these two temporal roles. No city identifier or city graph is supplied to the model.

The study uses 50 development cities and ten external cities over 2023–2025. Ten city folds and three training seeds yield 210 main fits across PhysTok and six learning baselines. Five frozen deployment populations distinguish city membership and target year. Evaluation pools errors across legal checkpoint exposures and measures uncertainty with paired city clusters.

## Research contributions and findings

- **Protocol:** five city–year populations with explicit FIT/VAL/TEST permissions, frozen checkpoint exposures, pooled risk, and paired city-cluster intervals.
- **Finding:** four of six pairwise orders reverse between Spatial and Future-Year within the four-model cohort sharing a training-budget policy. Among all six learning baselines, XLinear moves from rank 2 to rank 5.
- **Method:** schema-organized histories and issue-time states enter a shared encoder and a joint trajectory readout. PhysTok reduces Spatial RMSE by **6.94%** relative to Informer under matched training-opportunity policies; the family interval excludes zero.

PhysTok has the lowest observed mean pooled RMSE in all five populations. The four shifted-population comparisons with TimeXer have city-cluster family intervals that include zero. The repository preserves these distinctions, city-level heterogeneity, and cases where MAE favors another model. See [results](docs/RESULTS.md).

## Repository structure

| Directory | Contents |
|---|---|
| `phystok/models/` | Frozen Full model, scalar baseline adapters, matched controls |
| `phystok/data/` | Causal window construction, fold transforms, local input rebuilding |
| `phystok/training/` | Permutation sampler, opportunity budgets, optimizer, VAL selection and exact resume |
| `phystok/evaluation/` | Frozen prediction, exposure checks and pooled metrics |
| `phystok/statistics/` | Results verification and paired city-cluster analysis |
| `configs/`, `experiments/` | Frozen scientific recipes and ablation entry points |
| `results/` | Aggregate CSVs and their provenance, with no observations or predictions |
| `data/metadata/` | City roster and split metadata, with no measurements |
| `audits/`, `tests/` | Reusable integrity tools and synthetic tests |

## Environment setup

Use Python 3.11 or 3.12 in a new environment. The CPU checks were performed with Python 3.11 and PyTorch 2.7.1 CPU. The original training runtime was Python 3.12 / PyTorch 2.7.0a0 / CUDA 12.8 on RTX 4090 GPUs with 48 GB. The portable CPU environment validates functionality; it does not certify bitwise agreement with that GPU runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python scripts/smoke.py
python scripts/evaluate.py --synthetic-smoke
```

For training on a GPU, obtain a compatible PyTorch CUDA wheel from the [official installation selector](https://pytorch.org/get-started/locally/). The frozen recipe uses FP32, disabled AMP and TF32, deterministic algorithms, and an explicit cuBLAS workspace. A single fit uses one device; independent fits can be scheduled on separate devices.

For raw-data reconstruction, also install the preprocessing extras:

```bash
python -m pip install -e '.[data]'
```

## Data availability

**No datasets are distributed.** This includes raw CNEMC/ERA5 records, cleaned time series, input matrices, feature banks, checkpoints, optimizer states and prediction Parquet files. Researchers must obtain the source data themselves and confirm their usage rights. Preprocessing code reconstructs inputs locally from those user-supplied files. Original releases or catalog metadata may change; exact historical reconstruction may be limited by source availability.

Start with [data sources, access and licenses](docs/DATA_SOURCES.md), [schema](docs/DATA_SCHEMA.md), and `python scripts/prepare_data.py --help`. The historical feature contract is retrospective: archived meteorology and complete-month target quality control do not establish operational release-time availability.

## Reproducing the main experiments

After rebuilding the local input banks, task indices, fold transforms and their bound data contract:

```bash
# Inspect all 210 jobs; does not train.
python scripts/run_benchmark.py --output runs/benchmark_v2

# One model/seed/fold. Use the data contract generated for YOUR rebuilt inputs.
python scripts/train.py --config configs/benchmark_v2.json \
  --data-root ../local_data --data-contract ../local_data/data_contract.json \
  --model phystok --seed 20260826 --fold 0 \
  --device cuda:0 --output runs/benchmark_v2/fits/phystok/20260826/fold0

# Run missing jobs, freeze all 210 accepted fits, then evaluate all populations.
python scripts/run_benchmark.py --execute --config configs/benchmark_v2.json \
  --data-root ../local_data --data-contract ../local_data/data_contract.json \
  --device cuda:0 --output runs/benchmark_v2
```

The default benchmark plan trains 180 baseline fits before the 30 PhysTok fits, then freezes the complete cohort before evaluating TEST populations. It performs no HPO. A failed attempt is not silently retried: use a complete identity-bound resume checkpoint or a new reviewed attempt. Run-level TEST and external targets never select weights. The study's earlier recipe development inspected TEST; this is a fixed, Fold0-developed, non-nested evaluation, not a claim of independence from all historical research decisions.

Five populations reuse the same model/seed/fold checkpoint and transform. Future-Year has eight legal FIT-checkpoint exposures per physical target, internal Spatial and Joint have one, and external populations have ten. Metrics pool errors across exposures; they do not average predictions into an ensemble. Sample SD is computed across three seed-level ten-fold pooled metrics (`ddof=1`), never across folds or cities. Persistence has no invented training seeds.

TimeXer and XLinear source redistribution is unresolved at the frozen versions. Their code is intentionally omitted. [Third-party instructions](docs/THIRD_PARTY_LICENSES.md) specify exact commits, file hashes and the adapter installation procedure after the user obtains appropriate permission. Other vendored components retain their upstream license notices.

## Reproducing ablations

`python scripts/run_ablation.py --help` exposes the frozen capacity, meteorological, Delayed, Strict Late Fusion A, and Plain-State experiments. [Experiment protocols](docs/EXPERIMENT_PROTOCOL.md) distinguish their scopes and reference reuse. Capacity and ABC use seed 20260826 and ten folds; the three architecture controls use three seeds and ten folds. Full references come from the main run; they are not reselected on TEST.

ABC arms modify both meteorological trajectories and issue-time state. Donor intervention is retrospective. Delayed's prespecified Joint RMSE comparison does not establish a clear independent difference. The controls vary more than a single unconstrained interpretation of “fusion”; their exact inputs, attention and readout are documented.

## Results verification without datasets

```bash
python scripts/reproduce_tables.py --help
python scripts/reproduce_tables.py --mode verify
```

The aggregate CSVs and sufficient statistics support re-pooling errors, three-seed statistics, paired contrasts and the published city-level summaries. They contain no labels, individual residuals or timestamped predictions. Full bootstrap reconstruction is a separate CPU mode described in [statistical analysis](docs/STATISTICS.md).

**Results Verification** checks numerical consistency of the released aggregate evidence. **Full Reproduction** additionally requires obtaining data, reconstructing inputs, training, generating predictions and auditing those predictions. Aggregate verification cannot establish the correctness of the original labels, preprocessing or raw prediction records.

## Limitations and verification scope

- Results are conditional on the observed city roster, years, seeds, measurement support and frozen optimization recipes. A city-cluster interval is not a guarantee for arbitrary cities or initializations.
- Some baselines have different native input adapters and model-specific recipes. The full baseline table is not a causal test of token schema alone.
- Capacity and ABC are single-seed checks. An interval crossing zero does not demonstrate statistical equivalence.
- The test suite generates clearly synthetic data at runtime. These fixtures are not research observations or a replacement benchmark.
- Full raw-data and GPU reproduction has not been executed while preparing this public candidate. See [reproducibility status](docs/REPRODUCIBILITY.md) for the exact checks performed and unresolved access restrictions.

## Citation

Use `CITATION.cff` or cite the manuscript title and authors. No acceptance, DOI, publication page range, or release URL is asserted.

## License and release status

This is a **local candidate pending author approval**. A license for project-authored code has not been selected. See `LICENSE`, [ownership and license choices](docs/LICENSE_REVIEW.md), and [third-party licenses](docs/THIRD_PARTY_LICENSES.md). Candidate MIT and Apache-2.0 texts are proposals, not grants. Any eventual code license does not grant permission to redistribute CNEMC, ERA5 or other data. The repository is currently private at https://github.com/TheKingOfFPS/PhysTok. Public release and the final license remain subject to author approval. See [private staging status](docs/PRIVATE_REPOSITORY_STATUS.md); the November 8 audit records the earlier local-only preparation.
