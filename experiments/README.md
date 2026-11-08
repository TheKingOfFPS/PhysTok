# Frozen ablation runs

The public matrix contains **140 new fits**, in addition to the 210 main fits:
30 capacity fits, 20 weather-intervention fits, and 90 representation-control
fits. It does not fit extra delayed-interaction timings. Generated plans are
software instructions; they are not evidence that the experiments were rerun.

All experiments consume the same user-rebuilt city arrays, split selectors and
bound data contract as the main benchmark. `configs/ablations.json` contains the
actual frozen model-specific recipes, seeds, folds, scenarios, parameter counts
and uncertainty definitions. No HPO or TEST-driven candidate selection occurs.
The runner seeds CPU initialization through the main training entry point and
retains its FP32, FIT/VAL selection and qualified stopping rules.

```bash
python scripts/run_ablation.py plan
python scripts/run_ablation.py fit --variant phystok_small --fold 0 --seed 20260826 \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --output runs/ablations/phystok_small/fold0/seed20260826 --device cuda:0
python scripts/run_ablation.py evaluate --variant phystok_small \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --training-manifest runs/ablations/phystok_small/fold0/seed20260826/SUCCESS.json \
  --scenario Spatial --output predictions/phystok_small/fold0/Spatial --device cuda:0
```

Repeat each new fit/evaluation only over its frozen matrix. `plan --output`
creates a JSON task list without starting training. Use a fresh output directory
for a new attempt; the runner does not silently overwrite an existing fit.

## Reusing selected main references

`abc_a`, `phystok_full`, `itransformer_original` and `timexer_original` are
reference names, never new fitting variants. Use their **seed 20260826** main
checkpoint selected on VAL in the same fold. The runner rejects a different
model/seed/benchmark or a non-reference control checkpoint.

```bash
python scripts/run_ablation.py evaluate --variant abc_a \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --training-manifest runs/main/phystok/fold0/seed20260826/SUCCESS.json \
  --scenario Spatial --output predictions/abc_a/fold0/Spatial --device cuda:0
```

The main model must already be accepted and its checkpoint frozen. A reference
can reuse its existing predictions instead of generating them again, provided
the corresponding checkpoint, transform, legal support and prediction hashes
are checked by the evaluation/audit tools. No checkpoints or raw predictions
are supplied by the repository.

## Interpretation and validation limits

The model-count/shape, generated-batch and donor-mapping tests are software
checks. They do not reproduce the actual 50-city inputs, historical checkpoints
or frozen experiment outcomes. Full reproduction requires researchers' lawful
source data and, for TimeXer, confirmed use rights to its pinned source.
Summary verification uses the separately provided aggregate result CSVs.
