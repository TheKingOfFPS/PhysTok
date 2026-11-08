# ABC meteorological interventions

This is a **single-seed, Spatial-only** experiment: seed 20260826, ten folds.
Arm A reuses the corresponding Full main checkpoints. Arms B and C each train
10 new Full-architecture models on their own intervened FIT input and select on
correspondingly intervened VAL input. The experiment is not an inference-only
replacement applied to an A-trained model.

- **A**: aligned input and original fold-local preprocessing.
- **B**: the 39 meteorological channels in transport/dispersion,
  moisture/stability and precipitation/radiation are replaced by FIT-only
  region × local-month × local-hour means. Each of the 72 history hours uses
  its own conditioning key. This is done before refitting those channels'
  FIT-only imputation/scaling. Pollution, calendar and target statistics remain.
- **C**: the same 39 channels come from a coherent 72-hour donor episode in
  the same recipient city, month and local hour. The donor center is obtained
  by half-month rotation: 14–16 days away. The complete donor window is gathered
  rather than rotating each history timestamp separately. The issue-time state
  weather values are replaced through the final history row too. Pollution,
  calendar, timestamps, targets and target support stay with the recipient.

C's donor may be later than the recipient issue, so C is an explicitly
**retrospective intervention**, not an operationally available forecast input.
Donors stay within their recipient city and role; no cross-city FIT/VAL/TEST
mixing occurs. This does not make later donor weather available in real-time.

After rebuilding the main inputs and binding a data contract, prepare B once per
fold, then fit/evaluate its own model:

```bash
python scripts/run_ablation.py prepare --variant abc_b --fold 0 \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --prepared-root prepared/abc_b
python scripts/run_ablation.py fit --variant abc_b --fold 0 --seed 20260826 \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --prepared-root prepared/abc_b --output runs/abc_b/fold0 --device cuda:0
python scripts/run_ablation.py evaluate --variant abc_b \
  --data-root /path/to/rebuilt_inputs --contract /path/to/bound_contract.json \
  --prepared-root prepared/abc_b --training-manifest runs/abc_b/fold0/SUCCESS.json \
  --scenario Spatial --output predictions/abc_b/fold0 --device cuda:0
```

For C use `--variant abc_c` with its own fit/evaluation output and omit
`--prepared-root`. To score A use `--variant abc_a` with the seed826 Full main
`SUCCESS.json` as described in `experiments/README.md`. A cannot be refitted
through this interface.

B preparation computes the same frozen formulas but does not compare against
private historical climatology/preprocessing arrays: those arrays are not
redistributed. Generated software tests cannot establish equality to unavailable
historical observation values. Result verification can recheck the published
aggregate ABC contrasts and their recorded single-seed intervals. The retained
bootstrap definition is in `configs/ablations.json` (10,000 draws, seed 20260926).
