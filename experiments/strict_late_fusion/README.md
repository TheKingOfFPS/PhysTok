# Strict Late Fusion A

Frozen variant: `strict_late_fusion`; three seeds 20260826/27/28, ten folds and
five scenarios. All four shared-weight encoder layers block history/state
interaction. A dense head first combines all 60 encoded outputs, with hidden
width 461 and 27,409,183 total parameters. This changes both interaction policy
and readout; it is not a readout-preserving timing-only control.

Use `scripts/run_ablation.py`; the exact recipe and 20-endpoint uncertainty
family (two reference contrasts × five scenarios × two metrics) are in
`configs/ablations.json`. The frozen analysis uses 100,000 city-cluster draws,
seed 20261023, and Bonferroni family intervals. Keep all endpoints, including
those crossing zero. Full's supported Spatial comparison should not be
extrapolated to an all-population independent component benefit.
