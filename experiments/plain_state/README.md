# Plain-State Flattening

Frozen variant: `plain_state_flatten`; three seeds 20260826/27/28, ten folds and
five scenarios. Six independent generic slots each project the complete ordered
68-feature state vector, with independent slot identities. History family
identity, all-open encoder and Full's history-only head remain. Parameter count:
27,449,733 (approximately 0.17% above Full).

Use `scripts/run_ablation.py`; the exact recipe and family definition are in
`configs/ablations.json`. The analysis uses 100,000 city-cluster draws, seed
20261024 and a ten-endpoint Bonferroni family (five scenarios × RMSE/MAE).
Spatial is the designated primary population informed by earlier results; retain
that disclosure and all family endpoints. MAE effects vary by population.
