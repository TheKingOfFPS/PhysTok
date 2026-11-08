# T8 capacity controls

Variants: `phystok_small` (width 256; 7,021,701 parameters),
`itransformer_scaled` (width 1400; 27,284,497), `timexer_scaled` (width 1072;
27,316,513). Each has seed 20260826, ten folds and five scenarios. Reference
Full/original models reuse the same main-benchmark seed and fold.

Use `python scripts/run_ablation.py plan --variant phystok_small`, then the
`fit`/`evaluate` commands in `experiments/README.md`. All exact recipes are in
`configs/ablations.json`; width changes do not introduce a new HPO stage.
TimeXer requires locally installed authorized upstream source as described in
`baselines/README.md`. T8 remains single-seed supporting evidence; it does not
supply three-seed uncertainty or prove that parameter count alone explains
main-benchmark performance.
