# Checkpoint and exposure tools

`phystok.training.app.freeze_cohort` and `validate_cohort` require the complete 210-fit main grid, best-VAL checkpoint hashes, qualified stopping, and consistent recipes/source/data. `validate_evaluation_binding` rejects stale prediction caches. `phystok.evaluation.audits.checkpoint_exposures` checks that populations share a selected checkpoint and transform.

`tests/test_public_runner_guards.py` provides positive and negative generated metadata cases. No checkpoint tensors are distributed. Released hashes identify historical records but cannot verify the absent tensor files by themselves.
