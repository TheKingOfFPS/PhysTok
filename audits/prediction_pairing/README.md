# Prediction pairing tools

Run `python scripts/audit_predictions.py LEFT.parquet RIGHT.parquet` on user-generated files. The tool requires identical benchmark, seed, fold, scenario, physical task/sample IDs and true labels across the full ordered support. It never deletes unpaired records. The evaluator separately binds row selectors to the local city/time task index.

These files are model prediction exposures, not independent city samples. City-cluster statistics use one weight shared across a city's hours, fold exposures and seeds. Aggregate CSV verification does not perform this raw-prediction audit. `tests/test_training_and_audits.py` creates temporary synthetic files and verifies label-mismatch rejection.
