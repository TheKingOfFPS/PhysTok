# Data integrity tools

`phystok.data.rebuild.verify_local_inputs` validates all bound city-level input files and fold artifacts. `phystok.evaluation.audits.audit_timestamps` checks issue/target/predictor legality, including internal same-year membership. The original numerical data audits can be invoked with `scripts/prepare_data.py ... audit` on user-rebuilt inputs.

No data, preprocessing statistics, private PASS receipts or raw-data access credentials are included. Runtime-generated fixtures in `tests/test_data.py` verify the actual loader and counterexamples.
