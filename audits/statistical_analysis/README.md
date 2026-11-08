# Aggregate statistical verification

The executable entry point is `python scripts/reproduce_tables.py`. See `docs/STATISTICS.md` for estimands, city clustering, exposure handling, seed SD, and the separate frozen multiplicity families. `docs/RESULTS.md` describes what the manuscript claims and the limits of each comparison.

`VERIFICATION_RECEIPT.json` records the included snapshot's successful complete bootstrap reconstruction and ten synthetic arithmetic/failure-gate unit tests. It is a small reproducibility receipt, not an export of the internal audit/log archive. Re-running the command writes a new user-owned receipt under `outputs/`.

The result manifest records each released table's historical relative source artifact/hash and aggregation definition. No runtime command follows those historical source paths. No raw observation, feature, prediction, checkpoint, server configuration, or private path is included.

The minimum city sufficient-statistic group has 7,092 legal records; ABC city groups have at least 16,003. Grouped error sums are not individual observations. Their candidate release still requires the authors' ownership and source-term review; this audit does not grant dataset redistribution rights.
