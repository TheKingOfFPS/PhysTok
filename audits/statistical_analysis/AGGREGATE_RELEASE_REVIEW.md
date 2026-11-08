# Candidate aggregate-result release review

This is a local candidate review, not legal approval or a dataset license grant.

## Included information

- Author-generated pooled error metrics and sample SD, paired differences, conditional intervals, model ranks, and descriptive city/region comparisons already used in the manuscript.
- City/seed/scenario and fold/seed/scenario error sufficient statistics: only n, SSE, SAE, and derived error metrics. Counts include the specified repeated checkpoint exposures. The smallest city group contains 7,092 legal predictions; ABC's minimum is 16,003. No true values, predictions, timestamps, observation samples, features, or station-level series occur in these tables.
- City/region/fold membership and checkpoint-exposure counts; these identify the evaluation design rather than the data values.
- Sanitized run/prediction identity hashes and aggregate training counters. GPU UUIDs, server paths, hostnames, user names, private manifests, and referenced binary files were not copied. Hashes support binding if an authorized person later possesses an artifact; hashes alone do not prove its contents or correctness.

`results/PROVENANCE.json` gives each public result's historical relative source identifier, source hash, seed/fold scope, and aggregation definition. These are documentary identifiers, not paths read at runtime. Original audit documents, downloaded raw logs, bootstrap arrays, input matrices, checkpoints, and prediction Parquet were excluded.

## Permission boundary

These tables are research-output aggregates rather than a repackaged source dataset; a reader cannot reconstruct the original hourly observations or predictions from these error sums in general. That property does not itself settle copyright, database rights, source access terms, employment/institution ownership, collaborator consent, or manuscript disclosure rules.

No third-party data redistribution license is asserted. Final publication still requires the authors/institution to approve sharing their aggregate research results and to confirm source terms permit the proposed aggregate publication. The repository-wide license and data-source reviews control this release decision. If a rights holder restricts an aggregate table, withhold that table and clearly reduce the available verification scope rather than silently claiming full reproduction.

## Actual verification

The arithmetic verifier independently pools the released sufficient statistics and reconstructs all published conditional intervals using the original random generators, family sizes, and seeds. It does not reread or authenticate the original private prediction/label records. Consequently this release supports **Results Verification**, while **Full Reproduction** separately requires legally obtained data and end-to-end training/prediction audits.
