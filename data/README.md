# Data availability

There are no training data, processed time series, raw grids, prediction files,
checkpoints or fitted preprocessing statistics in this directory.

`metadata/` contains the frozen city/region roster, city indices, ten fold
assignments and pre-specified preprocessing/evaluation rules. Coordinates and
city identifiers are design metadata, not measurement records. The original
manifest's empirical missingness column and unrelated legacy AQ-grid key are
excluded. CNEMC station registries are third-party data and must be obtained by
the researcher, rather than copied from this repository.

See [verified source/access instructions](../docs/DATA_SOURCES.md) and
[the array and task schema](../docs/DATA_SCHEMA.md). Running formal training
requires researchers to obtain the raw data lawfully and reconstruct the input
stores locally. The package's software license does not license those data.
