# Data sources and local reconstruction

This repository distributes **no dataset**, including no raw observations,
processed time series, model-input matrices, feature caches, predictions, or
weights. The metadata in `data/metadata/` describe the authors' fixed city and
fold choices and contain no observed concentrations or meteorological values.
The code license does not grant rights to third-party data.

## Sources verified on November 8, 2026

| Source | Verified entry | Access and rights |
|---|---|---|
| CNEMC | [China National Environmental Monitoring Centre](https://www.cnemc.cn/) and its linked [national real-time air-quality platform](https://air.cnemc.cn/) | Official source of the observations; a current real-time portal is not a promise of a complete historical bulk archive. Obtain permission applicable to your use. |
| QuotSoft historical CNEMC archive | [Historical air-quality data and file documentation](https://quotsoft.net/air/) | The page identifies CNEMC as the national data source and documents daily station files and the station registry. A clear general redistribution license was not established. Confirm your right to obtain and use the archive; no mirror is supplied here. |
| ERA5 single levels | [Official CDS dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) | Register your own CDS account and accept the dataset terms. The download page currently labels a CC-BY license; record the actual accepted license and required attribution. |
| ERA5 pressure levels | [Official CDS dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) | Obtain the 925 and 850 hPa fields from CDS with the same account/terms process. |
| CDS API | [Official setup instructions](https://cds.climate.copernicus.eu/how-to-api) | Use your own credentials outside the repository. The official instructions require accepting dataset terms manually before API download. |

The QuotSoft page documents national station files named
`china_sites_YYYYMMDD.csv` and registry files under its archive download entry.
The documented daily URL pattern is
`https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv`; replace the date rather
than treating this literal template as a file. Missing dates and changes to
provider snapshots can affect exact numerical reconstruction. Do not substitute
rounded archived city means for the station-derived unrounded target.

## Required source coverage

The frozen study uses December 2022 as history and targets in 2023–2025.
CNEMC preparation expects 1,127 complete daily station files for
2022-12-01 through 2025-12-31 plus the original registry version named
`站点列表-2022.02.13起.csv`. The registry must contain station code/name, city,
longitude, latitude, and the control-site flag. The package does not provide
that third-party registry. Retain its source/version and checksum locally.

Create this user-owned workspace outside the code repository:

```text
local_data/
  01_raw/air_quality_cnemc_quotsoft/
    sites/daily/2022/china_sites_YYYYMMDD.csv
    sites/daily/2023/china_sites_YYYYMMDD.csv
    sites/daily/2024/china_sites_YYYYMMDD.csv
    sites/daily/2025/china_sites_YYYYMMDD.csv
    station_registry/source_files/站点列表-2022.02.13起.csv
```

The archived CNEMC `date` and `hour` are interpreted in Asia/Shanghai. Use the
hourly pollutant rows (`PM2.5`, `PM10`, `SO2`, `NO2`, `O3`, `CO`), not their
24-hour companions. PM2.5 and other listed concentrations are in µg/m³ except
CO, which is in mg/m³. The frozen station QC excludes control sites, converts
only PM2.5 values exactly 10,000 to missing, retains other extremes, and averages
finite station values without rounding. See the preserved canonical builder.

The ERA5 acquisition scripts request hourly 0.25° reanalysis for the frozen
regional bounds (3° halo, outward grid snap), with a December 2022 boundary
month. Three streams are retained: single-level instantaneous fields
(2 m temperature/dewpoint, surface pressure, 10 m u/v wind, boundary-layer
height, total cloud cover), precipitation/radiation accumulated fields, and
925/850 hPa temperature, humidity, u/v wind and geopotential. The raw GRIB files
and their local acquisition audits are generated in the user workspace.

## Commands

Install the package and the preprocessing extras described in the main README,
then run the following from the repository. These are real preparation commands;
they can be expensive and the ERA5 download contacts CDS only without
`--plan-only`.

```bash
python scripts/prepare_data.py --data-root ../local_data init
# First obtain the CNEMC files and registry lawfully at the paths above.
python scripts/prepare_data.py --data-root ../local_data cnemc
python scripts/prepare_data.py --data-root ../local_data era5 --plan-only
# Review requests, configure your own CDS account, accept the terms, then:
python scripts/prepare_data.py --data-root ../local_data era5
python scripts/prepare_data.py --data-root ../local_data internal
python scripts/prepare_data.py --data-root ../local_data external
python scripts/prepare_data.py --data-root ../local_data audit
python scripts/prepare_data.py --data-root ../local_data bind
```

`init` installs only frozen protocol metadata. `cnemc` reconstructs the station
coverage ledger, monthly Q95 support, eligibility and canonical city means.
`internal` executes the original numerical ERA5 canonicalization, physical
transforms, city interpolation, exact timestamp/city merge, task index,
72-feature construction, selectors and FIT-only transforms. `external` applies
the same frozen physical/feature functions to the ten external cities and fits
no external preprocessing. `bind` writes `local_data/data_contract.json` for the
locally rebuilt metadata. Pass that contract to training/evaluation; it does not
certify equality to unavailable original archive snapshots.

The historical pre-city-extraction gate also referenced obsolete CAMS assets.
The public orchestrator instead checks the actual reconstructed ERA5 timeline,
fields, pressure levels and all 50 city interpolations with the original
extraction functions. Its receipt explicitly identifies itself as a new local
audit. The physical formulas and downstream output gates are unchanged. No
historical internal PASS receipt is supplied or fabricated.

## Reproduction limits

Raw-data-to-training runs were not repeated for this release, because no data
are distributed and no new benchmark training is authorized. Unit tests use
runtime-generated synthetic arrays, marked as such. Source snapshot changes,
registry changes, unavailable archive days, or revised ERA5 files can change
results even when the same code is run. If the required historical archive or
registry cannot be obtained lawfully, full data reconstruction is unavailable;
Results Verification remains usable on the permitted aggregate CSV files.
Do not fill missing source days from other datasets while describing the run as
the frozen benchmark. Document any deliberate new dataset or protocol instead.
