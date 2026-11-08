#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


CANONICAL_ID = "cnemc_city_air_quality_canonical_v1"
START_LOCAL = pd.Timestamp("2022-12-01 00:00:00")
END_LOCAL = pd.Timestamp("2025-12-31 23:00:00")
POLLUTANTS = ("PM2.5", "PM10", "SO2", "NO2", "O3", "CO")

VAR_PREFIX = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "O3": "o3",
    "CO": "co",
}

UNITS = {
    "PM2.5": "ug m-3",
    "PM10": "ug m-3",
    "SO2": "ug m-3",
    "NO2": "ug m-3",
    "O3": "ug m-3",
    "CO": "mg m-3",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv_flexible(path: Path, **kwargs):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise last if last is not None else RuntimeError("CSV decode failed")


def hj_required_count(n_expected: float) -> float:
    if not np.isfinite(n_expected) or n_expected <= 0:
        return np.nan
    n = int(n_expected)
    if n >= 4:
        return float(math.ceil(0.75 * n))
    return float(math.ceil(0.50 * n))


def round_half_up_nonnegative(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = np.isfinite(x)
    out[m] = np.floor(x[m] + 0.5)
    return out


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    raw_root = root / "01_raw" / "air_quality_cnemc_quotsoft"
    sites_root = raw_root / "sites" / "daily"

    eligibility_path = root / "05_manifests" / "cnemc_station_eligibility_v1.csv"
    policy_path = root / "05_manifests" / "cnemc_station_qc_policy_v1.json"

    lifecycle_root = (
        root / "06_processed" / "audits"
        / "cnemc_station_lifecycle_validity_sensitivity_v1"
    )
    monthly_expected_path = (
        lifecycle_root / "monthly_expected_station_count_sensitivity.csv"
    )

    if not eligibility_path.exists() or not policy_path.exists():
        raise SystemExit("run freeze_cnemc_station_qc_policy.py first")
    if not monthly_expected_path.exists():
        raise SystemExit("missing lifecycle sensitivity artifact")

    eligibility = pd.read_csv(eligibility_path, encoding="utf-8-sig")
    eligible = eligibility.loc[eligibility["eligible_city_station"].eq(True)].copy()
    eligible["station_id"] = eligible["station_id"].astype(str)

    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)

    city_table = (
        eligible[
            ["region", "city_en", "city_zh"]
        ]
        .drop_duplicates()
        .sort_values(["region", "city_en"])
        .reset_index(drop=True)
    )
    if len(city_table) != 50:
        raise RuntimeError(f"expected 50 cities, found {len(city_table)}")

    city_keys = list(
        city_table[
            ["region", "city_en", "city_zh"]
        ].itertuples(index=False, name=None)
    )

    station_ids = eligible["station_id"].tolist()
    station_pos = {sid: i for i, sid in enumerate(station_ids)}
    city_station_ids = {
        key: eligible.loc[
            eligible["region"].eq(key[0])
            & eligible["city_en"].eq(key[1])
            & eligible["city_zh"].eq(key[2]),
            "station_id",
        ].tolist()
        for key in city_keys
    }
    city_station_pos = {
        key: np.array(
            [station_pos[sid] for sid in city_station_ids[key]],
            dtype=np.int64,
        )
        for key in city_keys
    }

    local_index = pd.date_range(START_LOCAL, END_LOCAL, freq="h")
    n_time = len(local_index)
    n_city = len(city_keys)

    # UTC coordinate is stored timezone-naive but explicitly documented as UTC.
    utc_index = (
        local_index.tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
        .tz_localize(None)
    )

    shape = (n_time, n_city)

    means = {
        p: np.full(shape, np.nan, dtype=np.float32)
        for p in POLLUTANTS
    }
    n_valid = {
        p: np.zeros(shape, dtype=np.int16)
        for p in POLLUTANTS
    }

    pm25_median = np.full(shape, np.nan, dtype=np.float32)
    pm25_std = np.full(shape, np.nan, dtype=np.float32)
    pm25_min = np.full(shape, np.nan, dtype=np.float32)
    pm25_max = np.full(shape, np.nan, dtype=np.float32)
    pm25_q25 = np.full(shape, np.nan, dtype=np.float32)
    pm25_q75 = np.full(shape, np.nan, dtype=np.float32)
    pm25_iqr = np.full(shape, np.nan, dtype=np.float32)
    pm25_mad = np.full(shape, np.nan, dtype=np.float32)
    pm25_exact10000_removed = np.zeros(shape, dtype=np.int8)

    for d in tqdm(
        pd.date_range(
            START_LOCAL.normalize(),
            END_LOCAL.normalize(),
            freq="D",
        ),
        desc="Build CNEMC canonical",
        unit="day",
    ):
        ymd = d.strftime("%Y%m%d")
        path = sites_root / str(d.year) / f"china_sites_{ymd}.csv"
        if not path.exists():
            raise RuntimeError(f"missing raw site file: {path}")

        usecols = ["date", "hour", "type"] + station_ids
        df = read_csv_flexible(path, usecols=usecols)
        df["type"] = df["type"].astype(str).str.strip()

        for pollutant in POLLUTANTS:
            block = df.loc[df["type"].eq(pollutant)].copy()
            if block.empty:
                continue

            hours = block["hour"].astype(int).to_numpy()
            if np.any((hours < 0) | (hours > 23)):
                raise RuntimeError(
                    f"{path.name}: unexpected hour labels {sorted(set(hours))}"
                )

            matrix = (
                block[station_ids]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )

            # Rows need not be complete or ordered; map each archived hour
            # independently into the canonical local-clock grid.
            day_start = (d.normalize() - START_LOCAL.normalize()).days * 24
            time_indices = day_start + hours

            for ci, key in enumerate(city_keys):
                vals = matrix[:, city_station_pos[key]].copy()

                if pollutant == "PM2.5":
                    saturation = np.isfinite(vals) & (vals == 10000.0)
                    removed = saturation.sum(axis=1).astype(np.int8)
                    vals[saturation] = np.nan
                    pm25_exact10000_removed[time_indices, ci] = removed

                finite = np.isfinite(vals)
                nv = finite.sum(axis=1).astype(np.int16)
                n_valid[pollutant][time_indices, ci] = nv

                nonempty = nv > 0
                if not np.any(nonempty):
                    continue

                with np.errstate(all="ignore"):
                    row_mean = np.nanmean(vals, axis=1)
                means[pollutant][time_indices, ci] = row_mean.astype(np.float32)

                if pollutant == "PM2.5":
                    with np.errstate(all="ignore"):
                        med = np.nanmedian(vals, axis=1)
                        std = np.nanstd(vals, axis=1, ddof=0)
                        minv = np.nanmin(vals, axis=1)
                        maxv = np.nanmax(vals, axis=1)
                        q25 = np.nanpercentile(vals, 25, axis=1)
                        q75 = np.nanpercentile(vals, 75, axis=1)
                        mad = np.nanmedian(
                            np.abs(vals - med[:, None]),
                            axis=1,
                        )

                    empty = ~nonempty
                    for arr in (med, std, minv, maxv, q25, q75, mad):
                        arr[empty] = np.nan

                    pm25_median[time_indices, ci] = med.astype(np.float32)
                    pm25_std[time_indices, ci] = std.astype(np.float32)
                    pm25_min[time_indices, ci] = minv.astype(np.float32)
                    pm25_max[time_indices, ci] = maxv.astype(np.float32)
                    pm25_q25[time_indices, ci] = q25.astype(np.float32)
                    pm25_q75[time_indices, ci] = q75.astype(np.float32)
                    pm25_iqr[time_indices, ci] = (q75 - q25).astype(np.float32)
                    pm25_mad[time_indices, ci] = mad.astype(np.float32)

    # Candidate station-count validity companions.
    monthly = pd.read_csv(
        monthly_expected_path,
        encoding="utf-8-sig",
    )
    month_key = local_index.strftime("%Y-%m")

    expected_cov50 = np.full(shape, np.nan, dtype=np.float32)
    expected_mode = np.full(shape, np.nan, dtype=np.float32)
    expected_q95 = np.full(shape, np.nan, dtype=np.float32)

    monthly_lookup = {}
    for row in monthly.itertuples(index=False):
        monthly_lookup[
            (row.region, row.city_en, row.city_zh, row.month)
        ] = (
            float(row.active_count_cov50),
            float(row.expected_count_mode),
            float(row.expected_count_q95),
        )

    # December 2022 was not in lifecycle audit. Use number of stations with
    # monthly PM2.5 archive coverage >=0.50 from the earlier roster audit.
    coverage_path = (
        root / "06_processed" / "audits"
        / "cnemc_station_registry_50city_roster_v1"
        / "station_monthly_pm25_coverage.csv"
    )
    coverage = pd.read_csv(coverage_path, encoding="utf-8-sig")
    coverage = coverage.loc[
        coverage["control_site"].ne("Y")
        & coverage["month"].eq("2022-12")
    ].copy()

    dec22_expected = (
        coverage.assign(
            active=coverage["pm25_archive_coverage"].ge(0.50)
        )
        .groupby(["region", "city_en", "city_zh"])["active"]
        .sum()
        .to_dict()
    )

    for ci, key in enumerate(city_keys):
        for ti, m in enumerate(month_key):
            if m == "2022-12":
                e = float(dec22_expected.get(key, np.nan))
                expected_cov50[ti, ci] = e
                expected_mode[ti, ci] = np.nan
                expected_q95[ti, ci] = np.nan
            else:
                vals = monthly_lookup.get((*key, m))
                if vals is None:
                    continue
                expected_cov50[ti, ci] = vals[0]
                expected_mode[ti, ci] = vals[1]
                expected_q95[ti, ci] = vals[2]

    def make_validity(expected):
        required = np.full_like(expected, np.nan, dtype=np.float32)
        finite = np.isfinite(expected)
        flat = expected[finite]
        required_vals = np.array(
            [hj_required_count(x) for x in flat],
            dtype=np.float32,
        )
        required[finite] = required_vals

        valid = (
            np.isfinite(means["PM2.5"])
            & np.isfinite(required)
            & (n_valid["PM2.5"] >= required)
        )
        coverage_fraction = np.full_like(expected, np.nan, dtype=np.float32)
        ok = np.isfinite(expected) & (expected > 0)
        coverage_fraction[ok] = (
            n_valid["PM2.5"][ok] / expected[ok]
        ).astype(np.float32)
        return required, valid, coverage_fraction

    req_cov50, valid_cov50, frac_cov50 = make_validity(expected_cov50)
    req_mode, valid_mode, frac_mode = make_validity(expected_mode)
    req_q95, valid_q95, frac_q95 = make_validity(expected_q95)

    pm25_reported = round_half_up_nonnegative(
        means["PM2.5"].astype(np.float64)
    ).astype(np.float32)

    coords = {
        "time": utc_index.to_numpy(dtype="datetime64[ns]"),
        "city": np.arange(n_city, dtype=np.int16),
    }

    data_vars = {
        "local_time": (
            ("time",),
            local_index.to_numpy(dtype="datetime64[ns]"),
        ),
        "local_year": (
            ("time",),
            local_index.year.to_numpy(dtype=np.int16),
        ),
        "local_month": (
            ("time",),
            local_index.month.to_numpy(dtype=np.int8),
        ),
        "local_hour": (
            ("time",),
            local_index.hour.to_numpy(dtype=np.int8),
        ),
        "region": (
            ("city",),
            city_table["region"].astype(str).to_numpy(dtype=str),
        ),
        "city_en": (
            ("city",),
            city_table["city_en"].astype(str).to_numpy(dtype=str),
        ),
        "city_zh": (
            ("city",),
            city_table["city_zh"].astype(str).to_numpy(dtype=str),
        ),
    }

    for pollutant in POLLUTANTS:
        prefix = VAR_PREFIX[pollutant]
        data_vars[f"{prefix}_city_mean"] = (
            ("time", "city"),
            means[pollutant],
            {
                "units": UNITS[pollutant],
                "aggregation": "arithmetic mean of finite eligible non-control stations",
            },
        )
        data_vars[f"{prefix}_n_valid"] = (
            ("time", "city"),
            n_valid[pollutant],
        )

    data_vars.update(
        {
            "pm25_city_mean_reported_precision": (
                ("time", "city"),
                pm25_reported,
                {
                    "units": "ug m-3",
                    "note": "floor(unrounded_mean + 0.5); consistency companion only",
                },
            ),
            "pm25_city_median": (("time", "city"), pm25_median),
            "pm25_city_std": (("time", "city"), pm25_std),
            "pm25_city_min": (("time", "city"), pm25_min),
            "pm25_city_max": (("time", "city"), pm25_max),
            "pm25_city_q25": (("time", "city"), pm25_q25),
            "pm25_city_q75": (("time", "city"), pm25_q75),
            "pm25_city_iqr": (("time", "city"), pm25_iqr),
            "pm25_city_mad": (("time", "city"), pm25_mad),
            "pm25_exact10000_removed_count": (
                ("time", "city"),
                pm25_exact10000_removed,
            ),
            "pm25_expected_n_active_cov50": (
                ("time", "city"),
                expected_cov50,
            ),
            "pm25_required_n_hj_cov50": (
                ("time", "city"),
                req_cov50,
            ),
            "pm25_coverage_fraction_cov50": (
                ("time", "city"),
                frac_cov50,
            ),
            "pm25_valid_hj_cov50": (
                ("time", "city"),
                valid_cov50,
            ),
            "pm25_expected_n_mode": (
                ("time", "city"),
                expected_mode,
            ),
            "pm25_required_n_hj_mode": (
                ("time", "city"),
                req_mode,
            ),
            "pm25_coverage_fraction_mode": (
                ("time", "city"),
                frac_mode,
            ),
            "pm25_valid_hj_mode": (
                ("time", "city"),
                valid_mode,
            ),
            "pm25_expected_n_q95": (
                ("time", "city"),
                expected_q95,
            ),
            "pm25_required_n_hj_q95": (
                ("time", "city"),
                req_q95,
            ),
            "pm25_coverage_fraction_q95": (
                ("time", "city"),
                frac_q95,
            ),
            "pm25_valid_hj_q95": (
                ("time", "city"),
                valid_q95,
            ),
        }
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "canonical_id": CANONICAL_ID,
            "created_at_utc": utc_now(),
            "measurement_origin": (
                "China National Environmental Monitoring Centre (CNEMC) "
                "national monitoring network"
            ),
            "historical_retrieval_archive": "QuotSoft",
            "timezone_source": "Asia/Shanghai",
            "time_coordinate": "UTC",
            "protocol_year_coordinate": "local_year (Asia/Shanghai)",
            "station_qc_policy_id": policy["policy_id"],
            "station_qc_policy_path": str(policy_path),
            "station_qc_policy_sha256": sha256_file(policy_path),
            "raw_modified": "false",
            "imputation": "false",
            "statistical_outlier_clipping": "false",
            "canonical_validity_policy": (
                "city-hour aggregate values retained; station-count validity "
                "masks stored as companions, not applied destructively"
            ),
        },
    )

    for name in (
        "pm25_city_median",
        "pm25_city_std",
        "pm25_city_min",
        "pm25_city_max",
        "pm25_city_q25",
        "pm25_city_q75",
        "pm25_city_iqr",
        "pm25_city_mad",
    ):
        ds[name].attrs["units"] = "ug m-3"

    out_dir = (
        root / "03_canonical" / "air_quality_cnemc"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = out_dir / "city_air_quality_50cities.zarr"
    if zarr_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing canonical: {zarr_path}"
        )

    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dims == ("time", "city") and np.issubdtype(da.dtype, np.number):
            encoding[name] = {
                "chunks": (744, min(50, n_city)),
            }

    ds.to_zarr(
        zarr_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    manifest = {
        "canonical_id": CANONICAL_ID,
        "created_at_utc": ds.attrs["created_at_utc"],
        "output": str(zarr_path),
        "dimensions": {
            "time": n_time,
            "city": n_city,
        },
        "local_time_start": str(local_index[0]),
        "local_time_end": str(local_index[-1]),
        "utc_time_start": str(utc_index[0]),
        "utc_time_end": str(utc_index[-1]),
        "pollutants": list(POLLUTANTS),
        "eligible_station_count": int(len(eligible)),
        "station_qc_policy_id": policy["policy_id"],
        "station_qc_policy_sha256": sha256_file(policy_path),
        "important": {
            "2025_values_processed_deterministically": True,
            "2025_distribution_statistics_reported": False,
            "2025_used_for_model_selection": False,
            "pm25_exact10000_rule_pre_frozen_on_2023_2024": True,
            "city_hour_values_not_dropped_by_validity_gate": True,
        },
    }

    manifest_path = out_dir / "city_air_quality_50cities_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
