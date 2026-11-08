#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


RUN_ID = "central_china_external_cnemc_v1"
PROTOCOL_ID = "central_china_external_holdout_v1"

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

REGISTRY_REQUIRED_COLUMNS = (
    "监测点编码",
    "监测点名称",
    "城市",
    "经度",
    "纬度",
    "对照点",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build the frozen 10-city Central-China external CNEMC canonical "
            "from the already-acquired national QuotSoft/CNEMC daily site archive. "
            "No network download is performed."
        )
    )
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def read_csv_flexible(path: Path, **kwargs):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise last if last is not None else RuntimeError("CSV decode failed")


def norm_city(x: object) -> str:
    s = str(x).strip().replace(" ", "")
    if s.endswith("市"):
        s = s[:-1]
    return s


def quantile_higher(values: np.ndarray, q: float) -> int:
    v = np.asarray(values, dtype=np.int64)
    v = v[v > 0]
    if len(v) == 0:
        return 0
    try:
        return int(np.quantile(v, q, method="higher"))
    except TypeError:  # old numpy
        return int(np.quantile(v, q, interpolation="higher"))


def mode_positive(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=np.int64)
    v = v[v > 0]
    if len(v) == 0:
        return 0
    s = pd.Series(v)
    m = s.mode()
    return int(m.iloc[-1])  # deterministic higher mode on ties


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


def expected_daily_paths(sites_root: Path) -> list[Path]:
    dates = pd.date_range(
        START_LOCAL.normalize(),
        END_LOCAL.normalize(),
        freq="D",
    )
    paths = [
        sites_root / str(d.year) / f"china_sites_{d:%Y%m%d}.csv"
        for d in dates
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        head = "\n".join(str(x) for x in missing[:20])
        raise RuntimeError(
            f"missing {len(missing)} national QuotSoft daily site files; head:\n{head}"
        )
    if len(paths) != 1127:
        raise RuntimeError(f"expected 1127 dates, got {len(paths)}")
    return paths


def load_external_manifest(root: Path) -> pd.DataFrame:
    path = root / "05_manifests" / "external_central_china_10cities_v1.csv"
    if not path.exists():
        raise SystemExit(f"missing frozen external manifest: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "region", "priority", "province", "city", "city_zh",
        "latitude", "longitude",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"external manifest missing columns: {sorted(missing)}")
    if len(df) != 10:
        raise RuntimeError(f"external manifest rows={len(df)}, expected 10")
    if df["city"].duplicated().any() or df["city_zh"].duplicated().any():
        raise RuntimeError("duplicate external city identity")
    if df["region"].nunique() != 1 or df["region"].iloc[0] != "Central_China_External":
        raise RuntimeError("unexpected external region identity")
    return df.sort_values("priority").reset_index(drop=True)


def load_registry(root: Path) -> tuple[pd.DataFrame, Path]:
    path = (
        root
        / "01_raw"
        / "air_quality_cnemc_quotsoft"
        / "station_registry"
        / "source_files"
        / "站点列表-2022.02.13起.csv"
    )
    if not path.exists():
        raise SystemExit(f"missing frozen CNEMC station registry: {path}")
    reg = read_csv_flexible(path)
    missing = [c for c in REGISTRY_REQUIRED_COLUMNS if c not in reg.columns]
    if missing:
        raise RuntimeError(f"registry missing columns: {missing}")
    reg = reg.loc[:, list(REGISTRY_REQUIRED_COLUMNS)].copy()
    reg["监测点编码"] = reg["监测点编码"].astype(str).str.strip()
    reg["监测点名称"] = reg["监测点名称"].astype(str).str.strip()
    reg["城市"] = reg["城市"].astype(str).str.strip()
    reg["city_norm"] = reg["城市"].map(norm_city)
    reg["对照点"] = reg["对照点"].astype(str).str.strip().str.upper()
    if reg["监测点编码"].duplicated().any():
        raise RuntimeError("station registry IDs are not unique")
    return reg, path


def build_roster(
    ext: pd.DataFrame,
    reg: pd.DataFrame,
    frozen_policy_id: str,
) -> pd.DataFrame:
    city_map = {}
    for row in ext.itertuples(index=False):
        key = norm_city(row.city_zh)
        if key in city_map:
            raise RuntimeError(f"duplicate normalized external city: {key}")
        city_map[key] = row

    cand = reg.loc[reg["city_norm"].isin(city_map)].copy()
    if cand.empty:
        raise RuntimeError("no external city stations matched station registry")

    rows = []
    for r in cand.itertuples(index=False):
        meta = city_map[getattr(r, "city_norm")]
        rows.append(
            {
                "region": meta.region,
                "province": meta.province,
                "city_en": meta.city,
                "city_zh": meta.city_zh,
                "registry_city": getattr(r, "城市"),
                "station_id": getattr(r, "监测点编码"),
                "station_name": getattr(r, "监测点名称"),
                "longitude": float(getattr(r, "经度")),
                "latitude": float(getattr(r, "纬度")),
                "control_site": getattr(r, "对照点"),
                "eligible_city_station": getattr(r, "对照点") != "Y",
                "exclusion_reason": (
                    "CONTROL_SITE" if getattr(r, "对照点") == "Y" else ""
                ),
                "policy_id": frozen_policy_id,
            }
        )

    roster = pd.DataFrame(rows).sort_values(
        ["province", "city_en", "station_id"]
    ).reset_index(drop=True)

    got_cities = set(roster["city_en"])
    expected_cities = set(ext["city"])
    missing_cities = sorted(expected_cities - got_cities)
    if missing_cities:
        raise RuntimeError(
            f"external cities absent from station registry: {missing_cities}"
        )

    per_city = roster.groupby("city_en").agg(
        candidate_stations=("station_id", "size"),
        eligible_noncontrol=("eligible_city_station", "sum"),
    )
    bad = per_city.index[per_city["eligible_noncontrol"].eq(0)].tolist()
    if bad:
        raise RuntimeError(f"cities with zero eligible non-control stations: {bad}")

    return roster


def build_monthly_station_counts(
    daily_paths: list[Path],
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = roster.loc[roster["eligible_city_station"].eq(True)].copy()
    station_ids = eligible["station_id"].astype(str).tolist()
    station_set = set(station_ids)

    city_station_ids = {
        city: g["station_id"].astype(str).tolist()
        for city, g in eligible.groupby("city_en", sort=False)
    }
    city_meta = (
        eligible[
            ["region", "province", "city_en", "city_zh"]
        ]
        .drop_duplicates()
        .set_index("city_en")
        .to_dict(orient="index")
    )

    hourly_rows = []
    station_month_valid = defaultdict(int)
    month_source_rows = defaultdict(int)

    usecols = ["date", "hour", "type"] + station_ids

    for path in tqdm(
        daily_paths,
        desc="External CNEMC station-count audit",
        unit="day",
    ):
        df = read_csv_flexible(path, usecols=usecols)
        pm = df.loc[
            df["type"].astype(str).str.strip().eq("PM2.5")
        ].copy()
        if pm.empty:
            continue

        date_str = str(int(pm["date"].iloc[0]))
        month = f"{date_str[:4]}-{date_str[4:6]}"

        for city, ids in city_station_ids.items():
            block = pm[ids]
            finite = block.notna()
            nvalid = finite.sum(axis=1).to_numpy(dtype=np.int64)

            for sid in ids:
                station_month_valid[(sid, month)] += int(
                    finite[sid].sum()
                )
            month_source_rows[month] += len(pm)

            meta = city_meta[city]
            for hour, nv in zip(
                pm["hour"].to_numpy(),
                nvalid,
            ):
                hourly_rows.append(
                    {
                        "date": date_str,
                        "month": month,
                        "year": int(date_str[:4]),
                        "hour": int(hour),
                        "region": meta["region"],
                        "province": meta["province"],
                        "city_en": city,
                        "city_zh": meta["city_zh"],
                        "n_valid_noncontrol": int(nv),
                    }
                )

    hourly = pd.DataFrame(hourly_rows)
    if hourly.empty:
        raise RuntimeError("no external PM2.5 station-count rows built")

    inferred_rows = []
    for key, g in hourly.groupby(
        ["region", "province", "city_en", "city_zh", "month"],
        sort=True,
    ):
        values = g["n_valid_noncontrol"].to_numpy(dtype=np.int64)
        inferred_rows.append(
            {
                "region": key[0],
                "province": key[1],
                "city_en": key[2],
                "city_zh": key[3],
                "month": key[4],
                "expected_count_mode": mode_positive(values),
                "expected_count_q95": quantile_higher(values, 0.95),
                "expected_count_q99": quantile_higher(values, 0.99),
                "source_rows": int(len(g)),
                "zero_station_rows": int((values == 0).sum()),
            }
        )
    inferred = pd.DataFrame(inferred_rows)

    # ACTIVE_COV50, matching the old Dec-2022 boundary semantics.
    active_rows = []
    for city, ids in city_station_ids.items():
        meta = city_meta[city]
        months = sorted(hourly.loc[hourly["city_en"].eq(city), "month"].unique())
        for month in months:
            # All hours in a given month have the same source row count in the archive.
            source = int(
                hourly.loc[
                    hourly["city_en"].eq(city)
                    & hourly["month"].eq(month),
                    ["date", "hour"],
                ]
                .drop_duplicates()
                .shape[0]
            )
            active = 0
            for sid in ids:
                valid = station_month_valid[(sid, month)]
                frac = valid / source if source > 0 else np.nan
                if np.isfinite(frac) and frac >= 0.50:
                    active += 1
            active_rows.append(
                {
                    "region": meta["region"],
                    "province": meta["province"],
                    "city_en": city,
                    "city_zh": meta["city_zh"],
                    "month": month,
                    "active_count_cov50": int(active),
                }
            )
    active = pd.DataFrame(active_rows)

    monthly = active.merge(
        inferred,
        on=["region", "province", "city_en", "city_zh", "month"],
        how="outer",
        validate="one_to_one",
    )
    for c in (
        "active_count_cov50",
        "expected_count_mode",
        "expected_count_q95",
        "expected_count_q99",
    ):
        monthly[f"required_hj__{c}"] = monthly[c].map(hj_required_count)

    return hourly, monthly


def build_canonical(
    root: Path,
    ext: pd.DataFrame,
    roster: pd.DataFrame,
    daily_paths: list[Path],
    monthly_counts: pd.DataFrame,
    frozen_policy: dict,
    overwrite: bool,
):
    eligible = roster.loc[roster["eligible_city_station"].eq(True)].copy()
    ext_order = ext["city"].tolist()

    city_station_ids = {
        city: eligible.loc[
            eligible["city_en"].eq(city),
            "station_id",
        ].astype(str).tolist()
        for city in ext_order
    }
    if any(len(v) == 0 for v in city_station_ids.values()):
        bad = [k for k, v in city_station_ids.items() if not v]
        raise RuntimeError(f"zero-station cities: {bad}")

    all_station_ids = []
    seen = set()
    for city in ext_order:
        for sid in city_station_ids[city]:
            if sid not in seen:
                all_station_ids.append(sid)
                seen.add(sid)
    station_pos = {sid: i for i, sid in enumerate(all_station_ids)}
    city_station_pos = {
        city: np.asarray(
            [station_pos[sid] for sid in city_station_ids[city]],
            dtype=np.int64,
        )
        for city in ext_order
    }

    local_index = pd.date_range(START_LOCAL, END_LOCAL, freq="h")
    n_time = len(local_index)
    n_city = len(ext_order)
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

    usecols = ["date", "hour", "type"] + all_station_ids

    for path in tqdm(
        daily_paths,
        desc="Build external CNEMC canonical",
        unit="day",
    ):
        stem = path.stem
        ymd = stem.rsplit("_", 1)[-1]
        d = pd.Timestamp(
            f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        )

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
                block[all_station_ids]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            day_start = (d.normalize() - START_LOCAL.normalize()).days * 24
            time_indices = day_start + hours

            for ci, city in enumerate(ext_order):
                vals = matrix[:, city_station_pos[city]].copy()

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

    # Frozen target-hour validity: Q95 of positive hourly valid non-control count
    # for 2023-2025. December 2022 is input boundary only and retains only cov50.
    expected_cov50 = np.full(shape, np.nan, dtype=np.float32)
    expected_mode = np.full(shape, np.nan, dtype=np.float32)
    expected_q95 = np.full(shape, np.nan, dtype=np.float32)

    lookup = {}
    for row in monthly_counts.itertuples(index=False):
        lookup[(row.city_en, row.month)] = row

    month_keys = local_index.strftime("%Y-%m")
    for ci, city in enumerate(ext_order):
        for ti, m in enumerate(month_keys):
            row = lookup.get((city, m))
            if row is None:
                continue
            expected_cov50[ti, ci] = float(row.active_count_cov50)
            if m != "2022-12":
                expected_mode[ti, ci] = float(row.expected_count_mode)
                expected_q95[ti, ci] = float(row.expected_count_q95)

    def make_validity(expected):
        required = np.full_like(expected, np.nan, dtype=np.float32)
        finite = np.isfinite(expected)
        if finite.any():
            required[finite] = np.asarray(
                [hj_required_count(x) for x in expected[finite]],
                dtype=np.float32,
            )
        valid = (
            np.isfinite(means["PM2.5"])
            & np.isfinite(required)
            & (n_valid["PM2.5"] >= required)
        )
        frac = np.full_like(expected, np.nan, dtype=np.float32)
        ok = np.isfinite(expected) & (expected > 0)
        frac[ok] = (
            n_valid["PM2.5"][ok] / expected[ok]
        ).astype(np.float32)
        return required, valid, frac

    req_cov50, valid_cov50, frac_cov50 = make_validity(expected_cov50)
    req_mode, valid_mode, frac_mode = make_validity(expected_mode)
    req_q95, valid_q95, frac_q95 = make_validity(expected_q95)

    pm25_reported = round_half_up_nonnegative(
        means["PM2.5"].astype(np.float64)
    ).astype(np.float32)

    city_rows = ext.set_index("city").loc[ext_order]

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
            city_rows["region"].astype(str).to_numpy(dtype=str),
        ),
        "province": (
            ("city",),
            city_rows["province"].astype(str).to_numpy(dtype=str),
        ),
        "city_en": (
            ("city",),
            np.asarray(ext_order, dtype=str),
        ),
        "city_zh": (
            ("city",),
            city_rows["city_zh"].astype(str).to_numpy(dtype=str),
        ),
        "latitude": (
            ("city",),
            city_rows["latitude"].to_numpy(dtype=np.float64),
        ),
        "longitude": (
            ("city",),
            city_rows["longitude"].to_numpy(dtype=np.float64),
        ),
    }

    for pollutant in POLLUTANTS:
        prefix = VAR_PREFIX[pollutant]
        data_vars[f"{prefix}_city_mean"] = (
            ("time", "city"),
            means[pollutant],
            {
                "units": UNITS[pollutant],
                "aggregation": (
                    "arithmetic mean of finite eligible non-control stations"
                ),
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
                ("time", "city"), expected_cov50
            ),
            "pm25_required_n_hj_cov50": (
                ("time", "city"), req_cov50
            ),
            "pm25_coverage_fraction_cov50": (
                ("time", "city"), frac_cov50
            ),
            "pm25_valid_hj_cov50": (
                ("time", "city"), valid_cov50
            ),
            "pm25_expected_n_mode": (
                ("time", "city"), expected_mode
            ),
            "pm25_required_n_hj_mode": (
                ("time", "city"), req_mode
            ),
            "pm25_coverage_fraction_mode": (
                ("time", "city"), frac_mode
            ),
            "pm25_valid_hj_mode": (
                ("time", "city"), valid_mode
            ),
            "pm25_expected_n_q95": (
                ("time", "city"), expected_q95
            ),
            "pm25_required_n_hj_q95": (
                ("time", "city"), req_q95
            ),
            "pm25_coverage_fraction_q95": (
                ("time", "city"), frac_q95
            ),
            "pm25_valid_hj_q95": (
                ("time", "city"), valid_q95
            ),
        }
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "canonical_id": RUN_ID,
            "created_at_utc": utc_now(),
            "measurement_origin": (
                "China National Environmental Monitoring Centre (CNEMC) "
                "national monitoring network"
            ),
            "historical_retrieval_archive": "QuotSoft",
            "quotsoft_url": "https://quotsoft.net/air/",
            "external_protocol_id": PROTOCOL_ID,
            "timezone_source": "Asia/Shanghai",
            "time_coordinate": "UTC",
            "protocol_year_coordinate": "local_year (Asia/Shanghai)",
            "station_qc_policy_id": frozen_policy["policy_id"],
            "station_eligibility_rule": "control_site != 'Y'",
            "pm25_exact10000_rule": (
                "set derived station value to NaN before city aggregation"
            ),
            "imputation": "false",
            "statistical_outlier_clipping": "false",
            "external_values_used_for_model_selection": "false",
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

    out_dir = root / "03_canonical" / "air_quality_cnemc_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / "city_air_quality_central_china_10cities_v1.zarr"
    manifest_path = (
        out_dir / "city_air_quality_central_china_10cities_v1_manifest.json"
    )

    if zarr_path.exists():
        if not overwrite:
            raise SystemExit(
                f"refusing to overwrite external canonical: {zarr_path}"
            )
        import shutil
        shutil.rmtree(zarr_path)

    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dims == ("time", "city") and np.issubdtype(
            da.dtype, np.number
        ):
            encoding[name] = {
                "chunks": (744, min(10, n_city)),
            }

    ds.to_zarr(
        zarr_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    # Structural and missingness-only report; deliberately no concentration
    # distribution summaries are printed or stored.
    report = {
        "status": "PASS",
        "canonical_id": RUN_ID,
        "created_at_utc": ds.attrs["created_at_utc"],
        "output": str(zarr_path),
        "dimensions": {"time": int(n_time), "city": int(n_city)},
        "local_time_start": str(local_index[0]),
        "local_time_end": str(local_index[-1]),
        "utc_time_start": str(utc_index[0]),
        "utc_time_end": str(utc_index[-1]),
        "city_order": ext_order,
        "eligible_station_count": int(len(eligible)),
        "eligible_station_count_by_city": {
            city: int(len(city_station_ids[city]))
            for city in ext_order
        },
        "target_validity_mask": "pm25_valid_hj_q95",
        "external_values_used_for_model_selection": False,
        "external_model_predictions_read": False,
        "concentration_distribution_statistics_reported": False,
    }
    atomic_json(report, manifest_path)
    return zarr_path, manifest_path, report


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    protocol_path = (
        root / "05_manifests" / "external_central_china_protocol_v1.json"
    )
    protocol_audit = (
        root
        / "09_diagnostics"
        / "external_validation"
        / PROTOCOL_ID
        / "protocol_freeze"
        / "summary.json"
    )
    policy_path = (
        root / "05_manifests" / "cnemc_station_qc_policy_v1.json"
    )

    for p in (protocol_path, protocol_audit, policy_path):
        if not p.exists():
            raise SystemExit(f"missing frozen prerequisite: {p}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    psummary = json.loads(protocol_audit.read_text(encoding="utf-8"))
    frozen_policy = json.loads(policy_path.read_text(encoding="utf-8"))

    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("external protocol identity mismatch")
    if psummary.get("status") != "PASS":
        raise RuntimeError("74 protocol freeze did not PASS")
    if frozen_policy.get("policy_id") is None:
        raise RuntimeError("frozen CNEMC policy missing policy_id")

    ext = load_external_manifest(root)
    reg, registry_path = load_registry(root)

    sites_root = (
        root / "01_raw" / "air_quality_cnemc_quotsoft" / "sites" / "daily"
    )
    daily_paths = expected_daily_paths(sites_root)

    print("=" * 132)
    print("75a — CENTRAL-CHINA EXTERNAL CNEMC BUILD")
    print("=" * 132)
    print("Source: existing national QuotSoft/CNEMC daily station archive")
    print("Network download: NONE")
    print(f"Daily files: {len(daily_paths)}")
    print(f"Registry: {registry_path}")
    print()

    roster = build_roster(
        ext=ext,
        reg=reg,
        frozen_policy_id=frozen_policy["policy_id"],
    )

    audit_root = (
        root
        / "09_diagnostics"
        / "external_validation"
        / PROTOCOL_ID
        / "cnemc_preflight"
    )
    audit_root.mkdir(parents=True, exist_ok=True)

    roster_path = (
        root
        / "05_manifests"
        / "cnemc_station_eligibility_external_central_china_v1.csv"
    )
    if roster_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"external station roster exists: {roster_path}; "
            "use --overwrite only before external model inference"
        )
    atomic_csv(roster, roster_path)

    city_station = (
        roster.groupby(["province", "city_en", "city_zh"])
        .agg(
            candidate_stations=("station_id", "size"),
            eligible_noncontrol=("eligible_city_station", "sum"),
            control_stations=("control_site", lambda s: int((s == "Y").sum())),
        )
        .reset_index()
    )
    atomic_csv(city_station, audit_root / "city_station_counts.csv")

    hourly, monthly = build_monthly_station_counts(
        daily_paths=daily_paths,
        roster=roster,
    )
    atomic_csv(
        monthly,
        audit_root / "monthly_expected_station_count_external.csv",
    )

    # Data-only gates: every formal month must have positive Q95 expectation.
    formal = monthly.loc[~monthly["month"].eq("2022-12")].copy()
    expected_months = 10 * 36
    if len(formal) != expected_months:
        raise RuntimeError(
            f"external city-month rows={len(formal)}, expected={expected_months}"
        )
    bad_q95 = formal.loc[formal["expected_count_q95"].fillna(0).le(0)]
    if not bad_q95.empty:
        raise RuntimeError(
            "at least one external city-month has zero Q95 expected station count:\n"
            + bad_q95[
                ["city_en", "month", "expected_count_q95"]
            ].head(30).to_string(index=False)
        )

    zarr_path, manifest_path, report = build_canonical(
        root=root,
        ext=ext,
        roster=roster,
        daily_paths=daily_paths,
        monthly_counts=monthly,
        frozen_policy=frozen_policy,
        overwrite=args.overwrite,
    )

    preflight = {
        "status": "PASS",
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "protocol_id": PROTOCOL_ID,
        "quotsoft_url": "https://quotsoft.net/air/",
        "daily_archive_pattern": (
            "https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv"
        ),
        "national_raw_files_reused": 1127,
        "new_cnemc_network_download_performed": False,
        "registry_sha256": sha256_file(registry_path),
        "frozen_station_qc_policy_sha256": sha256_file(policy_path),
        "external_station_manifest": str(roster_path.relative_to(root)),
        "external_station_manifest_sha256": sha256_file(roster_path),
        "cities": 10,
        "formal_city_months": expected_months,
        "all_formal_city_months_positive_q95_expected_count": True,
        "external_model_predictions_read": False,
        "external_data_used_for_model_selection": False,
        "canonical": str(zarr_path.relative_to(root)),
        "canonical_manifest": str(manifest_path.relative_to(root)),
    }
    atomic_json(preflight, audit_root / "summary.json")

    print("\nCITY STATION COUNTS")
    print(city_station.to_string(index=False))
    print()
    print(f"External canonical: {zarr_path}")
    print(f"Manifest:           {manifest_path}")
    print(f"Audit:              {audit_root / 'summary.json'}")
    print("=" * 132)
    print("75a PASS — EXTERNAL CNEMC CANONICAL BUILT")
    print("=" * 132)


if __name__ == "__main__":
    main()
