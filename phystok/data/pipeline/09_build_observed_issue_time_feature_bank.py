#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


VERSION = "observed_issue_time_feature_bank_v1"
SCHEMA_ID = "national_pm25_observed_feature_schema_v1"
PARENT_SCHEMA_ID = "national_pm25_feature_schema_v1"
FORECAST_HORIZON_HOURS = 24
EXPECTED_CITIES = 50
EXPECTED_LEVELS = [925, 850]


FAMILY_SCHEMA = [
    {
        "family": "pm25_history",
        "status": "core_baseline",
        "role": "observed PM2.5 persistence, recent tendency, variability, and 24h recurrence",
        "features": [
            "pm25_now",
            "pm25_lag_1h",
            "pm25_lag_3h",
            "pm25_lag_6h",
            "pm25_lag_24h",
            "pm25_change_1h",
            "pm25_change_3h",
            "pm25_change_6h",
            "pm25_change_24h",
            "pm25_mean_3h",
            "pm25_mean_6h",
            "pm25_mean_12h",
            "pm25_mean_24h",
            "pm25_std_6h",
            "pm25_std_24h",
        ],
    },
    {
        "family": "calendar",
        "status": "core_baseline",
        "role": "transferable diurnal and annual phase in Asia/Shanghai local civil time",
        "features": [
            "hour_sin",
            "hour_cos",
            "annual_phase_sin",
            "annual_phase_cos",
        ],
    },
    {
        "family": "transport_dispersion",
        "status": "core",
        "role": "boundary-layer dilution, low-level guiding transport, and transport tendency from ERA5",
        "features": [
            "boundary_layer_height_now",
            "wind_speed_10m_now",
            "wind_speed_925_now",
            "wind_speed_850_now",
            "low_level_vector_mean_wind_now",
            "ventilation_10m_now",
            "ventilation_925_now",
            "ventilation_850_now",
            "low_level_vector_ventilation_now",
            "blh_change_3h",
            "blh_change_24h",
            "wind10_change_3h",
            "wind10_change_24h",
            "ventilation10_change_3h",
            "ventilation10_change_24h",
        ],
    },
    {
        "family": "moisture_stability_context",
        "status": "core",
        "role": "lower-tropospheric moisture, stability, shear/coherence, and stability-conditioned humidity from ERA5",
        "features": [
            "theta_diff_850_925_now",
            "rh_925_now",
            "rh_850_now",
            "rh_diff_850_925_now",
            "wind_alignment_925_850_now",
            "wind_shear_per_km_now",
            "theta_x_rh925",
            "theta_x_rh850",
        ],
    },
    {
        "family": "pollution_composition",
        "status": "core",
        "role": "observed co-pollutant / emission-chemistry state and recent composition dynamics",
        "features": [
            "carbon_monoxide_now",
            "sulphur_dioxide_now",
            "nitrogen_dioxide_now",
            "ozone_now",
            "co_change_3h",
            "co_change_24h",
            "so2_change_3h",
            "so2_change_24h",
            "no2_change_3h",
            "no2_change_24h",
        ],
    },
    {
        "family": "precipitation_radiation",
        "status": "secondary_physical",
        "role": "ERA5 wet-removal event state and radiative context",
        "features": [
            "precipitation_now",
            "precip_sum_3h",
            "precip_sum_6h",
            "precip_sum_12h",
            "precip_sum_24h",
            "rain_now_flag",
            "rain_any_3h",
            "rain_any_6h",
            "rain_any_12h",
            "rain_any_24h",
            "log1p_precipitation_now",
            "log1p_precip_sum_3h",
            "log1p_precip_sum_6h",
            "log1p_precip_sum_12h",
            "log1p_precip_sum_24h",
            "solar_radiation_now",
        ],
    },
    {
        "family": "pm10_state",
        "status": "optional_shortcut_sensitive",
        "role": "observed particulate-state proxy; retained for deployment realism and explicit without-PM10 ablation",
        "features": [
            "pm10_now",
            "pm10_change_3h",
            "pm10_change_24h",
        ],
    },
    {
        "family": "domain_pressure_context",
        "status": "diagnostic_only",
        "role": "absolute pressure / geographic-domain marker; excluded from ordinary core predictor mask",
        "features": [
            "surface_pressure_now",
        ],
    },
]


BINARY_FEATURES = {
    "rain_now_flag",
    "rain_any_3h",
    "rain_any_6h",
    "rain_any_12h",
    "rain_any_24h",
}

BOUNDED_DETERMINISTIC_FEATURES = {
    "hour_sin",
    "hour_cos",
    "annual_phase_sin",
    "annual_phase_cos",
    "wind_alignment_925_850_now",
}

DIAGNOSTIC_ONLY_FEATURES = {"surface_pressure_now"}

OPTIONAL_SHORTCUT_FEATURES = {
    "pm10_now",
    "pm10_change_3h",
    "pm10_change_24h",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def shift(a: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[lag:] = a[:-lag]
    return out


def rolling(a: np.ndarray, window: int, how: str) -> np.ndarray:
    r = pd.DataFrame(a).rolling(window=window, min_periods=window)
    if how == "mean":
        z = r.mean()
    elif how == "std":
        z = r.std(ddof=0)
    elif how == "sum":
        z = r.sum()
    else:
        raise ValueError(how)
    return z.to_numpy(dtype=np.float32)


def arr2d(da: xr.DataArray) -> np.ndarray:
    return np.asarray(
        da.transpose("time", "city").values,
        dtype=np.float32,
    )


def theta(ds: xr.Dataset, level: int) -> xr.DataArray:
    tk = ds["temperature_pressure_c"].sel(
        pressure_level_hpa=level
    ) + 273.15
    return tk * (1000.0 / float(level)) ** 0.2854


def low_level_vector_mean_wind(ds: xr.Dataset) -> xr.DataArray:
    u925 = ds["u_wind_pressure_ms"].sel(pressure_level_hpa=925)
    v925 = ds["v_wind_pressure_ms"].sel(pressure_level_hpa=925)
    u850 = ds["u_wind_pressure_ms"].sel(pressure_level_hpa=850)
    v850 = ds["v_wind_pressure_ms"].sel(pressure_level_hpa=850)
    um = 0.5 * (u925 + u850)
    vm = 0.5 * (v925 + v850)
    return np.hypot(um, vm)


def wind_alignment(ds: xr.Dataset) -> xr.DataArray:
    u925 = ds["u_wind_pressure_ms"].sel(pressure_level_hpa=925)
    v925 = ds["v_wind_pressure_ms"].sel(pressure_level_hpa=925)
    u850 = ds["u_wind_pressure_ms"].sel(pressure_level_hpa=850)
    v850 = ds["v_wind_pressure_ms"].sel(pressure_level_hpa=850)

    s925 = np.hypot(u925, v925)
    s850 = np.hypot(u850, v850)
    denom = s925 * s850

    c = (
        u925 * u850 + v925 * v850
    ) / xr.where(denom > 1e-12, denom, np.nan)

    return c.clip(min=-1.0, max=1.0)


def build_schema(
    parent_schema: dict[str, Any] | None,
    parent_schema_path: Path,
) -> dict[str, Any]:
    feature_order = [
        f for fam in FAMILY_SCHEMA for f in fam["features"]
    ]
    if len(feature_order) != 72:
        raise RuntimeError(f"expected 72 features, got {len(feature_order)}")
    if len(feature_order) != len(set(feature_order)):
        raise RuntimeError("duplicate feature names")

    records = []
    for fam in FAMILY_SCHEMA:
        for f in fam["features"]:
            if f in BINARY_FEATURES:
                transform_class = "deterministic_binary_no_scaling"
            elif f in BOUNDED_DETERMINISTIC_FEATURES:
                transform_class = "deterministic_bounded_no_scaling"
            elif f in DIAGNOSTIC_ONLY_FEATURES:
                transform_class = (
                    "diagnostic_continuous_fold_local_scaling_if_used"
                )
            else:
                transform_class = "continuous_fold_local_standardization"

            records.append(
                {
                    "feature": f,
                    "family": fam["family"],
                    "family_status": fam["status"],
                    "role": fam["role"],
                    "transform_class": transform_class,
                    "default_core_predictor": (
                        fam["status"]
                        in {"core_baseline", "core", "secondary_physical"}
                    ),
                    "optional_shortcut_sensitive": (
                        f in OPTIONAL_SHORTCUT_FEATURES
                    ),
                    "diagnostic_only": f in DIAGNOSTIC_ONLY_FEATURES,
                }
            )

    lineage = {
        "parent_schema_id": PARENT_SCHEMA_ID,
        "parent_schema_path": str(parent_schema_path),
        "parent_schema_present": parent_schema is not None,
        "definition_reuse_policy": (
            "Reuse the previously frozen deterministic 72-feature definitions "
            "as a pre-specified bank. Do not reopen target-aware feature discovery "
            "after replacing CAMS-era AQ semantics with CNEMC observations."
        ),
        "semantic_changes": {
            "pm25_history": (
                "CAMS-era PM2.5 source replaced by CNEMC station-derived "
                "unrounded city arithmetic mean."
            ),
            "pollution_composition": (
                "CAMS-era co-pollutants replaced by CNEMC station-derived "
                "city arithmetic means."
            ),
            "pm10_state": (
                "CAMS-era PM10 replaced by CNEMC station-derived city arithmetic mean."
            ),
            "meteorology_physics": (
                "ERA5-based deterministic physical definitions unchanged."
            ),
        },
    }

    if parent_schema is not None:
        parent_order = list(parent_schema.get("feature_order", []))
        if parent_order != feature_order:
            missing = sorted(set(parent_order) - set(feature_order))
            extra = sorted(set(feature_order) - set(parent_order))
            raise RuntimeError(
                "new feature names/order differ from frozen parent schema: "
                f"missing={missing}, extra={extra}"
            )
        parent_family = {
            r["feature"]: r["family"]
            for r in parent_schema.get("feature_records", [])
        }
        new_family = {r["feature"]: r["family"] for r in records}
        if parent_family and parent_family != new_family:
            raise RuntimeError("feature-family mapping differs from parent schema")
        lineage["parent_schema_fingerprint_sha256"] = parent_schema.get(
            "schema_fingerprint_sha256"
        )
        lineage["parent_schema_file_sha256"] = sha256_file(parent_schema_path)

    schema = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "feature_count": 72,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "feature_engineering_status": (
            "pre-specified deterministic rebuild after observed-target source correction"
        ),
        "lineage": lineage,
        "source_semantics": {
            "observed_air_quality": (
                "CNEMC non-control-station city arithmetic means after frozen "
                "station-level QC; retrieved historically via QuotSoft."
            ),
            "meteorology": "ERA5 city meteorology and deterministic physical derivatives",
            "calendar_timezone": "Asia/Shanghai",
        },
        "causality": {
            "rule": "every feature at row t uses information at t or earlier only",
            "target": "PM2.5 at t+24h; target is not materialized in this bank",
            "history_buffer": (
                "December 2022 remains predictor history/context only."
            ),
        },
        "missingness_policy": {
            "bank_construction": (
                "preserve source missingness; do not perform empirical imputation, "
                "statistical clipping, or sample deletion"
            ),
            "downstream": (
                "imputation/scaling/missingness handling is model-input-specific "
                "and must respect fold-local fit boundaries"
            ),
        },
        "families": FAMILY_SCHEMA,
        "feature_order": feature_order,
        "feature_records": records,
        "planned_ablation_masks": {
            "history_calendar_only": "pm25_history + calendar",
            "core_without_pm10": (
                "core baseline + core + secondary physical; PM10 and "
                "diagnostic pressure excluded"
            ),
            "core_plus_pm10": "core_without_pm10 + pm10_state",
            "all_with_pressure": "all 72 features",
            "meteorology_physics_only": (
                "calendar + transport_dispersion + moisture_stability_context "
                "+ precipitation_radiation"
            ),
            "pollution_history_only": (
                "pm25_history + calendar + pollution_composition + pm10_state"
            ),
        },
    }
    schema["schema_fingerprint_sha256"] = canonical_sha256(schema)
    return schema


def validate_source(ds: xr.Dataset) -> None:
    if int(ds.sizes.get("city", -1)) != EXPECTED_CITIES:
        raise RuntimeError(
            f"expected {EXPECTED_CITIES} cities, got {ds.sizes.get('city')}"
        )

    levels = [int(x) for x in ds["pressure_level_hpa"].values.tolist()]
    if levels != EXPECTED_LEVELS:
        raise RuntimeError(
            f"expected pressure levels {EXPECTED_LEVELS}, got {levels}"
        )

    required = {
        "local_time",
        "city_en",
        "region",
        "pm25_city_mean",
        "pm10_city_mean",
        "co_city_mean",
        "so2_city_mean",
        "no2_city_mean",
        "o3_city_mean",
        "boundary_layer_height_m",
        "wind_speed_10m_ms",
        "wind_speed_pressure_ms",
        "u_wind_pressure_ms",
        "v_wind_pressure_ms",
        "temperature_pressure_c",
        "relative_humidity_pressure_pct",
        "wind_shear_magnitude_850_925_ms",
        "delta_geopotential_height_850_925_m",
        "total_precipitation_mm",
        "surface_solar_radiation_downwards_wm2",
        "surface_pressure_hpa",
    }
    missing = required - set(ds.variables)
    if missing:
        raise RuntimeError(
            f"merged source missing required variables: {sorted(missing)}"
        )


def build_feature_arrays(
    ds: xr.Dataset,
) -> dict[str, np.ndarray]:
    pm25 = arr2d(ds["pm25_city_mean"])
    pm10 = arr2d(ds["pm10_city_mean"])
    co = arr2d(ds["co_city_mean"])
    so2 = arr2d(ds["so2_city_mean"])
    no2 = arr2d(ds["no2_city_mean"])
    ozone = arr2d(ds["o3_city_mean"])

    blh = arr2d(ds["boundary_layer_height_m"])
    ws10 = arr2d(ds["wind_speed_10m_ms"])
    ws925 = arr2d(
        ds["wind_speed_pressure_ms"].sel(pressure_level_hpa=925)
    )
    ws850 = arr2d(
        ds["wind_speed_pressure_ms"].sel(pressure_level_hpa=850)
    )
    llwind = arr2d(low_level_vector_mean_wind(ds))

    vent10 = blh * ws10
    vent925 = blh * ws925
    vent850 = blh * ws850
    llvent = blh * llwind

    rh925 = arr2d(
        ds["relative_humidity_pressure_pct"].sel(pressure_level_hpa=925)
    )
    rh850 = arr2d(
        ds["relative_humidity_pressure_pct"].sel(pressure_level_hpa=850)
    )
    theta_diff = arr2d(theta(ds, 850) - theta(ds, 925))

    precip = arr2d(ds["total_precipitation_mm"])
    ps3 = rolling(precip, 3, "sum")
    ps6 = rolling(precip, 6, "sum")
    ps12 = rolling(precip, 12, "sum")
    ps24 = rolling(precip, 24, "sum")

    local = pd.DatetimeIndex(ds["local_time"].values)
    hour = local.hour.to_numpy(dtype=float)
    hour_phase = 2.0 * np.pi * hour / 24.0

    doy = local.dayofyear.to_numpy(dtype=float) - 1.0
    hour_frac = hour / 24.0
    days_in_year = np.where(local.is_leap_year, 366.0, 365.0)
    annual_phase = 2.0 * np.pi * ((doy + hour_frac) / days_in_year)

    def broadcast(v: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(v, dtype=np.float32)[:, None],
            pm25.shape,
        ).copy()

    dz = arr2d(ds["delta_geopotential_height_850_925_m"])
    shear = arr2d(ds["wind_shear_magnitude_850_925_ms"])
    shear_per_km = np.full_like(shear, np.nan, dtype=np.float32)
    ok_dz = np.isfinite(dz) & (np.abs(dz) > 1e-6) & np.isfinite(shear)
    shear_per_km[ok_dz] = shear[ok_dz] / dz[ok_dz] * 1000.0

    features: dict[str, np.ndarray] = {
        "pm25_now": pm25,
        "pm25_lag_1h": shift(pm25, 1),
        "pm25_lag_3h": shift(pm25, 3),
        "pm25_lag_6h": shift(pm25, 6),
        "pm25_lag_24h": shift(pm25, 24),
        "pm25_change_1h": pm25 - shift(pm25, 1),
        "pm25_change_3h": pm25 - shift(pm25, 3),
        "pm25_change_6h": pm25 - shift(pm25, 6),
        "pm25_change_24h": pm25 - shift(pm25, 24),
        "pm25_mean_3h": rolling(pm25, 3, "mean"),
        "pm25_mean_6h": rolling(pm25, 6, "mean"),
        "pm25_mean_12h": rolling(pm25, 12, "mean"),
        "pm25_mean_24h": rolling(pm25, 24, "mean"),
        "pm25_std_6h": rolling(pm25, 6, "std"),
        "pm25_std_24h": rolling(pm25, 24, "std"),

        "hour_sin": broadcast(np.sin(hour_phase)),
        "hour_cos": broadcast(np.cos(hour_phase)),
        "annual_phase_sin": broadcast(np.sin(annual_phase)),
        "annual_phase_cos": broadcast(np.cos(annual_phase)),

        "boundary_layer_height_now": blh,
        "wind_speed_10m_now": ws10,
        "wind_speed_925_now": ws925,
        "wind_speed_850_now": ws850,
        "low_level_vector_mean_wind_now": llwind,
        "ventilation_10m_now": vent10,
        "ventilation_925_now": vent925,
        "ventilation_850_now": vent850,
        "low_level_vector_ventilation_now": llvent,
        "blh_change_3h": blh - shift(blh, 3),
        "blh_change_24h": blh - shift(blh, 24),
        "wind10_change_3h": ws10 - shift(ws10, 3),
        "wind10_change_24h": ws10 - shift(ws10, 24),
        "ventilation10_change_3h": vent10 - shift(vent10, 3),
        "ventilation10_change_24h": vent10 - shift(vent10, 24),

        "theta_diff_850_925_now": theta_diff,
        "rh_925_now": rh925,
        "rh_850_now": rh850,
        "rh_diff_850_925_now": rh850 - rh925,
        "wind_alignment_925_850_now": arr2d(wind_alignment(ds)),
        "wind_shear_per_km_now": shear_per_km,
        "theta_x_rh925": theta_diff * rh925,
        "theta_x_rh850": theta_diff * rh850,

        "carbon_monoxide_now": co,
        "sulphur_dioxide_now": so2,
        "nitrogen_dioxide_now": no2,
        "ozone_now": ozone,
        "co_change_3h": co - shift(co, 3),
        "co_change_24h": co - shift(co, 24),
        "so2_change_3h": so2 - shift(so2, 3),
        "so2_change_24h": so2 - shift(so2, 24),
        "no2_change_3h": no2 - shift(no2, 3),
        "no2_change_24h": no2 - shift(no2, 24),

        "precipitation_now": precip,
        "precip_sum_3h": ps3,
        "precip_sum_6h": ps6,
        "precip_sum_12h": ps12,
        "precip_sum_24h": ps24,
        "rain_now_flag": (precip > 0).astype(np.float32),
        "rain_any_3h": (ps3 > 0).astype(np.float32),
        "rain_any_6h": (ps6 > 0).astype(np.float32),
        "rain_any_12h": (ps12 > 0).astype(np.float32),
        "rain_any_24h": (ps24 > 0).astype(np.float32),
        "log1p_precipitation_now": np.log1p(
            np.maximum(precip, 0)
        ).astype(np.float32),
        "log1p_precip_sum_3h": np.log1p(
            np.maximum(ps3, 0)
        ).astype(np.float32),
        "log1p_precip_sum_6h": np.log1p(
            np.maximum(ps6, 0)
        ).astype(np.float32),
        "log1p_precip_sum_12h": np.log1p(
            np.maximum(ps12, 0)
        ).astype(np.float32),
        "log1p_precip_sum_24h": np.log1p(
            np.maximum(ps24, 0)
        ).astype(np.float32),
        "solar_radiation_now": arr2d(
            ds["surface_solar_radiation_downwards_wm2"]
        ),

        "pm10_now": pm10,
        "pm10_change_3h": pm10 - shift(pm10, 3),
        "pm10_change_24h": pm10 - shift(pm10, 24),

        "surface_pressure_now": arr2d(ds["surface_pressure_hpa"]),
    }

    return features


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    source = (
        root / "04_derived" / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )
    task_index = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1.parquet"
    )
    parent_schema_path = (
        root / "05_manifests" / "features"
        / PARENT_SCHEMA_ID / "feature_schema.json"
    )

    schema_dir = (
        root / "05_manifests" / "features" / SCHEMA_ID
    )
    schema_path = schema_dir / "feature_schema.json"
    feature_manifest_path = schema_dir / "feature_manifest.csv"

    output_dir = (
        root / "04_derived" / "issue_time_features" / SCHEMA_ID
    )
    output_zarr = output_dir / "issue_time_feature_bank.zarr"
    build_report_path = output_dir / "build_report.json"

    if not source.exists():
        raise SystemExit(f"missing merged source: {source}")
    if not task_index.exists():
        raise SystemExit(f"missing task index: {task_index}")

    parent_schema = None
    if parent_schema_path.exists():
        parent_schema = json.loads(
            parent_schema_path.read_text(encoding="utf-8")
        )

    schema = build_schema(parent_schema, parent_schema_path)

    schema_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(schema, schema_path)
    pd.DataFrame(schema["feature_records"]).to_csv(
        feature_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    ds = xr.open_zarr(source, consolidated=True, chunks=None)
    try:
        validate_source(ds)

        times = pd.DatetimeIndex(ds["time"].values)
        local = pd.DatetimeIndex(ds["local_time"].values)

        if len(times) != 27040:
            raise RuntimeError(f"expected 27040 source hours, got {len(times)}")
        if not np.all((local - times) == pd.Timedelta(hours=8)):
            raise RuntimeError("local_time != UTC + 8h")

        features = build_feature_arrays(ds)
        if list(features) != schema["feature_order"]:
            raise RuntimeError("feature order differs from schema")
        if any(a.shape != (27040, 50) for a in features.values()):
            bad = {
                k: v.shape
                for k, v in features.items()
                if v.shape != (27040, 50)
            }
            raise RuntimeError(f"feature shape mismatch: {bad}")

        # Task index issue_time_index must directly address this same source grid.
        task = pd.read_parquet(
            task_index,
            columns=[
                "issue_time_index",
                "issue_time_utc",
                "target_time_utc",
                "forecast_horizon_hours",
            ],
        )
        if not np.all(task["forecast_horizon_hours"].to_numpy() == 24):
            raise RuntimeError("task index horizon is not uniformly 24h")

        unique_issue = task[
            ["issue_time_index", "issue_time_utc"]
        ].drop_duplicates()
        ip = unique_issue["issue_time_index"].to_numpy(dtype=np.int64)
        if np.any((ip < 0) | (ip >= len(times))):
            raise RuntimeError("task issue_time_index outside source range")
        indexed_time = times[ip]
        supplied_time = pd.DatetimeIndex(unique_issue["issue_time_utc"])
        if not np.array_equal(
            indexed_time.to_numpy(dtype="datetime64[ns]"),
            supplied_time.to_numpy(dtype="datetime64[ns]"),
        ):
            raise RuntimeError("task issue_time_index does not match issue_time_utc")

        # Missingness is preserved; only count it, never fit/impute/delete here.
        dev_mask = (local.year >= 2023) & (local.year <= 2024)
        future_mask = local.year == 2025

        missing_dev = {}
        missing_future = {}
        for name, a in features.items():
            missing_dev[name] = int(
                np.size(a[dev_mask]) - np.isfinite(a[dev_mask]).sum()
            )
            # Availability-only diagnostic; no 2025 magnitude statistics.
            missing_future[name] = int(
                np.size(a[future_mask]) - np.isfinite(a[future_mask]).sum()
            )

        data_vars = {
            name: (
                ("time", "city"),
                a.astype(np.float32, copy=False),
            )
            for name, a in features.items()
        }

        out_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ds["time"].values,
                "city": ds["city"].values,
            },
            attrs={
                "artifact_version": VERSION,
                "schema_id": SCHEMA_ID,
                "schema_fingerprint_sha256": (
                    schema["schema_fingerprint_sha256"]
                ),
                "construction": (
                    "global deterministic causal feature bank; no target "
                    "materialized; no fitted data-dependent transform"
                ),
                "time_coordinate": "UTC",
                "calendar_timezone": "Asia/Shanghai",
                "source": str(source),
                "task_index": str(task_index),
                "2025_policy": (
                    "mechanically constructed using issue-time observations "
                    "and ERA5 predictors; no 2025 target magnitude is read here"
                ),
            },
        )

        for meta_name in ("local_time", "local_year", "local_month", "local_hour"):
            if meta_name in ds:
                out_ds[meta_name] = ds[meta_name]
        for meta_name in ("city_en", "city_zh", "region"):
            if meta_name in ds:
                out_ds[meta_name] = ds[meta_name]

        # Do not inherit V3-specific storage encoding from source variables.
        for name in out_ds.variables:
            out_ds[name].encoding = {}

        if output_zarr.exists():
            if not args.overwrite:
                raise RuntimeError(
                    f"output exists: {output_zarr}; use --overwrite intentionally"
                )
            shutil.rmtree(output_zarr)

        encoding = {}
        for name, da in out_ds.data_vars.items():
            if da.dims == ("time", "city") and np.issubdtype(da.dtype, np.number):
                encoding[name] = {"chunks": (744, 50)}
            elif da.dims == ("time",):
                encoding[name] = {"chunks": (744,)}

        kwargs = dict(
            store=output_zarr,
            mode="w",
            consolidated=True,
            encoding=encoding,
        )
        params = inspect.signature(xr.Dataset.to_zarr).parameters
        if "zarr_format" in params:
            kwargs["zarr_format"] = 2
        elif "zarr_version" in params:
            kwargs["zarr_version"] = 2
        else:
            raise RuntimeError("cannot force stable Zarr V2 output")

        out_ds.to_zarr(**kwargs)

        report = {
            "version": VERSION,
            "status": "PASS",
            "created_at_utc": now_utc(),
            "source": str(source),
            "output": str(output_zarr),
            "schema": str(schema_path),
            "schema_id": SCHEMA_ID,
            "schema_fingerprint_sha256": schema[
                "schema_fingerprint_sha256"
            ],
            "parent_schema_id": PARENT_SCHEMA_ID,
            "parent_schema_present": parent_schema is not None,
            "parent_schema_fingerprint_sha256": (
                parent_schema.get("schema_fingerprint_sha256")
                if parent_schema is not None
                else None
            ),
            "feature_count": 72,
            "source_hours": int(len(times)),
            "cities": int(ds.sizes["city"]),
            "full_grid_feature_cells": int(len(times) * 50 * 72),
            "target_materialized": False,
            "feature_nonfinite_counts_2023_2024": missing_dev,
            "feature_nonfinite_counts_2025_availability_only": missing_future,
            "fit_or_selection_performed": False,
            "2025_target_magnitudes_read": False,
            "storage_format": "Zarr V2",
            "next_step": (
                "audit bank lineage/causality/missingness, then build fold-local "
                "imputation/scaling and scenario selectors"
            ),
        }
        atomic_json(report, build_report_path)

    finally:
        ds.close()


if __name__ == "__main__":
    main()
