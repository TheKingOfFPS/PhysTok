#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


BUILD_VERSION = "city_meteorology_extraction_v1"

EXPECTED_HOURS = 27048
EXPECTED_START = pd.Timestamp("2022-12-01 00:00:00")
EXPECTED_END = pd.Timestamp("2025-12-31 23:00:00")
EXPECTED_REGIONS = {
    "North_China_Plain",
    "Northeast",
    "Sichuan_Basin",
    "South_China",
    "Yangtze_River_Delta_Extended",
}
PRESSURE_LEVELS = [925, 850]

SINGLE_SOURCE_VARS = [
    "temperature_2m_c",
    "dewpoint_temperature_2m_c",
    "surface_pressure_hpa",
    "u_wind_10m_ms",
    "v_wind_10m_ms",
    "boundary_layer_height_m",
    "total_cloud_cover_pct",
    "total_precipitation_mm",
    "surface_solar_radiation_downwards_wm2",
]

PRESSURE_PRIMARY_VARS = [
    "temperature_pressure_c",
    "relative_humidity_pressure_pct",
    "u_wind_pressure_ms",
    "v_wind_pressure_ms",
    "geopotential_height_pressure_m",
]

PAIR_SOURCE_VARS = [
    "delta_u_850_925_ms",
    "delta_v_850_925_ms",
    "delta_wind_speed_850_925_ms",
    "wind_shear_magnitude_850_925_ms",
    "delta_temperature_850_925_c",
    "delta_geopotential_height_850_925_m",
]

REQUIRED_SOURCE_VARS = set(
    SINGLE_SOURCE_VARS
    + PRESSURE_PRIMARY_VARS
    + PAIR_SOURCE_VARS
    + [
        "pressure_level_above_ground_mask",
        "pressure_pair_850_925_above_ground_mask",
    ]
)

CITY_SINGLE_VARS = (
    SINGLE_SOURCE_VARS
    + ["wind_speed_10m_ms", "wind_direction_10m_deg"]
)

CITY_PRESSURE_VARS = (
    PRESSURE_PRIMARY_VARS
    + ["wind_speed_pressure_ms", "wind_direction_pressure_deg"]
)

SUPPORT_VARS = [
    "pressure_level_valid_weight_sum",
    "pressure_level_valid_corner_count",
    "pressure_level_interpolation_renormalized_mask",
    "pressure_pair_850_925_valid_weight_sum",
    "pressure_pair_850_925_valid_corner_count",
    "pressure_pair_850_925_interpolation_renormalized_mask",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def expected_times() -> pd.DatetimeIndex:
    return pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")


def ensure_exact_timeline(times: Any, label: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(times)
    exp = expected_times()
    if len(idx) != EXPECTED_HOURS:
        raise RuntimeError(
            f"{label}: hours={len(idx)} != {EXPECTED_HOURS}"
        )
    if idx.has_duplicates:
        raise RuntimeError(f"{label}: duplicate timestamps")
    if not idx.is_monotonic_increasing:
        raise RuntimeError(f"{label}: non-monotonic timeline")
    if not idx.equals(exp):
        raise RuntimeError(
            f"{label}: timeline does not exactly match "
            f"{EXPECTED_START} -> {EXPECTED_END}"
        )
    return idx


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 50:
        raise RuntimeError(
            f"Expected 50 cities in manifest, got {len(rows)}"
        )

    required = {
        "region",
        "city",
        "city_zh",
        "latitude",
        "longitude",
    }
    if not rows:
        raise RuntimeError("City manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            f"City manifest missing columns: {sorted(missing)}"
        )

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                **r,
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            }
        )

    cities = [r["city"] for r in out]
    if len(set(cities)) != 50:
        raise RuntimeError("Duplicate city names in manifest")

    regions = {r["region"] for r in out}
    if regions != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Manifest region set mismatch: {sorted(regions)}"
        )

    counts = pd.Series(
        [r["region"] for r in out]
    ).value_counts()
    bad = {
        region: int(counts.get(region, 0))
        for region in EXPECTED_REGIONS
        if int(counts.get(region, 0)) != 10
    }
    if bad:
        raise RuntimeError(
            f"Expected 10 cities per region; got {bad}"
        )
    return out


def check_pre_extraction_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Required pre-extraction audit report not found: {path}"
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    status = report.get("overall_status")
    fail_gates = report.get("fail_gates", [])
    review_gates = report.get("review_gates", [])

    if fail_gates:
        raise RuntimeError(
            f"Pre-extraction audit has FAIL gates: {fail_gates}"
        )
    if status not in {"PASS", "REVIEW_REQUIRED"}:
        raise RuntimeError(
            f"Pre-extraction audit status is not acceptable: {status!r}"
        )

    unexpected_review = sorted(
        set(review_gates) - {"pressure_validity"}
    )
    if unexpected_review:
        raise RuntimeError(
            "Pre-extraction audit contains unexpected REVIEW gates: "
            f"{unexpected_review}"
        )

    return report


def bracket_axis(
    coords: np.ndarray,
    value: float,
    tol: float = 1e-10,
) -> tuple[float, float]:
    arr = np.sort(
        np.asarray(coords, dtype=np.float64)
    )

    if value < arr[0] - tol or value > arr[-1] + tol:
        raise RuntimeError(
            f"Coordinate {value} outside "
            f"[{arr[0]}, {arr[-1]}]"
        )

    nearest = int(np.argmin(np.abs(arr - value)))
    if abs(float(arr[nearest]) - value) <= tol:
        v = float(arr[nearest])
        return v, v

    j = int(np.searchsorted(arr, value, side="right"))
    if j <= 0 or j >= len(arr):
        raise RuntimeError(
            f"Cannot bracket coordinate {value}"
        )
    return float(arr[j - 1]), float(arr[j])


def coordinate_index(
    coords: np.ndarray,
    value: float,
    tol: float = 1e-8,
) -> int:
    arr = np.asarray(coords, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - value)))
    if abs(float(arr[idx]) - value) > tol:
        raise RuntimeError(
            f"Grid coordinate {value} not found; "
            f"nearest={float(arr[idx])}"
        )
    return idx


def bilinear_support(
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    lat: float,
    lon: float,
) -> list[dict[str, float]]:
    south, north = bracket_axis(lat_values, lat)
    west, east = bracket_axis(lon_values, lon)

    if south == north:
        lat_terms = [(south, 1.0)]
    else:
        lat_terms = [
            (south, (north - lat) / (north - south)),
            (north, (lat - south) / (north - south)),
        ]

    if west == east:
        lon_terms = [(west, 1.0)]
    else:
        lon_terms = [
            (west, (east - lon) / (east - west)),
            (east, (lon - west) / (east - west)),
        ]

    support: list[dict[str, float]] = []
    for la, wla in lat_terms:
        for lo, wlo in lon_terms:
            w = float(wla * wlo)
            if w > 1e-15:
                support.append(
                    {
                        "latitude": float(la),
                        "longitude": float(lo),
                        "weight": w,
                    }
                )

    total = sum(x["weight"] for x in support)
    if not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(
            f"Bilinear weights sum to {total}, not 1"
        )
    if not (1 <= len(support) <= 4):
        raise RuntimeError(
            f"Unexpected support size: {len(support)}"
        )
    return support


def wind_speed_direction(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    direction = np.mod(
        np.degrees(np.arctan2(-u, -v)),
        360.0,
    )
    direction = np.where(
        direction >= 360.0, 0.0, direction
    )
    direction = np.where(
        speed <= 1e-12, 0.0, direction
    )
    return speed, direction


def circular_abs_diff_deg(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return np.abs(
        (a - b + 180.0) % 360.0 - 180.0
    )


def ordinary_weighted(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    values: (..., corner)
    weights: (corner,)
    """
    if values.shape[-1] != weights.shape[0]:
        raise RuntimeError(
            "ordinary_weighted: corner dimension mismatch"
        )
    return np.sum(
        values * weights.reshape(
            (1,) * (values.ndim - 1) + (-1,)
        ),
        axis=-1,
    )


def masked_weighted(
    values: np.ndarray,
    base_weights: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    values: (..., corner)
    mask: same shape as values OR broadcast-compatible to values
    base_weights: (corner,)

    Returns:
        interpolated
        valid_weight_sum (... without corner)
        valid_corner_count (... without corner)
    """
    if values.shape[-1] != base_weights.shape[0]:
        raise RuntimeError(
            f"{label}: corner dimension mismatch"
        )

    bw_shape = (
        (1,) * (values.ndim - 1)
        + (base_weights.shape[0],)
    )
    bw = base_weights.reshape(bw_shape)
    m = np.asarray(mask, dtype=np.float64)

    if m.shape != values.shape:
        try:
            m = np.broadcast_to(m, values.shape)
        except ValueError as e:
            raise RuntimeError(
                f"{label}: mask cannot broadcast from "
                f"{np.asarray(mask).shape} to {values.shape}"
            ) from e

    weighted_mask = bw * m
    denom = np.sum(weighted_mask, axis=-1)
    corner_count = np.sum(m > 0.5, axis=-1).astype(np.uint8)

    if np.any(denom <= 0.0):
        bad = int(np.count_nonzero(denom <= 0.0))
        raise RuntimeError(
            f"{label}: zero valid interpolation support "
            f"for {bad} samples"
        )

    out = np.sum(
        values * weighted_mask,
        axis=-1,
    ) / denom

    return out, denom, corner_count


def select_city_corners(
    ds: xr.Dataset,
    support: list[dict[str, float]],
) -> xr.Dataset:
    lat_values = np.asarray(
        ds["latitude"].values,
        dtype=np.float64,
    )
    lon_values = np.asarray(
        ds["longitude"].values,
        dtype=np.float64,
    )

    lat_idx = np.array(
        [
            coordinate_index(
                lat_values,
                p["latitude"],
            )
            for p in support
        ],
        dtype=np.int64,
    )
    lon_idx = np.array(
        [
            coordinate_index(
                lon_values,
                p["longitude"],
            )
            for p in support
        ],
        dtype=np.int64,
    )

    li = xr.DataArray(
        lat_idx,
        dims="corner",
        coords={"corner": np.arange(len(support))},
    )
    lj = xr.DataArray(
        lon_idx,
        dims="corner",
        coords={"corner": np.arange(len(support))},
    )

    return ds[list(REQUIRED_SOURCE_VARS)].isel(
        latitude=li,
        longitude=lj,
    ).load()


def da_time_corner(da: xr.DataArray) -> np.ndarray:
    return np.asarray(
        da.transpose("time", "corner").values,
        dtype=np.float64,
    )


def da_time_pressure_corner(
    da: xr.DataArray,
) -> np.ndarray:
    return np.asarray(
        da.transpose(
            "time",
            "pressure_level_hpa",
            "corner",
        ).values,
        dtype=np.float64,
    )


def extract_city(
    ds: xr.Dataset,
    city_row: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    city = city_row["city"]
    lat = float(city_row["latitude"])
    lon = float(city_row["longitude"])

    support = bilinear_support(
        np.asarray(ds["latitude"].values),
        np.asarray(ds["longitude"].values),
        lat,
        lon,
    )
    weights = np.array(
        [p["weight"] for p in support],
        dtype=np.float64,
    )

    corners = select_city_corners(
        ds,
        support,
    )
    try:
        out: dict[str, np.ndarray] = {}

        # ----------------------------------------------------------
        # Single-level variables: ordinary bilinear interpolation.
        # ----------------------------------------------------------
        for var in SINGLE_SOURCE_VARS:
            x = da_time_corner(corners[var])
            out[var] = ordinary_weighted(x, weights)

        speed10, dir10 = wind_speed_direction(
            out["u_wind_10m_ms"],
            out["v_wind_10m_ms"],
        )
        out["wind_speed_10m_ms"] = speed10
        out["wind_direction_10m_deg"] = dir10

        # ----------------------------------------------------------
        # Pressure masks at the four positive-weight support corners.
        # shape: time x pressure x corner
        # ----------------------------------------------------------
        pressure_mask = da_time_pressure_corner(
            corners["pressure_level_above_ground_mask"]
        )
        pressure_mask_u8 = (
            pressure_mask > 0.5
        ).astype(np.uint8)

        pair_mask = da_time_corner(
            corners[
                "pressure_pair_850_925_above_ground_mask"
            ]
        )
        pair_mask_u8 = (
            pair_mask > 0.5
        ).astype(np.uint8)

        levels = [
            int(x)
            for x in ds[
                "pressure_level_hpa"
            ].values.tolist()
        ]
        if levels != PRESSURE_LEVELS:
            raise RuntimeError(
                f"{city}: pressure levels/order={levels}, "
                f"expected={PRESSURE_LEVELS}"
            )
        i925 = levels.index(925)
        i850 = levels.index(850)

        # The frozen policy relies on the full audit result:
        # 850 hPa must have complete city support everywhere.
        if not np.all(
            pressure_mask_u8[:, i850, :] == 1
        ):
            bad = int(
                np.count_nonzero(
                    np.any(
                        pressure_mask_u8[:, i850, :] == 0,
                        axis=1,
                    )
                )
            )
            raise RuntimeError(
                f"{city}: 850 hPa has invalid bilinear support "
                f"for {bad} hours. The audited extraction policy "
                "must be reviewed before continuing."
            )

        # Support diagnostics.
        base_w = weights.reshape(1, 1, -1)
        valid_weight = np.sum(
            base_w * pressure_mask_u8,
            axis=-1,
        )
        valid_count = np.sum(
            pressure_mask_u8,
            axis=-1,
        ).astype(np.uint8)

        if np.any(valid_weight <= 0.0):
            raise RuntimeError(
                f"{city}: zero pressure-level valid support"
            )

        out[
            "pressure_level_valid_weight_sum"
        ] = valid_weight
        out[
            "pressure_level_valid_corner_count"
        ] = valid_count
        out[
            "pressure_level_interpolation_renormalized_mask"
        ] = (
            valid_weight < (1.0 - 1e-12)
        ).astype(np.uint8)

        pair_valid_weight = np.sum(
            weights.reshape(1, -1) * pair_mask_u8,
            axis=-1,
        )
        pair_valid_count = np.sum(
            pair_mask_u8,
            axis=-1,
        ).astype(np.uint8)

        if np.any(pair_valid_weight <= 0.0):
            raise RuntimeError(
                f"{city}: zero 850/925 pair valid support"
            )

        out[
            "pressure_pair_850_925_valid_weight_sum"
        ] = pair_valid_weight
        out[
            "pressure_pair_850_925_valid_corner_count"
        ] = pair_valid_count
        out[
            "pressure_pair_850_925_interpolation_renormalized_mask"
        ] = (
            pair_valid_weight < (1.0 - 1e-12)
        ).astype(np.uint8)

        # Pair mask is expected to equal the 925 hPa mask.
        if not np.array_equal(
            pair_mask_u8,
            pressure_mask_u8[:, i925, :],
        ):
            raise RuntimeError(
                f"{city}: pair mask != 925 hPa support mask"
            )

        # ----------------------------------------------------------
        # Pressure-level primary variables.
        #
        # 850 hPa: ordinary bilinear.
        # 925 hPa: validity-aware renormalized bilinear.
        # ----------------------------------------------------------
        pressure_outputs: dict[str, np.ndarray] = {}
        pair_primary: dict[str, dict[int, np.ndarray]] = {}

        for var in PRESSURE_PRIMARY_VARS:
            x = da_time_pressure_corner(
                corners[var]
            )
            y = np.empty(
                (EXPECTED_HOURS, len(PRESSURE_LEVELS)),
                dtype=np.float64,
            )

            # 925: masked/renormalized.
            y[:, i925], _, _ = masked_weighted(
                x[:, i925, :],
                weights,
                pressure_mask_u8[:, i925, :],
                f"{city}/{var}/925",
            )

            # 850: ordinary bilinear because the full audit
            # established 4/4 valid support for all 50 cities.
            y[:, i850] = ordinary_weighted(
                x[:, i850, :],
                weights,
            )
            pressure_outputs[var] = y

            # For pair comparisons, both pressure levels use the
            # SAME pair-valid renormalized support.
            if var in {
                "temperature_pressure_c",
                "u_wind_pressure_ms",
                "v_wind_pressure_ms",
                "geopotential_height_pressure_m",
            }:
                pair_primary[var] = {}
                for level, idx in (
                    (925, i925),
                    (850, i850),
                ):
                    pv, _, _ = masked_weighted(
                        x[:, idx, :],
                        weights,
                        pair_mask_u8,
                        f"{city}/{var}/{level}/pair",
                    )
                    pair_primary[var][level] = pv

        for var, values in pressure_outputs.items():
            out[var] = values

        wsp, wdir = wind_speed_direction(
            out["u_wind_pressure_ms"],
            out["v_wind_pressure_ms"],
        )
        out["wind_speed_pressure_ms"] = wsp
        out["wind_direction_pressure_deg"] = wdir

        # ----------------------------------------------------------
        # 850-925 pair / vertical features.
        # Interpolate the already-defined grid-level derived fields
        # with pair-valid renormalized support.
        # ----------------------------------------------------------
        for var in PAIR_SOURCE_VARS:
            x = da_time_corner(corners[var])
            y, _, _ = masked_weighted(
                x,
                weights,
                pair_mask_u8,
                f"{city}/{var}/pair",
            )
            out[var] = y

        # Independent linear invariants under common pair support.
        invariant_pairs = [
            (
                "delta_u_850_925_ms",
                "u_wind_pressure_ms",
            ),
            (
                "delta_v_850_925_ms",
                "v_wind_pressure_ms",
            ),
            (
                "delta_temperature_850_925_c",
                "temperature_pressure_c",
            ),
            (
                "delta_geopotential_height_850_925_m",
                "geopotential_height_pressure_m",
            ),
        ]
        invariant_max_error: dict[str, float] = {}
        for delta_var, primary_var in invariant_pairs:
            expected_delta = (
                pair_primary[primary_var][850]
                - pair_primary[primary_var][925]
            )
            err = float(
                np.max(
                    np.abs(
                        out[delta_var] - expected_delta
                    )
                )
            )
            invariant_max_error[delta_var] = err
            invariant_tol = (
                1e-4
                if delta_var == "delta_geopotential_height_850_925_m"
                else 1e-5
            )
            if not math.isfinite(err) or err > invariant_tol:
                raise RuntimeError(
                    f"{city}: pair invariant failed for "
                    f"{delta_var}; max_error={err}; "
                    f"tolerance={invariant_tol}"
                )

        # All scientific output variables must be finite.
        for name, arr in out.items():
            if not np.all(np.isfinite(arr)):
                bad = int(
                    np.count_nonzero(~np.isfinite(arr))
                )
                raise RuntimeError(
                    f"{city}/{name}: {bad} non-finite values"
                )

        mapping = {
            "city": city,
            "city_zh": city_row["city_zh"],
            "region": city_row["region"],
            "city_latitude": lat,
            "city_longitude": lon,
            "support_corner_count": len(support),
            "support_json": json.dumps(
                support,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "925_renormalized_hours": int(
                out[
                    "pressure_level_interpolation_renormalized_mask"
                ][:, i925].sum()
            ),
            "925_renormalized_fraction": float(
                out[
                    "pressure_level_interpolation_renormalized_mask"
                ][:, i925].mean()
            ),
            "925_min_valid_weight_sum": float(
                out[
                    "pressure_level_valid_weight_sum"
                ][:, i925].min()
            ),
            "925_min_valid_corner_count": int(
                out[
                    "pressure_level_valid_corner_count"
                ][:, i925].min()
            ),
            "850_renormalized_hours": int(
                out[
                    "pressure_level_interpolation_renormalized_mask"
                ][:, i850].sum()
            ),
            "850_min_valid_weight_sum": float(
                out[
                    "pressure_level_valid_weight_sum"
                ][:, i850].min()
            ),
            "pair_renormalized_hours": int(
                out[
                    "pressure_pair_850_925_interpolation_renormalized_mask"
                ].sum()
            ),
            "pair_renormalized_fraction": float(
                out[
                    "pressure_pair_850_925_interpolation_renormalized_mask"
                ].mean()
            ),
            "pair_min_valid_weight_sum": float(
                out[
                    "pressure_pair_850_925_valid_weight_sum"
                ].min()
            ),
            "pair_min_valid_corner_count": int(
                out[
                    "pressure_pair_850_925_valid_corner_count"
                ].min()
            ),
            "max_linear_pair_invariant_error": max(
                invariant_max_error.values()
            ),
        }

        return out, mapping

    finally:
        corners.close()


def allocate_outputs(
    n_cities: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    for var in CITY_SINGLE_VARS + PAIR_SOURCE_VARS:
        out[var] = np.empty(
            (EXPECTED_HOURS, n_cities),
            dtype=np.float64,
        )

    for var in CITY_PRESSURE_VARS:
        out[var] = np.empty(
            (
                EXPECTED_HOURS,
                n_cities,
                len(PRESSURE_LEVELS),
            ),
            dtype=np.float64,
        )

    out[
        "pressure_level_valid_weight_sum"
    ] = np.empty(
        (
            EXPECTED_HOURS,
            n_cities,
            len(PRESSURE_LEVELS),
        ),
        dtype=np.float64,
    )
    out[
        "pressure_level_valid_corner_count"
    ] = np.empty(
        (
            EXPECTED_HOURS,
            n_cities,
            len(PRESSURE_LEVELS),
        ),
        dtype=np.uint8,
    )
    out[
        "pressure_level_interpolation_renormalized_mask"
    ] = np.empty(
        (
            EXPECTED_HOURS,
            n_cities,
            len(PRESSURE_LEVELS),
        ),
        dtype=np.uint8,
    )

    out[
        "pressure_pair_850_925_valid_weight_sum"
    ] = np.empty(
        (EXPECTED_HOURS, n_cities),
        dtype=np.float64,
    )
    out[
        "pressure_pair_850_925_valid_corner_count"
    ] = np.empty(
        (EXPECTED_HOURS, n_cities),
        dtype=np.uint8,
    )
    out[
        "pressure_pair_850_925_interpolation_renormalized_mask"
    ] = np.empty(
        (EXPECTED_HOURS, n_cities),
        dtype=np.uint8,
    )

    return out


def assign_city_output(
    dest: dict[str, np.ndarray],
    city_index: int,
    city_out: dict[str, np.ndarray],
) -> None:
    for var in CITY_SINGLE_VARS + PAIR_SOURCE_VARS:
        dest[var][:, city_index] = city_out[var]

    for var in CITY_PRESSURE_VARS:
        dest[var][:, city_index, :] = city_out[var]

    for var in SUPPORT_VARS:
        arr = city_out[var]
        if arr.ndim == 2:
            dest[var][:, city_index, :] = arr
        elif arr.ndim == 1:
            dest[var][:, city_index] = arr
        else:
            raise RuntimeError(
                f"{var}: unexpected ndim={arr.ndim}"
            )


def copy_direct_attrs(
    source_example: xr.Dataset,
    ds_out: xr.Dataset,
) -> None:
    # Variables that are spatially interpolated directly from
    # deterministic grid-level derived fields.
    direct_vars = (
        SINGLE_SOURCE_VARS
        + PRESSURE_PRIMARY_VARS
        + PAIR_SOURCE_VARS
    )
    for var in direct_vars:
        if var in source_example and var in ds_out:
            ds_out[var].attrs.update(
                dict(source_example[var].attrs)
            )

    for var in SINGLE_SOURCE_VARS:
        ds_out[var].attrs[
            "city_spatial_interpolation"
        ] = "ordinary bilinear"

    for var in PRESSURE_PRIMARY_VARS:
        ds_out[var].attrs[
            "city_spatial_interpolation"
        ] = (
            "850 hPa: ordinary bilinear; "
            "925 hPa: above-ground-mask-aware "
            "renormalized bilinear"
        )

    for var in PAIR_SOURCE_VARS:
        ds_out[var].attrs[
            "city_spatial_interpolation"
        ] = (
            "850/925 pair-mask-aware renormalized bilinear"
        )

    ds_out["wind_speed_10m_ms"].attrs.update(
        {
            "units": "m/s",
            "formula": "sqrt(u_wind_10m_ms^2 + v_wind_10m_ms^2)",
            "city_spatial_interpolation": (
                "derive after bilinear interpolation of u/v"
            ),
        }
    )
    ds_out["wind_direction_10m_deg"].attrs.update(
        {
            "units": "degree",
            "convention": "meteorological FROM direction",
            "formula": (
                "mod(degrees(atan2(-u,-v)),360); "
                "calm speed<=1e-12 -> 0 degree"
            ),
            "city_spatial_interpolation": (
                "derive after bilinear interpolation of u/v; "
                "angles are not directly averaged"
            ),
        }
    )
    ds_out["wind_speed_pressure_ms"].attrs.update(
        {
            "units": "m/s",
            "formula": (
                "sqrt(u_wind_pressure_ms^2 + "
                "v_wind_pressure_ms^2)"
            ),
            "city_spatial_interpolation": (
                "derive after level-specific interpolation of u/v"
            ),
        }
    )
    ds_out["wind_direction_pressure_deg"].attrs.update(
        {
            "units": "degree",
            "convention": "meteorological FROM direction",
            "formula": (
                "mod(degrees(atan2(-u,-v)),360); "
                "calm speed<=1e-12 -> 0 degree"
            ),
            "city_spatial_interpolation": (
                "derive after level-specific interpolation of u/v; "
                "angles are not directly averaged"
            ),
        }
    )

    ds_out[
        "pressure_level_valid_weight_sum"
    ].attrs.update(
        {
            "description": (
                "Sum of original bilinear weights belonging to "
                "above-ground support corners before renormalization."
            ),
            "range": "[0,1]",
            "diagnostic_role": (
                "provenance/support-quality variable; not automatically "
                "selected as a model feature"
            ),
        }
    )
    ds_out[
        "pressure_level_valid_corner_count"
    ].attrs.update(
        {
            "description": (
                "Number of positive-weight bilinear support corners "
                "that are above ground for this pressure level."
            ),
            "range": "0..4",
        }
    )
    ds_out[
        "pressure_level_interpolation_renormalized_mask"
    ].attrs.update(
        {
            "description": (
                "1 if pressure-level interpolation required "
                "renormalization because at least one positive-weight "
                "support corner was below ground; 0 otherwise."
            ),
            "range": "{0,1}",
        }
    )
    ds_out[
        "pressure_pair_850_925_valid_weight_sum"
    ].attrs.update(
        {
            "description": (
                "Sum of original bilinear weights valid for the "
                "joint 850/925 pair support before renormalization."
            ),
            "range": "[0,1]",
        }
    )
    ds_out[
        "pressure_pair_850_925_valid_corner_count"
    ].attrs.update(
        {
            "description": (
                "Number of positive-weight bilinear support corners "
                "valid for the 850/925 pair."
            ),
            "range": "0..4",
        }
    )
    ds_out[
        "pressure_pair_850_925_interpolation_renormalized_mask"
    ].attrs.update(
        {
            "description": (
                "1 if 850/925 pair-feature interpolation required "
                "support renormalization; 0 otherwise."
            ),
            "range": "{0,1}",
        }
    )


def build_dataset(
    outputs: dict[str, np.ndarray],
    manifest: list[dict[str, Any]],
    source_example: xr.Dataset,
    audit_report_path: Path,
) -> xr.Dataset:
    coords = {
        "time": expected_times().values,
        "city": np.asarray(
            [r["city"] for r in manifest],
            dtype=str,
        ),
        "pressure_level_hpa": np.asarray(
            PRESSURE_LEVELS,
            dtype=np.int16,
        ),
        "city_zh": (
            "city",
            np.asarray(
                [r["city_zh"] for r in manifest],
                dtype=str,
            ),
        ),
        "region": (
            "city",
            np.asarray(
                [r["region"] for r in manifest],
                dtype=str,
            ),
        ),
        "latitude": (
            "city",
            np.asarray(
                [r["latitude"] for r in manifest],
                dtype=np.float64,
            ),
        ),
        "longitude": (
            "city",
            np.asarray(
                [r["longitude"] for r in manifest],
                dtype=np.float64,
            ),
        ),
    }

    data_vars: dict[str, Any] = {}

    for var in CITY_SINGLE_VARS + PAIR_SOURCE_VARS:
        data_vars[var] = (
            ("time", "city"),
            outputs[var],
        )

    for var in CITY_PRESSURE_VARS:
        data_vars[var] = (
            ("time", "city", "pressure_level_hpa"),
            outputs[var],
        )

    data_vars[
        "pressure_level_valid_weight_sum"
    ] = (
        ("time", "city", "pressure_level_hpa"),
        outputs["pressure_level_valid_weight_sum"],
    )
    data_vars[
        "pressure_level_valid_corner_count"
    ] = (
        ("time", "city", "pressure_level_hpa"),
        outputs["pressure_level_valid_corner_count"],
    )
    data_vars[
        "pressure_level_interpolation_renormalized_mask"
    ] = (
        ("time", "city", "pressure_level_hpa"),
        outputs[
            "pressure_level_interpolation_renormalized_mask"
        ],
    )

    for var in [
        "pressure_pair_850_925_valid_weight_sum",
        "pressure_pair_850_925_valid_corner_count",
        "pressure_pair_850_925_interpolation_renormalized_mask",
    ]:
        data_vars[var] = (
            ("time", "city"),
            outputs[var],
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "build_version": BUILD_VERSION,
            "created_at_utc": now_utc(),
            "source": (
                "ERA5 deterministic physical derived regional grids"
            ),
            "source_root": (
                "04_derived/era5_physical/<region>.zarr"
            ),
            "city_manifest": (
                "05_manifests/selected_cities_v2.csv"
            ),
            "pre_extraction_audit": str(
                audit_report_path
            ),
            "time_basis": "UTC",
            "time_range": (
                "2022-12-01 00:00 through "
                "2025-12-31 23:00 hourly"
            ),
            "single_level_policy": "ordinary bilinear",
            "pressure_850_policy": (
                "ordinary bilinear; full pre-extraction audit "
                "must confirm complete 850 hPa city support"
            ),
            "pressure_925_policy": (
                "above-ground-mask-aware renormalized bilinear"
            ),
            "pressure_pair_policy": (
                "joint 850/925 pair-mask-aware renormalized bilinear"
            ),
            "wind_policy": (
                "interpolate u/v first, then derive speed and "
                "meteorological FROM direction; never bilinear-average "
                "direction angles"
            ),
            "vertical_pair_semantics": (
                "grid-level deterministic 850-925 derived fields are "
                "interpolated using a common pair-valid support"
            ),
            "no_data_dependent_processing": (
                "no imputation, clipping, scaling, smoothing, "
                "feature selection, climatology fitting, or "
                "train/test transformation"
            ),
            "split_before_fit_boundary": (
                "all future data-dependent transformations remain "
                "downstream of split construction"
            ),
        },
    )

    copy_direct_attrs(
        source_example,
        ds_out,
    )
    return ds_out


def final_gate(
    ds: xr.Dataset,
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    ensure_exact_timeline(ds["time"].values, "city meteorology")

    if int(ds.sizes.get("city", -1)) != 50:
        raise RuntimeError(
            f"Final city count={ds.sizes.get('city')} != 50"
        )
    if [
        str(x)
        for x in ds["city"].values.tolist()
    ] != [
        r["city"]
        for r in manifest
    ]:
        raise RuntimeError(
            "Final city order differs from manifest"
        )
    if [
        int(x)
        for x in ds[
            "pressure_level_hpa"
        ].values.tolist()
    ] != PRESSURE_LEVELS:
        raise RuntimeError(
            "Final pressure level order mismatch"
        )

    # Every scientific field must be finite.
    nonfinite: list[dict[str, Any]] = []
    for name, da in ds.data_vars.items():
        arr = np.asarray(da.values)
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        if bad:
            nonfinite.append(
                {
                    "variable": name,
                    "nonfinite_count": bad,
                }
            )
    if nonfinite:
        raise RuntimeError(
            f"Non-finite output detected: {nonfinite[:5]}"
        )

    levels = [
        int(x)
        for x in ds[
            "pressure_level_hpa"
        ].values.tolist()
    ]
    i925 = levels.index(925)
    i850 = levels.index(850)

    # 850 must remain fully supported under the audited policy.
    pcount = np.asarray(
        ds[
            "pressure_level_valid_corner_count"
        ].values
    )
    prenorm = np.asarray(
        ds[
            "pressure_level_interpolation_renormalized_mask"
        ].values
    )
    if np.any(prenorm[:, :, i850] != 0):
        raise RuntimeError(
            "850 hPa unexpectedly required renormalization"
        )

    # Pair support and 925 support must be identical.
    pair_count = np.asarray(
        ds[
            "pressure_pair_850_925_valid_corner_count"
        ].values
    )
    pair_renorm = np.asarray(
        ds[
            "pressure_pair_850_925_interpolation_renormalized_mask"
        ].values
    )
    if not np.array_equal(
        pair_count,
        pcount[:, :, i925],
    ):
        raise RuntimeError(
            "Pair valid-corner count != 925 hPa count"
        )
    if not np.array_equal(
        pair_renorm,
        prenorm[:, :, i925],
    ):
        raise RuntimeError(
            "Pair renormalization mask != 925 hPa mask"
        )

    # Wind formula invariants at city level.
    u10 = np.asarray(
        ds["u_wind_10m_ms"].values
    )
    v10 = np.asarray(
        ds["v_wind_10m_ms"].values
    )
    speed10, dir10 = wind_speed_direction(u10, v10)

    speed10_err = float(
        np.max(
            np.abs(
                np.asarray(
                    ds["wind_speed_10m_ms"].values
                )
                - speed10
            )
        )
    )
    dir10_err = float(
        np.max(
            circular_abs_diff_deg(
                np.asarray(
                    ds["wind_direction_10m_deg"].values
                ),
                dir10,
            )
        )
    )

    up = np.asarray(
        ds["u_wind_pressure_ms"].values
    )
    vp = np.asarray(
        ds["v_wind_pressure_ms"].values
    )
    wsp, wdir = wind_speed_direction(up, vp)

    wsp_err = float(
        np.max(
            np.abs(
                np.asarray(
                    ds["wind_speed_pressure_ms"].values
                )
                - wsp
            )
        )
    )
    wdir_err = float(
        np.max(
            circular_abs_diff_deg(
                np.asarray(
                    ds["wind_direction_pressure_deg"].values
                ),
                wdir,
            )
        )
    )

    if max(
        speed10_err,
        wsp_err,
    ) > 1e-10:
        raise RuntimeError(
            "City wind-speed invariant failed"
        )
    if max(
        dir10_err,
        wdir_err,
    ) > 1e-10:
        raise RuntimeError(
            "City wind-direction invariant failed"
        )

    renorm_925 = prenorm[:, :, i925]
    affected_idx = np.where(
        renorm_925.any(axis=0)
    )[0]
    affected_cities = [
        str(ds["city"].values[i])
        for i in affected_idx
    ]

    pweight = np.asarray(
        ds[
            "pressure_level_valid_weight_sum"
        ].values
    )
    pair_weight = np.asarray(
        ds[
            "pressure_pair_850_925_valid_weight_sum"
        ].values
    )

    return {
        "status": "PASS",
        "hours": EXPECTED_HOURS,
        "cities": 50,
        "pressure_levels_hpa": PRESSURE_LEVELS,
        "output_variable_count": len(ds.data_vars),
        "850_renormalized_samples": int(
            prenorm[:, :, i850].sum()
        ),
        "925_renormalized_samples": int(
            renorm_925.sum()
        ),
        "925_affected_city_count": len(
            affected_cities
        ),
        "925_affected_cities": affected_cities,
        "925_min_valid_weight_sum": float(
            pweight[:, :, i925].min()
        ),
        "925_min_valid_corner_count": int(
            pcount[:, :, i925].min()
        ),
        "pair_renormalized_samples": int(
            pair_renorm.sum()
        ),
        "pair_min_valid_weight_sum": float(
            pair_weight.min()
        ),
        "pair_min_valid_corner_count": int(
            pair_count.min()
        ),
        "wind_speed_10m_max_formula_error": (
            speed10_err
        ),
        "wind_direction_10m_max_circular_error_deg": (
            dir10_err
        ),
        "wind_speed_pressure_max_formula_error": (
            wsp_err
        ),
        "wind_direction_pressure_max_circular_error_deg": (
            wdir_err
        ),
    }


def zarr_encoding(ds: xr.Dataset) -> dict[str, Any]:
    enc: dict[str, Any] = {}
    for name, da in ds.data_vars.items():
        if da.dims == (
            "time",
            "city",
            "pressure_level_hpa",
        ):
            enc[name] = {"chunks": (168, 10, 2)}
        elif da.dims == ("time", "city"):
            enc[name] = {"chunks": (168, 10)}
    return enc


def print_smoke(
    city: str,
    city_out: dict[str, np.ndarray],
    mapping: dict[str, Any],
) -> None:
    print("\n" + "=" * 88)
    print(f"CITY METEOROLOGY SMOKE TEST: {city}")
    print("=" * 88)
    print("support:", mapping["support_json"])
    print(
        "925 renormalized:",
        mapping["925_renormalized_hours"],
        f"hours ({mapping['925_renormalized_fraction']:.4%})",
    )
    print(
        "925 min valid support:",
        f"weight={mapping['925_min_valid_weight_sum']:.6f},",
        f"corners={mapping['925_min_valid_corner_count']}",
    )
    print(
        "850 renormalized:",
        mapping["850_renormalized_hours"],
        "hours",
    )
    print(
        "pair renormalized:",
        mapping["pair_renormalized_hours"],
        f"hours ({mapping['pair_renormalized_fraction']:.4%})",
    )
    print(
        "pair min valid support:",
        f"weight={mapping['pair_min_valid_weight_sum']:.6f},",
        f"corners={mapping['pair_min_valid_corner_count']}",
    )
    print(
        "max linear pair invariant error:",
        f"{mapping['max_linear_pair_invariant_error']:.3e}",
    )
    print(
        "scientific variables finite:",
        all(
            np.all(np.isfinite(v))
            for v in city_out.values()
        ),
    )
    print("SMOKE TEST: PASS")
    print("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract audited city-level hourly meteorology from "
            "regional ERA5 physical-derived grids."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=(
            Path.cwd()
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing formal output Zarr store.",
    )
    ap.add_argument(
        "--smoke-city",
        type=str,
        default=None,
        help=(
            "Extract and validate one city only; do not write "
            "the formal 50-city output."
        ),
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()

    manifest_path = (
        root
        / "05_manifests"
        / "selected_cities_v2.csv"
    )
    audit_report_path = (
        root
        / "09_diagnostics"
        / "pre_city_extraction"
        / "full_diagnostic_report.json"
    )
    source_root = (
        root
        / "04_derived"
        / "era5_physical"
    )
    output_root = (
        root
        / "04_derived"
        / "city_meteorology"
    )
    output_store = (
        output_root
        / "city_meteorology_50cities.zarr"
    )
    build_report_path = (
        output_root
        / "city_meteorology_50cities_build_report.json"
    )
    mapping_path = (
        root
        / "05_manifests"
        / "city_meteorology_interpolation_mapping.csv"
    )

    manifest = load_manifest(manifest_path)
    pre_audit = check_pre_extraction_audit(
        audit_report_path
    )

    by_city = {
        r["city"]: r
        for r in manifest
    }
    if (
        args.smoke_city is not None
        and args.smoke_city not in by_city
    ):
        raise RuntimeError(
            f"Unknown --smoke-city {args.smoke_city!r}"
        )

    # Open all five source stores and validate their shared timeline/schema.
    stores: dict[str, xr.Dataset] = {}
    try:
        for region in sorted(EXPECTED_REGIONS):
            path = source_root / f"{region}.zarr"
            if not path.exists():
                raise RuntimeError(
                    f"Missing source store: {path}"
                )
            ds = xr.open_zarr(
                str(path),
                consolidated=True,
                chunks=None,
            )
            ensure_exact_timeline(
                ds["time"].values,
                f"{region} ERA5 physical derived",
            )

            missing = REQUIRED_SOURCE_VARS - set(
                ds.data_vars
            )
            if missing:
                raise RuntimeError(
                    f"{region}: missing required variables "
                    f"{sorted(missing)}"
                )

            levels = [
                int(x)
                for x in ds[
                    "pressure_level_hpa"
                ].values.tolist()
            ]
            if levels != PRESSURE_LEVELS:
                raise RuntimeError(
                    f"{region}: pressure levels/order={levels}, "
                    f"expected={PRESSURE_LEVELS}"
                )
            stores[region] = ds

        # ----------------------------------------------------------
        # Smoke mode.
        # ----------------------------------------------------------
        if args.smoke_city is not None:
            row = by_city[args.smoke_city]
            ds = stores[row["region"]]
            city_out, mapping = extract_city(
                ds,
                row,
            )
            print_smoke(
                args.smoke_city,
                city_out,
                mapping,
            )
            return

        # ----------------------------------------------------------
        # Formal 50-city extraction.
        # ----------------------------------------------------------
        if output_store.exists():
            if not args.overwrite:
                raise RuntimeError(
                    f"Output exists: {output_store}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_store)

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        mapping_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arrays = allocate_outputs(
            len(manifest)
        )
        mapping_rows: list[dict[str, Any]] = []

        print(
            "\nCITY METEOROLOGY EXTRACTION",
            flush=True,
        )
        print(
            "Policy: single=bilinear | "
            "850=bilinear | "
            "925=mask-aware renormalized | "
            "pair=pair-mask-aware renormalized",
            flush=True,
        )

        for i, row in enumerate(manifest):
            city = row["city"]
            region = row["region"]
            print(
                f"[{i+1:02d}/50] {region}/{city}",
                flush=True,
            )
            city_out, mapping = extract_city(
                stores[region],
                row,
            )
            assign_city_output(
                arrays,
                i,
                city_out,
            )
            mapping_rows.append(mapping)

            if mapping["925_renormalized_hours"]:
                print(
                    "   925 REVIEW-handled: "
                    f"{mapping['925_renormalized_hours']} h "
                    f"({mapping['925_renormalized_fraction']:.4%}), "
                    f"min corners="
                    f"{mapping['925_min_valid_corner_count']}, "
                    f"min valid weight="
                    f"{mapping['925_min_valid_weight_sum']:.6f}",
                    flush=True,
                )

        example = stores[
            sorted(EXPECTED_REGIONS)[0]
        ]
        ds_out = build_dataset(
            arrays,
            manifest,
            example,
            audit_report_path,
        )

        gate = final_gate(
            ds_out,
            manifest,
        )

        # Write formal store only after in-memory final gate passes.
        ds_out.to_zarr(
            str(output_store),
            mode="w",
            consolidated=True,
            encoding=zarr_encoding(ds_out),
        )

        # Reopen and repeat key structural checks after serialization.
        verify = xr.open_zarr(
            str(output_store),
            consolidated=True,
            chunks=None,
        )
        try:
            verify_gate = final_gate(
                verify,
                manifest,
            )
        finally:
            verify.close()

        write_csv(
            mapping_rows,
            mapping_path,
        )

        build_report = {
            "build_version": BUILD_VERSION,
            "created_at_utc": now_utc(),
            "status": "PASS",
            "source_root": str(source_root),
            "output_store": str(output_store),
            "manifest": str(manifest_path),
            "interpolation_mapping": str(mapping_path),
            "pre_extraction_audit_report": str(
                audit_report_path
            ),
            "pre_extraction_audit_status": (
                pre_audit.get("overall_status")
            ),
            "pre_extraction_review_gates": (
                pre_audit.get("review_gates", [])
            ),
            "policy": {
                "single_level": "ordinary bilinear",
                "850_hpa": (
                    "ordinary bilinear; requires complete audited "
                    "support"
                ),
                "925_hpa": (
                    "above-ground-mask-aware renormalized bilinear"
                ),
                "850_925_pair": (
                    "pair-mask-aware renormalized bilinear"
                ),
                "wind": (
                    "interpolate u/v first, derive speed/direction "
                    "afterward"
                ),
                "wind_direction": (
                    "never directly bilinear-average angles"
                ),
                "data_dependent_processing": "none",
            },
            "pre_write_gate": gate,
            "post_write_gate": verify_gate,
            "city_mapping_summary": mapping_rows,
        }
        atomic_json(
            build_report,
            build_report_path,
        )

        run_root = (
            root
            / "08_runs"
            / "city_meteorology_extraction"
        )
        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        run_path = (
            run_root
            / (
                "run_"
                + datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                + ".json"
            )
        )
        atomic_json(
            build_report,
            run_path,
        )

        ds_out.close()

        print("\n" + "=" * 92)
        print("CITY METEOROLOGY EXTRACTION: PASS")
        print("=" * 92)
        print(f"Store: {output_store}")
        print(f"Hours: {gate['hours']}")
        print(f"Cities: {gate['cities']}")
        print(
            "925 affected cities:",
            gate["925_affected_cities"],
        )
        print(
            "925 renormalized samples:",
            gate["925_renormalized_samples"],
        )
        print(
            "925 min valid support:",
            f"weight={gate['925_min_valid_weight_sum']:.6f},",
            f"corners={gate['925_min_valid_corner_count']}",
        )
        print(
            "850 renormalized samples:",
            gate["850_renormalized_samples"],
        )
        print(
            "Pair renormalized samples:",
            gate["pair_renormalized_samples"],
        )
        print(f"Mapping: {mapping_path}")
        print(f"Report: {build_report_path}")
        print(f"Run report: {run_path}")
        print("=" * 92)

    finally:
        for ds in stores.values():
            try:
                ds.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
