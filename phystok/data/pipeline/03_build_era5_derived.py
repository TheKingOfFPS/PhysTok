#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    import zarr  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Missing zarr. Activate era5_data or install with "
        "`conda install -c conda-forge zarr numcodecs`."
    ) from exc


DERIVED_VERSION = "era5_physical_derived_v2"

# Fixed physical constants / exact conversion constants.
KELVIN_OFFSET = 273.15
PA_PER_HPA = 100.0
MM_PER_M = 1000.0
SECONDS_PER_HOUR = 3600.0
PERCENT_PER_FRACTION = 100.0
STANDARD_GRAVITY_MS2 = 9.80665
CALM_WIND_EPS_MS = 1.0e-12

PRESSURE_LEVELS_HPA = [925, 850]
EXPECTED_FULL_HOURS = 27048
DEFAULT_CHUNK_HOURS = 168

EXPECTED_CANONICAL_VARIABLES = {
    "boundary_layer_height",
    "dewpoint_temperature_2m",
    "geopotential_pressure",
    "relative_humidity_pressure",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "temperature_2m",
    "temperature_pressure",
    "total_cloud_cover",
    "total_precipitation",
    "u_wind_10m",
    "u_wind_pressure",
    "v_wind_10m",
    "v_wind_pressure",
}

EXPECTED_CANONICAL_UNITS = {
    "temperature_2m": "K",
    "dewpoint_temperature_2m": "K",
    "surface_pressure": "Pa",
    "u_wind_10m": "m s**-1",
    "v_wind_10m": "m s**-1",
    "boundary_layer_height": "m",
    "total_cloud_cover": "(0 - 1)",
    "total_precipitation": "m",
    "surface_solar_radiation_downwards": "J m**-2",
    "temperature_pressure": "K",
    "relative_humidity_pressure": "%",
    "u_wind_pressure": "m s**-1",
    "v_wind_pressure": "m s**-1",
    "geopotential_pressure": "m**2 s**-2",
}

DERIVED_VARIABLES = {
    # Single-level physical-ready variables.
    "temperature_2m_c",
    "dewpoint_temperature_2m_c",
    "surface_pressure_hpa",
    "u_wind_10m_ms",
    "v_wind_10m_ms",
    "wind_speed_10m_ms",
    "wind_direction_10m_deg",
    "boundary_layer_height_m",
    "total_cloud_cover_pct",
    "total_precipitation_mm",
    "surface_solar_radiation_downwards_wm2",

    # Pressure-level variables retaining the pressure dimension.
    "temperature_pressure_c",
    "relative_humidity_pressure_pct",
    "u_wind_pressure_ms",
    "v_wind_pressure_ms",
    "wind_speed_pressure_ms",
    "wind_direction_pressure_deg",
    "geopotential_height_pressure_m",

    # Pressure-level physical-validity masks.
    "pressure_level_above_ground_mask",
    "pressure_pair_850_925_above_ground_mask",

    # 850 minus 925 hPa deterministic layer contrasts.
    "delta_u_850_925_ms",
    "delta_v_850_925_ms",
    "delta_wind_speed_850_925_ms",
    "wind_shear_magnitude_850_925_ms",
    "delta_temperature_850_925_c",
    "delta_geopotential_height_850_925_m",
}


FORMULA_MANIFEST: dict[str, dict[str, str]] = {
    "temperature_2m_c": {
        "source": "temperature_2m",
        "formula": "T_C = T_K - 273.15",
        "source_unit": "K",
        "output_unit": "degC",
    },
    "dewpoint_temperature_2m_c": {
        "source": "dewpoint_temperature_2m",
        "formula": "Td_C = Td_K - 273.15",
        "source_unit": "K",
        "output_unit": "degC",
    },
    "surface_pressure_hpa": {
        "source": "surface_pressure",
        "formula": "p_hPa = p_Pa / 100",
        "source_unit": "Pa",
        "output_unit": "hPa",
    },
    "u_wind_10m_ms": {
        "source": "u_wind_10m",
        "formula": "identity",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "v_wind_10m_ms": {
        "source": "v_wind_10m",
        "formula": "identity",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "wind_speed_10m_ms": {
        "source": "u_wind_10m, v_wind_10m",
        "formula": "speed = sqrt(u^2 + v^2)",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "wind_direction_10m_deg": {
        "source": "u_wind_10m, v_wind_10m",
        "formula": "direction_from_deg = mod(degrees(atan2(-u, -v)), 360); calm winds set to 0 deg by convention",
        "source_unit": "m s-1",
        "output_unit": "degree",
    },
    "boundary_layer_height_m": {
        "source": "boundary_layer_height",
        "formula": "identity",
        "source_unit": "m",
        "output_unit": "m",
    },
    "total_cloud_cover_pct": {
        "source": "total_cloud_cover",
        "formula": "cloud_pct = cloud_fraction * 100",
        "source_unit": "0-1",
        "output_unit": "%",
    },
    "total_precipitation_mm": {
        "source": "total_precipitation",
        "formula": "tp_mm = tp_m * 1000",
        "source_unit": "m water equivalent per hour",
        "output_unit": "mm water equivalent per hour",
    },
    "surface_solar_radiation_downwards_wm2": {
        "source": "surface_solar_radiation_downwards",
        "formula": "ssrd_mean_Wm2 = ssrd_Jm2 / 3600",
        "source_unit": "J m-2 accumulated over the hour ending at validity time",
        "output_unit": "W m-2 mean over that hour",
    },
    "temperature_pressure_c": {
        "source": "temperature_pressure",
        "formula": "T_C = T_K - 273.15",
        "source_unit": "K",
        "output_unit": "degC",
    },
    "relative_humidity_pressure_pct": {
        "source": "relative_humidity_pressure",
        "formula": "identity",
        "source_unit": "%",
        "output_unit": "%",
    },
    "u_wind_pressure_ms": {
        "source": "u_wind_pressure",
        "formula": "identity",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "v_wind_pressure_ms": {
        "source": "v_wind_pressure",
        "formula": "identity",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "wind_speed_pressure_ms": {
        "source": "u_wind_pressure, v_wind_pressure",
        "formula": "speed = sqrt(u^2 + v^2)",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "wind_direction_pressure_deg": {
        "source": "u_wind_pressure, v_wind_pressure",
        "formula": "direction_from_deg = mod(degrees(atan2(-u, -v)), 360); calm winds set to 0 deg by convention",
        "source_unit": "m s-1",
        "output_unit": "degree",
    },
    "geopotential_height_pressure_m": {
        "source": "geopotential_pressure",
        "formula": "geopotential_height_m = geopotential_m2s2 / 9.80665",
        "source_unit": "m2 s-2",
        "output_unit": "m (geopotential metres)",
    },
    "pressure_level_above_ground_mask": {
        "source": "surface_pressure_hpa, pressure_level_hpa",
        "formula": "mask(t,p,lat,lon) = 1[surface_pressure_hpa(t,lat,lon) >= pressure_level_hpa(p)]",
        "source_unit": "hPa, hPa",
        "output_unit": "1",
    },
    "pressure_pair_850_925_above_ground_mask": {
        "source": "surface_pressure_hpa",
        "formula": "mask_pair(t,lat,lon) = 1[surface_pressure_hpa(t,lat,lon) >= 925]",
        "source_unit": "hPa",
        "output_unit": "1",
    },
    "delta_u_850_925_ms": {
        "source": "u_wind_pressure_ms",
        "formula": "delta_u_850_925 = u_850 - u_925",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "delta_v_850_925_ms": {
        "source": "v_wind_pressure_ms",
        "formula": "delta_v_850_925 = v_850 - v_925",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "delta_wind_speed_850_925_ms": {
        "source": "wind_speed_pressure_ms",
        "formula": "delta_speed_850_925 = speed_850 - speed_925",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "wind_shear_magnitude_850_925_ms": {
        "source": "u_wind_pressure_ms, v_wind_pressure_ms",
        "formula": "shear_mag_850_925 = sqrt((u_850-u_925)^2 + (v_850-v_925)^2)",
        "source_unit": "m s-1",
        "output_unit": "m s-1",
    },
    "delta_temperature_850_925_c": {
        "source": "temperature_pressure_c",
        "formula": "delta_T_850_925 = T_850 - T_925",
        "source_unit": "degC",
        "output_unit": "degC",
    },
    "delta_geopotential_height_850_925_m": {
        "source": "geopotential_height_pressure_m",
        "formula": "delta_Z_850_925 = Z_850 - Z_925",
        "source_unit": "m",
        "output_unit": "m",
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def zarr_write_kwargs() -> dict[str, Any]:
    params = inspect.signature(xr.Dataset.to_zarr).parameters
    if "zarr_format" in params:
        return {"zarr_format": 2}
    if "zarr_version" in params:
        return {"zarr_version": 2}
    return {}


def _units(da: xr.DataArray) -> str:
    return str(da.attrs.get("units", ""))


def _assert_exact_canonical_schema(ds: xr.Dataset, region: str) -> None:
    got = set(ds.data_vars)
    if got != EXPECTED_CANONICAL_VARIABLES:
        raise RuntimeError(
            f"{region}: canonical variable set mismatch; "
            f"expected={sorted(EXPECTED_CANONICAL_VARIABLES)} "
            f"got={sorted(got)}"
        )

    for name, expected_unit in EXPECTED_CANONICAL_UNITS.items():
        got_unit = _units(ds[name])
        if got_unit != expected_unit:
            raise RuntimeError(
                f"{region}: canonical unit mismatch for {name}; "
                f"got={got_unit!r} expected={expected_unit!r}"
            )

    required_dims = {"time", "latitude", "longitude", "pressure_level_hpa"}
    missing_dims = required_dims - set(ds.dims)
    if missing_dims:
        raise RuntimeError(
            f"{region}: canonical dimensions missing: {sorted(missing_dims)}"
        )

    levels = [int(x) for x in ds.pressure_level_hpa.values.tolist()]
    if levels != PRESSURE_LEVELS_HPA:
        raise RuntimeError(
            f"{region}: pressure levels/order mismatch; "
            f"got={levels}, expected={PRESSURE_LEVELS_HPA}"
        )

    times = pd.DatetimeIndex(ds.time.values)
    if len(times) != EXPECTED_FULL_HOURS:
        raise RuntimeError(
            f"{region}: expected {EXPECTED_FULL_HOURS:,} canonical hours, "
            f"got {len(times):,}"
        )
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise RuntimeError(
            f"{region}: canonical time axis is duplicated or non-monotonic"
        )

    expected = pd.date_range(
        "2022-12-01 00:00:00",
        "2025-12-31 23:00:00",
        freq="h",
    )
    if not times.equals(expected):
        missing = expected.difference(times)[:10].tolist()
        extra = times.difference(expected)[:10].tolist()
        raise RuntimeError(
            f"{region}: canonical hourly timeline mismatch; "
            f"missing_head={missing}, extra_head={extra}"
        )

    # Pressure variables must carry the pressure dimension.
    for name in (
        "temperature_pressure",
        "relative_humidity_pressure",
        "u_wind_pressure",
        "v_wind_pressure",
        "geopotential_pressure",
    ):
        if "pressure_level_hpa" not in ds[name].dims:
            raise RuntimeError(
                f"{region}: {name} lacks pressure_level_hpa dimension"
            )

    # Single-level variables must not accidentally carry pressure.
    for name in EXPECTED_CANONICAL_VARIABLES - {
        "temperature_pressure",
        "relative_humidity_pressure",
        "u_wind_pressure",
        "v_wind_pressure",
        "geopotential_pressure",
    }:
        if "pressure_level_hpa" in ds[name].dims:
            raise RuntimeError(
                f"{region}: single-level {name} unexpectedly has pressure dimension"
            )


def _assert_canonical_build_report(root: Path, region: str) -> dict[str, Any]:
    path = (
        root / "03_canonical" / "era5"
        / "_build_reports" / f"{region}.json"
    )
    if not path.exists():
        raise RuntimeError(f"{region}: canonical build report missing: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(
            f"{region}: canonical report not PASS: {path}"
        )
    if report.get("months") != 37:
        raise RuntimeError(
            f"{region}: canonical report months={report.get('months')} != 37"
        )
    if report.get("hours") != EXPECTED_FULL_HOURS:
        raise RuntimeError(
            f"{region}: canonical report hours={report.get('hours')} "
            f"!= {EXPECTED_FULL_HOURS}"
        )
    if report.get("is_full_37_month_archive") is not True:
        raise RuntimeError(
            f"{region}: canonical report is_full_37_month_archive != true"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "build_version": report.get("build_version"),
        "status": report.get("status"),
        "months": report.get("months"),
        "hours": report.get("hours"),
    }


# -----------------------------------------------------------------------------
# Explicit physical transformation functions.
# These are intentionally simple and audit-friendly.
# -----------------------------------------------------------------------------


def kelvin_to_celsius(x: xr.DataArray) -> xr.DataArray:
    """Exact offset conversion: degC = K - 273.15."""
    return x - KELVIN_OFFSET


def pa_to_hpa(x: xr.DataArray) -> xr.DataArray:
    """Pressure conversion: hPa = Pa / 100."""
    return x / PA_PER_HPA


def metres_water_to_mm(x: xr.DataArray) -> xr.DataArray:
    """Water-equivalent depth conversion: mm = m * 1000."""
    return x * MM_PER_M


def hourly_energy_to_mean_flux_wm2(x: xr.DataArray) -> xr.DataArray:
    """
    ERA5 SSRD is J m-2 accumulated over the one-hour processing interval
    ending at validity time. Dividing by 3600 s gives the mean W m-2
    over that hour.
    """
    return x / SECONDS_PER_HOUR


def fraction_to_percent(x: xr.DataArray) -> xr.DataArray:
    """Percentage conversion: percent = fraction * 100."""
    return x * PERCENT_PER_FRACTION


def geopotential_to_height_m(x: xr.DataArray) -> xr.DataArray:
    """
    ECMWF geopotential height:
        h = geopotential / g0
    where g0 = 9.80665 m s-2.
    """
    return x / STANDARD_GRAVITY_MS2


def wind_speed_ms(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Wind speed magnitude: sqrt(u^2 + v^2)."""
    return np.hypot(u, v)


def meteorological_wind_direction_deg(
    u: xr.DataArray,
    v: xr.DataArray,
) -> xr.DataArray:
    """
    Meteorological FROM direction, clockwise from true north:
        dir = mod(degrees(atan2(-u, -v)), 360)

    Cardinal checks:
      wind from north: u= 0, v=-V ->   0 deg
      wind from east : u=-V, v= 0 ->  90 deg
      wind from south: u= 0, v=+V -> 180 deg
      wind from west : u=+V, v= 0 -> 270 deg

    Direction is physically undefined at exactly calm wind. To avoid
    injecting NaN into an otherwise finite deterministic layer, this
    builder stores 0 deg when speed <= 1e-12 m/s and records that convention.
    Final models should prefer u/v or circular sin/cos if direction is used.
    """
    speed = wind_speed_ms(u, v)
    direction = np.mod(
        np.degrees(np.arctan2(-u, -v)),
        360.0,
    )

    # Numerical canonicalization:
    # mathematically the result is in [0, 360), but finite-precision
    # arithmetic may round a value infinitesimally below 360 to exactly
    # 360.0. Since 360 deg is physically identical to 0 deg, normalize
    # it to the unique canonical representation 0 deg.
    direction = xr.where(direction >= 360.0, 0.0, direction)

    return xr.where(speed <= CALM_WIND_EPS_MS, 0.0, direction)



def pressure_level_above_ground_mask(
    surface_pressure_hpa: xr.DataArray,
    pressure_level_hpa: xr.DataArray,
) -> xr.DataArray:
    """
    Deterministic pressure-level validity mask.

    A pressure surface is at or above the local model surface when:
        surface_pressure_hpa >= pressure_level_hpa

    Returns uint8 with dimensions normalized to:
        time x pressure_level_hpa x latitude x longitude

    1 = pressure level is at/above the local model surface
    0 = pressure level lies below the local model surface
    """
    mask = (
        surface_pressure_hpa >= pressure_level_hpa
    ).astype(np.uint8)

    return mask.transpose(
        "time",
        "pressure_level_hpa",
        "latitude",
        "longitude",
    )


def pressure_pair_850_925_above_ground_mask(
    surface_pressure_hpa: xr.DataArray,
) -> xr.DataArray:
    """
    Deterministic joint-validity mask for 850/925 hPa contrasts.

    Since 925 hPa is the higher pressure / lower-altitude surface,
    both 925 and 850 hPa are at or above ground iff:
        surface_pressure_hpa >= 925 hPa

    Returns uint8 with dimensions:
        time x latitude x longitude
    """
    return (surface_pressure_hpa >= 925.0).astype(np.uint8)


def _with_attrs(
    da: xr.DataArray,
    *,
    name: str,
    units: str,
    source: str,
    formula: str,
    note: str | None = None,
) -> xr.DataArray:
    out = da.rename(name)
    attrs = {
        "units": units,
        "source_variable": source,
        "formula": formula,
        "derived_version": DERIVED_VERSION,
        "transformation_class": "deterministic_physical",
        "data_dependent": "false",
    }
    if note:
        attrs["note"] = note
    out.attrs = attrs
    return out


def build_derived_chunk(block: xr.Dataset) -> xr.Dataset:
    # ---- Single-level conversions / identities ----
    t2m_c = _with_attrs(
        kelvin_to_celsius(block["temperature_2m"]),
        name="temperature_2m_c",
        units="degC",
        source="temperature_2m",
        formula="T_C = T_K - 273.15",
    )
    d2m_c = _with_attrs(
        kelvin_to_celsius(block["dewpoint_temperature_2m"]),
        name="dewpoint_temperature_2m_c",
        units="degC",
        source="dewpoint_temperature_2m",
        formula="Td_C = Td_K - 273.15",
    )
    sp_hpa = _with_attrs(
        pa_to_hpa(block["surface_pressure"]),
        name="surface_pressure_hpa",
        units="hPa",
        source="surface_pressure",
        formula="p_hPa = p_Pa / 100",
    )

    u10 = _with_attrs(
        block["u_wind_10m"],
        name="u_wind_10m_ms",
        units="m s-1",
        source="u_wind_10m",
        formula="identity",
    )
    v10 = _with_attrs(
        block["v_wind_10m"],
        name="v_wind_10m_ms",
        units="m s-1",
        source="v_wind_10m",
        formula="identity",
    )
    ws10 = _with_attrs(
        wind_speed_ms(block["u_wind_10m"], block["v_wind_10m"]),
        name="wind_speed_10m_ms",
        units="m s-1",
        source="u_wind_10m, v_wind_10m",
        formula="sqrt(u^2 + v^2)",
    )
    wd10 = _with_attrs(
        meteorological_wind_direction_deg(
            block["u_wind_10m"],
            block["v_wind_10m"],
        ),
        name="wind_direction_10m_deg",
        units="degree",
        source="u_wind_10m, v_wind_10m",
        formula="mod(degrees(atan2(-u, -v)), 360)",
        note="meteorological FROM direction; calm wind <=1e-12 m/s stored as 0 deg by convention",
    )

    blh = _with_attrs(
        block["boundary_layer_height"],
        name="boundary_layer_height_m",
        units="m",
        source="boundary_layer_height",
        formula="identity",
    )
    tcc = _with_attrs(
        fraction_to_percent(block["total_cloud_cover"]),
        name="total_cloud_cover_pct",
        units="%",
        source="total_cloud_cover",
        formula="cloud_pct = cloud_fraction * 100",
    )
    tp = _with_attrs(
        metres_water_to_mm(block["total_precipitation"]),
        name="total_precipitation_mm",
        units="mm",
        source="total_precipitation",
        formula="tp_mm = tp_m * 1000",
        note="hourly accumulation over the hour ending at validity time; no daily aggregation",
    )
    ssrd = _with_attrs(
        hourly_energy_to_mean_flux_wm2(
            block["surface_solar_radiation_downwards"]
        ),
        name="surface_solar_radiation_downwards_wm2",
        units="W m-2",
        source="surface_solar_radiation_downwards",
        formula="ssrd_mean_Wm2 = ssrd_Jm2 / 3600",
        note="mean flux over the one-hour accumulation interval ending at validity time",
    )

    # ---- Pressure-level conversions / identities ----
    tp_c = _with_attrs(
        kelvin_to_celsius(block["temperature_pressure"]),
        name="temperature_pressure_c",
        units="degC",
        source="temperature_pressure",
        formula="T_C = T_K - 273.15",
    )
    rhp = _with_attrs(
        block["relative_humidity_pressure"],
        name="relative_humidity_pressure_pct",
        units="%",
        source="relative_humidity_pressure",
        formula="identity",
    )
    up = _with_attrs(
        block["u_wind_pressure"],
        name="u_wind_pressure_ms",
        units="m s-1",
        source="u_wind_pressure",
        formula="identity",
    )
    vp = _with_attrs(
        block["v_wind_pressure"],
        name="v_wind_pressure_ms",
        units="m s-1",
        source="v_wind_pressure",
        formula="identity",
    )
    wsp = _with_attrs(
        wind_speed_ms(
            block["u_wind_pressure"],
            block["v_wind_pressure"],
        ),
        name="wind_speed_pressure_ms",
        units="m s-1",
        source="u_wind_pressure, v_wind_pressure",
        formula="sqrt(u^2 + v^2)",
    )
    wdp = _with_attrs(
        meteorological_wind_direction_deg(
            block["u_wind_pressure"],
            block["v_wind_pressure"],
        ),
        name="wind_direction_pressure_deg",
        units="degree",
        source="u_wind_pressure, v_wind_pressure",
        formula="mod(degrees(atan2(-u, -v)), 360)",
        note="meteorological FROM direction; calm wind <=1e-12 m/s stored as 0 deg by convention",
    )
    ghp = _with_attrs(
        geopotential_to_height_m(block["geopotential_pressure"]),
        name="geopotential_height_pressure_m",
        units="m",
        source="geopotential_pressure",
        formula="geopotential_height = geopotential / 9.80665",
        note="geopotential metres; not geometric altitude",
    )

    # ---- Pressure-level physical-validity masks ----
    plev_mask = _with_attrs(
        pressure_level_above_ground_mask(
            sp_hpa,
            block["pressure_level_hpa"],
        ),
        name="pressure_level_above_ground_mask",
        units="1",
        source="surface_pressure_hpa, pressure_level_hpa",
        formula=(
            "1[surface_pressure_hpa >= pressure_level_hpa]"
        ),
        note=(
            "1 = pressure surface is at or above the local model surface; "
            "0 = pressure surface lies below the local model surface. "
            "Numerical pressure-level fields are retained regardless of mask."
        ),
    )
    pair_mask = _with_attrs(
        pressure_pair_850_925_above_ground_mask(sp_hpa),
        name="pressure_pair_850_925_above_ground_mask",
        units="1",
        source="surface_pressure_hpa",
        formula="1[surface_pressure_hpa >= 925 hPa]",
        note=(
            "Joint physical-validity mask for all deterministic 850-minus-925 "
            "contrasts; 1 means both 925 and 850 hPa are at or above the "
            "local model surface. Derived contrast values are retained "
            "regardless of mask."
        ),
    )

    # ---- 850 minus 925 deterministic layer contrasts ----
    u850 = up.sel(pressure_level_hpa=850, drop=True)
    u925 = up.sel(pressure_level_hpa=925, drop=True)
    v850 = vp.sel(pressure_level_hpa=850, drop=True)
    v925 = vp.sel(pressure_level_hpa=925, drop=True)
    s850 = wsp.sel(pressure_level_hpa=850, drop=True)
    s925 = wsp.sel(pressure_level_hpa=925, drop=True)
    t850 = tp_c.sel(pressure_level_hpa=850, drop=True)
    t925 = tp_c.sel(pressure_level_hpa=925, drop=True)
    z850 = ghp.sel(pressure_level_hpa=850, drop=True)
    z925 = ghp.sel(pressure_level_hpa=925, drop=True)

    du = _with_attrs(
        u850 - u925,
        name="delta_u_850_925_ms",
        units="m s-1",
        source="u_wind_pressure_ms",
        formula="u_850 - u_925",
    )
    dv = _with_attrs(
        v850 - v925,
        name="delta_v_850_925_ms",
        units="m s-1",
        source="v_wind_pressure_ms",
        formula="v_850 - v_925",
    )
    dspeed = _with_attrs(
        s850 - s925,
        name="delta_wind_speed_850_925_ms",
        units="m s-1",
        source="wind_speed_pressure_ms",
        formula="speed_850 - speed_925",
        note="signed scalar-speed contrast; distinct from vector shear magnitude",
    )
    shear = _with_attrs(
        np.hypot(u850 - u925, v850 - v925),
        name="wind_shear_magnitude_850_925_ms",
        units="m s-1",
        source="u_wind_pressure_ms, v_wind_pressure_ms",
        formula="sqrt((u_850-u_925)^2 + (v_850-v_925)^2)",
        note="vector wind difference magnitude between 850 and 925 hPa",
    )
    dt = _with_attrs(
        t850 - t925,
        name="delta_temperature_850_925_c",
        units="degC",
        source="temperature_pressure_c",
        formula="T_850 - T_925",
    )
    dz = _with_attrs(
        z850 - z925,
        name="delta_geopotential_height_850_925_m",
        units="m",
        source="geopotential_height_pressure_m",
        formula="Z_850 - Z_925",
    )

    out = xr.Dataset(
        {
            x.name: x
            for x in (
                t2m_c,
                d2m_c,
                sp_hpa,
                u10,
                v10,
                ws10,
                wd10,
                blh,
                tcc,
                tp,
                ssrd,
                tp_c,
                rhp,
                up,
                vp,
                wsp,
                wdp,
                ghp,
                plev_mask,
                pair_mask,
                du,
                dv,
                dspeed,
                shear,
                dt,
                dz,
            )
        }
    )

    # Preserve normalized canonical coordinates exactly.
    out.coords["time"].attrs.update(
        {
            "timezone": "UTC",
            "semantics": "ERA5 validity time; hourly",
        }
    )
    out.coords["latitude"].attrs["units"] = "degrees_north"
    out.coords["longitude"].attrs["units"] = "degrees_east"
    out.coords["pressure_level_hpa"].attrs.update(
        {"units": "hPa", "positive": "down"}
    )

    out.attrs.update(
        {
            "derived_version": DERIVED_VERSION,
            "layer": "deterministic physical-ready ERA5",
            "time_resolution": "1 hour",
            "time_timezone": "UTC",
            "spatial_policy": "full regional ERA5 grid retained; no city extraction or spatial interpolation",
            "transformation_policy": "deterministic physical conversions and same-timestamp algebra only",
            "data_dependent_transformations": "none",
            "daily_aggregation": "none",
            "imputation": "none",
            "scaling": "none",
            "clipping": "none",
            "feature_selection": "none",
            "pressure_level_below_ground_policy": (
                "retain numerical pressure-level values; expose deterministic "
                "above-ground masks; do not replace, NaN, clip, or substitute "
                "10 m variables"
            ),
        }
    )

    return out


def assert_chunk_finite(ds: xr.Dataset, label: str) -> None:
    for name, da in ds.data_vars.items():
        a = np.asarray(da.values)
        if a.size == 0:
            raise RuntimeError(f"{label}: {name} is empty")
        finite = np.isfinite(a)
        if not finite.all():
            bad = int(a.size - finite.sum())
            raise RuntimeError(
                f"{label}: {name} contains {bad:,}/{a.size:,} "
                f"non-finite derived values"
            )


def update_ranges(
    ranges: dict[str, dict[str, float]],
    ds: xr.Dataset,
) -> None:
    for name, da in ds.data_vars.items():
        a = np.asarray(da.values)
        mn = float(np.min(a))
        mx = float(np.max(a))
        if name not in ranges:
            ranges[name] = {"min": mn, "max": mx}
        else:
            ranges[name]["min"] = min(ranges[name]["min"], mn)
            ranges[name]["max"] = max(ranges[name]["max"], mx)


def first_write_encoding(
    ds: xr.Dataset,
    chunk_hours: int,
) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        chunks = []
        for d in da.dims:
            if d == "time":
                chunks.append(min(chunk_hours, da.sizes[d]))
            else:
                chunks.append(da.sizes[d])
        encoding[name] = {"chunks": tuple(chunks)}
    encoding["time"] = {
        "chunks": (min(chunk_hours, ds.sizes["time"]),)
    }
    return encoding


def append_to_zarr(
    ds: xr.Dataset,
    store: Path,
    *,
    first: bool,
    chunk_hours: int,
) -> None:
    kwargs = {
        "store": str(store),
        "consolidated": False,
        **zarr_write_kwargs(),
    }
    if first:
        ds.to_zarr(
            mode="w",
            encoding=first_write_encoding(ds, chunk_hours),
            **kwargs,
        )
    else:
        ds.to_zarr(
            mode="a",
            append_dim="time",
            **kwargs,
        )


def verify_resume_state(
    store: Path,
    state: dict[str, Any],
) -> None:
    if not store.exists():
        raise RuntimeError(
            f"State exists but derived Zarr store is missing: {store}"
        )
    ds = xr.open_zarr(str(store), consolidated=False, chunks=None)
    try:
        actual = int(ds.sizes.get("time", -1))
        expected = int(state.get("hours_written", -1))
        if actual != expected:
            raise RuntimeError(
                f"Resume refused: store hours={actual}, "
                f"state hours_written={expected}. "
                f"Previous append may have been interrupted; use --overwrite."
            )
        if actual > 0:
            times = pd.DatetimeIndex(ds.time.values)
            if times.has_duplicates or not times.is_monotonic_increasing:
                raise RuntimeError(
                    "Resume refused: derived time axis is invalid"
                )
            expected_last = pd.Timestamp(
                state["last_time_utc"]
            )
            if pd.Timestamp(times[-1]) != expected_last:
                raise RuntimeError(
                    f"Resume refused: last store time={times[-1]} "
                    f"!= state last_time_utc={expected_last}"
                )
    finally:
        ds.close()


def final_gate(
    canonical: xr.Dataset,
    store: Path,
    region: str,
    ranges: dict[str, dict[str, float]],
) -> dict[str, Any]:
    derived = xr.open_zarr(str(store), consolidated=False, chunks=None)
    try:
        if set(derived.data_vars) != DERIVED_VARIABLES:
            raise RuntimeError(
                f"{region}: final derived variable set mismatch; "
                f"expected={sorted(DERIVED_VARIABLES)} "
                f"got={sorted(derived.data_vars)}"
            )

        for coord in ("time", "latitude", "longitude", "pressure_level_hpa"):
            a = np.asarray(canonical.coords[coord].values)
            b = np.asarray(derived.coords[coord].values)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise RuntimeError(
                    f"{region}: final coordinate mismatch for {coord}"
                )

        if int(derived.sizes["time"]) != EXPECTED_FULL_HOURS:
            raise RuntimeError(
                f"{region}: final derived hours="
                f"{derived.sizes['time']} != {EXPECTED_FULL_HOURS}"
            )

        # Audit formula-semantic invariants from the chunk-wise range
        # accumulator, avoiding an unnecessary full reread of the large Zarr.
        wd10_range = ranges["wind_direction_10m_deg"]
        wd_min = float(wd10_range["min"])
        wd_max = float(wd10_range["max"])
        if wd_min < 0.0 or wd_max >= 360.0:
            raise RuntimeError(
                f"{region}: 10m wind direction outside [0,360): "
                f"{wd_min}..{wd_max}"
            )

        wdp_range = ranges["wind_direction_pressure_deg"]
        wdp_min = float(wdp_range["min"])
        wdp_max = float(wdp_range["max"])
        if wdp_min < 0.0 or wdp_max >= 360.0:
            raise RuntimeError(
                f"{region}: pressure wind direction outside [0,360): "
                f"{wdp_min}..{wdp_max}"
            )

        # Mask schema and exact semantics.
        plev_mask = derived["pressure_level_above_ground_mask"]
        pair_mask = derived["pressure_pair_850_925_above_ground_mask"]

        expected_plev_dims = (
            "time",
            "pressure_level_hpa",
            "latitude",
            "longitude",
        )
        expected_pair_dims = ("time", "latitude", "longitude")

        if tuple(plev_mask.dims) != expected_plev_dims:
            raise RuntimeError(
                f"{region}: pressure_level_above_ground_mask dims="
                f"{plev_mask.dims}, expected={expected_plev_dims}"
            )
        if tuple(pair_mask.dims) != expected_pair_dims:
            raise RuntimeError(
                f"{region}: pressure_pair_850_925_above_ground_mask dims="
                f"{pair_mask.dims}, expected={expected_pair_dims}"
            )

        plev_mask_range = ranges["pressure_level_above_ground_mask"]
        pair_mask_range = ranges["pressure_pair_850_925_above_ground_mask"]

        for name, rr in (
            ("pressure_level_above_ground_mask", plev_mask_range),
            ("pressure_pair_850_925_above_ground_mask", pair_mask_range),
        ):
            if rr["min"] < 0.0 or rr["max"] > 1.0:
                raise RuntimeError(
                    f"{region}: {name} outside binary range [0,1]: {rr}"
                )

        # The 850/925 pair mask must be exactly identical to the 925-hPa
        # slice of the general pressure-level mask.
        p925 = plev_mask.sel(pressure_level_hpa=925, drop=True)
        if not np.array_equal(
            np.asarray(p925.values),
            np.asarray(pair_mask.values),
        ):
            raise RuntimeError(
                f"{region}: pair mask is not exactly equal to the "
                f"925-hPa slice of pressure-level mask"
            )

        # Monotonic pressure-level validity for our two levels:
        # if 925 hPa is above ground, 850 hPa must also be above ground.
        p850 = plev_mask.sel(pressure_level_hpa=850, drop=True)
        violation = (
            np.asarray(p925.values, dtype=np.uint8)
            > np.asarray(p850.values, dtype=np.uint8)
        )
        if violation.any():
            raise RuntimeError(
                f"{region}: impossible mask state found: "
                f"925 hPa valid while 850 hPa invalid"
            )

        return {
            "status": "PASS",
            "region": region,
            "store": str(store),
            "hours": int(derived.sizes["time"]),
            "time_start_utc": str(pd.Timestamp(derived.time.values[0])),
            "time_end_utc": str(pd.Timestamp(derived.time.values[-1])),
            "grid": {
                "latitude": int(derived.sizes["latitude"]),
                "longitude": int(derived.sizes["longitude"]),
                "grid_cells": int(
                    derived.sizes["latitude"]
                    * derived.sizes["longitude"]
                ),
            },
            "pressure_levels_hpa": [
                int(x)
                for x in derived.pressure_level_hpa.values.tolist()
            ],
            "variables": sorted(derived.data_vars),
            "ranges": ranges,
        }
    finally:
        derived.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build deterministic physical-ready ERA5 derived regional "
            "Zarr stores from the audited canonical layer."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )
    ap.add_argument(
        "--region",
        action="append",
        default=[],
        help="Repeat to limit regions; default builds all canonical regions.",
    )
    ap.add_argument(
        "--chunk-hours",
        type=int,
        default=DEFAULT_CHUNK_HOURS,
        help="In-memory processing chunk size; default 168 h.",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Per-region development/smoke-test limit; safe to resume later.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete selected region derived output/state and rebuild.",
    )
    args = ap.parse_args()

    if args.chunk_hours < 1:
        raise RuntimeError("--chunk-hours must be >= 1")
    if args.max_chunks is not None and args.max_chunks < 1:
        raise RuntimeError("--max-chunks must be >= 1")

    root = args.root.expanduser().resolve()
    canonical_root = root / "03_canonical" / "era5"
    output_root = root / "04_derived" / "era5_physical"
    state_root = output_root / "_build_state"
    reports_root = output_root / "_build_reports"
    metadata_root = output_root / "_metadata"

    for p in (output_root, state_root, reports_root, metadata_root):
        p.mkdir(parents=True, exist_ok=True)

    # Formula manifest is itself a reproducibility artifact.
    formula_payload = {
        "derived_version": DERIVED_VERSION,
        "constants": {
            "kelvin_offset": KELVIN_OFFSET,
            "pa_per_hpa": PA_PER_HPA,
            "mm_per_m": MM_PER_M,
            "seconds_per_hour": SECONDS_PER_HOUR,
            "percent_per_fraction": PERCENT_PER_FRACTION,
            "standard_gravity_ms2": STANDARD_GRAVITY_MS2,
            "calm_wind_epsilon_ms": CALM_WIND_EPS_MS,
        },
        "policy": {
            "data_dependent_transformations": "none",
            "spatial_interpolation": "none",
            "daily_aggregation": "none",
            "time_resolution": "hourly",
            "timezone": "UTC",
            "pressure_contrast_sign": "850 minus 925 hPa",
            "pressure_level_validity_rule": (
                "surface_pressure_hpa >= pressure_level_hpa"
            ),
            "below_ground_policy": (
                "retain numerical values and expose masks; no NaN/removal/"
                "clipping/substitution in the derived layer"
            ),
        },
        "variables": FORMULA_MANIFEST,
    }
    formula_path = metadata_root / "formula_manifest.json"
    atomic_json(formula_payload, formula_path)

    regions_all = sorted(
        p.stem
        for p in canonical_root.glob("*.zarr")
        if p.is_dir()
    )
    if len(regions_all) != 5:
        raise RuntimeError(
            f"Expected 5 canonical region stores, found "
            f"{len(regions_all)}: {regions_all}"
        )

    selected = sorted(set(args.region)) if args.region else regions_all
    unknown = sorted(set(selected) - set(regions_all))
    if unknown:
        raise RuntimeError(f"Unknown regions: {unknown}")

    run_report: dict[str, Any] = {
        "status": "RUNNING",
        "derived_version": DERIVED_VERSION,
        "started_at_utc": now_utc(),
        "root": str(root),
        "formula_manifest": str(formula_path),
        "formula_manifest_sha256": sha256_file(formula_path),
        "chunk_hours": args.chunk_hours,
        "selected_regions": selected,
        "regions": {},
    }

    for region in selected:
        print("\n" + "=" * 110)
        print(f"REGION: {region}")
        print("=" * 110, flush=True)

        canonical_path = canonical_root / f"{region}.zarr"
        store = output_root / f"{region}.zarr"
        state_path = state_root / f"{region}.json"
        report_path = reports_root / f"{region}.json"

        canonical_report = _assert_canonical_build_report(root, region)

        canonical = xr.open_zarr(
            str(canonical_path),
            consolidated=True,
            chunks=None,
        )
        try:
            _assert_exact_canonical_schema(canonical, region)

            if args.overwrite:
                if store.exists():
                    shutil.rmtree(store)
                state_path.unlink(missing_ok=True)
                report_path.unlink(missing_ok=True)

            state = None
            if state_path.exists():
                state = json.loads(
                    state_path.read_text(encoding="utf-8")
                )

            if store.exists() != (state is not None):
                raise RuntimeError(
                    f"{region}: derived store/state existence mismatch; "
                    f"use --overwrite to rebuild safely"
                )

            if state is None:
                state = {
                    "derived_version": DERIVED_VERSION,
                    "region": region,
                    "canonical_path": str(canonical_path),
                    "canonical_report": canonical_report,
                    "formula_manifest_sha256": sha256_file(formula_path),
                    "hours_written": 0,
                    "last_time_utc": None,
                    "ranges": {},
                    "created_at_utc": now_utc(),
                }
            else:
                if state.get("derived_version") != DERIVED_VERSION:
                    raise RuntimeError(
                        f"{region}: derived version changed; use --overwrite"
                    )
                if (
                    state.get("formula_manifest_sha256")
                    != sha256_file(formula_path)
                ):
                    raise RuntimeError(
                        f"{region}: formula manifest changed; use --overwrite"
                    )
                verify_resume_state(store, state)

            start = int(state["hours_written"])
            total = int(canonical.sizes["time"])
            pending_slices: list[tuple[int, int]] = []
            cursor = start
            while cursor < total:
                end = min(cursor + args.chunk_hours, total)
                pending_slices.append((cursor, end))
                cursor = end

            if args.max_chunks is not None:
                pending_slices = pending_slices[: args.max_chunks]

            if not pending_slices:
                print(f"[NO PENDING HOURS] {region}", flush=True)

            for idx, (a, b) in enumerate(pending_slices, start=1):
                t0 = pd.Timestamp(canonical.time.values[a])
                t1 = pd.Timestamp(canonical.time.values[b - 1])
                print(
                    f"[{idx}/{len(pending_slices)}] "
                    f"DERIVE {region} hours[{a}:{b}] "
                    f"{t0} -> {t1}",
                    flush=True,
                )

                block = canonical.isel(time=slice(a, b)).load()
                derived = build_derived_chunk(block)
                assert_chunk_finite(
                    derived,
                    f"{region} hours[{a}:{b}]",
                )

                ranges = state.get("ranges", {})
                update_ranges(ranges, derived)

                first = not store.exists()
                append_to_zarr(
                    derived,
                    store,
                    first=first,
                    chunk_hours=args.chunk_hours,
                )

                written = b
                state["hours_written"] = written
                state["last_time_utc"] = str(
                    pd.Timestamp(canonical.time.values[b - 1])
                )
                state["ranges"] = ranges
                state["updated_at_utc"] = now_utc()
                atomic_json(state, state_path)

                derived.close()
                block.close()

                print(
                    f"[PASS CHUNK] {region} "
                    f"written={written:,}/{total:,}",
                    flush=True,
                )

            complete = int(state["hours_written"]) == total
            if complete:
                region_report = final_gate(
                    canonical=canonical,
                    store=store,
                    region=region,
                    ranges=state["ranges"],
                )
                region_report.update(
                    {
                        "derived_version": DERIVED_VERSION,
                        "canonical_source": str(canonical_path),
                        "canonical_report": canonical_report,
                        "formula_manifest": str(formula_path),
                        "formula_manifest_sha256": sha256_file(
                            formula_path
                        ),
                        "finished_at_utc": now_utc(),
                        "is_full_archive": True,
                    }
                )
                atomic_json(region_report, report_path)

                try:
                    zarr.consolidate_metadata(str(store))
                    print(
                        f"[METADATA CONSOLIDATED] {store}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] metadata consolidation skipped: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                run_report["regions"][region] = region_report
                print(
                    f"[REGION PASS] {region}: "
                    f"hours={region_report['hours']:,} "
                    f"variables={len(region_report['variables'])}",
                    flush=True,
                )
            else:
                partial = {
                    "status": "PARTIAL_PASS",
                    "region": region,
                    "store": str(store),
                    "hours_written": int(state["hours_written"]),
                    "hours_total": total,
                    "last_time_utc": state["last_time_utc"],
                    "derived_version": DERIVED_VERSION,
                    "is_full_archive": False,
                }
                atomic_json(partial, report_path)
                run_report["regions"][region] = partial
                print(
                    f"[REGION PARTIAL PASS] {region}: "
                    f"{state['hours_written']:,}/{total:,} h",
                    flush=True,
                )

        finally:
            canonical.close()

    full = (
        set(run_report["regions"]) == set(regions_all)
        and all(
            r.get("is_full_archive") is True
            for r in run_report["regions"].values()
        )
    )
    run_report["status"] = "PASS" if full else "PARTIAL_PASS"
    run_report["finished_at_utc"] = now_utc()

    run_dir = root / "08_runs" / "era5_derived"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / (
        "build_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    atomic_json(run_report, run_path)

    print("\n" + "=" * 110)
    print("ERA5 PHYSICAL DERIVED BUILD:", run_report["status"])
    print("Formula manifest:", formula_path)
    print("Run report:", run_path)
    print("=" * 110)


if __name__ == "__main__":
    main()
