#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import inspect
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message=r"In a future version of xarray the default value for compat will change.*",
    category=FutureWarning,
)

try:
    import cfgrib  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Missing cfgrib. Activate the era5_data environment first."
    ) from exc

try:
    import zarr  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Missing zarr. Install it in era5_data, e.g. "
        "`conda install -c conda-forge zarr numcodecs`."
    ) from exc


BUILD_VERSION = "era5_canonical_v3"
GRID_DEG = 0.25
STREAMS = ("single_instant", "single_accum", "pressure_925_850")
PRESSURE_LEVEL_ORDER_HPA = [925, 850]
TIME_CHUNK_HOURS = 168

STREAM_DIRS = {
    "single_instant": "01_raw/era5_single_levels",
    "single_accum": "01_raw/era5_single_levels_accum",
    "pressure_925_850": "01_raw/era5_pressure_levels",
}

EXPECTED_SHORTNAMES = {
    "single_instant": {"2t", "2d", "sp", "10u", "10v", "blh", "tcc"},
    "single_accum": {"tp", "ssrd"},
    "pressure_925_850": {"t", "r", "u", "v", "z"},
}

# cfgrib may expose a human-readable xarray variable name while preserving
# the GRIB shortName in attrs. These fallbacks make the parser tolerant to
# common cfgrib naming conventions.
NAME_TO_SHORTNAME_FALLBACK = {
    "t2m": "2t",
    "d2m": "2d",
    "sp": "sp",
    "u10": "10u",
    "v10": "10v",
    "blh": "blh",
    "tcc": "tcc",
    "tp": "tp",
    "ssrd": "ssrd",
    "t": "t",
    "r": "r",
    "u": "u",
    "v": "v",
    "z": "z",
}

CANONICAL_NAMES = {
    "2t": "temperature_2m",
    "2d": "dewpoint_temperature_2m",
    "sp": "surface_pressure",
    "10u": "u_wind_10m",
    "10v": "v_wind_10m",
    "blh": "boundary_layer_height",
    "tcc": "total_cloud_cover",
    "tp": "total_precipitation",
    "ssrd": "surface_solar_radiation_downwards",
    "t": "temperature_pressure",
    "r": "relative_humidity_pressure",
    "u": "u_wind_pressure",
    "v": "v_wind_pressure",
    "z": "geopotential_pressure",
}

EXPECTED_UNITS = {
    "2t": "K",
    "2d": "K",
    "sp": "Pa",
    "10u": "m s**-1",
    "10v": "m s**-1",
    "blh": "m",
    "tcc": "(0 - 1)",
    "tp": "m",
    "ssrd": "J m**-2",
    "t": "K",
    "r": "%",
    "u": "m s**-1",
    "v": "m s**-1",
    "z": "m**2 s**-2",
}


@dataclass(frozen=True)
class Job:
    region: str
    scope: str
    year: int
    month: int
    stream: str
    expected_hours: int


@dataclass(frozen=True)
class Bounds:
    region: str
    north: float
    west: float
    south: float
    east: float
    n_lat: int
    n_lon: int
    grid_cells: int


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def expected_month_times(year: int, month: int) -> pd.DatetimeIndex:
    hours = calendar.monthrange(year, month)[1] * 24
    return pd.date_range(
        start=f"{year:04d}-{month:02d}-01 00:00:00",
        periods=hours,
        freq="h",
    )


def load_jobs(path: Path) -> list[Job]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"region", "scope", "year", "month", "stream", "expected_hours"}
    if not rows:
        raise ValueError(f"Empty jobs manifest: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Jobs manifest missing columns: {sorted(missing)}")

    jobs: list[Job] = []
    for i, r in enumerate(rows, start=2):
        try:
            j = Job(
                region=r["region"].strip(),
                scope=r["scope"].strip(),
                year=int(r["year"]),
                month=int(r["month"]),
                stream=r["stream"].strip(),
                expected_hours=int(r["expected_hours"]),
            )
        except Exception as exc:
            raise ValueError(f"Bad jobs row at CSV line {i}: {r}") from exc
        if j.stream not in STREAMS:
            raise ValueError(f"Unexpected stream at line {i}: {j.stream}")
        expected = calendar.monthrange(j.year, j.month)[1] * 24
        if j.expected_hours != expected:
            raise ValueError(
                f"{j.region} {month_key(j.year, j.month)} {j.stream}: "
                f"expected_hours={j.expected_hours}, calendar says {expected}"
            )
        jobs.append(j)
    return jobs


def load_bounds(path: Path) -> dict[str, Bounds]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "region", "north", "west", "south", "east",
        "n_lat", "n_lon", "grid_cells",
    }
    if not rows:
        raise ValueError(f"Empty bounds manifest: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Bounds manifest missing columns: {sorted(missing)}")

    out: dict[str, Bounds] = {}
    for r in rows:
        b = Bounds(
            region=r["region"].strip(),
            north=float(r["north"]),
            west=float(r["west"]),
            south=float(r["south"]),
            east=float(r["east"]),
            n_lat=int(r["n_lat"]),
            n_lon=int(r["n_lon"]),
            grid_cells=int(r["grid_cells"]),
        )
        if b.n_lat * b.n_lon != b.grid_cells:
            raise ValueError(f"Bad grid cell count for {b.region}: {b}")
        out[b.region] = b
    return out


def raw_path(root: Path, j: Job) -> Path:
    tag = f"{j.region}_{j.year:04d}-{j.month:02d}"
    return root / STREAM_DIRS[j.stream] / f"{tag}_{j.stream}.grib"


def audit_path(root: Path, j: Job) -> Path:
    tag = f"{j.region}_{j.year:04d}-{j.month:02d}"
    return (
        root / "02_audit" / "era5_bulk" / j.region
        / f"{tag}_{j.stream}_audit.json"
    )


def verify_raw_against_audit(
    root: Path,
    j: Job,
    verify_sha: bool,
) -> dict[str, Any]:
    raw = raw_path(root, j)
    audit = audit_path(root, j)

    if not raw.exists():
        raise RuntimeError(f"RAW MISSING: {raw}")
    if not audit.exists():
        raise RuntimeError(f"AUDIT MISSING: {audit}")

    report = json.loads(audit.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(
            f"RAW AUDIT NOT PASS: {audit}; status={report.get('status')!r}"
        )

    if int(report.get("expected_hours", -1)) != j.expected_hours:
        raise RuntimeError(
            f"RAW AUDIT HOUR MISMATCH: {audit}; "
            f"audit={report.get('expected_hours')} manifest={j.expected_hours}"
        )

    if report.get("stream") != j.stream:
        raise RuntimeError(
            f"RAW AUDIT STREAM MISMATCH: {audit}; "
            f"audit={report.get('stream')!r} manifest={j.stream!r}"
        )

    if verify_sha:
        expected_sha = report.get("sha256")
        if not expected_sha:
            raise RuntimeError(f"RAW AUDIT HAS NO SHA256: {audit}")
        actual_sha = sha256_file(raw)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"RAW SHA256 MISMATCH: {raw}\n"
                f"  audit : {expected_sha}\n"
                f"  actual: {actual_sha}"
            )

    return report


def _shortname(name: str, da: xr.DataArray) -> str | None:
    short = da.attrs.get("GRIB_shortName")
    if short is not None:
        return str(short)
    return NAME_TO_SHORTNAME_FALLBACK.get(name)


def _get_units(da: xr.DataArray) -> str:
    return str(
        da.attrs.get(
            "GRIB_units",
            da.attrs.get("units", ""),
        )
    )


def _pressure_dim(da: xr.DataArray) -> str | None:
    for d in da.dims:
        if d in {"isobaricInhPa", "isobaricInPa"}:
            return d
    for c in da.coords:
        if c in {"isobaricInhPa", "isobaricInPa"}:
            return c
    return None


def _extract_time_values(da: xr.DataArray) -> np.ndarray:
    # ERA5 accumulated variables are defined by validity time. For instantaneous
    # variables valid_time equals time, so using valid_time consistently avoids
    # semantic drift between streams.
    if "valid_time" in da.coords:
        vt = da.coords["valid_time"]
        if vt.ndim == 1 and "time" in da.dims and vt.size == da.sizes["time"]:
            return np.asarray(vt.values, dtype="datetime64[ns]")
        if vt.ndim == 1 and "valid_time" in da.dims:
            return np.asarray(vt.values, dtype="datetime64[ns]")

    if "time" in da.coords and "time" in da.dims:
        return np.asarray(da.coords["time"].values, dtype="datetime64[ns]")

    raise RuntimeError(
        f"Cannot determine 1-D validity time for variable "
        f"{da.name!r}; dims={da.dims}, coords={list(da.coords)}"
    )


def _decode_accumulated_time_step(
    work: xr.DataArray,
    stream: str,
    short: str,
    year: int,
    month: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Decode ERA5 accumulated single-level fields.

    cfgrib represents ERA5 forecast accumulations as:
        time x step x latitude x longitude

    For ERA5 HRES, forecast initializations are normally 06/18 UTC and
    steps 1..12 h provide the hourly accumulated fields.  The physically
    meaningful canonical timestamp is GRIB validity time (time + step).

    Values are NOT differenced here: ERA5 hourly accumulated fields are
    already defined over the interval from the previous hourly step to
    the current validity time.
    """
    if "time" not in work.dims or "step" not in work.dims:
        raise RuntimeError(
            f"{stream} {short}: accumulated field must expose time and step; "
            f"dims={work.dims}"
        )
    if "valid_time" not in work.coords:
        raise RuntimeError(
            f"{stream} {short}: accumulated field has no valid_time coordinate; "
            f"coords={list(work.coords)}"
        )

    vt = work.coords["valid_time"]
    if set(vt.dims) != {"time", "step"} or vt.ndim != 2:
        raise RuntimeError(
            f"{stream} {short}: expected 2-D valid_time(time,step); "
            f"valid_time.dims={vt.dims} shape={vt.shape}"
        )

    # Semantic gate: ERA5 hourly accumulated retrieval should be steps 1..12 h.
    step_values = np.asarray(work.coords["step"].values)
    try:
        step_hours = (
            step_values.astype("timedelta64[s]").astype(np.int64) / 3600.0
        )
    except Exception as exc:
        raise RuntimeError(
            f"{stream} {short}: cannot interpret step coordinate as hours: "
            f"{step_values!r}"
        ) from exc

    rounded = np.rint(step_hours).astype(int)
    if not np.allclose(step_hours, rounded, atol=1e-9):
        raise RuntimeError(
            f"{stream} {short}: non-integral forecast steps in hours: "
            f"{step_hours.tolist()}"
        )
    expected_steps = list(range(1, 13))
    if sorted(rounded.tolist()) != expected_steps:
        raise RuntimeError(
            f"{stream} {short}: unexpected forecast steps; "
            f"got={sorted(rounded.tolist())}, expected={expected_steps}"
        )

    # cfgrib should give 06/18 UTC initialization times for these ERA5
    # accumulated fields. Keep this as a hard semantic gate.
    init_times = pd.DatetimeIndex(
        np.asarray(work.coords["time"].values, dtype="datetime64[ns]")
    )
    init_hours = sorted(set(int(h) for h in init_times.hour.tolist()))
    if not set(init_hours).issubset({6, 18}):
        raise RuntimeError(
            f"{stream} {short}: unexpected ERA5 forecast initialization hours "
            f"{init_hours}; expected subset of [6, 18]"
        )

    values = np.asarray(
        work.transpose("time", "step", "latitude", "longitude").values
    )
    flat_values = values.reshape(
        values.shape[0] * values.shape[1],
        values.shape[2],
        values.shape[3],
    )
    flat_times = np.asarray(vt.values, dtype="datetime64[ns]").reshape(-1)

    if len(flat_times) != flat_values.shape[0]:
        raise RuntimeError(
            f"{stream} {short}: flattened validity/value length mismatch "
            f"{len(flat_times)} vs {flat_values.shape[0]}"
        )

    # cfgrib can represent a sparse GRIB time×step collection as a dense
    # xarray hypercube. At monthly boundaries this creates structurally absent
    # time/step combinations whose entire latitude×longitude field is NaN.
    #
    # Drop ONLY completely empty spatial fields. A partially non-finite field
    # is never treated as a structural gap: that is a real data-integrity
    # failure and must stop the build.
    row_finite = np.isfinite(flat_values).reshape(flat_values.shape[0], -1)
    row_any_finite = row_finite.any(axis=1)
    row_all_finite = row_finite.all(axis=1)

    partial_rows = row_any_finite & ~row_all_finite
    if partial_rows.any():
        bad_idx = np.flatnonzero(partial_rows)[:10].tolist()
        raise RuntimeError(
            f"{stream} {short}: partially non-finite accumulated spatial "
            f"fields encountered after cfgrib expansion; "
            f"flat_row_indices_head={bad_idx}. "
            f"Only fully empty structural time×step slots may be dropped."
        )

    structural_empty = ~row_any_finite
    structural_empty_count = int(structural_empty.sum())

    flat_values = flat_values[row_any_finite, :, :]
    flat_times = flat_times[row_any_finite]

    order = np.argsort(flat_times)
    flat_times = flat_times[order]
    flat_values = flat_values[order, :, :]
    actual_times = pd.DatetimeIndex(flat_times)

    if actual_times.has_duplicates:
        dup = actual_times[actual_times.duplicated()].unique()[:10].tolist()
        raise RuntimeError(
            f"{stream} {short}: duplicate validity times after removing "
            f"structural empty time×step slots; head={dup}"
        )

    expected_times = expected_month_times(year, month)
    if len(actual_times) != len(expected_times):
        raise RuntimeError(
            f"{stream} {short}: effective validity length {len(actual_times)} != "
            f"{len(expected_times)} for {year:04d}-{month:02d}; "
            f"cfgrib_dense_slots={len(row_any_finite)}, "
            f"structural_empty_slots={structural_empty_count}"
        )
    if not actual_times.equals(expected_times):
        missing = expected_times.difference(actual_times)[:10].tolist()
        extra = actual_times.difference(expected_times)[:10].tolist()
        raise RuntimeError(
            f"{stream} {short}: effective validity time mismatch for "
            f"{year:04d}-{month:02d}; "
            f"structural_empty_slots={structural_empty_count}, "
            f"missing_head={missing}, extra_head={extra}"
        )

    return flat_values, actual_times


def canonicalize_da(
    da: xr.DataArray,
    short: str,
    stream: str,
    year: int,
    month: int,
    bounds: Bounds,
) -> xr.DataArray:
    work = da

    pressure = stream == "pressure_925_850"
    accumulated = stream == "single_accum"
    p_dim = _pressure_dim(work) if pressure else None

    # ERA5 accumulated fields legitimately carry a non-scalar step dimension.
    allowed_non_scalar = {"time", "latitude", "longitude"}
    if pressure and p_dim:
        allowed_non_scalar.add(p_dim)
    if accumulated:
        allowed_non_scalar.add("step")

    # Squeeze only truly scalar auxiliary dimensions.
    for d in list(work.dims):
        if d not in allowed_non_scalar:
            if work.sizes[d] == 1:
                work = work.isel({d: 0}, drop=True)
            else:
                raise RuntimeError(
                    f"{stream} {short}: unexpected non-scalar dimension "
                    f"{d}={work.sizes[d]}; dims={work.dims}"
                )

    if "latitude" not in work.dims or "longitude" not in work.dims:
        raise RuntimeError(
            f"{stream} {short}: latitude/longitude dimensions missing; "
            f"dims={work.dims}"
        )
    if "time" not in work.dims:
        raise RuntimeError(
            f"{stream} {short}: time dimension missing; dims={work.dims}"
        )

    expected_times = expected_month_times(year, month)

    lat = np.asarray(work.coords["latitude"].values, dtype=np.float64)
    lon = np.asarray(work.coords["longitude"].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise RuntimeError(
            f"{stream} {short}: expected 1-D lat/lon; "
            f"lat_shape={lat.shape} lon_shape={lon.shape}"
        )
    if len(lat) != bounds.n_lat or len(lon) != bounds.n_lon:
        raise RuntimeError(
            f"{stream} {short}: grid shape mismatch; "
            f"got={len(lat)}x{len(lon)} expected={bounds.n_lat}x{bounds.n_lon}"
        )

    tol = GRID_DEG + 1e-3
    if (
        abs(float(np.max(lat)) - bounds.north) > tol
        or abs(float(np.min(lat)) - bounds.south) > tol
        or abs(float(np.min(lon)) - bounds.west) > tol
        or abs(float(np.max(lon)) - bounds.east) > tol
    ):
        raise RuntimeError(
            f"{stream} {short}: grid bbox mismatch; "
            f"got N/W/S/E="
            f"{np.max(lat):.6f}/{np.min(lon):.6f}/"
            f"{np.min(lat):.6f}/{np.max(lon):.6f}, "
            f"expected≈"
            f"{bounds.north:.6f}/{bounds.west:.6f}/"
            f"{bounds.south:.6f}/{bounds.east:.6f}"
        )

    expected_unit = EXPECTED_UNITS[short]
    unit = _get_units(work)
    if unit and unit != expected_unit:
        raise RuntimeError(
            f"{stream} {short}: unit changed during cfgrib decode; "
            f"got={unit!r} expected={expected_unit!r}"
        )

    if pressure:
        # Pressure-level variables are instantaneous hourly fields.
        times = _extract_time_values(work)
        actual_times = pd.DatetimeIndex(times)
        if len(actual_times) != len(expected_times):
            raise RuntimeError(
                f"{stream} {short}: time length {len(actual_times)} != "
                f"{len(expected_times)} for {year:04d}-{month:02d}"
            )
        if not actual_times.equals(expected_times):
            missing = expected_times.difference(actual_times)[:10].tolist()
            extra = actual_times.difference(expected_times)[:10].tolist()
            raise RuntimeError(
                f"{stream} {short}: validity time mismatch for "
                f"{year:04d}-{month:02d}; "
                f"missing_head={missing}, extra_head={extra}"
            )

        if p_dim is None or p_dim not in work.dims:
            raise RuntimeError(
                f"{stream} {short}: pressure dimension missing; dims={work.dims}"
            )
        levels = np.asarray(work.coords[p_dim].values, dtype=np.float64)
        if p_dim == "isobaricInPa":
            levels = levels / 100.0
        levels_i = np.rint(levels).astype(np.int16)
        if set(levels_i.tolist()) != set(PRESSURE_LEVEL_ORDER_HPA):
            raise RuntimeError(
                f"{stream} {short}: pressure levels mismatch; "
                f"got={levels_i.tolist()} expected={PRESSURE_LEVEL_ORDER_HPA}"
            )

        arr = np.asarray(
            work.transpose("time", p_dim, "latitude", "longitude").values
        )

        order = [
            int(np.where(levels_i == p)[0][0])
            for p in PRESSURE_LEVEL_ORDER_HPA
        ]
        arr = arr[:, order, :, :]

        lat_order = np.argsort(lat)[::-1]
        lon_order = np.argsort(lon)
        arr = arr[:, :, lat_order, :][:, :, :, lon_order]
        lat = lat[lat_order]
        lon = lon[lon_order]

        dims = ("time", "pressure_level_hpa", "latitude", "longitude")
        coords = {
            "time": expected_times.values.astype("datetime64[ns]"),
            "pressure_level_hpa": np.asarray(
                PRESSURE_LEVEL_ORDER_HPA, dtype=np.int16
            ),
            "latitude": lat,
            "longitude": lon,
        }

    elif accumulated:
        arr, actual_times = _decode_accumulated_time_step(
            work=work,
            stream=stream,
            short=short,
            year=year,
            month=month,
        )

        lat_order = np.argsort(lat)[::-1]
        lon_order = np.argsort(lon)
        arr = arr[:, lat_order, :][:, :, lon_order]
        lat = lat[lat_order]
        lon = lon[lon_order]

        dims = ("time", "latitude", "longitude")
        coords = {
            "time": actual_times.values.astype("datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        }

    else:
        # Instantaneous single-level fields.
        times = _extract_time_values(work)
        actual_times = pd.DatetimeIndex(times)
        if len(actual_times) != len(expected_times):
            raise RuntimeError(
                f"{stream} {short}: time length {len(actual_times)} != "
                f"{len(expected_times)} for {year:04d}-{month:02d}"
            )
        if not actual_times.equals(expected_times):
            missing = expected_times.difference(actual_times)[:10].tolist()
            extra = actual_times.difference(expected_times)[:10].tolist()
            raise RuntimeError(
                f"{stream} {short}: validity time mismatch for "
                f"{year:04d}-{month:02d}; "
                f"missing_head={missing}, extra_head={extra}"
            )

        arr = np.asarray(
            work.transpose("time", "latitude", "longitude").values
        )
        lat_order = np.argsort(lat)[::-1]
        lon_order = np.argsort(lon)
        arr = arr[:, lat_order, :][:, :, lon_order]
        lat = lat[lat_order]
        lon = lon[lon_order]

        dims = ("time", "latitude", "longitude")
        coords = {
            "time": expected_times.values.astype("datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        }

    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        raise RuntimeError(
            f"{stream} {short}: decoded canonical array has zero finite values"
        )
    if not finite.all():
        bad = int(arr.size - finite.sum())
        raise RuntimeError(
            f"{stream} {short}: decoded canonical array contains "
            f"{bad:,}/{arr.size:,} non-finite values"
        )

    canonical_name = CANONICAL_NAMES[short]
    out = xr.DataArray(
        arr,
        dims=dims,
        coords=coords,
        name=canonical_name,
        attrs={
            "units": expected_unit,
            "source": "ERA5 CDS GRIB",
            "source_shortName": short,
            "canonical_policy": (
                "decoded only; accumulated fields flattened by GRIB validity "
                "time; no unit conversion, interpolation, imputation, scaling, "
                "clipping, or derived features"
            ),
        },
    )
    return out

def load_stream_dataset(
    path: Path,
    stream: str,
    year: int,
    month: int,
    bounds: Bounds,
) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    expected_short = EXPECTED_SHORTNAMES[stream]

    # indexpath="" prevents creation of persistent .idx files next to immutable raw GRIB.
    datasets = cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )

    found: dict[str, xr.DataArray] = {}
    try:
        for ds in datasets:
            for name, da in ds.data_vars.items():
                short = _shortname(name, da)
                if short not in expected_short:
                    continue
                if short in found:
                    raise RuntimeError(
                        f"{stream}: duplicate shortName={short!r} across cfgrib groups in {path}"
                    )
                found[short] = canonicalize_da(
                    da=da,
                    short=short,
                    stream=stream,
                    year=year,
                    month=month,
                    bounds=bounds,
                )
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass

    if set(found) != expected_short:
        raise RuntimeError(
            f"{stream}: cfgrib variable set mismatch for {path}; "
            f"expected={sorted(expected_short)} got={sorted(found)}"
        )

    out = xr.Dataset({da.name: da for da in found.values()})
    ranges: dict[str, dict[str, float]] = {}
    for name, da in out.data_vars.items():
        values = np.asarray(da.values)
        ranges[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return out, ranges


def assert_same_core_coords(datasets: list[xr.Dataset], label: str) -> None:
    ref = datasets[0]
    for i, ds in enumerate(datasets[1:], start=1):
        for coord in ("time", "latitude", "longitude"):
            a = np.asarray(ref.coords[coord].values)
            b = np.asarray(ds.coords[coord].values)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise RuntimeError(
                    f"{label}: coordinate mismatch for {coord} "
                    f"between stream 0 and stream {i}"
                )


def build_month_dataset(
    root: Path,
    region: str,
    year: int,
    month: int,
    jobs_by_key: dict[tuple[str, int, int, str], Job],
    bounds: Bounds,
    verify_sha: bool,
) -> tuple[xr.Dataset, dict[str, Any]]:
    stream_datasets: list[xr.Dataset] = []
    source_records: list[dict[str, Any]] = []
    ranges: dict[str, dict[str, float]] = {}

    for stream in STREAMS:
        key = (region, year, month, stream)
        if key not in jobs_by_key:
            raise RuntimeError(f"Missing manifest job: {key}")
        j = jobs_by_key[key]
        raw = raw_path(root, j)
        audit = audit_path(root, j)

        report = verify_raw_against_audit(root, j, verify_sha=verify_sha)
        ds, stream_ranges = load_stream_dataset(
            path=raw,
            stream=stream,
            year=year,
            month=month,
            bounds=bounds,
        )
        stream_datasets.append(ds)
        ranges.update(stream_ranges)

        source_records.append(
            {
                "stream": stream,
                "raw_path": str(raw),
                "audit_path": str(audit),
                "raw_sha256": report.get("sha256"),
                "raw_size_bytes": int(report.get("size_bytes", raw.stat().st_size)),
                "expected_hours": j.expected_hours,
            }
        )

    assert_same_core_coords(
        stream_datasets,
        f"{region} {year:04d}-{month:02d}",
    )

    month_ds = xr.merge(stream_datasets, compat="equals", join="exact")
    expected_hours = calendar.monthrange(year, month)[1] * 24
    if month_ds.sizes.get("time") != expected_hours:
        raise RuntimeError(
            f"{region} {year:04d}-{month:02d}: merged time size "
            f"{month_ds.sizes.get('time')} != {expected_hours}"
        )
    if month_ds.sizes.get("latitude") != bounds.n_lat:
        raise RuntimeError(
            f"{region} {year:04d}-{month:02d}: merged n_lat mismatch"
        )
    if month_ds.sizes.get("longitude") != bounds.n_lon:
        raise RuntimeError(
            f"{region} {year:04d}-{month:02d}: merged n_lon mismatch"
        )
    if month_ds.sizes.get("pressure_level_hpa") != 2:
        raise RuntimeError(
            f"{region} {year:04d}-{month:02d}: pressure dimension != 2"
        )

    month_ds.coords["time"].attrs.update(
        {
            "timezone": "UTC",
            "semantics": "ERA5 validity time; hourly",
        }
    )
    month_ds.coords["latitude"].attrs["units"] = "degrees_north"
    month_ds.coords["longitude"].attrs["units"] = "degrees_east"
    month_ds.coords["pressure_level_hpa"].attrs.update(
        {"units": "hPa", "positive": "down"}
    )

    month_ds.attrs.update(
        {
            "canonical_build_version": BUILD_VERSION,
            "region": region,
            "spatial_domain_policy": "continuous research domain derived from member-city geographic envelope + fixed transport halo; not an administrative boundary",
            "grid_resolution_degrees": GRID_DEG,
            "time_resolution": "1 hour",
            "time_timezone": "UTC",
            "raw_product_family": "ERA5",
            "raw_format": "GRIB",
            "canonical_transformations": "decode, coordinate normalization, stream merge only",
            "prohibited_here": "unit conversion; local-time conversion; interpolation; imputation; scaling; empirical clipping; feature selection; derived features",
        }
    )

    report = {
        "region": region,
        "month": f"{year:04d}-{month:02d}",
        "expected_hours": expected_hours,
        "shape": {
            "time": int(month_ds.sizes["time"]),
            "pressure_level_hpa": int(month_ds.sizes["pressure_level_hpa"]),
            "latitude": int(month_ds.sizes["latitude"]),
            "longitude": int(month_ds.sizes["longitude"]),
        },
        "time_start_utc": str(pd.Timestamp(month_ds.time.values[0])),
        "time_end_utc": str(pd.Timestamp(month_ds.time.values[-1])),
        "pressure_levels_hpa": [
            int(x) for x in month_ds.pressure_level_hpa.values.tolist()
        ],
        "variable_ranges_raw_units": ranges,
        "sources": source_records,
    }
    return month_ds, report


def zarr_write_kwargs() -> dict[str, Any]:
    params = inspect.signature(xr.Dataset.to_zarr).parameters
    if "zarr_format" in params:
        return {"zarr_format": 2}
    if "zarr_version" in params:
        return {"zarr_version": 2}
    return {}


def first_write_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        chunks = []
        for d in da.dims:
            if d == "time":
                chunks.append(min(TIME_CHUNK_HOURS, da.sizes[d]))
            else:
                # The regional grids are small; keeping the full spatial field
                # in one chunk is convenient for transport-context access.
                chunks.append(da.sizes[d])
        encoding[name] = {"chunks": tuple(chunks)}
    encoding["time"] = {"chunks": (min(TIME_CHUNK_HOURS, ds.sizes["time"]),)}
    return encoding


def append_month_to_zarr(
    ds: xr.Dataset,
    store: Path,
    first: bool,
) -> None:
    kwargs = {
        "store": str(store),
        "consolidated": False,
        **zarr_write_kwargs(),
    }

    if first:
        ds.to_zarr(
            mode="w",
            encoding=first_write_encoding(ds),
            **kwargs,
        )
    else:
        ds.to_zarr(
            mode="a",
            append_dim="time",
            **kwargs,
        )


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_resume_state(
    store: Path,
    state: dict[str, Any],
) -> None:
    completed = state.get("completed_months", [])
    expected_hours = int(state.get("hours_written", -1))
    if not store.exists():
        raise RuntimeError(
            f"State exists but Zarr store is missing: {store}"
        )

    ds = xr.open_zarr(str(store), consolidated=False)
    try:
        actual_hours = int(ds.sizes.get("time", -1))
        if actual_hours != expected_hours:
            raise RuntimeError(
                f"Resume refused: Zarr time size={actual_hours}, "
                f"state hours_written={expected_hours}. "
                f"The previous write may have been interrupted; "
                f"rerun this region with --overwrite."
            )

        times = pd.DatetimeIndex(ds.time.values)
        if len(times) and (
            not times.is_monotonic_increasing
            or times.has_duplicates
        ):
            raise RuntimeError(
                f"Resume refused: non-monotonic/duplicate time in {store}"
            )

        if completed:
            last = pd.Period(completed[-1], freq="M")
            expected_last = expected_month_times(last.year, last.month)[-1]
            if len(times) == 0 or pd.Timestamp(times[-1]) != expected_last:
                raise RuntimeError(
                    f"Resume refused: last Zarr timestamp={times[-1] if len(times) else None}, "
                    f"state last month={completed[-1]} expected_last={expected_last}"
                )
    finally:
        ds.close()


def final_region_gate(
    store: Path,
    region: str,
    completed_months: list[str],
    bounds: Bounds,
) -> dict[str, Any]:
    ds = xr.open_zarr(str(store), consolidated=False)
    try:
        times = pd.DatetimeIndex(ds.time.values)
        if times.has_duplicates:
            raise RuntimeError(f"{region}: duplicate canonical timestamps")
        if not times.is_monotonic_increasing:
            raise RuntimeError(f"{region}: canonical time is not monotonic")

        expected_parts = []
        for m in completed_months:
            p = pd.Period(m, freq="M")
            expected_parts.append(expected_month_times(p.year, p.month))
        expected = expected_parts[0]
        for part in expected_parts[1:]:
            expected = expected.append(part)

        if not times.equals(expected):
            missing = expected.difference(times)[:10].tolist()
            extra = times.difference(expected)[:10].tolist()
            raise RuntimeError(
                f"{region}: final hourly timeline mismatch; "
                f"missing_head={missing}, extra_head={extra}"
            )

        expected_vars = set(CANONICAL_NAMES.values())
        got_vars = set(ds.data_vars)
        if got_vars != expected_vars:
            raise RuntimeError(
                f"{region}: final variable set mismatch; "
                f"expected={sorted(expected_vars)} got={sorted(got_vars)}"
            )

        if int(ds.sizes["latitude"]) != bounds.n_lat:
            raise RuntimeError(f"{region}: final n_lat mismatch")
        if int(ds.sizes["longitude"]) != bounds.n_lon:
            raise RuntimeError(f"{region}: final n_lon mismatch")
        if [int(x) for x in ds.pressure_level_hpa.values.tolist()] != PRESSURE_LEVEL_ORDER_HPA:
            raise RuntimeError(f"{region}: final pressure level order mismatch")

        return {
            "status": "PASS",
            "region": region,
            "store": str(store),
            "months": len(completed_months),
            "completed_months": completed_months,
            "hours": int(ds.sizes["time"]),
            "time_start_utc": str(pd.Timestamp(times[0])),
            "time_end_utc": str(pd.Timestamp(times[-1])),
            "grid": {
                "latitude": int(ds.sizes["latitude"]),
                "longitude": int(ds.sizes["longitude"]),
                "grid_cells": int(ds.sizes["latitude"] * ds.sizes["longitude"]),
            },
            "pressure_levels_hpa": [
                int(x) for x in ds.pressure_level_hpa.values.tolist()
            ],
            "variables": sorted(ds.data_vars),
        }
    finally:
        ds.close()


def parse_month_filter(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    p = pd.Period(text, freq="M")
    return p.year, p.month


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build leakage-neutral hourly ERA5 regional canonical Zarr stores "
            "from acquisition-audited GRIB files."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )
    ap.add_argument("--halo-deg", type=float, default=3.0)
    ap.add_argument(
        "--region",
        action="append",
        default=[],
        help="Repeat to limit regions; default builds all five.",
    )
    ap.add_argument(
        "--start-month",
        default=None,
        help="Optional YYYY-MM filter; default uses all manifest months.",
    )
    ap.add_argument(
        "--end-month",
        default=None,
        help="Optional YYYY-MM filter; default uses all manifest months.",
    )
    ap.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Per-region development limit. Safe to resume later.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete selected region canonical store/state and rebuild it.",
    )
    ap.add_argument(
        "--skip-raw-sha",
        action="store_true",
        help="Skip re-hashing GRIB against its PASS audit. Faster but weaker.",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    jobs_path = (
        root / "05_manifests"
        / f"era5_acquisition_jobs_halo{args.halo_deg:g}.csv"
    )
    bounds_path = (
        root / "05_manifests"
        / f"era5_region_bounds_halo{args.halo_deg:g}.csv"
    )
    cities_path = root / "05_manifests" / "selected_cities_v2.csv"

    jobs = load_jobs(jobs_path)
    bounds_map = load_bounds(bounds_path)

    # Full acquisition closure invariants.
    if len(jobs) != 555:
        raise RuntimeError(
            f"Expected the frozen 555-job acquisition manifest, got {len(jobs)}"
        )

    regions_all = sorted({j.region for j in jobs})
    if len(regions_all) != 5:
        raise RuntimeError(f"Expected 5 regions, got {regions_all}")

    for region in regions_all:
        rjobs = [j for j in jobs if j.region == region]
        if len(rjobs) != 111:
            raise RuntimeError(
                f"{region}: expected 111 jobs, got {len(rjobs)}"
            )

    selected_regions = sorted(set(args.region)) if args.region else regions_all
    unknown = sorted(set(selected_regions) - set(regions_all))
    if unknown:
        raise RuntimeError(f"Unknown regions: {unknown}")

    start_filter = parse_month_filter(args.start_month)
    end_filter = parse_month_filter(args.end_month)
    if start_filter and end_filter and start_filter > end_filter:
        raise RuntimeError("--start-month > --end-month")

    jobs_by_key = {
        (j.region, j.year, j.month, j.stream): j for j in jobs
    }
    if len(jobs_by_key) != len(jobs):
        raise RuntimeError("Duplicate acquisition job keys in manifest")

    canonical_root = root / "03_canonical" / "era5"
    reports_root = canonical_root / "_build_reports"
    state_root = canonical_root / "_build_state"
    canonical_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    run_report: dict[str, Any] = {
        "status": "RUNNING",
        "build_version": BUILD_VERSION,
        "started_at_utc": now_utc(),
        "root": str(root),
        "jobs_manifest": str(jobs_path),
        "jobs_manifest_sha256": sha256_file(jobs_path),
        "bounds_manifest": str(bounds_path),
        "bounds_manifest_sha256": sha256_file(bounds_path),
        "cities_manifest": str(cities_path),
        "cities_manifest_sha256": sha256_file(cities_path) if cities_path.exists() else None,
        "verify_raw_sha": not args.skip_raw_sha,
        "selected_regions": selected_regions,
        "regions": {},
    }

    for region in selected_regions:
        print("\n" + "=" * 110)
        print(f"REGION: {region}")
        print("=" * 110, flush=True)

        bounds = bounds_map[region]
        store = canonical_root / f"{region}.zarr"
        state_path = state_root / f"{region}.json"
        report_path = reports_root / f"{region}.json"

        if args.overwrite:
            if store.exists():
                shutil.rmtree(store)
            state_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)

        state = read_state(state_path)
        if store.exists() != (state is not None):
            raise RuntimeError(
                f"{region}: store/state existence mismatch. "
                f"Use --overwrite to rebuild this region safely."
            )

        if state is None:
            state = {
                "build_version": BUILD_VERSION,
                "region": region,
                "store": str(store),
                "created_at_utc": now_utc(),
                "completed_months": [],
                "hours_written": 0,
                "month_reports": [],
            }
        else:
            if state.get("build_version") != BUILD_VERSION:
                raise RuntimeError(
                    f"{region}: state build version differs; use --overwrite"
                )
            verify_resume_state(store, state)

        completed = list(state["completed_months"])
        completed_set = set(completed)

        region_months = sorted({
            (j.year, j.month)
            for j in jobs
            if j.region == region
        })

        if start_filter:
            region_months = [x for x in region_months if x >= start_filter]
        if end_filter:
            region_months = [x for x in region_months if x <= end_filter]

        pending = [
            ym for ym in region_months
            if month_key(*ym) not in completed_set
        ]
        if args.max_months is not None:
            pending = pending[: args.max_months]

        if not pending:
            print(f"[NO PENDING MONTHS] {region}", flush=True)

        for idx, (year, month) in enumerate(pending, start=1):
            mk = month_key(year, month)
            print(
                f"\n[{idx}/{len(pending)}] BUILD {region} {mk}",
                flush=True,
            )
            try:
                month_ds, month_report = build_month_dataset(
                    root=root,
                    region=region,
                    year=year,
                    month=month,
                    jobs_by_key=jobs_by_key,
                    bounds=bounds,
                    verify_sha=not args.skip_raw_sha,
                )

                first = not store.exists()
                append_month_to_zarr(
                    ds=month_ds,
                    store=store,
                    first=first,
                )
                month_ds.close()

                completed.append(mk)
                state["completed_months"] = completed
                state["hours_written"] = int(
                    sum(
                        calendar.monthrange(
                            pd.Period(x, freq="M").year,
                            pd.Period(x, freq="M").month,
                        )[1] * 24
                        for x in completed
                    )
                )
                state["month_reports"].append(month_report)
                state["updated_at_utc"] = now_utc()
                atomic_json(state, state_path)

                print(
                    f"[PASS MONTH] {region} {mk} "
                    f"hours={month_report['expected_hours']} "
                    f"canonical_hours={state['hours_written']}",
                    flush=True,
                )
            except Exception as exc:
                fail = {
                    "status": "FAIL",
                    "build_version": BUILD_VERSION,
                    "failed_at_utc": now_utc(),
                    "region": region,
                    "month": mk,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_months_before_failure": completed,
                    "store": str(store),
                }
                atomic_json(fail, report_path)
                print(
                    f"\n[FAIL] {region} {mk}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        # Gate whatever contiguous set is currently represented in the store.
        if completed:
            region_report = final_region_gate(
                store=store,
                region=region,
                completed_months=completed,
                bounds=bounds,
            )
            region_report.update(
                {
                    "build_version": BUILD_VERSION,
                    "finished_at_utc": now_utc(),
                    "verify_raw_sha": not args.skip_raw_sha,
                    "is_full_37_month_archive": len(completed) == 37,
                }
            )
            atomic_json(region_report, report_path)
            run_report["regions"][region] = region_report

            print(
                f"\n[REGION PASS] {region}: "
                f"months={region_report['months']} "
                f"hours={region_report['hours']:,} "
                f"store={store}",
                flush=True,
            )

            # Consolidation is metadata-only and may make repeated open_zarr faster.
            try:
                zarr.consolidate_metadata(str(store))
                print(f"[METADATA CONSOLIDATED] {store}", flush=True)
            except Exception as exc:
                print(
                    f"[WARN] metadata consolidation skipped: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    full = (
        set(run_report["regions"]) == set(regions_all)
        and all(
            r.get("is_full_37_month_archive")
            for r in run_report["regions"].values()
        )
    )
    run_report["status"] = "PASS" if full else "PARTIAL_PASS"
    run_report["finished_at_utc"] = now_utc()

    run_dir = root / "08_runs" / "era5_canonical"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / (
        "build_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    atomic_json(run_report, run_path)

    print("\n" + "=" * 110)
    print("ERA5 CANONICAL BUILD:", run_report["status"])
    print("Run report:", run_path)
    if full:
        print(
            "Expected full archive: 5 regions x 37 months; "
            "each region has 27,048 hourly UTC timestamps."
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
