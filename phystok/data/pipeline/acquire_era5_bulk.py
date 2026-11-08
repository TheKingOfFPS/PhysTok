#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cdsapi  # type: ignore
except ImportError:
    cdsapi = None  # type: ignore

try:
    from eccodes import (
        codes_get,
        codes_get_array,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )
except ImportError:
    codes_get = codes_get_array = codes_get_values = None  # type: ignore
    codes_grib_new_from_file = codes_release = None  # type: ignore


SINGLE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"

SINGLE_INSTANT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "boundary_layer_height",
    "total_cloud_cover",
]

SINGLE_ACCUM_VARIABLES = [
    "total_precipitation",
    "surface_solar_radiation_downwards",
]

PRESSURE_VARIABLES = [
    "temperature",
    "relative_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
PRESSURE_LEVELS = ["925", "850"]
TIMES = [f"{h:02d}:00" for h in range(24)]
GRID_DEG = 0.25

EXPECTED_SINGLE_INSTANT = {"2t", "2d", "sp", "10u", "10v", "blh", "tcc"}
EXPECTED_SINGLE_ACCUM = {"tp", "ssrd"}
EXPECTED_PRESSURE = {"t", "r", "u", "v", "z"}
EXPECTED_PRESSURE_LEVELS = {850, 925}

EXPECTED_UNITS = {
    "2t": {"K"},
    "2d": {"K"},
    "sp": {"Pa"},
    "10u": {"m s**-1"},
    "10v": {"m s**-1"},
    "blh": {"m"},
    "tcc": {"(0 - 1)"},
    "tp": {"m"},
    "ssrd": {"J m**-2"},
    "t": {"K"},
    "r": {"%"},
    "u": {"m s**-1"},
    "v": {"m s**-1"},
    "z": {"m**2 s**-2"},
}

STREAMS = ("single_instant", "single_accum", "pressure_925_850")


@dataclass(frozen=True)
class City:
    region: str
    priority: int
    city: str
    city_zh: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class RegionBounds:
    region: str
    city_count: int
    city_names: str
    city_lat_min: float
    city_lat_max: float
    city_lon_min: float
    city_lon_max: float
    halo_deg: float
    north: float
    west: float
    south: float
    east: float
    n_lat: int
    n_lon: int
    grid_cells: int


@dataclass(frozen=True)
class Job:
    region: str
    scope: str
    year: int
    month: int
    stream: str
    north: float
    west: float
    south: float
    east: float
    expected_hours: int
    target: Path
    audit_path: Path
    provenance_path: Path


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_get(gid: Any, key: str, default: Any = None) -> Any:
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def safe_get_array(gid: Any, key: str) -> np.ndarray | None:
    try:
        return np.asarray(codes_get_array(gid, key))
    except Exception:
        return None


def validity_key(gid: Any) -> str:
    vdate = safe_get(gid, "validityDate", safe_get(gid, "dataDate"))
    vtime = safe_get(gid, "validityTime", safe_get(gid, "dataTime"))
    if vdate is None or vtime is None:
        return "UNKNOWN"
    return f"{int(vdate):08d}-{int(vtime):04d}"


def month_validities(year: int, month: int) -> list[str]:
    ndays = calendar.monthrange(year, month)[1]
    return [f"{year:04d}{month:02d}{day:02d}-{hour:02d}00" for day in range(1, ndays + 1) for hour in range(24)]


def snap_down(value: float, step: float = GRID_DEG) -> float:
    return round(math.floor((value + 1e-12) / step) * step, 6)


def snap_up(value: float, step: float = GRID_DEG) -> float:
    return round(math.ceil((value - 1e-12) / step) * step, 6)


def grid_count(lo: float, hi: float, step: float = GRID_DEG) -> int:
    return int(round((hi - lo) / step)) + 1


def check_grib_magic(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing/empty GRIB: {path}")
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != b"GRIB":
        raise RuntimeError(f"Not GRIB by magic bytes: {path}; first4={magic!r}")


# -----------------------------------------------------------------------------
# City selection and region geometry
# -----------------------------------------------------------------------------


def load_cities(path: Path) -> list[City]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"region", "priority", "city", "city_zh", "latitude", "longitude"}
    if not rows:
        raise ValueError(f"Empty city CSV: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"City CSV missing columns: {sorted(missing)}")

    cities: list[City] = []
    for i, r in enumerate(rows, start=2):
        try:
            city = City(
                region=r["region"].strip(),
                priority=int(r["priority"]),
                city=r["city"].strip(),
                city_zh=r["city_zh"].strip(),
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
            )
        except Exception as exc:
            raise ValueError(f"Bad city row at CSV line {i}: {r}") from exc
        if not (-90 <= city.latitude <= 90 and -180 <= city.longitude <= 180):
            raise ValueError(f"Invalid coordinate: {city}")
        cities.append(city)

    if len(cities) != 50:
        raise ValueError(f"Expected 50 cities, got {len(cities)}")
    if len({c.city for c in cities}) != 50:
        raise ValueError("City names are not unique")

    counts = Counter(c.region for c in cities)
    if len(counts) != 5 or any(v != 10 for v in counts.values()):
        raise ValueError(f"Expected exactly 5 regions x 10 cities; got {dict(counts)}")

    for region in sorted(counts):
        priorities = sorted(c.priority for c in cities if c.region == region)
        if priorities != list(range(1, 11)):
            raise ValueError(f"{region}: priorities must be exactly 1..10; got {priorities}")

    return cities


def build_region_bounds(cities: list[City], halo_deg: float) -> list[RegionBounds]:
    if halo_deg < 0:
        raise ValueError("halo_deg must be >= 0")

    regions = sorted({c.region for c in cities})
    out: list[RegionBounds] = []
    for region in regions:
        rc = sorted((c for c in cities if c.region == region), key=lambda x: x.priority)
        lats = [c.latitude for c in rc]
        lons = [c.longitude for c in rc]

        south = snap_down(min(lats) - halo_deg)
        north = snap_up(max(lats) + halo_deg)
        west = snap_down(min(lons) - halo_deg)
        east = snap_up(max(lons) + halo_deg)

        if not (north > south and east > west):
            raise ValueError(f"Invalid bbox for {region}: N/W/S/E={north, west, south, east}")
        if not all(south <= c.latitude <= north and west <= c.longitude <= east for c in rc):
            raise AssertionError(f"Region bbox does not contain all member cities: {region}")

        n_lat = grid_count(south, north)
        n_lon = grid_count(west, east)
        out.append(
            RegionBounds(
                region=region,
                city_count=len(rc),
                city_names="|".join(c.city for c in rc),
                city_lat_min=min(lats),
                city_lat_max=max(lats),
                city_lon_min=min(lons),
                city_lon_max=max(lons),
                halo_deg=halo_deg,
                north=north,
                west=west,
                south=south,
                east=east,
                n_lat=n_lat,
                n_lon=n_lon,
                grid_cells=n_lat * n_lon,
            )
        )
    return out


def bounds_fingerprint(bounds: list[RegionBounds]) -> str:
    payload = [asdict(b) for b in sorted(bounds, key=lambda x: x.region)]
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:16]


# -----------------------------------------------------------------------------
# CDS requests
# -----------------------------------------------------------------------------


def base_month_request(year: int, month: int, b: RegionBounds) -> dict[str, Any]:
    ndays = calendar.monthrange(year, month)[1]
    return {
        "product_type": ["reanalysis"],
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": TIMES,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [b.north, b.west, b.south, b.east],
        "grid": [GRID_DEG, GRID_DEG],
    }


def build_request(job: Job) -> tuple[str, dict[str, Any]]:
    b = RegionBounds(
        region=job.region,
        city_count=0,
        city_names="",
        city_lat_min=0,
        city_lat_max=0,
        city_lon_min=0,
        city_lon_max=0,
        halo_deg=0,
        north=job.north,
        west=job.west,
        south=job.south,
        east=job.east,
        n_lat=0,
        n_lon=0,
        grid_cells=0,
    )
    req = base_month_request(job.year, job.month, b)
    if job.stream == "single_instant":
        req["variable"] = SINGLE_INSTANT_VARIABLES
        dataset = SINGLE_DATASET
    elif job.stream == "single_accum":
        req["variable"] = SINGLE_ACCUM_VARIABLES
        dataset = SINGLE_DATASET
    elif job.stream == "pressure_925_850":
        req["variable"] = PRESSURE_VARIABLES
        req["pressure_level"] = PRESSURE_LEVELS
        dataset = PRESSURE_DATASET
    else:
        raise ValueError(job.stream)
    return dataset, req


def retrieve_atomic(client: Any, dataset: str, request: dict[str, Any], target: Path, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[CDS] attempt={attempt}/{retries} dataset={dataset} target={target}", flush=True)
            client.retrieve(dataset, request, str(part))
            check_grib_magic(part)
            part.replace(target)
            return
        except Exception as exc:
            last_exc = exc
            print(f"[CDS] ERROR attempt={attempt}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"CDS retrieval failed after {retries} attempts: {target}") from last_exc


# -----------------------------------------------------------------------------
# Monthly GRIB hard audit
# -----------------------------------------------------------------------------


def audit_grib(
    path: Path,
    stream: str,
    expected_validities: list[str],
    requested_area: tuple[float, float, float, float],
    member_cities: list[City],
) -> dict[str, Any]:
    if codes_grib_new_from_file is None or codes_get_values is None:
        raise RuntimeError("ecCodes Python bindings are required for GRIB audit; activate the era5_data environment")
    check_grib_magic(path)

    pressure = stream == "pressure_925_850"
    if stream == "single_instant":
        expected_shortnames = EXPECTED_SINGLE_INSTANT
        expected_step_type = "instant"
    elif stream == "single_accum":
        expected_shortnames = EXPECTED_SINGLE_ACCUM
        expected_step_type = "accum"
    elif stream == "pressure_925_850":
        expected_shortnames = EXPECTED_PRESSURE
        expected_step_type = "instant"
    else:
        raise ValueError(stream)

    by_short = Counter()
    timestamps_by_key: dict[str, set[str]] = defaultdict(set)
    level_pairs = Counter()
    units_by_short: dict[str, set[str]] = defaultdict(set)
    step_types_by_short: dict[str, set[str]] = defaultdict(set)
    level_types_by_key: dict[str, set[str]] = defaultdict(set)
    ranges: dict[str, dict[str, float]] = {}
    range_acc: dict[str, list[float]] = {}
    first_grid_summary: dict[str, Any] | None = None
    grid_signature: tuple[Any, ...] | None = None
    message_count = 0
    all_finite = True

    with path.open("rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            message_count += 1
            try:
                short = str(safe_get(gid, "shortName", "UNKNOWN"))
                units = str(safe_get(gid, "units", ""))
                level_type = str(safe_get(gid, "typeOfLevel", "UNKNOWN"))
                level_raw = safe_get(gid, "level")
                try:
                    level = int(round(float(level_raw)))
                except Exception:
                    level = None
                validity = validity_key(gid)
                step_type = str(safe_get(gid, "stepType", "UNKNOWN"))
                grid_type = str(safe_get(gid, "gridType", "UNKNOWN"))

                if short not in expected_shortnames:
                    raise RuntimeError(f"{stream}: unexpected shortName={short!r} in message {message_count}")
                if not units or units.strip().lower() in {"undefined", "unknown", "none", "nan"}:
                    raise RuntimeError(f"{stream}: invalid units={units!r} for {short}")
                if units not in EXPECTED_UNITS[short]:
                    raise RuntimeError(f"{stream}: unexpected units for {short}: {units!r}; expected={EXPECTED_UNITS[short]}")
                if step_type != expected_step_type:
                    raise RuntimeError(f"{stream}: unexpected stepType for {short}: {step_type!r}; expected={expected_step_type!r}")

                values = np.asarray(codes_get_values(gid), dtype=np.float64)
                finite = np.isfinite(values)
                finite_count = int(finite.sum())
                total_count = int(values.size)
                if total_count == 0 or finite_count == 0:
                    raise RuntimeError(f"{stream}: {short}@{level} has zero finite values")
                if finite_count != total_count:
                    all_finite = False
                    raise RuntimeError(
                        f"{stream}: {short}@{level} contains non-finite values: finite={finite_count}/{total_count}"
                    )
                vv = values[finite]
                mn, mx = float(vv.min()), float(vv.max())

                by_short[short] += 1
                units_by_short[short].add(units)
                step_types_by_short[short].add(step_type)

                key = f"{short}@{level}" if pressure else short
                timestamps_by_key[key].add(validity)
                level_types_by_key[key].add(level_type)
                if pressure:
                    level_pairs[(short, level)] += 1
                    if level_type not in {"isobaricInhPa", "isobaricInPa"}:
                        raise RuntimeError(f"{stream}: unexpected pressure typeOfLevel={level_type!r} for {key}")

                if key not in range_acc:
                    range_acc[key] = [mn, mx]
                else:
                    range_acc[key][0] = min(range_acc[key][0], mn)
                    range_acc[key][1] = max(range_acc[key][1], mx)

                sig = (
                    grid_type,
                    safe_get(gid, "Ni"),
                    safe_get(gid, "Nj"),
                    safe_get(gid, "latitudeOfFirstGridPointInDegrees"),
                    safe_get(gid, "longitudeOfFirstGridPointInDegrees"),
                    safe_get(gid, "latitudeOfLastGridPointInDegrees"),
                    safe_get(gid, "longitudeOfLastGridPointInDegrees"),
                    safe_get(gid, "iDirectionIncrementInDegrees"),
                    safe_get(gid, "jDirectionIncrementInDegrees"),
                )
                if grid_signature is None:
                    grid_signature = sig
                elif sig != grid_signature:
                    raise RuntimeError(f"{stream}: grid geometry changed across messages")

                if first_grid_summary is None:
                    lats = safe_get_array(gid, "latitudes")
                    lons = safe_get_array(gid, "longitudes")
                    if lats is None or lons is None or not lats.size or not lons.size:
                        raise RuntimeError(f"{stream}: failed to decode grid coordinates")
                    ulats = np.unique(np.round(lats.astype(float), 6))
                    ulons = np.unique(np.round(lons.astype(float), 6))
                    first_grid_summary = {
                        "grid_type": grid_type,
                        "data_points": total_count,
                        "latitude_min": float(lats.min()),
                        "latitude_max": float(lats.max()),
                        "longitude_min": float(lons.min()),
                        "longitude_max": float(lons.max()),
                        "unique_latitude_count": int(len(ulats)),
                        "unique_longitude_count": int(len(ulons)),
                        "latitude_head": [float(x) for x in ulats[:5]],
                        "latitude_tail": [float(x) for x in ulats[-5:]],
                        "longitude_head": [float(x) for x in ulons[:5]],
                        "longitude_tail": [float(x) for x in ulons[-5:]],
                    }
            finally:
                codes_release(gid)

    if message_count == 0 or first_grid_summary is None:
        raise RuntimeError(f"{stream}: no GRIB messages decoded")

    present = set(by_short)
    if present != expected_shortnames:
        raise RuntimeError(f"{stream}: shortName set mismatch; expected={sorted(expected_shortnames)} got={sorted(present)}")

    expected_ts = set(expected_validities)
    expected_hours = len(expected_validities)
    time_gate: dict[str, Any] = {}

    if pressure:
        levels_found = {
            int(level)
            for (short, level), count in level_pairs.items()
            if short in expected_shortnames and level is not None and count > 0
        }
        if levels_found != EXPECTED_PRESSURE_LEVELS:
            raise RuntimeError(f"{stream}: pressure levels mismatch; expected={sorted(EXPECTED_PRESSURE_LEVELS)} got={sorted(levels_found)}")
        for short in sorted(expected_shortnames):
            for level in sorted(EXPECTED_PRESSURE_LEVELS):
                key = f"{short}@{level}"
                count = level_pairs[(short, level)]
                actual = timestamps_by_key.get(key, set())
                if count != expected_hours or actual != expected_ts:
                    missing = sorted(expected_ts - actual)[:10]
                    extra = sorted(actual - expected_ts)[:10]
                    raise RuntimeError(
                        f"{stream}: hourly gate failed for {key}: messages={count}/{expected_hours}, "
                        f"unique={len(actual)}/{expected_hours}, missing_head={missing}, extra_head={extra}"
                    )
                time_gate[key] = {"message_count": count, "unique_validity_count": len(actual)}
    else:
        levels_found = set()
        for short in sorted(expected_shortnames):
            count = by_short[short]
            actual = timestamps_by_key.get(short, set())
            if count != expected_hours or actual != expected_ts:
                missing = sorted(expected_ts - actual)[:10]
                extra = sorted(actual - expected_ts)[:10]
                raise RuntimeError(
                    f"{stream}: hourly gate failed for {short}: messages={count}/{expected_hours}, "
                    f"unique={len(actual)}/{expected_hours}, missing_head={missing}, extra_head={extra}"
                )
            time_gate[short] = {"message_count": count, "unique_validity_count": len(actual)}

    # Returned geometry must contain every selected city in the region. We do not
    # demand bit-identical requested N/W/S/E because CDS/MIR can represent grid
    # endpoints with small floating differences.
    lat_min = float(first_grid_summary["latitude_min"])
    lat_max = float(first_grid_summary["latitude_max"])
    lon_min = float(first_grid_summary["longitude_min"])
    lon_max = float(first_grid_summary["longitude_max"])
    outside = [
        c.city
        for c in member_cities
        if not (lat_min - 1e-6 <= c.latitude <= lat_max + 1e-6 and lon_min - 1e-6 <= c.longitude <= lon_max + 1e-6)
    ]
    if outside:
        raise RuntimeError(f"{stream}: returned grid does not contain member cities: {outside}")

    north, west, south, east = requested_area
    # Returned edge should be close to requested snapped area (at most one grid
    # step discrepancy tolerated; larger discrepancy is a likely area/order bug).
    edge_deltas = {
        "north": abs(lat_max - north),
        "south": abs(lat_min - south),
        "west": abs(lon_min - west),
        "east": abs(lon_max - east),
    }
    if any(v > GRID_DEG + 1e-3 for v in edge_deltas.values()):
        raise RuntimeError(f"{stream}: returned bbox differs too much from requested area: {edge_deltas}")

    for key, (mn, mx) in range_acc.items():
        ranges[key] = {"min": mn, "max": mx}

    return {
        "status": "PASS",
        "stream": stream,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "message_count": message_count,
        "expected_hours": expected_hours,
        "shortname_counts": dict(sorted(by_short.items())),
        "units": {k: sorted(v) for k, v in sorted(units_by_short.items())},
        "step_types": {k: sorted(v) for k, v in sorted(step_types_by_short.items())},
        "level_types": {k: sorted(v) for k, v in sorted(level_types_by_key.items())},
        "pressure_levels_found": sorted(levels_found),
        "time_gate": time_gate,
        "requested_area_north_west_south_east": [north, west, south, east],
        "returned_grid": first_grid_summary,
        "returned_edge_deltas_deg": edge_deltas,
        "all_values_finite": all_finite,
        "ranges": ranges,
    }


# -----------------------------------------------------------------------------
# Job generation and execution
# -----------------------------------------------------------------------------


def stream_dir(root: Path, stream: str) -> Path:
    if stream == "single_instant":
        return root / "01_raw" / "era5_single_levels"
    if stream == "single_accum":
        return root / "01_raw" / "era5_single_levels_accum"
    if stream == "pressure_925_850":
        return root / "01_raw" / "era5_pressure_levels"
    raise ValueError(stream)


def make_jobs(
    root: Path,
    bounds: list[RegionBounds],
    start_month: str,
    end_month: str,
    only_regions: set[str] | None,
    only_streams: set[str] | None,
    include_dec2022_boundary: bool,
) -> list[Job]:
    sy, sm = map(int, start_month.split("-"))
    ey, em = map(int, end_month.split("-"))
    if (sy, sm) > (ey, em):
        raise ValueError("start_month > end_month")

    months: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

    if include_dec2022_boundary and (sy, sm) == (2023, 1):
        months = [(2022, 12)] + months

    jobs: list[Job] = []
    for b in sorted(bounds, key=lambda x: x.region):
        if only_regions and b.region not in only_regions:
            continue
        for year, month in months:
            expected_hours = calendar.monthrange(year, month)[1] * 24
            for stream in STREAMS:
                if only_streams and stream not in only_streams:
                    continue
                tag = f"{b.region}_{year:04d}-{month:02d}"
                target = stream_dir(root, stream) / f"{tag}_{stream}.grib"
                audit_path = root / "02_audit" / "era5_bulk" / b.region / f"{tag}_{stream}_audit.json"
                provenance_path = root / "01_raw" / "provenance" / "era5_bulk" / b.region / f"{tag}_{stream}_request.json"
                jobs.append(
                    Job(
                        region=b.region,
                        scope=("local_time_boundary" if (year, month) == (2022, 12) and include_dec2022_boundary else "core"),
                        year=year,
                        month=month,
                        stream=stream,
                        north=b.north,
                        west=b.west,
                        south=b.south,
                        east=b.east,
                        expected_hours=expected_hours,
                        target=target,
                        audit_path=audit_path,
                        provenance_path=provenance_path,
                    )
                )
    return jobs


def prior_pass_is_valid(job: Job) -> bool:
    if not job.target.exists() or not job.audit_path.exists():
        return False
    try:
        report = json.loads(job.audit_path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "PASS"
            and report.get("sha256") == sha256_file(job.target)
            and int(report.get("expected_hours", -1)) == job.expected_hours
        )
    except Exception:
        return False


def run_job(
    job: Job,
    cities_by_region: dict[str, list[City]],
    retries: int,
    timeout_seconds: int,
    overwrite: bool,
) -> dict[str, Any]:
    dataset, request = build_request(job)
    expected_validities = month_validities(job.year, job.month)
    member_cities = cities_by_region[job.region]

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "stream": job.stream,
        "region": job.region,
        "year": job.year,
        "month": job.month,
        "expected_hours": job.expected_hours,
        "request": request,
        "raw_policy": {
            "format": "GRIB",
            "timezone": "UTC",
            "grid_deg": [GRID_DEG, GRID_DEG],
            "unit_conversion": False,
            "time_zone_conversion": False,
            "interpolation": False,
            "imputation": False,
            "scaling": False,
            "derived_features": False,
        },
    }
    atomic_json(provenance, job.provenance_path)

    if not overwrite and prior_pass_is_valid(job):
        print(f"[SKIP PASS] {job.target}", flush=True)
        return {"status": "SKIP_PASS", "target": str(job.target)}

    # If a file exists but no trusted PASS audit exists, first try to audit it.
    if job.target.exists() and not overwrite:
        try:
            report = audit_grib(
                job.target,
                job.stream,
                expected_validities,
                (job.north, job.west, job.south, job.east),
                member_cities,
            )
            report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "region": job.region, "year": job.year, "month": job.month})
            atomic_json(report, job.audit_path)
            print(f"[RECOVERED PASS] {job.target}", flush=True)
            return {"status": "RECOVERED_PASS", "target": str(job.target)}
        except Exception as exc:
            print(f"[RE-DOWNLOAD] existing file failed audit: {job.target}: {exc}", file=sys.stderr, flush=True)
            job.target.unlink(missing_ok=True)

    if cdsapi is None:
        raise RuntimeError("cdsapi is required for ERA5 retrieval; activate the era5_data environment")
    client = cdsapi.Client(timeout=timeout_seconds, sleep_max=30, retry_max=20)
    retrieve_atomic(client, dataset, request, job.target, retries=retries)
    report = audit_grib(
        job.target,
        job.stream,
        expected_validities,
        (job.north, job.west, job.south, job.east),
        member_cities,
    )
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "region": job.region, "year": job.year, "month": job.month})
    atomic_json(report, job.audit_path)
    print(f"[PASS] {job.region} {job.year:04d}-{job.month:02d} {job.stream} bytes={job.target.stat().st_size:,}", flush=True)
    return {"status": "PASS", "target": str(job.target), "sha256": report["sha256"]}


def write_plan(root: Path, cities_path: Path, cities: list[City], bounds: list[RegionBounds], jobs: list[Job], halo_deg: float) -> tuple[Path, Path]:
    plan_dir = root / "05_manifests"
    bounds_path = plan_dir / f"era5_region_bounds_halo{halo_deg:g}.csv"
    job_path = plan_dir / f"era5_acquisition_jobs_halo{halo_deg:g}.csv"

    bounds_rows = [asdict(b) for b in bounds]
    atomic_csv(bounds_rows, bounds_path, list(bounds_rows[0].keys()))

    job_rows = []
    for j in jobs:
        job_rows.append({
            "region": j.region,
            "scope": j.scope,
            "year": j.year,
            "month": f"{j.month:02d}",
            "stream": j.stream,
            "north": j.north,
            "west": j.west,
            "south": j.south,
            "east": j.east,
            "expected_hours": j.expected_hours,
            "target": str(j.target),
            "audit_path": str(j.audit_path),
            "provenance_path": str(j.provenance_path),
        })
    atomic_csv(job_rows, job_path, list(job_rows[0].keys()) if job_rows else ["region"])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_manifest": str(cities_path),
        "city_manifest_sha256": sha256_file(cities_path),
        "city_count": len(cities),
        "region_count": len(bounds),
        "halo_deg": halo_deg,
        "grid_deg": GRID_DEG,
        "bounds_fingerprint": bounds_fingerprint(bounds),
        "job_count": len(jobs),
        "streams": list(STREAMS),
        "bounds_csv": str(bounds_path),
        "jobs_csv": str(job_path),
    }
    atomic_json(summary, plan_dir / f"era5_acquisition_plan_halo{halo_deg:g}.json")
    return bounds_path, job_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Formal region x month ERA5 GRIB acquisition with strict monthly audits.")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--cities", type=Path, default=None, help="Default: <root>/05_manifests/selected_cities_v2.csv")
    ap.add_argument("--halo-deg", type=float, default=3.0, help="Fixed physical acquisition halo before 0.25-degree outward snap. Default: 3.0")
    ap.add_argument("--start-month", default="2023-01")
    ap.add_argument("--end-month", default="2025-12")
    ap.add_argument("--region", action="append", default=[], help="Repeat to limit regions")
    ap.add_argument("--stream", action="append", choices=STREAMS, default=[], help="Repeat to limit streams")
    ap.add_argument("--plan-only", action="store_true", help="Write/print manifests only; do not contact CDS")
    ap.add_argument("--no-local-time-boundary", action="store_true", help="Disable extra 2022-12 boundary month used to preserve full Asia/Shanghai 2023 start")
    ap.add_argument("--max-files", type=int, default=None, help="Limit first N jobs after planning, useful for smoke tests")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent CDS requests. Keep small; default=1")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=1200)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    cities_path = (args.cities or (root / "05_manifests" / "selected_cities_v2.csv")).expanduser().resolve()

    cities = load_cities(cities_path)
    bounds = build_region_bounds(cities, args.halo_deg)
    available_regions = {b.region for b in bounds}
    only_regions = set(args.region) if args.region else None
    if only_regions:
        unknown = only_regions - available_regions
        if unknown:
            raise ValueError(f"Unknown --region values: {sorted(unknown)}; available={sorted(available_regions)}")
    only_streams = set(args.stream) if args.stream else None

    jobs = make_jobs(
        root=root,
        bounds=bounds,
        start_month=args.start_month,
        end_month=args.end_month,
        only_regions=only_regions,
        only_streams=only_streams,
        include_dec2022_boundary=not args.no_local_time_boundary,
    )
    bounds_path, job_path = write_plan(root, cities_path, cities, bounds, jobs, args.halo_deg)

    print("=" * 120)
    print("ERA5 FORMAL ACQUISITION PLAN")
    print("=" * 120)
    print(f"Root               : {root}")
    print(f"Cities             : {cities_path}")
    print(f"Cities SHA256      : {sha256_file(cities_path)}")
    print(f"Halo               : {args.halo_deg} deg")
    print(f"Grid               : {GRID_DEG} x {GRID_DEG} deg")
    print(f"Bounds fingerprint : {bounds_fingerprint(bounds)}")
    print(f"Bounds CSV         : {bounds_path}")
    print(f"Jobs CSV           : {job_path}")
    print(f"Jobs               : {len(jobs)}")
    print()
    for b in bounds:
        print(
            f"{b.region:30s} cities={b.city_count:2d} "
            f"base_lat=[{b.city_lat_min:.4f},{b.city_lat_max:.4f}] "
            f"base_lon=[{b.city_lon_min:.4f},{b.city_lon_max:.4f}] "
            f"N/W/S/E=[{b.north:.2f},{b.west:.2f},{b.south:.2f},{b.east:.2f}] "
            f"grid={b.n_lat}x{b.n_lon}={b.grid_cells:,}"
        )

    # Full formal acquisition contains 540 core 2023-2025 jobs plus, by
    # default, 15 Dec-2022 boundary jobs (5 regions x 3 streams) so that
    # UTC -> Asia/Shanghai conversion can retain local 2023-01-01 00:00-07:00.
    if args.start_month == "2023-01" and args.end_month == "2025-12" and not only_regions and not only_streams:
        core_jobs = [j for j in jobs if j.scope == "core"]
        boundary_jobs = [j for j in jobs if j.scope == "local_time_boundary"]
        if len(core_jobs) != 540:
            raise AssertionError(f"Expected 540 core jobs, got {len(core_jobs)}")
        print("\n[COUNT GATE] 5 regions x 36 months x 3 streams = 540 core jobs: PASS")
        if args.no_local_time_boundary:
            if boundary_jobs:
                raise AssertionError("Boundary jobs present despite --no-local-time-boundary")
        else:
            if len(boundary_jobs) != 15:
                raise AssertionError(f"Expected 15 Dec-2022 boundary jobs, got {len(boundary_jobs)}")
            print("[BOUNDARY GATE] 5 regions x 1 boundary month x 3 streams = 15 extra jobs: PASS")
            print("[TOTAL] 540 core + 15 boundary = 555 physical GRIB files")

    if args.plan_only:
        print("\nPLAN ONLY: no CDS requests were sent.")
        return

    run_jobs = jobs[: args.max_files] if args.max_files is not None else jobs
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 4:
        raise ValueError("--workers > 4 intentionally blocked; CDS bulk concurrency should remain conservative")

    cities_by_region = {r: sorted([c for c in cities if c.region == r], key=lambda x: x.priority) for r in available_regions}

    run_manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "cities": str(cities_path),
        "cities_sha256": sha256_file(cities_path),
        "halo_deg": args.halo_deg,
        "bounds_fingerprint": bounds_fingerprint(bounds),
        "start_month": args.start_month,
        "end_month": args.end_month,
        "include_dec2022_boundary": not args.no_local_time_boundary,
        "workers": args.workers,
        "job_count_planned": len(jobs),
        "job_count_this_run": len(run_jobs),
        "regions": sorted(only_regions or available_regions),
        "streams": sorted(only_streams or set(STREAMS)),
        "argv": sys.argv,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = root / "08_runs" / "era5_acquisition" / f"run_{run_id}.json"
    atomic_json(run_manifest, run_path)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if args.workers == 1:
        for idx, job in enumerate(run_jobs, start=1):
            print("\n" + "=" * 120)
            print(f"JOB {idx}/{len(run_jobs)}: {job.region} {job.year:04d}-{job.month:02d} {job.stream} scope={job.scope}")
            print("=" * 120)
            try:
                results.append(run_job(job, cities_by_region, args.retries, args.timeout_seconds, args.overwrite))
            except Exception as exc:
                failure = {
                    "region": job.region,
                    "scope": job.scope,
                    "year": job.year,
                    "month": job.month,
                    "stream": job.stream,
                    "target": str(job.target),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[FAIL] {failure}", file=sys.stderr, flush=True)
                break
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_to_job = {
                ex.submit(run_job, job, cities_by_region, args.retries, args.timeout_seconds, args.overwrite): job
                for job in run_jobs
            }
            for fut in as_completed(fut_to_job):
                job = fut_to_job[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    failure = {
                        "region": job.region,
                        "scope": job.scope,
                        "year": job.year,
                        "month": job.month,
                        "stream": job.stream,
                        "target": str(job.target),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(f"[FAIL] {failure}", file=sys.stderr, flush=True)

    run_manifest.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    })
    atomic_json(run_manifest, run_path)

    print("\n" + "=" * 120)
    print(f"ERA5 ACQUISITION RUN: {'PASS' if not failures else 'FAIL'}")
    print(f"Run record: {run_path}")
    print(f"Completed : {len(results)}")
    print(f"Failures  : {len(failures)}")
    print("=" * 120)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
