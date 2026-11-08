#!/usr/bin/env python3
from __future__ import annotations

"""
76 — Build Central-China external frozen inputs for whole-region holdout evaluation.

Scientific contract
-------------------
This program performs NO model training, NO checkpoint loading/selection, and NO
external fitting of preprocessing statistics.  It externalizes the already-frozen
50-city preprocessing pipeline to the 10-city Central-China holdout namespace:

    audited external ERA5 GRIB
      -> ERA5 canonical
      -> physical-derived ERA5
      -> city-level meteorology
      -> exact CNEMC + ERA5 merge
      -> split-agnostic +24 h task index
      -> frozen 72-feature issue-time bank

The physics / feature formulas are not reimplemented here.  The frozen source
modules are imported and called for the scientifically non-trivial stages.
External-specific code is limited to namespace routing, 10-city structural gates,
merge/index plumbing, and provenance/audit.

The final feature bank is RAW / unscaled.  Fold-k imputation/scaling remains the
responsibility of the later frozen-checkpoint evaluator and MUST come only from:

    06_processed/fold_artifacts/
      national_pm25_ood_split_v1/
      national_pm25_observed_feature_schema_v1/fold_k/

This script intentionally never reads those fold-local statistics.
"""

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Frozen identities
# =============================================================================

EXTERNAL_PROTOCOL_ID = "central_china_external_holdout_v1"
FEATURE_SCHEMA_ID = "national_pm25_observed_feature_schema_v1"
FEATURE_SCHEMA_PARENT_ID = "national_pm25_feature_schema_v1"
RUN_ID = "central_china_external_frozen_inputs_v1"

EXPECTED_EXTERNAL_CITIES = 10
EXPECTED_ERA5_JOBS = 111
EXPECTED_ERA5_MONTHS = 37
EXPECTED_ERA5_HOURS = 27_048
EXPECTED_MERGED_HOURS = 27_040
EXPECTED_FEATURES = 72
EXPECTED_PRESSURE_LEVELS = [925, 850]
HORIZON_HOURS = 24
LOOKBACK_HOURS = 72

TARGET_LOCAL_START = pd.Timestamp("2023-01-01 00:00:00")
TARGET_LOCAL_END = pd.Timestamp("2025-12-31 23:00:00")
EXPECTED_TARGET_HOURS = 8760 + 8784 + 8760
EXPECTED_TASK_ROWS = EXPECTED_EXTERNAL_CITIES * EXPECTED_TARGET_HOURS

# These are source-of-truth scripts from the frozen benchmark pipeline.
SOURCE_FILES: dict[str, tuple[str, ...]] = {
    # The live project uses unversioned filenames, while the internal frozen
    # build-version constants are v3/v2.  Accept both basenames, then audit
    # the imported module constants below so filename aliases cannot silently
    # change scientific preprocessing.
    "era5_canonical": (
        "02_build_era5_canonical.py",
        "02_build_era5_canonical_v3.py",
    ),
    "era5_derived": (
        "03_build_era5_derived.py",
        "03_build_era5_derived_v2.py",
    ),
    "city_meteorology": (
        "05_extract_city_meteorology.py",
    ),
    "merge_reference": (
        "06_build_observed_era5_merged_patched.py",
        "06_build_observed_era5_merged.py",
    ),
    "task_reference": (
        "07_build_pm25_24h_task_index.py",
    ),
    "feature_bank": (
        "09_build_observed_issue_time_feature_bank.py",
    ),
}

OPTIONAL_PROVENANCE_FILES = {
    "protocol_freeze": "74_freeze_central_china_external_protocol.py",
    "external_cnemc": "75a_build_central_china_external_cnemc.py",
    "external_era5_acquisition": "75b_acquire_central_china_external_era5.py",
}


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the frozen 10-city Central-China external ERA5/CNEMC merged "
            "dataset, +24 h task index, and exact 72-feature issue-time bank."
        )
    )
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing frozen pipeline scripts. Default searches: "
            "directory of this script, <root>, <root>/scripts, <root>/pipeline."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace 76-owned external outputs only. Original 50-city benchmark "
            "artifacts are never touched."
        ),
    )
    p.add_argument(
        "--skip-raw-sha",
        action="store_true",
        help=(
            "Trust PASS ERA5 acquisition audits without re-hashing all 111 GRIBs. "
            "Default is strict SHA verification."
        ),
    )
    p.add_argument(
        "--derived-chunk-hours",
        type=int,
        default=168,
        help="Chunk size for ERA5 physical derivation. Default: 168 h.",
    )
    p.add_argument(
        "--stop-after",
        choices=(
            "prerequisites",
            "canonical",
            "derived",
            "city_meteorology",
            "merged",
            "task_index",
            "feature_bank",
        ),
        default=None,
        help="Development/debug gate. Omit for full 76 PASS.",
    )
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
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


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON prerequisite: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def zarr_root_metadata_sha256(path: Path) -> str | None:
    for name in ("zarr.json", ".zmetadata", ".zgroup"):
        p = path / name
        if p.exists() and p.is_file():
            return sha256_file(p)
    return None


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_build_path(final_path: Path, overwrite: bool) -> Path:
    """Return a sibling .building path and enforce output ownership semantics."""
    building = final_path.with_name(final_path.name + ".building")
    if building.exists():
        remove_path(building)
    if final_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"76-owned output already exists: {final_path}; "
                "rerun with --overwrite only if intentional"
            )
        remove_path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return building


def finalize_build_path(building: Path, final_path: Path) -> None:
    if final_path.exists():
        raise RuntimeError(f"refusing to replace existing final path: {final_path}")
    building.replace(final_path)


def import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and some import-time reflection expect the module to be
    # registered before execution.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_search_dirs(root: Path, source_dir: Path | None) -> list[Path]:
    """Return deterministic high-priority directories for frozen source scripts."""
    search_dirs: list[Path] = []
    if source_dir is not None:
        search_dirs.append(source_dir.expanduser().resolve())
    search_dirs.extend(
        [
            Path(__file__).resolve().parent,
            root,
            # Actual frozen project layout.
            root / "07_scripts" / "data_pipeline",
            root / "07_scripts" / "external_validation",
            root / "07_scripts" / "modeling",
            # Backward-compatible fallbacks for ad-hoc checkouts.
            root / "scripts",
            root / "pipeline",
        ]
    )

    dedup: list[Path] = []
    seen: set[Path] = set()
    for d in search_dirs:
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            dedup.append(d)
    return dedup


def _recursive_exact_matches(root: Path, filename: str) -> list[Path]:
    """Find exact-basename matches below ROOT without guessing renamed scripts."""
    if not root.exists():
        return []

    # Avoid generated/output trees where copied scripts should never be treated
    # as authoritative pipeline sources.
    skip_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        # Never resolve archived or generated copies as authoritative source.
        "99_archive",
        "99achieve",
        "09_diagnostics",
    }

    matches: list[Path] = []
    for p in root.rglob(filename):
        try:
            rel = p.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if any(part in skip_parts for part in rel.parts[:-1]):
            continue
        if p.is_file():
            matches.append(p.resolve())
    return sorted(set(matches), key=lambda x: str(x))


def _resolve_one_source(
    root: Path,
    source_dir: Path | None,
    names: tuple[str, ...],
    *,
    required: bool,
    label: str,
) -> Path | None:
    # 1) Preserve the original fast / explicit lookup order.
    search_dirs = _source_search_dirs(root, source_dir)
    for d in search_dirs:
        for fn in names:
            p = d / fn
            if p.is_file():
                return p.resolve()

    # 2) Fall back to recursive exact-basename lookup anywhere under project root.
    # This is intentionally exact-name only: no fuzzy matching of scientific code.
    matches: list[Path] = []
    for fn in names:
        matches.extend(_recursive_exact_matches(root, fn))
    matches = sorted(set(matches), key=lambda x: str(x))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple frozen source scripts match {label}; refusing to choose silently. "
            f"matches={[str(x) for x in matches]}. "
            "Re-run with --source-dir pointing to the authoritative source directory."
        )

    if required:
        raise RuntimeError(
            f"cannot find frozen source script for {label}; candidates={names}; "
            f"direct_search={[str(x) for x in search_dirs]}; "
            f"also recursively searched project root={root}. "
            "If the source scripts live outside the project tree, pass --source-dir /path/to/their/directory."
        )
    return None


def resolve_source_files(root: Path, source_dir: Path | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for key, names in SOURCE_FILES.items():
        hit = _resolve_one_source(
            root,
            source_dir,
            names,
            required=True,
            label=key,
        )
        assert hit is not None
        out[key] = hit
    return out


def find_optional_source(root: Path, source_dir: Path | None, filename: str) -> Path | None:
    return _resolve_one_source(
        root,
        source_dir,
        (filename,),
        required=False,
        label=filename,
    )


def force_zarr_v2_kwargs() -> dict[str, Any]:
    params = inspect.signature(xr.Dataset.to_zarr).parameters
    if "zarr_format" in params:
        return {"zarr_format": 2}
    if "zarr_version" in params:
        return {"zarr_version": 2}
    raise RuntimeError("cannot force stable Zarr V2 output with installed xarray")


# =============================================================================
# Paths
# =============================================================================

def build_paths(root: Path) -> dict[str, Path]:
    ext03 = root / "03_canonical" / "external_validation" / EXTERNAL_PROTOCOL_ID
    ext04 = root / "04_derived" / "external_validation" / EXTERNAL_PROTOCOL_ID
    ext06 = root / "06_processed" / "external_validation" / EXTERNAL_PROTOCOL_ID
    diag = (
        root / "09_diagnostics" / "external_validation" / EXTERNAL_PROTOCOL_ID
        / "input_construction_76"
    )

    return {
        "external_manifest": root / "05_manifests" / "external_central_china_10cities_v1.csv",
        "protocol": root / "05_manifests" / "external_central_china_protocol_v1.json",
        "protocol_audit": (
            root / "09_diagnostics" / "external_validation" / EXTERNAL_PROTOCOL_ID
            / "protocol_freeze" / "summary.json"
        ),
        "cnemc_preflight": (
            root / "09_diagnostics" / "external_validation" / EXTERNAL_PROTOCOL_ID
            / "cnemc_preflight" / "summary.json"
        ),
        "cnemc_fallback": (
            root / "03_canonical" / "air_quality_cnemc_external"
            / "city_air_quality_central_china_10cities_v1.zarr"
        ),
        "era5_jobs": root / "05_manifests" / "era5_external_central_china_acquisition_jobs_halo3.csv",
        "era5_bounds": root / "05_manifests" / "era5_external_central_china_region_bounds_halo3.csv",
        "benchmark_policy": root / "05_manifests" / "benchmark_validity_policy_v1.json",
        "frozen_schema": (
            root / "05_manifests" / "features" / FEATURE_SCHEMA_ID / "feature_schema.json"
        ),
        "parent_schema": (
            root / "05_manifests" / "features" / FEATURE_SCHEMA_PARENT_ID / "feature_schema.json"
        ),
        "era5_canonical_root": ext03 / "era5",
        "era5_derived_root": ext04 / "era5_physical",
        "city_meteo": ext04 / "city_meteorology" / "city_meteorology_10cities.zarr",
        "city_meteo_report": ext04 / "city_meteorology" / "build_report.json",
        "city_meteo_mapping": ext04 / "city_meteorology" / "interpolation_mapping.csv",
        "merged": ext04 / "observed_era5_merged" / "observed_era5_10cities_v1.zarr",
        "merged_manifest": ext04 / "observed_era5_merged" / "manifest.json",
        "task": ext06 / "task_index" / "pm25_24h_task_index_external_v1.parquet",
        "task_manifest": ext06 / "task_index" / "pm25_24h_task_index_external_v1_manifest.json",
        "feature_bank": (
            ext04 / "issue_time_features" / FEATURE_SCHEMA_ID / "issue_time_feature_bank.zarr"
        ),
        "feature_report": ext04 / "issue_time_features" / FEATURE_SCHEMA_ID / "build_report.json",
        "diag": diag,
        "summary": diag / "summary.json",
    }


# =============================================================================
# Frozen prerequisite gates
# =============================================================================

def load_external_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"missing frozen external manifest: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"region", "city", "city_zh", "province", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"external city manifest missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_EXTERNAL_CITIES:
        raise RuntimeError(f"external city count={len(df)} != {EXPECTED_EXTERNAL_CITIES}")
    if df["city"].astype(str).duplicated().any():
        raise RuntimeError("external manifest contains duplicate city names")
    if df[["latitude", "longitude"]].isna().any().any():
        raise RuntimeError("external manifest contains missing coordinates")
    if df["region"].astype(str).nunique() != 1:
        raise RuntimeError(
            "76 currently expects the frozen whole-region holdout to contain exactly one region"
        )
    return df.reset_index(drop=True)


def gate_frozen_prerequisites(
    root: Path,
    p: dict[str, Path],
    ext: pd.DataFrame,
    mod02: ModuleType,
    verify_raw_sha: bool,
) -> dict[str, Any]:
    protocol = read_json(p["protocol"])
    pa = read_json(p["protocol_audit"])
    if protocol.get("protocol_id") != EXTERNAL_PROTOCOL_ID:
        raise RuntimeError("external protocol identity mismatch")
    if pa.get("status") != "PASS":
        raise RuntimeError("74 protocol freeze did not PASS")

    frozen_order = protocol.get("external_city_order")
    if frozen_order is None:
        # Older protocol payloads may store it only in the audit summary.
        frozen_order = pa.get("external_city_order")
    manifest_order = ext["city"].astype(str).tolist()
    if frozen_order is not None and list(frozen_order) != manifest_order:
        raise RuntimeError(
            "external manifest row order differs from the frozen protocol city order"
        )

    cnemc = read_json(p["cnemc_preflight"])
    if cnemc.get("status") != "PASS":
        raise RuntimeError("75a CNEMC preflight/canonical did not PASS")
    if int(cnemc.get("cities", -1)) != EXPECTED_EXTERNAL_CITIES:
        raise RuntimeError("75a city count mismatch")
    cnemc_rel = cnemc.get("canonical")
    aq_path = (root / cnemc_rel).resolve() if cnemc_rel else p["cnemc_fallback"]
    if not aq_path.exists():
        raise RuntimeError(f"external CNEMC canonical missing: {aq_path}")

    jobs = mod02.load_jobs(p["era5_jobs"])
    bounds = mod02.load_bounds(p["era5_bounds"])
    if len(jobs) != EXPECTED_ERA5_JOBS:
        raise RuntimeError(f"external ERA5 jobs={len(jobs)} != {EXPECTED_ERA5_JOBS}")
    regions = sorted({j.region for j in jobs})
    if len(regions) != 1:
        raise RuntimeError(f"expected one external ERA5 region, got {regions}")
    region = regions[0]
    if region not in bounds or len(bounds) != 1:
        raise RuntimeError("external ERA5 bounds/job region identity mismatch")

    months = sorted({(j.year, j.month) for j in jobs})
    if len(months) != EXPECTED_ERA5_MONTHS:
        raise RuntimeError(
            f"external ERA5 distinct months={len(months)} != {EXPECTED_ERA5_MONTHS}"
        )
    if months[0] != (2022, 12) or months[-1] != (2025, 12):
        raise RuntimeError(f"unexpected external ERA5 month extent: {months[0]}..{months[-1]}")
    streams = sorted({j.stream for j in jobs})
    if set(streams) != set(mod02.STREAMS):
        raise RuntimeError(f"external ERA5 stream set mismatch: {streams}")
    for ym in months:
        n = sum((j.year, j.month) == ym for j in jobs)
        if n != 3:
            raise RuntimeError(f"external ERA5 {ym}: expected 3 streams, got {n}")

    # 75b closure gate.  We verify exactly the files referenced by the frozen
    # external jobs manifest.  This does not read any model output.
    for i, job in enumerate(jobs, start=1):
        mod02.verify_raw_against_audit(
            root=root,
            j=job,
            verify_sha=verify_raw_sha,
        )
        if i % 20 == 0 or i == len(jobs):
            print(f"  ERA5 raw audit gate: {i}/{len(jobs)}", flush=True)

    policy = read_json(p["benchmark_policy"])
    if policy.get("status") != "FROZEN":
        raise RuntimeError("benchmark validity policy is not FROZEN")
    if int(policy["task"]["forecast_horizon_hours"]) != HORIZON_HOURS:
        raise RuntimeError("benchmark policy horizon mismatch")

    schema = read_json(p["frozen_schema"])
    feature_order = list(schema.get("feature_order", []))
    if len(feature_order) != EXPECTED_FEATURES or len(set(feature_order)) != EXPECTED_FEATURES:
        raise RuntimeError("frozen observed feature schema is not exactly 72 unique features")

    return {
        "protocol": protocol,
        "protocol_audit": pa,
        "cnemc_preflight": cnemc,
        "aq_path": aq_path,
        "jobs": jobs,
        "bounds": bounds,
        "region": region,
        "months": months,
        "frozen_schema": schema,
    }


# =============================================================================
# Stage A — audited raw ERA5 -> external canonical
# =============================================================================

def build_external_era5_canonical(
    *,
    root: Path,
    out_root: Path,
    region: str,
    months: list[tuple[int, int]],
    jobs: list[Any],
    bounds_map: dict[str, Any],
    mod02: ModuleType,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / f"{region}.zarr"
    report_path = out_root / f"{region}_build_report.json"

    if final.exists() and report_path.exists() and not overwrite:
        report = read_json(report_path)
        if report.get("status") != "PASS":
            raise RuntimeError(f"existing canonical report is not PASS: {report_path}")
        gate = mod02.final_region_gate(
            final,
            region,
            [mod02.month_key(y, m) for y, m in months],
            bounds_map[region],
        )
        return final, {**report, "revalidated": gate}

    building = prepare_build_path(final, overwrite=overwrite)
    if report_path.exists() and overwrite:
        report_path.unlink()

    jobs_by_key = {(j.region, j.year, j.month, j.stream): j for j in jobs}
    if len(jobs_by_key) != len(jobs):
        raise RuntimeError("duplicate keys in external ERA5 jobs")

    completed: list[str] = []
    try:
        for idx, (year, month) in enumerate(months, start=1):
            mk = mod02.month_key(year, month)
            print(f"  canonical {idx:02d}/{len(months)} {region} {mk}", flush=True)
            month_ds, _ = mod02.build_month_dataset(
                root=root,
                region=region,
                year=year,
                month=month,
                jobs_by_key=jobs_by_key,
                bounds=bounds_map[region],
                # Strict SHA was already performed once in prerequisite gate.
                verify_sha=False,
            )
            mod02.append_month_to_zarr(
                ds=month_ds,
                store=building,
                first=not building.exists(),
            )
            month_ds.close()
            completed.append(mk)

        gate = mod02.final_region_gate(
            building,
            region,
            completed,
            bounds_map[region],
        )
        if gate["months"] != EXPECTED_ERA5_MONTHS or gate["hours"] != EXPECTED_ERA5_HOURS:
            raise RuntimeError("external ERA5 canonical full-archive gate failed")
        mod02.zarr.consolidate_metadata(str(building))
        finalize_build_path(building, final)

        report = {
            **gate,
            "status": "PASS",
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "external_namespace": True,
            "source_builder": str(Path(mod02.__file__).resolve()),
            "source_builder_sha256": sha256_file(Path(mod02.__file__).resolve()),
            "data_dependent_fit_performed": False,
        }
        atomic_json(report, report_path)
        return final, report
    except Exception:
        # Leave no apparently valid final artifact after a failed stage.
        if building.exists():
            remove_path(building)
        raise


# =============================================================================
# Stage B — canonical ERA5 -> exact physical-derived ERA5
# =============================================================================

def build_external_era5_derived(
    *,
    canonical_path: Path,
    out_root: Path,
    region: str,
    mod02: ModuleType,
    mod03: ModuleType,
    overwrite: bool,
    chunk_hours: int,
) -> tuple[Path, dict[str, Any]]:
    if chunk_hours < 1:
        raise ValueError("derived chunk hours must be >= 1")
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / f"{region}.zarr"
    report_path = out_root / f"{region}_build_report.json"
    formula_path = out_root / "formula_manifest.json"

    canonical = xr.open_zarr(canonical_path, consolidated=True, chunks=None)
    try:
        mod03._assert_exact_canonical_schema(canonical, region)

        if final.exists() and report_path.exists() and not overwrite:
            old = read_json(report_path)
            if old.get("status") != "PASS":
                raise RuntimeError(f"existing derived report is not PASS: {report_path}")
            # Recompute ranges only if necessary would be expensive.  The report
            # already records the exact ranges used by source final_gate.
            gate = mod03.final_gate(canonical, final, region, old["ranges"])
            return final, {**old, "revalidated": gate}

        building = prepare_build_path(final, overwrite=overwrite)
        if report_path.exists() and overwrite:
            report_path.unlink()

        formula_payload = {
            "derived_version": mod03.DERIVED_VERSION,
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "source_builder_sha256": sha256_file(Path(mod03.__file__).resolve()),
            "constants": {
                "kelvin_offset": mod03.KELVIN_OFFSET,
                "pa_per_hpa": mod03.PA_PER_HPA,
                "mm_per_m": mod03.MM_PER_M,
                "seconds_per_hour": mod03.SECONDS_PER_HOUR,
                "percent_per_fraction": mod03.PERCENT_PER_FRACTION,
                "standard_gravity_ms2": mod03.STANDARD_GRAVITY_MS2,
                "calm_wind_epsilon_ms": mod03.CALM_WIND_EPS_MS,
            },
            "policy": {
                "data_dependent_transformations": "none",
                "spatial_interpolation": "none",
                "daily_aggregation": "none",
                "time_resolution": "hourly",
                "timezone": "UTC",
                "pressure_contrast_sign": "850 minus 925 hPa",
                "pressure_level_validity_rule": "surface_pressure_hpa >= pressure_level_hpa",
                "below_ground_policy": (
                    "retain numerical values and expose masks; no NaN/removal/"
                    "clipping/substitution in the derived layer"
                ),
            },
            "variables": mod03.FORMULA_MANIFEST,
        }
        atomic_json(formula_payload, formula_path)

        ranges: dict[str, dict[str, float]] = {}
        try:
            n = int(canonical.sizes["time"])
            for start in range(0, n, chunk_hours):
                stop = min(start + chunk_hours, n)
                block = canonical.isel(time=slice(start, stop)).load()
                derived = mod03.build_derived_chunk(block)
                mod03.assert_chunk_finite(
                    derived,
                    f"{region} external hours {start}:{stop}",
                )
                mod03.update_ranges(ranges, derived)
                mod03.append_to_zarr(
                    derived,
                    building,
                    first=not building.exists(),
                    chunk_hours=chunk_hours,
                )
                derived.close()
                block.close()
                if stop == n or stop % (24 * 28) == 0:
                    print(f"  derived {stop:,}/{n:,} h", flush=True)

            gate = mod03.final_gate(canonical, building, region, ranges)
            mod02.zarr.consolidate_metadata(str(building))
            finalize_build_path(building, final)

            report = {
                **gate,
                "status": "PASS",
                "external_protocol_id": EXTERNAL_PROTOCOL_ID,
                "canonical_input": str(canonical_path),
                "canonical_root_metadata_sha256": zarr_root_metadata_sha256(canonical_path),
                "source_builder": str(Path(mod03.__file__).resolve()),
                "source_builder_sha256": sha256_file(Path(mod03.__file__).resolve()),
                "formula_manifest": str(formula_path),
                "formula_manifest_sha256": sha256_file(formula_path),
                "data_dependent_fit_performed": False,
            }
            atomic_json(report, report_path)
            return final, report
        except Exception:
            if building.exists():
                remove_path(building)
            raise
    finally:
        canonical.close()


# =============================================================================
# Stage C — physical ERA5 grid -> exact 10-city meteorology
# =============================================================================

def external_city_meteorology_gate(
    ds: xr.Dataset,
    manifest_rows: list[dict[str, Any]],
    mod05: ModuleType,
) -> dict[str, Any]:
    mod05.ensure_exact_timeline(ds["time"].values, "external city meteorology")
    if int(ds.sizes.get("city", -1)) != EXPECTED_EXTERNAL_CITIES:
        raise RuntimeError("external city meteorology city-count mismatch")
    city_order = [str(x) for x in ds["city"].values.tolist()]
    expected_order = [str(r["city"]) for r in manifest_rows]
    if city_order != expected_order:
        raise RuntimeError("external city meteorology city order differs from frozen manifest")
    levels = [int(x) for x in ds["pressure_level_hpa"].values.tolist()]
    if levels != EXPECTED_PRESSURE_LEVELS:
        raise RuntimeError("external city meteorology pressure-level order mismatch")

    nonfinite: list[dict[str, Any]] = []
    for name, da in ds.data_vars.items():
        a = np.asarray(da.values)
        if np.issubdtype(a.dtype, np.number):
            bad = int(np.count_nonzero(~np.isfinite(a)))
            if bad:
                nonfinite.append({"variable": name, "nonfinite_count": bad})
    if nonfinite:
        raise RuntimeError(f"non-finite external meteorology output: {nonfinite[:8]}")

    i925 = levels.index(925)
    i850 = levels.index(850)
    pcount = np.asarray(ds["pressure_level_valid_corner_count"].values)
    prenorm = np.asarray(ds["pressure_level_interpolation_renormalized_mask"].values)
    pair_count = np.asarray(ds["pressure_pair_850_925_valid_corner_count"].values)
    pair_renorm = np.asarray(
        ds["pressure_pair_850_925_interpolation_renormalized_mask"].values
    )

    if np.any(prenorm[:, :, i850] != 0):
        raise RuntimeError(
            "external 850 hPa support requires renormalization; frozen extraction "
            "assumption violated — stop before model inference"
        )
    if not np.array_equal(pair_count, pcount[:, :, i925]):
        raise RuntimeError("external pair valid-corner count != 925 hPa count")
    if not np.array_equal(pair_renorm, prenorm[:, :, i925]):
        raise RuntimeError("external pair renormalization mask != 925 hPa mask")

    u10 = np.asarray(ds["u_wind_10m_ms"].values)
    v10 = np.asarray(ds["v_wind_10m_ms"].values)
    speed10, dir10 = mod05.wind_speed_direction(u10, v10)
    speed10_err = float(
        np.max(np.abs(np.asarray(ds["wind_speed_10m_ms"].values) - speed10))
    )
    dir10_err = float(
        np.max(
            mod05.circular_abs_diff_deg(
                np.asarray(ds["wind_direction_10m_deg"].values), dir10
            )
        )
    )

    up = np.asarray(ds["u_wind_pressure_ms"].values)
    vp = np.asarray(ds["v_wind_pressure_ms"].values)
    wsp, wdir = mod05.wind_speed_direction(up, vp)
    wsp_err = float(
        np.max(np.abs(np.asarray(ds["wind_speed_pressure_ms"].values) - wsp))
    )
    wdir_err = float(
        np.max(
            mod05.circular_abs_diff_deg(
                np.asarray(ds["wind_direction_pressure_deg"].values), wdir
            )
        )
    )
    if max(speed10_err, wsp_err, dir10_err, wdir_err) > 1e-10:
        raise RuntimeError("external city-level wind formula invariant failed")

    affected_idx = np.where(prenorm[:, :, i925].any(axis=0))[0]
    pweight = np.asarray(ds["pressure_level_valid_weight_sum"].values)
    pair_weight = np.asarray(ds["pressure_pair_850_925_valid_weight_sum"].values)
    return {
        "status": "PASS",
        "hours": int(ds.sizes["time"]),
        "cities": EXPECTED_EXTERNAL_CITIES,
        "pressure_levels_hpa": levels,
        "output_variable_count": len(ds.data_vars),
        "850_renormalized_samples": int(prenorm[:, :, i850].sum()),
        "925_renormalized_samples": int(prenorm[:, :, i925].sum()),
        "925_affected_cities": [city_order[i] for i in affected_idx],
        "925_min_valid_weight_sum": float(pweight[:, :, i925].min()),
        "925_min_valid_corner_count": int(pcount[:, :, i925].min()),
        "pair_renormalized_samples": int(pair_renorm.sum()),
        "pair_min_valid_weight_sum": float(pair_weight.min()),
        "pair_min_valid_corner_count": int(pair_count.min()),
        "wind_speed_10m_max_formula_error": speed10_err,
        "wind_direction_10m_max_circular_error_deg": dir10_err,
        "wind_speed_pressure_max_formula_error": wsp_err,
        "wind_direction_pressure_max_circular_error_deg": wdir_err,
    }


def build_external_city_meteorology(
    *,
    derived_path: Path,
    final: Path,
    report_path: Path,
    mapping_path: Path,
    ext: pd.DataFrame,
    mod02: ModuleType,
    mod05: ModuleType,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for r in ext.to_dict("records"):
        manifest_rows.append(
            {
                **r,
                "city": str(r["city"]),
                "city_zh": str(r["city_zh"]),
                "region": str(r["region"]),
                "province": str(r["province"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            }
        )

    if final.exists() and report_path.exists() and not overwrite:
        old = read_json(report_path)
        if old.get("status") != "PASS":
            raise RuntimeError("existing external city meteorology report is not PASS")
        ds = xr.open_zarr(final, consolidated=True, chunks=None)
        try:
            gate = external_city_meteorology_gate(ds, manifest_rows, mod05)
        finally:
            ds.close()
        return final, {**old, "revalidated": gate}

    building = prepare_build_path(final, overwrite=overwrite)
    if report_path.exists() and overwrite:
        report_path.unlink()
    if mapping_path.exists() and overwrite:
        mapping_path.unlink()

    source = xr.open_zarr(derived_path, consolidated=True, chunks=None)
    try:
        mod05.ensure_exact_timeline(source["time"].values, "external ERA5 physical derived")
        outputs = mod05.allocate_outputs(EXPECTED_EXTERNAL_CITIES)
        mappings: list[dict[str, Any]] = []

        for ci, row in enumerate(manifest_rows):
            print(
                f"  city meteorology {ci + 1:02d}/{EXPECTED_EXTERNAL_CITIES}: {row['city']}",
                flush=True,
            )
            city_out, mapping = mod05.extract_city(source, row)
            mod05.assign_city_output(outputs, ci, city_out)
            mapping = {**mapping, "province": row["province"], "region": row["region"]}
            mappings.append(mapping)

        # build_dataset uses the exact source implementation of variable semantics
        # and attrs.  We only patch external provenance / province metadata.
        ds_out = mod05.build_dataset(
            outputs,
            manifest_rows,
            source,
            report_path,
        )
        ds_out.attrs.update(
            {
                "external_protocol_id": EXTERNAL_PROTOCOL_ID,
                "external_holdout": "true",
                "source_root": str(derived_path),
                "city_manifest": "05_manifests/external_central_china_10cities_v1.csv",
                "source_builder_sha256": sha256_file(Path(mod05.__file__).resolve()),
            }
        )
        ds_out = ds_out.assign_coords(
            province=(
                "city",
                np.asarray([r["province"] for r in manifest_rows], dtype=str),
            )
        )

        gate = external_city_meteorology_gate(ds_out, manifest_rows, mod05)
        for name in ds_out.variables:
            ds_out[name].encoding = {}
        ds_out.to_zarr(
            str(building),
            mode="w",
            consolidated=False,
            encoding=mod05.zarr_encoding(ds_out),
            **force_zarr_v2_kwargs(),
        )
        mod02.zarr.consolidate_metadata(str(building))
        ds_out.close()
        finalize_build_path(building, final)

        atomic_csv(pd.DataFrame(mappings), mapping_path)
        report = {
            **gate,
            "status": "PASS",
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "source": str(derived_path),
            "output": str(final),
            "mapping": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "source_builder": str(Path(mod05.__file__).resolve()),
            "source_builder_sha256": sha256_file(Path(mod05.__file__).resolve()),
            "data_dependent_fit_performed": False,
            "frozen_policy_violation": False,
        }
        atomic_json(report, report_path)
        return final, report
    except Exception:
        if building.exists():
            remove_path(building)
        raise
    finally:
        source.close()


# =============================================================================
# Stage D — exact CNEMC + ERA5 merge
# =============================================================================

def build_external_merge(
    *,
    root: Path,
    aq_path: Path,
    meteo_path: Path,
    out_path: Path,
    manifest_path: Path,
    mod02: ModuleType,
    mod06: ModuleType,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    if out_path.exists() and manifest_path.exists() and not overwrite:
        old = read_json(manifest_path)
        if old.get("status") != "PASS":
            raise RuntimeError("existing external merge manifest is not PASS")
        ds = xr.open_zarr(out_path, consolidated=True, chunks=None)
        try:
            if int(ds.sizes.get("time", -1)) != EXPECTED_MERGED_HOURS:
                raise RuntimeError("existing external merged time length mismatch")
            if int(ds.sizes.get("city", -1)) != EXPECTED_EXTERNAL_CITIES:
                raise RuntimeError("existing external merged city count mismatch")
        finally:
            ds.close()
        return out_path, old

    building = prepare_build_path(out_path, overwrite=overwrite)
    if manifest_path.exists() and overwrite:
        manifest_path.unlink()

    aq = xr.open_zarr(aq_path, consolidated=True, chunks=None)
    met = xr.open_zarr(meteo_path, consolidated=True, chunks=None)
    try:
        for label, ds in (("AQ", aq), ("METEO", met)):
            if "time" not in ds.dims or "city" not in ds.dims:
                raise RuntimeError(f"{label} must contain time and city dimensions")
        if "city_en" not in aq.variables:
            raise RuntimeError("external CNEMC canonical lacks city_en")

        aq_city = mod06.as_str_list(aq["city_en"].values)
        # External city meteorology deliberately uses the string city coordinate.
        met_city = [str(x) for x in np.asarray(met["city"].values).tolist()]
        if len(aq_city) != EXPECTED_EXTERNAL_CITIES or len(met_city) != EXPECTED_EXTERNAL_CITIES:
            raise RuntimeError("external AQ/meteo city count mismatch")
        if len(set(aq_city)) != len(aq_city) or len(set(met_city)) != len(met_city):
            raise RuntimeError("duplicate city labels in external merge inputs")
        if set(aq_city) != set(met_city):
            raise RuntimeError(
                f"external AQ/meteo city sets differ; "
                f"AQ-only={sorted(set(aq_city) - set(met_city))}, "
                f"meteo-only={sorted(set(met_city) - set(aq_city))}"
            )

        met_pos = {c: i for i, c in enumerate(met_city)}
        met_order = [met_pos[c] for c in aq_city]

        aq_time = pd.DatetimeIndex(aq["time"].values)
        met_time = pd.DatetimeIndex(met["time"].values)
        common = aq_time.intersection(met_time)
        if len(common) != EXPECTED_MERGED_HOURS:
            raise RuntimeError(
                f"external AQ/ERA5 UTC intersection={len(common)} != {EXPECTED_MERGED_HOURS}"
            )
        if common[0] != pd.Timestamp("2022-12-01 00:00:00"):
            raise RuntimeError(f"unexpected external merge start={common[0]}")
        if common[-1] != pd.Timestamp("2025-12-31 15:00:00"):
            raise RuntimeError(f"unexpected external merge end={common[-1]}")

        aq_sel = aq.sel(time=common)
        met_sel = met.sel(time=common).isel(city=met_order)
        met_sel = met_sel.assign_coords(
            time=aq_sel["time"].values,
            city=aq_sel["city"].values,
        )

        drop_from_meteo = [
            x
            for x in (
                "city_en",
                "city_zh",
                "region",
                "province",
                "latitude",
                "longitude",
                "local_time",
                "local_year",
                "local_month",
                "local_hour",
            )
            if x in met_sel.variables
        ]
        met_clean = met_sel.drop_vars(drop_from_meteo)
        collisions = sorted(set(aq_sel.data_vars) & set(met_clean.data_vars))
        rename_map = {name: f"meteo_{name}" for name in collisions}
        if rename_map:
            met_clean = met_clean.rename(rename_map)

        merged = xr.merge(
            [aq_sel, met_clean],
            compat="override",
            join="exact",
            combine_attrs="drop_conflicts",
        )
        merged.attrs.update(
            {
                "build_id": "external_observed_era5_merged_v1",
                "source_reference_build_id": getattr(mod06, "BUILD_ID", None),
                "created_at_utc": utc_now(),
                "external_protocol_id": EXTERNAL_PROTOCOL_ID,
                "air_quality_source": "CNEMC national monitoring network; QuotSoft archive",
                "meteorology_source": "ERA5",
                "time_coordinate": "UTC",
                "protocol_year_coordinate": "local_year (Asia/Shanghai)",
                "merge_rule": "exact UTC-hour intersection + exact city_en identity alignment",
                "forecast_target_definition": (
                    "not materialized here; downstream task is fixed 24h-ahead "
                    "pm25_city_mean with target-hour validity applied separately"
                ),
                "target_leakage_note": (
                    "contemporaneous observed AQ and ERA5 predictors only; "
                    "no future-shifted target variable is created"
                ),
                "canonical_values_dropped_by_validity_gate": "false",
                "external_data_used_for_model_selection": "false",
            }
        )

        local = pd.DatetimeIndex(merged["local_time"].values)
        utc = pd.DatetimeIndex(merged["time"].values)
        if not np.all((local - utc) == pd.Timedelta(hours=8)):
            raise RuntimeError("external merged local_time != UTC + 8h")
        if [str(x) for x in merged["city_en"].values.tolist()] != aq_city:
            raise RuntimeError("external merged city order differs from CNEMC canonical")

        for name in merged.variables:
            merged[name].encoding = {}
        encoding: dict[str, Any] = {}
        for name, da in merged.data_vars.items():
            if da.dims == ("time", "city") and np.issubdtype(da.dtype, np.number):
                encoding[name] = {"chunks": (744, EXPECTED_EXTERNAL_CITIES)}
        merged.to_zarr(
            str(building),
            mode="w",
            consolidated=False,
            encoding=encoding,
            **force_zarr_v2_kwargs(),
        )
        mod02.zarr.consolidate_metadata(str(building))
        merged.close()
        finalize_build_path(building, out_path)

        manifest = {
            "status": "PASS",
            "build_id": "external_observed_era5_merged_v1",
            "created_at_utc": utc_now(),
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "output": str(out_path),
            "dimensions": {"time": EXPECTED_MERGED_HOURS, "city": EXPECTED_EXTERNAL_CITIES},
            "utc_start": str(common[0]),
            "utc_end": str(common[-1]),
            "city_order": aq_city,
            "air_quality_input": str(aq_path),
            "meteorology_input": str(meteo_path),
            "air_quality_root_metadata_sha256": zarr_root_metadata_sha256(aq_path),
            "meteorology_root_metadata_sha256": zarr_root_metadata_sha256(meteo_path),
            "source_reference": str(Path(mod06.__file__).resolve()),
            "source_reference_sha256": sha256_file(Path(mod06.__file__).resolve()),
            "meteorology_collision_renames": rename_map,
            "important": {
                "time_alignment": "exact UTC-hour intersection",
                "city_alignment": "city_en identity, explicitly reordered",
                "aq_only_boundary_hours_removed": 8,
                "meteo_only_boundary_hours_removed": 8,
                "future_shifted_target_created": False,
                "validity_gate_applied_destructively": False,
                "external_data_used_for_model_selection": False,
            },
        }
        atomic_json(manifest, manifest_path)
        return out_path, manifest
    except Exception:
        if building.exists():
            remove_path(building)
        raise
    finally:
        aq.close()
        met.close()


# =============================================================================
# Stage E — split-agnostic external +24 h task index
# =============================================================================

def build_external_task_index(
    *,
    merged_path: Path,
    out_path: Path,
    manifest_path: Path,
    policy_path: Path,
    source07: Path,
    overwrite: bool,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    if out_path.exists() and manifest_path.exists() and not overwrite:
        old = read_json(manifest_path)
        task = pd.read_parquet(out_path)
        if old.get("status") != "PASS" or len(task) != EXPECTED_TASK_ROWS:
            raise RuntimeError("existing external task index did not pass structural gate")
        return out_path, old, task

    if out_path.exists() and overwrite:
        out_path.unlink()
    if manifest_path.exists() and overwrite:
        manifest_path.unlink()

    policy = read_json(policy_path)
    if policy.get("status") != "FROZEN":
        raise RuntimeError("benchmark validity policy is not frozen")
    if int(policy["task"]["forecast_horizon_hours"]) != HORIZON_HOURS:
        raise RuntimeError("benchmark validity policy horizon mismatch")

    ds = xr.open_zarr(merged_path, consolidated=True, chunks=None)
    try:
        utc = pd.DatetimeIndex(ds["time"].values)
        local = pd.DatetimeIndex(ds["local_time"].values)
        if not np.all((local - utc) == pd.Timedelta(hours=8)):
            raise RuntimeError("external task source local/UTC mismatch")

        target_mask = (local >= TARGET_LOCAL_START) & (local <= TARGET_LOCAL_END)
        target_pos = np.flatnonzero(target_mask)
        if len(target_pos) != EXPECTED_TARGET_HOURS:
            raise RuntimeError(
                f"external target hours={len(target_pos)} != {EXPECTED_TARGET_HOURS}"
            )

        issue_utc = utc[target_pos] - pd.Timedelta(hours=HORIZON_HOURS)
        issue_local = local[target_pos] - pd.Timedelta(hours=HORIZON_HOURS)
        utc_to_pos = {ts: i for i, ts in enumerate(utc)}
        issue_pos = np.asarray([utc_to_pos.get(ts, -1) for ts in issue_utc], dtype=np.int32)
        if np.any(issue_pos < 0):
            bad = issue_utc[issue_pos < 0]
            raise RuntimeError(f"external issue time missing from merged store; first={bad[0]}")
        if int(issue_pos.min()) < LOOKBACK_HOURS - 1:
            raise RuntimeError("external earliest issue lacks full 72 h structural history")

        city_en = [str(x) for x in ds["city_en"].values.tolist()]
        city_zh = [str(x) for x in ds["city_zh"].values.tolist()]
        region = [str(x) for x in ds["region"].values.tolist()]
        province = (
            [str(x) for x in ds["province"].values.tolist()]
            if "province" in ds.variables
            else [""] * len(city_en)
        )
        if len(city_en) != EXPECTED_EXTERNAL_CITIES:
            raise RuntimeError("external task source city count mismatch")

        # Deliberately read only the frozen target-validity mask, not target magnitude.
        valid = np.asarray(
            ds["pm25_valid_hj_q95"].isel(time=target_pos).values,
            dtype=bool,
        )

        rows: list[dict[str, Any]] = []
        external_task_row = 0
        for h, tp in enumerate(target_pos):
            t_utc = utc[tp]
            t_local = local[tp]
            i_utc = issue_utc[h]
            i_local = issue_local[h]
            ip = int(issue_pos[h])
            scenario = "External-Spatial" if int(t_local.year) in (2023, 2024) else "External-Joint"
            for ci in range(EXPECTED_EXTERNAL_CITIES):
                rows.append(
                    {
                        "external_task_row": external_task_row,
                        "sample_id": f"{t_local:%Y%m%d%H}_{city_en[ci]}",
                        "city_index": ci,
                        "city_en": city_en[ci],
                        "city_zh": city_zh[ci],
                        "province": province[ci],
                        "region": region[ci],
                        "external_scenario": scenario,
                        "issue_time_index": ip,
                        "target_time_index": int(tp),
                        "issue_time_utc": i_utc,
                        "target_time_utc": t_utc,
                        "issue_time_local": i_local,
                        "target_time_local": t_local,
                        "target_local_year": int(t_local.year),
                        "target_local_month": int(t_local.month),
                        "target_local_hour": int(t_local.hour),
                        "forecast_horizon_hours": HORIZON_HOURS,
                        "target_valid_benchmark": bool(valid[h, ci]),
                    }
                )
                external_task_row += 1

        task = pd.DataFrame(rows)
        if len(task) != EXPECTED_TASK_ROWS:
            raise RuntimeError(f"external task rows={len(task)} != {EXPECTED_TASK_ROWS}")
        if not np.array_equal(
            task["external_task_row"].to_numpy(np.int64),
            np.arange(len(task), dtype=np.int64),
        ):
            raise RuntimeError("external_task_row is not exact row identity")
        if task["sample_id"].duplicated().any():
            raise RuntimeError("external task sample_id is not unique")
        if not np.array_equal(
            task["target_time_index"].to_numpy(np.int64),
            task["issue_time_index"].to_numpy(np.int64) + HORIZON_HOURS,
        ):
            raise RuntimeError("external target_time_index != issue_time_index + 24")

        atomic_parquet(task, out_path)
        counts: dict[str, Any] = {}
        for scenario, g in task.groupby("external_scenario", sort=False):
            counts[scenario] = {
                "all_structural_samples": int(len(g)),
                "benchmark_valid_samples": int(g["target_valid_benchmark"].sum()),
                "retention_fraction": float(g["target_valid_benchmark"].mean()),
                "years": sorted(g["target_local_year"].unique().astype(int).tolist()),
            }

        manifest = {
            "status": "PASS",
            "index_id": "pm25_24h_task_index_external_v1",
            "created_at_utc": utc_now(),
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "output": str(out_path),
            "rows": int(len(task)),
            "cities": EXPECTED_EXTERNAL_CITIES,
            "forecast_horizon_hours": HORIZON_HOURS,
            "target_local_start": str(TARGET_LOCAL_START),
            "target_local_end": str(TARGET_LOCAL_END),
            "target_validity_variable": "pm25_valid_hj_q95",
            "target_value_materialized_in_index": False,
            "counts_by_external_scenario": counts,
            "important": {
                "issue_time_exactly_target_minus_24h": True,
                "all_issue_times_exist_in_merged_store": True,
                "all_issue_times_have_72h_structural_history": True,
                "external_target_magnitudes_read_for_index": False,
                "external_data_used_for_model_selection": False,
                "task_index_is_split_agnostic": True,
                "external_temporal_scenario_defined": False,
            },
            "provenance": {
                "merged_store": str(merged_path),
                "benchmark_policy": str(policy_path),
                "benchmark_policy_sha256": sha256_file(policy_path),
                "source_reference": str(source07),
                "source_reference_sha256": sha256_file(source07),
            },
        }
        atomic_json(manifest, manifest_path)
        return out_path, manifest, task
    finally:
        ds.close()


# =============================================================================
# Stage F — exact frozen 72-feature issue-time bank
# =============================================================================

def build_external_feature_bank(
    *,
    merged_path: Path,
    task_path: Path,
    final: Path,
    report_path: Path,
    frozen_schema_path: Path,
    parent_schema_path: Path,
    mod02: ModuleType,
    mod09: ModuleType,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    frozen_schema = read_json(frozen_schema_path)
    order = list(frozen_schema["feature_order"])
    if len(order) != EXPECTED_FEATURES:
        raise RuntimeError("frozen feature schema is not 72-dimensional")

    # Strong gate: current source builder must still reconstruct exactly the same
    # frozen schema identity.  This catches accidental edits before data build.
    parent_schema = read_json(parent_schema_path) if parent_schema_path.exists() else None
    rebuilt_schema = mod09.build_schema(parent_schema, parent_schema_path)
    # Do not compare the regenerated schema fingerprint directly: the schema
    # payload records parent_schema_path, so relocating the project directory
    # can change that hash without changing any scientific definition.  Compare
    # the scientific identity fields exactly, while the authoritative frozen
    # fingerprint remains the one already stored in frozen_schema_path.
    if list(rebuilt_schema["feature_order"]) != order:
        raise RuntimeError("09 source builder feature order differs from frozen schema")
    if rebuilt_schema.get("feature_records") != frozen_schema.get("feature_records"):
        raise RuntimeError("09 source builder feature records differ from frozen schema")
    if rebuilt_schema.get("families") != frozen_schema.get("families"):
        raise RuntimeError("09 source builder family definitions differ from frozen schema")

    if final.exists() and report_path.exists() and not overwrite:
        old = read_json(report_path)
        if old.get("status") != "PASS":
            raise RuntimeError("existing external feature bank report is not PASS")
        ds = xr.open_zarr(final, consolidated=True, chunks=None)
        try:
            if set(ds.data_vars) < set(order):
                raise RuntimeError("existing external feature bank misses frozen features")
            for name in order:
                if ds[name].shape != (EXPECTED_MERGED_HOURS, EXPECTED_EXTERNAL_CITIES):
                    raise RuntimeError(f"existing feature shape mismatch: {name}={ds[name].shape}")
        finally:
            ds.close()
        return final, old

    building = prepare_build_path(final, overwrite=overwrite)
    if report_path.exists() and overwrite:
        report_path.unlink()

    ds = xr.open_zarr(merged_path, consolidated=True, chunks=None)
    try:
        # Reuse source validation by changing only its cardinality constant for
        # this external invocation.  All variable / pressure-level checks remain
        # exactly the source implementation.
        old_expected = mod09.EXPECTED_CITIES
        mod09.EXPECTED_CITIES = EXPECTED_EXTERNAL_CITIES
        try:
            mod09.validate_source(ds)
        finally:
            mod09.EXPECTED_CITIES = old_expected

        times = pd.DatetimeIndex(ds["time"].values)
        local = pd.DatetimeIndex(ds["local_time"].values)
        if len(times) != EXPECTED_MERGED_HOURS:
            raise RuntimeError("external feature-bank source hours mismatch")
        if not np.all((local - times) == pd.Timedelta(hours=8)):
            raise RuntimeError("external feature-bank source local_time != UTC + 8h")

        features = mod09.build_feature_arrays(ds)
        if list(features) != order:
            raise RuntimeError("external feature order differs from frozen schema")
        bad_shape = {
            k: v.shape
            for k, v in features.items()
            if v.shape != (EXPECTED_MERGED_HOURS, EXPECTED_EXTERNAL_CITIES)
        }
        if bad_shape:
            raise RuntimeError(f"external 72-feature shape mismatch: {bad_shape}")

        task = pd.read_parquet(
            task_path,
            columns=[
                "issue_time_index",
                "issue_time_utc",
                "target_time_utc",
                "forecast_horizon_hours",
            ],
        )
        if not np.all(task["forecast_horizon_hours"].to_numpy() == HORIZON_HOURS):
            raise RuntimeError("external task horizon is not uniformly 24 h")
        unique_issue = task[["issue_time_index", "issue_time_utc"]].drop_duplicates()
        ip = unique_issue["issue_time_index"].to_numpy(dtype=np.int64)
        if np.any((ip < 0) | (ip >= len(times))):
            raise RuntimeError("external task issue_time_index outside feature-bank source")
        indexed_time = times[ip]
        supplied_time = pd.DatetimeIndex(unique_issue["issue_time_utc"])
        if not np.array_equal(
            indexed_time.to_numpy(dtype="datetime64[ns]"),
            supplied_time.to_numpy(dtype="datetime64[ns]"),
        ):
            raise RuntimeError("external task issue_time_index does not match issue_time_utc")

        dev_mask = (local.year >= 2023) & (local.year <= 2024)
        future_mask = local.year == 2025
        missing_dev: dict[str, int] = {}
        missing_future: dict[str, int] = {}
        for name, a in features.items():
            missing_dev[name] = int(np.size(a[dev_mask]) - np.isfinite(a[dev_mask]).sum())
            missing_future[name] = int(
                np.size(a[future_mask]) - np.isfinite(a[future_mask]).sum()
            )

        data_vars = {
            name: (("time", "city"), a.astype(np.float32, copy=False))
            for name, a in features.items()
        }
        out_ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": ds["time"].values, "city": ds["city"].values},
            attrs={
                "artifact_version": "external_observed_issue_time_feature_bank_v1",
                "external_protocol_id": EXTERNAL_PROTOCOL_ID,
                "schema_id": FEATURE_SCHEMA_ID,
                "schema_fingerprint_sha256": frozen_schema["schema_fingerprint_sha256"],
                "construction": (
                    "global deterministic causal feature bank; no target materialized; "
                    "no fitted data-dependent transform"
                ),
                "time_coordinate": "UTC",
                "calendar_timezone": "Asia/Shanghai",
                "source": str(merged_path),
                "task_index": str(task_path),
                "preprocessing_fit": "FORBIDDEN; use original fold-k FIT statistics only in evaluator",
                "2025_policy": (
                    "mechanically constructed from forecasting-time observations and ERA5; "
                    "not used for model selection"
                ),
                "source_builder_sha256": sha256_file(Path(mod09.__file__).resolve()),
            },
        )
        for meta_name in ("local_time", "local_year", "local_month", "local_hour"):
            if meta_name in ds:
                out_ds[meta_name] = ds[meta_name]
        for meta_name in ("city_en", "city_zh", "region", "province"):
            if meta_name in ds:
                out_ds[meta_name] = ds[meta_name]

        for name in out_ds.variables:
            out_ds[name].encoding = {}
        encoding: dict[str, Any] = {}
        for name, da in out_ds.data_vars.items():
            if da.dims == ("time", "city") and np.issubdtype(da.dtype, np.number):
                encoding[name] = {"chunks": (744, EXPECTED_EXTERNAL_CITIES)}
            elif da.dims == ("time",):
                encoding[name] = {"chunks": (744,)}

        out_ds.to_zarr(
            str(building),
            mode="w",
            consolidated=False,
            encoding=encoding,
            **force_zarr_v2_kwargs(),
        )
        mod02.zarr.consolidate_metadata(str(building))
        out_ds.close()
        finalize_build_path(building, final)

        # Reopen final artifact and assert exact feature identity / order.
        check = xr.open_zarr(final, consolidated=True, chunks=None)
        try:
            present_order = [name for name in order if name in check.data_vars]
            if present_order != order:
                raise RuntimeError("final external feature bank lost frozen feature order identity")
            for name in order:
                if check[name].shape != (EXPECTED_MERGED_HOURS, EXPECTED_EXTERNAL_CITIES):
                    raise RuntimeError(f"final feature shape mismatch: {name}")
            city_labels = [str(x) for x in check["city_en"].values.tolist()]
            if len(city_labels) != EXPECTED_EXTERNAL_CITIES:
                raise RuntimeError("final external feature bank city metadata mismatch")
        finally:
            check.close()

        report = {
            "status": "PASS",
            "version": "external_observed_issue_time_feature_bank_v1",
            "created_at_utc": utc_now(),
            "external_protocol_id": EXTERNAL_PROTOCOL_ID,
            "source": str(merged_path),
            "output": str(final),
            "schema": str(frozen_schema_path),
            "schema_id": FEATURE_SCHEMA_ID,
            "schema_fingerprint_sha256": frozen_schema["schema_fingerprint_sha256"],
            "feature_count": EXPECTED_FEATURES,
            "source_hours": EXPECTED_MERGED_HOURS,
            "cities": EXPECTED_EXTERNAL_CITIES,
            "full_grid_feature_cells": int(
                EXPECTED_MERGED_HOURS * EXPECTED_EXTERNAL_CITIES * EXPECTED_FEATURES
            ),
            "target_materialized": False,
            "feature_nonfinite_counts_2023_2024": missing_dev,
            "feature_nonfinite_counts_2025_availability_only": missing_future,
            "fit_or_selection_performed": False,
            "external_preprocessing_statistics_fitted": False,
            "external_data_used_for_model_selection": False,
            "storage_format": "Zarr V2",
            "source_builder": str(Path(mod09.__file__).resolve()),
            "source_builder_sha256": sha256_file(Path(mod09.__file__).resolve()),
        }
        atomic_json(report, report_path)
        return final, report
    except Exception:
        if building.exists():
            remove_path(building)
        raise
    finally:
        ds.close()
        gc.collect()


# =============================================================================
# Final whole-pipeline audit
# =============================================================================

def final_audit(
    *,
    root: Path,
    p: dict[str, Path],
    prereq: dict[str, Any],
    sources: dict[str, Path],
    optional_sources: dict[str, Path | None],
    canonical_path: Path,
    derived_path: Path,
    city_meteo_path: Path,
    merged_path: Path,
    task_path: Path,
    feature_bank_path: Path,
    stage_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task = pd.read_parquet(task_path)
    valid = task.loc[task["target_valid_benchmark"].astype(bool)].copy()
    scenario_counts = (
        valid.groupby("external_scenario", sort=False)
        .size()
        .astype(int)
        .to_dict()
    )
    if set(scenario_counts) != {"External-Spatial", "External-Joint"}:
        raise RuntimeError(f"unexpected final external scenarios: {scenario_counts}")

    bank = xr.open_zarr(feature_bank_path, consolidated=True, chunks=None)
    merged = xr.open_zarr(merged_path, consolidated=True, chunks=None)
    try:
        schema = prereq["frozen_schema"]
        order = list(schema["feature_order"])
        if len(order) != EXPECTED_FEATURES:
            raise RuntimeError("final audit frozen feature count mismatch")
        if any(name not in bank.data_vars for name in order):
            raise RuntimeError("final audit feature bank misses frozen features")
        if int(bank.sizes["time"]) != EXPECTED_MERGED_HOURS:
            raise RuntimeError("final audit feature-bank time mismatch")
        if int(bank.sizes["city"]) != EXPECTED_EXTERNAL_CITIES:
            raise RuntimeError("final audit feature-bank city mismatch")

        merged_city = [str(x) for x in merged["city_en"].values.tolist()]
        bank_city = [str(x) for x in bank["city_en"].values.tolist()]
        task_city = (
            task[["city_index", "city_en"]]
            .drop_duplicates()
            .sort_values("city_index")["city_en"]
            .astype(str)
            .tolist()
        )
        frozen_city = prereq["protocol"].get(
            "external_city_order",
            prereq["protocol_audit"].get("external_city_order"),
        )
        if merged_city != bank_city or merged_city != task_city:
            raise RuntimeError("final audit city-axis identity mismatch across merged/task/bank")
        if frozen_city is not None and merged_city != list(frozen_city):
            raise RuntimeError("final audit city axis differs from frozen external protocol")

        # Structural gather gate used by the future ExternalFrozenFoldCache.
        rows = valid["external_task_row"].to_numpy(np.int64)
        issue = task.iloc[rows]["issue_time_index"].to_numpy(np.int64)
        city = task.iloc[rows]["city_index"].to_numpy(np.int64)
        if int(issue.min()) < LOOKBACK_HOURS - 1:
            raise RuntimeError("final valid task includes issue without 72 h history")
        if np.any(city < 0) or np.any(city >= EXPECTED_EXTERNAL_CITIES):
            raise RuntimeError("final valid task city_index outside external bank")
    finally:
        bank.close()
        merged.close()

    source_provenance = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in sources.items()
    }
    for key, path in optional_sources.items():
        source_provenance[key] = (
            {"path": str(path), "sha256": sha256_file(path)} if path is not None else None
        )

    summary = {
        "status": "PASS",
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "external_protocol_id": EXTERNAL_PROTOCOL_ID,
        "scientific_role": (
            "data-construction gate for Whole-Region External Holdout Evaluation; "
            "no training/checkpoint selection/inference performed"
        ),
        "frozen_contract": {
            "external_city_count": EXPECTED_EXTERNAL_CITIES,
            "external_scenarios": ["External-Spatial", "External-Joint"],
            "external_temporal_scenario_defined": False,
            "forecast_horizon_hours": HORIZON_HOURS,
            "lookback_hours": LOOKBACK_HOURS,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_schema_fingerprint_sha256": prereq["frozen_schema"][
                "schema_fingerprint_sha256"
            ],
            "feature_count": EXPECTED_FEATURES,
            "prediction_averaging_across_folds": False,
        },
        "sealed_boundaries": {
            "model_checkpoint_read": False,
            "model_inference_run": False,
            "training_run": False,
            "checkpoint_reselection": False,
            "external_preprocessing_fit": False,
            "external_scaler_fit": False,
            "external_imputation_fit": False,
            "external_data_used_for_model_selection": False,
            "target_magnitude_materialized_in_task_index": False,
        },
        "inputs": {
            "external_city_manifest": str(p["external_manifest"]),
            "external_city_manifest_sha256": sha256_file(p["external_manifest"]),
            "external_protocol": str(p["protocol"]),
            "external_protocol_sha256": sha256_file(p["protocol"]),
            "external_cnemc_canonical": str(prereq["aq_path"]),
            "external_cnemc_root_metadata_sha256": zarr_root_metadata_sha256(
                prereq["aq_path"]
            ),
            "external_era5_jobs_manifest": str(p["era5_jobs"]),
            "external_era5_jobs_sha256": sha256_file(p["era5_jobs"]),
            "external_era5_bounds_manifest": str(p["era5_bounds"]),
            "external_era5_bounds_sha256": sha256_file(p["era5_bounds"]),
            "benchmark_validity_policy": str(p["benchmark_policy"]),
            "benchmark_validity_policy_sha256": sha256_file(p["benchmark_policy"]),
            "frozen_feature_schema": str(p["frozen_schema"]),
            "frozen_feature_schema_sha256": sha256_file(p["frozen_schema"]),
        },
        "outputs": {
            "era5_canonical": str(canonical_path),
            "era5_canonical_root_metadata_sha256": zarr_root_metadata_sha256(canonical_path),
            "era5_physical": str(derived_path),
            "era5_physical_root_metadata_sha256": zarr_root_metadata_sha256(derived_path),
            "city_meteorology": str(city_meteo_path),
            "city_meteorology_root_metadata_sha256": zarr_root_metadata_sha256(city_meteo_path),
            "merged": str(merged_path),
            "merged_root_metadata_sha256": zarr_root_metadata_sha256(merged_path),
            "task_index": str(task_path),
            "task_index_sha256": sha256_file(task_path),
            "feature_bank": str(feature_bank_path),
            "feature_bank_root_metadata_sha256": zarr_root_metadata_sha256(feature_bank_path),
        },
        "sample_counts": {
            "task_rows_all_structural": int(len(task)),
            "task_rows_valid_total": int(len(valid)),
            "valid_by_external_scenario": scenario_counts,
        },
        "stage_reports": stage_reports,
        "source_code_provenance": source_provenance,
        "next_gate": (
            "77 may now build ExternalFrozenFoldCache from this external task/bank, "
            "while reading preprocessing/target statistics ONLY from original fold-k FIT artifacts."
        ),
    }
    atomic_json(summary, p["summary"])
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    p = build_paths(root)
    p["diag"].mkdir(parents=True, exist_ok=True)

    print("=" * 126)
    print("76 — CENTRAL-CHINA EXTERNAL FROZEN INPUT CONSTRUCTION")
    print("=" * 126)
    print(f"Root: {root}")
    print("NO training | NO checkpoint read | NO external preprocessing fit")
    print()

    sources = resolve_source_files(root, args.source_dir)
    optional_sources = {
        key: find_optional_source(root, args.source_dir, fn)
        for key, fn in OPTIONAL_PROVENANCE_FILES.items()
    }

    # Import only frozen scientific modules actually executed by 76.
    mod02 = import_module_from_path("_frozen76_era5_canonical", sources["era5_canonical"])
    mod03 = import_module_from_path("_frozen76_era5_derived", sources["era5_derived"])
    mod05 = import_module_from_path("_frozen76_city_meteorology", sources["city_meteorology"])
    mod06 = import_module_from_path("_frozen76_merge_reference", sources["merge_reference"])
    mod09 = import_module_from_path("_frozen76_feature_bank", sources["feature_bank"])

    # Filename aliases are allowed only because the live project keeps the
    # frozen v3/v2 implementations under unversioned basenames.  Audit the
    # semantic build-version constants before touching any data.
    build_version = getattr(mod02, "BUILD_VERSION", None)
    if build_version != "era5_canonical_v3":
        raise RuntimeError(
            f"02 ERA5 canonical source has BUILD_VERSION={build_version!r}; "
            "expected frozen era5_canonical_v3"
        )
    derived_version = getattr(mod03, "DERIVED_VERSION", None)
    if derived_version != "era5_physical_derived_v2":
        raise RuntimeError(
            f"03 ERA5 derived source has DERIVED_VERSION={derived_version!r}; "
            "expected frozen era5_physical_derived_v2"
        )
    for required_name in (
        "pressure_level_above_ground_mask",
        "pressure_pair_850_925_above_ground_mask",
    ):
        if not hasattr(mod03, required_name):
            raise RuntimeError(
                f"03 ERA5 derived source missing frozen pressure-validity primitive: {required_name}"
            )

    print(
        "Frozen source versions: "
        f"02={build_version} | 03={derived_version}",
        flush=True,
    )

    ext = load_external_manifest(p["external_manifest"])

    print("[0/6] Frozen prerequisite / acquisition closure gates", flush=True)
    prereq = gate_frozen_prerequisites(
        root=root,
        p=p,
        ext=ext,
        mod02=mod02,
        verify_raw_sha=not args.skip_raw_sha,
    )
    print("  PASS", flush=True)
    if args.stop_after == "prerequisites":
        print("Stopped after prerequisite gate by request.")
        return

    region = prereq["region"]
    stage_reports: dict[str, dict[str, Any]] = {}

    print("[1/6] External ERA5 raw -> canonical", flush=True)
    canonical_path, canonical_report = build_external_era5_canonical(
        root=root,
        out_root=p["era5_canonical_root"],
        region=region,
        months=prereq["months"],
        jobs=prereq["jobs"],
        bounds_map=prereq["bounds"],
        mod02=mod02,
        overwrite=args.overwrite,
    )
    stage_reports["era5_canonical"] = canonical_report
    print(f"  PASS: {canonical_path}", flush=True)
    if args.stop_after == "canonical":
        return

    print("[2/6] ERA5 canonical -> frozen physical derivations", flush=True)
    derived_path, derived_report = build_external_era5_derived(
        canonical_path=canonical_path,
        out_root=p["era5_derived_root"],
        region=region,
        mod02=mod02,
        mod03=mod03,
        overwrite=args.overwrite,
        chunk_hours=args.derived_chunk_hours,
    )
    stage_reports["era5_derived"] = derived_report
    print(f"  PASS: {derived_path}", flush=True)
    if args.stop_after == "derived":
        return

    print("[3/6] Frozen city-level ERA5 extraction for external 10 cities", flush=True)
    city_meteo_path, city_meteo_report = build_external_city_meteorology(
        derived_path=derived_path,
        final=p["city_meteo"],
        report_path=p["city_meteo_report"],
        mapping_path=p["city_meteo_mapping"],
        ext=ext,
        mod02=mod02,
        mod05=mod05,
        overwrite=args.overwrite,
    )
    stage_reports["city_meteorology"] = city_meteo_report
    print(f"  PASS: {city_meteo_path}", flush=True)
    if args.stop_after == "city_meteorology":
        return

    print("[4/6] Exact external CNEMC + ERA5 merge", flush=True)
    merged_path, merged_report = build_external_merge(
        root=root,
        aq_path=prereq["aq_path"],
        meteo_path=city_meteo_path,
        out_path=p["merged"],
        manifest_path=p["merged_manifest"],
        mod02=mod02,
        mod06=mod06,
        overwrite=args.overwrite,
    )
    stage_reports["merged"] = merged_report
    print(f"  PASS: {merged_path}", flush=True)
    if args.stop_after == "merged":
        return

    print("[5/6] Split-agnostic external +24 h task index", flush=True)
    task_path, task_report, _ = build_external_task_index(
        merged_path=merged_path,
        out_path=p["task"],
        manifest_path=p["task_manifest"],
        policy_path=p["benchmark_policy"],
        source07=sources["task_reference"],
        overwrite=args.overwrite,
    )
    stage_reports["task_index"] = task_report
    print(f"  PASS: {task_path}", flush=True)
    if args.stop_after == "task_index":
        return

    print("[6/6] Frozen 72-feature issue-time bank", flush=True)
    feature_bank_path, feature_report = build_external_feature_bank(
        merged_path=merged_path,
        task_path=task_path,
        final=p["feature_bank"],
        report_path=p["feature_report"],
        frozen_schema_path=p["frozen_schema"],
        parent_schema_path=p["parent_schema"],
        mod02=mod02,
        mod09=mod09,
        overwrite=args.overwrite,
    )
    stage_reports["feature_bank"] = feature_report
    print(f"  PASS: {feature_bank_path}", flush=True)
    if args.stop_after == "feature_bank":
        return

    print("[FINAL] Whole-pipeline identity / structural audit", flush=True)
    summary = final_audit(
        root=root,
        p=p,
        prereq=prereq,
        sources=sources,
        optional_sources=optional_sources,
        canonical_path=canonical_path,
        derived_path=derived_path,
        city_meteo_path=city_meteo_path,
        merged_path=merged_path,
        task_path=task_path,
        feature_bank_path=feature_bank_path,
        stage_reports=stage_reports,
    )

    print("\n" + "=" * 126)
    print("76 PASS — CENTRAL-CHINA EXTERNAL FROZEN INPUTS READY")
    print("=" * 126)
    print(f"Task index  : {task_path}")
    print(f"Feature bank: {feature_bank_path}")
    print(f"Audit       : {p['summary']}")
    print(f"Valid rows  : {summary['sample_counts']['valid_by_external_scenario']}")
    print("Next: 77 ExternalFrozenFoldCache + D5 / Informer / TimeXer frozen-checkpoint inference")
    print("=" * 126)


if __name__ == "__main__":
    main()
