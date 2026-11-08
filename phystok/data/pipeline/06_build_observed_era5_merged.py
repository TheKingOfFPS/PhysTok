#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


BUILD_ID = "observed_era5_merged_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--aq", type=Path, default=None)
    p.add_argument("--meteo", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zarr_root_metadata_sha256(path: Path) -> str | None:
    for name in ("zarr.json", ".zmetadata", ".zgroup"):
        p = path / name
        if p.exists() and p.is_file():
            return sha256_file(p)
    return None


def as_str_list(a) -> list[str]:
    return [str(x) for x in np.asarray(a).tolist()]


def resolve_city_labels(
    ds: xr.Dataset,
    root: Path,
    aq_city_order: list[str] | None = None,
) -> tuple[list[str], str]:
    """
    Resolve city-axis labels without assuming the Zarr necessarily stores
    a variable literally named `city_en`.

    Priority:
      1) string-valued city metadata in the Zarr;
      2) selected_cities_v2.csv row order, but only when the Zarr city
         coordinate is the canonical integer 0..N-1 axis.

    The fallback is explicit and recorded in the output manifest.
    """
    candidates = (
        "city_en",
        "city_name",
        "city",
        "name_en",
        "city_english",
    )

    n_city = int(ds.sizes["city"])

    for name in candidates:
        if name not in ds.variables:
            continue
        da = ds[name]
        if da.dims != ("city",):
            continue

        vals = np.asarray(da.values)
        if vals.dtype.kind not in {"U", "S", "O"}:
            continue

        labels = [str(x).strip() for x in vals.tolist()]
        if len(labels) == n_city and len(set(labels)) == n_city:
            return labels, f"zarr:{name}"

    # Fallback: the original city_meteorology extraction used the selected
    # city manifest as the city-axis source, but did not persist city_en.
    city_coord = np.asarray(ds["city"].values)
    expected_coord = np.arange(n_city)
    if (
        city_coord.shape != expected_coord.shape
        or not np.array_equal(city_coord.astype(int), expected_coord)
    ):
        raise RuntimeError(
            "meteorology lacks a string city label variable and its city "
            "coordinate is not the canonical 0..N-1 axis; refusing to guess"
        )

    manifest_path = root / "05_manifests" / "selected_cities_v2.csv"
    if not manifest_path.exists():
        raise RuntimeError(
            "meteorology lacks city labels and selected_cities_v2.csv is missing"
        )

    selected = pd.read_csv(manifest_path, encoding="utf-8-sig")
    city_col = next(
        (
            c for c in selected.columns
            if str(c).strip().lower()
            in {
                "city_en",
                "city",
                "city_name",
                "city_english",
                "name_en",
            }
        ),
        None,
    )
    if city_col is None:
        raise RuntimeError(
            "cannot resolve English city column in selected_cities_v2.csv"
        )

    labels = selected[city_col].astype(str).str.strip().tolist()
    if len(labels) != n_city or len(set(labels)) != n_city:
        raise RuntimeError(
            "selected_cities_v2.csv does not provide exactly one unique "
            "English label per meteorology city-axis entry"
        )

    if aq_city_order is not None and set(labels) != set(aq_city_order):
        raise RuntimeError(
            "manifest-derived meteorology city labels do not match AQ city set"
        )

    return labels, "selected_cities_v2.csv:row_order"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    aq_path = (
        args.aq.expanduser().resolve()
        if args.aq is not None
        else root
        / "03_canonical"
        / "air_quality_cnemc"
        / "city_air_quality_50cities.zarr"
    )
    meteo_path = (
        args.meteo.expanduser().resolve()
        if args.meteo is not None
        else root
        / "04_derived"
        / "city_meteorology"
        / "city_meteorology_50cities.zarr"
    )
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else root
        / "04_derived"
        / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out_path}")

    aq = xr.open_zarr(aq_path, consolidated=True)
    met = xr.open_zarr(meteo_path, consolidated=True)

    for name, ds in (("AQ", aq), ("METEO", met)):
        if "time" not in ds.dims or "city" not in ds.dims:
            raise RuntimeError(f"{name} must contain time and city dimensions")

    if "city_en" not in aq.variables:
        raise RuntimeError("AQ canonical lacks city_en")

    aq_city = as_str_list(aq["city_en"].values)
    met_city, meteo_city_labels_source = resolve_city_labels(
        met,
        root,
        aq_city_order=aq_city,
    )

    if len(aq_city) != len(set(aq_city)):
        raise RuntimeError("AQ city_en contains duplicates")
    if len(met_city) != len(set(met_city)):
        raise RuntimeError("meteorology city_en contains duplicates")
    if set(aq_city) != set(met_city):
        raise RuntimeError(
            "AQ/meteo city sets differ: "
            f"AQ-only={sorted(set(aq_city)-set(met_city))}, "
            f"meteo-only={sorted(set(met_city)-set(aq_city))}"
        )

    met_pos = {c: i for i, c in enumerate(met_city)}
    met_order = [met_pos[c] for c in aq_city]

    aq_time = pd.DatetimeIndex(aq["time"].values)
    met_time = pd.DatetimeIndex(met["time"].values)
    common = aq_time.intersection(met_time)

    if len(common) != 27040:
        raise RuntimeError(
            f"unexpected AQ/ERA5 UTC intersection={len(common)}, expected=27040"
        )
    if common[0] != pd.Timestamp("2022-12-01 00:00:00"):
        raise RuntimeError(f"unexpected common start={common[0]}")
    if common[-1] != pd.Timestamp("2025-12-31 15:00:00"):
        raise RuntimeError(f"unexpected common end={common[-1]}")

    aq_sel = aq.sel(time=common)
    met_sel = met.sel(time=common).isel(city=met_order)

    # Force identical dimension coordinates after explicit identity-based reorder.
    met_sel = met_sel.assign_coords(
        time=aq_sel["time"].values,
        city=aq_sel["city"].values,
    )

    # AQ owns common metadata and protocol-local clock fields.
    drop_from_meteo = [
        x
        for x in (
            "city_en",
            "city_zh",
            "region",
            "local_time",
            "local_year",
            "local_month",
            "local_hour",
        )
        if x in met_sel.variables
    ]
    met_clean = met_sel.drop_vars(drop_from_meteo)

    # Preserve every meteorological data variable. If a name ever collides with
    # AQ, prefix only that meteorological variable rather than silently replacing.
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
            "build_id": BUILD_ID,
            "created_at_utc": utc_now(),
            "air_quality_source": (
                "CNEMC national monitoring network; historical retrieval via QuotSoft"
            ),
            "meteorology_source": "ERA5",
            "time_coordinate": "UTC",
            "protocol_year_coordinate": "local_year (Asia/Shanghai)",
            "merge_rule": "exact UTC-hour intersection + exact city_en identity alignment",
            "forecast_target_definition": (
                "not materialized here; downstream task is fixed 24h-ahead "
                "pm25_city_mean with target-hour validity applied separately"
            ),
            "target_leakage_note": (
                "this merged store contains contemporaneous observed AQ and ERA5 "
                "predictors only; no future-shifted target variable is created"
            ),
            "canonical_values_dropped_by_validity_gate": "false",
            "2025_used_for_model_selection": "false",
            "storage_format": "Zarr V2 (forced for stable string metadata)",
        }
    )

    # Inputs include Zarr V3 artifacts. xarray preserves storage-layer
    # encoding metadata when variables are loaded, including V3-only keys
    # such as `serializer`. Those keys are incompatible with Zarr V2.
    #
    # Clear inherited storage encodings before writing. This changes only
    # on-disk representation, never values, dimensions, coordinates, attrs,
    # or scientific semantics. We then specify our own V2 chunk encoding.
    for name in merged.variables:
        merged[name].encoding = {}

    # Keep practical monthly time chunks for ordinary time-city variables.
    encoding = {}
    for name, da in merged.data_vars.items():
        if da.dims == ("time", "city") and np.issubdtype(da.dtype, np.number):
            encoding[name] = {"chunks": (744, min(50, merged.sizes["city"]))}

    # Write Zarr V2 deliberately. The current environment defaults to
    # Zarr V3, whose fixed-length UTF32 string dtype and consolidated metadata
    # are still marked unstable by zarr-python. This merged artifact contains
    # city/region string metadata, so V2 is the safer reproducible format here.
    import inspect

    to_zarr_params = inspect.signature(xr.Dataset.to_zarr).parameters
    zarr_kwargs = dict(
        store=out_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )
    if "zarr_format" in to_zarr_params:
        zarr_kwargs["zarr_format"] = 2
    elif "zarr_version" in to_zarr_params:
        zarr_kwargs["zarr_version"] = 2
    else:
        raise RuntimeError(
            "xarray Dataset.to_zarr exposes neither zarr_format nor "
            "zarr_version; cannot force stable Zarr V2 output"
        )

    merged.to_zarr(**zarr_kwargs)

    aq_vars = sorted(aq_sel.data_vars)
    meteo_vars_original = sorted(met_sel.data_vars)
    meteo_vars_merged = sorted(
        rename_map.get(v, v)
        for v in meteo_vars_original
        if v not in drop_from_meteo
    )

    policy_path = root / "05_manifests" / "cnemc_station_qc_policy_v1.json"

    manifest = {
        "build_id": BUILD_ID,
        "created_at_utc": merged.attrs["created_at_utc"],
        "output": str(out_path),
        "dimensions": {k: int(v) for k, v in merged.sizes.items()},
        "utc_start": str(common[0]),
        "utc_end": str(common[-1]),
        "city_count": int(merged.sizes["city"]),
        "meteorology_city_labels_source": meteo_city_labels_source,
        "air_quality_input": str(aq_path),
        "meteorology_input": str(meteo_path),
        "air_quality_root_metadata_sha256": zarr_root_metadata_sha256(aq_path),
        "meteorology_root_metadata_sha256": zarr_root_metadata_sha256(meteo_path),
        "station_qc_policy_sha256": (
            sha256_file(policy_path) if policy_path.exists() else None
        ),
        "air_quality_variables": aq_vars,
        "meteorology_variables_original": meteo_vars_original,
        "meteorology_variables_in_merged": meteo_vars_merged,
        "meteorology_collision_renames": rename_map,
        "storage_format": "Zarr V2",
        "important": {
            "time_alignment": "exact UTC-hour intersection",
            "city_alignment": "city_en identity, explicitly reordered",
            "aq_only_boundary_hours_removed": 8,
            "meteo_only_boundary_hours_removed": 8,
            "future_shifted_target_created": False,
            "validity_gate_applied_destructively": False,
            "2025_distribution_statistics_reported": False,
        },
    }

    manifest_path = out_path.parent / "observed_era5_50cities_v1_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
