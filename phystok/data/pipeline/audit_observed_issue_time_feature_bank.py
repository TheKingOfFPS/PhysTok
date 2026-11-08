#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


AUDIT_ID = "observed_issue_time_feature_bank_audit_v2"
SCHEMA_ID = "national_pm25_observed_feature_schema_v1"
PARENT_SCHEMA_ID = "national_pm25_feature_schema_v1"

EXPECTED_FAMILY_COUNTS = {
    "pm25_history": 15,
    "calendar": 4,
    "transport_dispersion": 15,
    "moisture_stability_context": 8,
    "pollution_composition": 10,
    "precipitation_radiation": 16,
    "pm10_state": 3,
    "domain_pressure_context": 1,
}

EXPECTED_MASK_COUNTS = {
    "history_calendar_only": 19,
    "core_without_pm10": 68,
    "core_plus_pm10": 71,
    "all_with_pressure": 72,
    "meteorology_physics_only": 43,
    "pollution_history_only": 32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    return p.parse_args()


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    schema_path = (
        root / "05_manifests" / "features"
        / SCHEMA_ID / "feature_schema.json"
    )
    parent_path = (
        root / "05_manifests" / "features"
        / PARENT_SCHEMA_ID / "feature_schema.json"
    )
    bank_path = (
        root / "04_derived" / "issue_time_features"
        / SCHEMA_ID / "issue_time_feature_bank.zarr"
    )
    build_report_path = (
        root / "04_derived" / "issue_time_features"
        / SCHEMA_ID / "build_report.json"
    )
    task_path = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1.parquet"
    )
    merged_path = (
        root / "04_derived" / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )

    for p in (schema_path, bank_path, build_report_path, task_path, merged_path):
        if not p.exists():
            raise SystemExit(f"missing prerequisite: {p}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    parent = (
        json.loads(parent_path.read_text(encoding="utf-8"))
        if parent_path.exists()
        else None
    )
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))

    failures = []
    warnings = []

    if schema.get("schema_id") != SCHEMA_ID:
        failures.append("wrong schema_id")
    if int(schema.get("feature_count", -1)) != 72:
        failures.append("feature_count != 72")

    order = list(schema["feature_order"])
    if len(order) != len(set(order)):
        failures.append("duplicate feature names")

    family_counts = {}
    for fam in schema["families"]:
        family_counts[fam["family"]] = len(fam["features"])
    if family_counts != EXPECTED_FAMILY_COUNTS:
        failures.append(
            f"family counts={family_counts}, expected={EXPECTED_FAMILY_COUNTS}"
        )

    if parent is not None:
        if list(parent.get("feature_order", [])) != order:
            failures.append("new feature order differs from frozen parent schema")
    else:
        warnings.append("parent feature schema not found; lineage comparison skipped")

    recs = {r["feature"]: r for r in schema["feature_records"]}

    def mask_by_families(families):
        return np.asarray(
            [recs[f]["family"] in families for f in order],
            dtype=bool,
        )

    default_core = np.asarray(
        [bool(recs[f]["default_core_predictor"]) for f in order],
        dtype=bool,
    )
    pm10 = mask_by_families({"pm10_state"})

    masks = {
        "history_calendar_only": mask_by_families(
            {"pm25_history", "calendar"}
        ),
        "core_without_pm10": default_core,
        "core_plus_pm10": default_core | pm10,
        "all_with_pressure": np.ones(72, dtype=bool),
        "meteorology_physics_only": mask_by_families(
            {
                "calendar",
                "transport_dispersion",
                "moisture_stability_context",
                "precipitation_radiation",
            }
        ),
        "pollution_history_only": mask_by_families(
            {
                "pm25_history",
                "calendar",
                "pollution_composition",
                "pm10_state",
            }
        ),
    }
    mask_counts = {k: int(v.sum()) for k, v in masks.items()}
    if mask_counts != EXPECTED_MASK_COUNTS:
        failures.append(
            f"mask counts={mask_counts}, expected={EXPECTED_MASK_COUNTS}"
        )

    bank = xr.open_zarr(bank_path, consolidated=True, chunks=None)
    merged = xr.open_zarr(merged_path, consolidated=True, chunks=None)
    try:
        if bank.sizes.get("time") != 27040:
            failures.append(f"bank time={bank.sizes.get('time')} expected=27040")
        if bank.sizes.get("city") != 50:
            failures.append(f"bank city={bank.sizes.get('city')} expected=50")

        missing_bank_features = [f for f in order if f not in bank]
        if missing_bank_features:
            failures.append(
                f"bank missing features: {missing_bank_features}"
            )

        if not np.array_equal(bank["time"].values, merged["time"].values):
            failures.append("bank UTC time coordinate differs from merged source")
        if not np.array_equal(bank["city"].values, merged["city"].values):
            failures.append("bank city coordinate differs from merged source")

        for name in ("city_en", "city_zh", "region", "local_time"):
            if name in bank and name in merged:
                if not np.array_equal(bank[name].values, merged[name].values):
                    failures.append(f"bank metadata differs for {name}")

        # Direct semantic identity for issue-time observed AQ state.
        direct_pairs = {
            "pm25_now": "pm25_city_mean",
            "pm10_now": "pm10_city_mean",
            "carbon_monoxide_now": "co_city_mean",
            "sulphur_dioxide_now": "so2_city_mean",
            "nitrogen_dioxide_now": "no2_city_mean",
            "ozone_now": "o3_city_mean",
            "boundary_layer_height_now": "boundary_layer_height_m",
            "wind_speed_10m_now": "wind_speed_10m_ms",
            "surface_pressure_now": "surface_pressure_hpa",
            "solar_radiation_now": "surface_solar_radiation_downwards_wm2",
            "precipitation_now": "total_precipitation_mm",
        }

        direct_mismatches = []
        for f, src in direct_pairs.items():
            # The feature bank deliberately stores deterministic continuous
            # predictors as float32 (same policy as the frozen parent bank).
            # Some ERA5 variables in the merged source remain float64.
            # Therefore identity must be checked against the exact float32
            # representation used by the builder, not against the original
            # float64 byte representation.
            a = np.asarray(bank[f].values, dtype=np.float32)
            b = np.asarray(merged[src].values, dtype=np.float32)
            if not np.array_equal(a, b, equal_nan=True):
                direct_mismatches.append((f, src))
        if direct_mismatches:
            failures.append(
                f"direct source identity failures after float32 canonicalization: "
                f"{direct_mismatches}"
            )

        # Explicit lag causality checks.
        lag_checks = {
            "pm25_lag_1h": 1,
            "pm25_lag_3h": 3,
            "pm25_lag_6h": 6,
            "pm25_lag_24h": 24,
        }
        pm25 = np.asarray(bank["pm25_now"].values)
        lag_failures = []
        for f, lag in lag_checks.items():
            got = np.asarray(bank[f].values)
            if not np.array_equal(
                got[lag:],
                pm25[:-lag],
                equal_nan=True,
            ):
                lag_failures.append(f)
        if lag_failures:
            failures.append(f"lag causality failures: {lag_failures}")

        task = pd.read_parquet(
            task_path,
            columns=[
                "issue_time_index",
                "issue_time_utc",
                "target_time_utc",
                "target_local_year",
                "target_valid_benchmark",
            ],
        )

        ip = task["issue_time_index"].to_numpy(dtype=np.int64)
        if np.any((ip < 0) | (ip >= bank.sizes["time"])):
            failures.append("task issue_time_index outside feature-bank range")

        # No target values are present in the feature bank.
        target_like = [
            name for name in bank.data_vars
            if name.lower().startswith("target")
        ]
        if target_like:
            failures.append(f"target-like variables present in bank: {target_like}")

        # Development feature missingness is diagnostic, not a validity failure.
        local = pd.DatetimeIndex(bank["local_time"].values)
        dev = (local.year >= 2023) & (local.year <= 2024)
        future = local.year == 2025

        missing_rows = []
        for f in order:
            a = np.asarray(bank[f].values)
            dev_total = int(np.size(a[dev]))
            dev_missing = int(dev_total - np.isfinite(a[dev]).sum())
            fut_total = int(np.size(a[future]))
            fut_missing = int(fut_total - np.isfinite(a[future]).sum())
            missing_rows.append(
                {
                    "feature": f,
                    "family": recs[f]["family"],
                    "development_missing_cells": dev_missing,
                    "development_total_cells": dev_total,
                    "development_missing_fraction": (
                        dev_missing / dev_total if dev_total else np.nan
                    ),
                    "future_2025_missing_cells_availability_only": fut_missing,
                    "future_2025_total_cells": fut_total,
                    "future_2025_missing_fraction_availability_only": (
                        fut_missing / fut_total if fut_total else np.nan
                    ),
                }
            )

        missing_df = pd.DataFrame(missing_rows)

        # Physical-only missingness should not suddenly depend on observed AQ.
        phys_mask = mask_by_families(
            {
                "calendar",
                "transport_dispersion",
                "moisture_stability_context",
                "precipitation_radiation",
            }
        )
        physical_features = [f for f, m in zip(order, phys_mask) if m]
        physical_nonfinite_dev = int(
            missing_df.loc[
                missing_df["feature"].isin(physical_features),
                "development_missing_cells",
            ].sum()
        )

        out_root = (
            root / "06_processed" / "audits" / AUDIT_ID
        )
        out_root.mkdir(parents=True, exist_ok=True)

        missing_df.to_csv(
            out_root / "feature_missingness.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            [
                {
                    "mask": name,
                    "feature_count": int(mask.sum()),
                    "features": ";".join(
                        f for f, keep in zip(order, mask) if keep
                    ),
                }
                for name, mask in masks.items()
            ]
        ).to_csv(
            out_root / "feature_masks.csv",
            index=False,
            encoding="utf-8-sig",
        )

        status = "PASS" if not failures else "FAIL"
        summary = {
            "audit_id": AUDIT_ID,
            "created_at_utc": now_utc(),
            "status": status,
            "failures": failures,
            "warnings": warnings,
            "schema_id": SCHEMA_ID,
            "schema_fingerprint_sha256": schema[
                "schema_fingerprint_sha256"
            ],
            "feature_count": 72,
            "family_counts": family_counts,
            "mask_counts": mask_counts,
            "parent_schema_present": parent is not None,
            "parent_feature_order_reused_exactly": (
                parent is not None
                and list(parent.get("feature_order", [])) == order
            ),
            "bank": {
                "time": int(bank.sizes["time"]),
                "city": int(bank.sizes["city"]),
                "target_variables_present": target_like,
                "direct_source_identity_checks": len(direct_pairs),
                "physical_feature_development_nonfinite_cells": (
                    physical_nonfinite_dev
                ),
            },
            "missingness_policy": {
                "nonfinite_features_are_preserved": True,
                "missingness_is_not_a_bank_failure": True,
                "no_imputation_performed": True,
                "no_sample_deletion_performed": True,
            },
            "sealed_future_policy": {
                "2025_feature_value_distributions_inspected": False,
                "2025_target_values_read": False,
                "2025_feature_missingness_summarized": True,
            },
            "build_report_status": build_report.get("status"),
            "outputs": [
                "feature_missingness.csv",
                "feature_masks.csv",
            ],
        }

        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        archive = out_root.parent / f"{AUDIT_ID}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out_root, arcname=out_root.name)

        if failures:
            raise SystemExit(2)

    finally:
        bank.close()
        merged.close()


if __name__ == "__main__":
    main()
