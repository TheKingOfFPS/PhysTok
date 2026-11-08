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


AUDIT_ID = "observed_fold_local_artifacts_audit_v1"
PROTOCOL_ID = "national_pm25_ood_split_v1"
SCHEMA_ID = "national_pm25_observed_feature_schema_v1"
BINDING_ID = "observed_task_split_binding_v1"
ELIGIBILITY_POLICY_ID = "model_sample_eligibility_policy_v1"
SCENARIOS = {
    "fit": "FIT",
    "validation": "VALIDATION",
    "spatial_ood_final": "SPATIAL_OOD_FINAL",
    "temporal_ood_final": "TEMPORAL_OOD_FINAL",
    "joint_ood_final": "JOINT_OOD_FINAL",
    "reserved": "RESERVED",
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

    artifact_root = (
        root / "06_processed" / "fold_artifacts"
        / PROTOCOL_ID / SCHEMA_ID
    )
    bank_path = (
        root / "04_derived" / "issue_time_features"
        / SCHEMA_ID / "issue_time_feature_bank.zarr"
    )
    merged_path = (
        root / "04_derived" / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )
    task_path = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1.parquet"
    )
    binding_path = (
        root / "06_processed" / "split_bindings"
        / PROTOCOL_ID / BINDING_ID / "fold_city_year_roles.parquet"
    )
    schema_path = (
        root / "05_manifests" / "features"
        / SCHEMA_ID / "feature_schema.json"
    )
    eligibility_path = (
        root / "05_manifests" / f"{ELIGIBILITY_POLICY_ID}.json"
    )

    for p in (
        artifact_root,
        bank_path,
        merged_path,
        task_path,
        binding_path,
        schema_path,
        eligibility_path,
    ):
        if not p.exists():
            raise SystemExit(f"missing prerequisite: {p}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    eligibility = json.loads(
        eligibility_path.read_text(encoding="utf-8")
    )
    task = pd.read_parquet(
        task_path,
        columns=[
            "city_index",
            "city_en",
            "issue_time_index",
            "target_time_index",
            "issue_time_local",
            "target_time_local",
            "target_local_year",
            "target_valid_benchmark",
        ],
    )
    binding = pd.read_parquet(binding_path)

    failures = []
    warnings = []

    issue_local = pd.DatetimeIndex(task["issue_time_local"])
    target_local = pd.DatetimeIndex(task["target_time_local"])
    target_valid = task["target_valid_benchmark"].to_numpy(dtype=bool)
    same_year = (
        issue_local.year.to_numpy()
        == target_local.year.to_numpy()
    )
    formal = target_valid & same_year

    bank = xr.open_zarr(bank_path, consolidated=True, chunks=None)
    merged = xr.open_zarr(merged_path, consolidated=True, chunks=None)

    try:
        city_en = [str(x) for x in bank["city_en"].values.tolist()]
        task_city = task["city_index"].to_numpy(dtype=np.int16)
        task_issue = task["issue_time_index"].to_numpy(dtype=np.int32)
        task_target = task["target_time_index"].to_numpy(dtype=np.int32)
        task_year = task["target_local_year"].to_numpy(dtype=np.int16)

        # binding matrix for exact scenario semantics
        years = [2023, 2024, 2025]
        ypos = {y: i for i, y in enumerate(years)}
        cpos = {c: i for i, c in enumerate(city_en)}
        smap = {
            s: i for i, s in enumerate(
                [
                    "FIT",
                    "VALIDATION",
                    "SPATIAL_OOD_FINAL",
                    "TEMPORAL_OOD_FINAL",
                    "JOINT_OOD_FINAL",
                    "RESERVED",
                ]
            )
        }
        mat = np.full((10, 50, 3), -1, dtype=np.int8)
        for r in binding.itertuples(index=False):
            mat[
                int(r.fold),
                cpos[str(r.city_en)],
                ypos[int(r.target_local_year)],
            ] = smap[str(r.scenario)]

        task_ypos = np.asarray(
            [ypos[int(y)] for y in task_year],
            dtype=np.int8,
        )

        selector_summary = []
        disjoint_all = True
        union_all = True
        exact_role_all = True
        no_2022_fit_issue = True
        no_2025_fit = True

        order = list(schema["feature_order"])
        if len(order) != 72:
            failures.append("schema feature count is not 72")

        # Recompute all 72 preprocessing stats for fold 0.
        recompute_fold0_failures = []

        for fold in range(10):
            fold_dir = artifact_root / f"fold_{fold}"
            for p in (
                fold_dir / "scenario_task_rows.npz",
                fold_dir / "preprocess_stats.npz",
                fold_dir / "target_stats.json",
                fold_dir / "fold_artifact.json",
            ):
                if not p.exists():
                    failures.append(f"fold {fold}: missing {p.name}")

            if failures:
                continue

            z = np.load(
                fold_dir / "scenario_task_rows.npz",
                allow_pickle=False,
            )

            seen = np.zeros(len(task), dtype=np.int8)
            fold_code = mat[fold, task_city, task_ypos]

            for key, scenario in SCENARIOS.items():
                rows = z[f"{key}_task_rows"].astype(np.int64)
                if len(np.unique(rows)) != len(rows):
                    failures.append(
                        f"fold {fold} {scenario}: duplicate task rows"
                    )
                if len(rows) and (
                    rows.min() < 0 or rows.max() >= len(task)
                ):
                    failures.append(
                        f"fold {fold} {scenario}: row outside task index"
                    )
                    continue

                seen[rows] += 1
                expected_code = smap[scenario]
                if not np.all(fold_code[rows] == expected_code):
                    exact_role_all = False
                    failures.append(
                        f"fold {fold} {scenario}: selector violates binding"
                    )
                if not np.all(formal[rows]):
                    failures.append(
                        f"fold {fold} {scenario}: contains ineligible rows"
                    )

                selector_summary.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "samples": int(len(rows)),
                    }
                )

                if scenario == "FIT":
                    iy = issue_local.year.to_numpy()[rows]
                    ty = target_local.year.to_numpy()[rows]
                    if not np.isin(iy, [2023, 2024]).all():
                        no_2022_fit_issue = False
                        failures.append(
                            f"fold {fold}: FIT issue year outside 2023/2024"
                        )
                    if not np.isin(ty, [2023, 2024]).all():
                        no_2025_fit = False
                        failures.append(
                            f"fold {fold}: FIT target year outside 2023/2024"
                        )

            eligible_seen = seen[formal]
            if not np.all(eligible_seen == 1):
                union_all = False
                failures.append(
                    f"fold {fold}: formal eligible rows do not form exact selector partition"
                )
            if np.any(seen[~formal] != 0):
                failures.append(
                    f"fold {fold}: ineligible rows appear in selectors"
                )
            if np.any(seen > 1):
                disjoint_all = False
                failures.append(
                    f"fold {fold}: scenario selectors overlap"
                )

            stats = np.load(
                fold_dir / "preprocess_stats.npz",
                allow_pickle=False,
            )
            required_arrays = [
                "impute_median_train",
                "feature_mean_after_impute_train",
                "feature_std_after_impute_train_raw",
                "feature_finite_count_train",
                "feature_missing_count_train",
                "transform_center",
                "transform_scale",
                "apply_standardization",
            ]
            for name in required_arrays:
                if name not in stats.files:
                    failures.append(
                        f"fold {fold}: preprocess stats missing {name}"
                    )
                    continue
                if len(stats[name]) != 72:
                    failures.append(
                        f"fold {fold}: {name} length !=72"
                    )

            for name in (
                "impute_median_train",
                "feature_mean_after_impute_train",
                "feature_std_after_impute_train_raw",
                "transform_center",
                "transform_scale",
            ):
                if name in stats.files and not np.isfinite(stats[name]).all():
                    failures.append(
                        f"fold {fold}: non-finite {name}"
                    )
            if "transform_scale" in stats.files and np.any(
                stats["transform_scale"] <= 0
            ):
                failures.append(
                    f"fold {fold}: non-positive transform scale"
                )

            if fold == 0:
                fit_rows = z["fit_task_rows"].astype(np.int64)
                ii = task_issue[fit_rows]
                ci = task_city[fit_rows]
                saved_med = stats["impute_median_train"]
                saved_mean = stats["feature_mean_after_impute_train"]
                saved_std = stats["feature_std_after_impute_train_raw"]
                saved_miss = stats["feature_missing_count_train"]

                for j, f in enumerate(order):
                    a = np.asarray(bank[f].values, dtype=np.float64)
                    x = a[ii, ci]
                    finite = np.isfinite(x)
                    if not finite.any():
                        recompute_fold0_failures.append(
                            f"{f}: no finite values"
                        )
                        continue
                    m = float(np.median(x[finite]))
                    xi = np.where(finite, x, m)
                    mu = float(np.mean(xi))
                    sd = float(np.std(xi, ddof=0))
                    miss = int((~finite).sum())

                    if not np.isclose(
                        saved_med[j], m, rtol=0, atol=1e-12
                    ):
                        recompute_fold0_failures.append(
                            f"{f}: median"
                        )
                    if not np.isclose(
                        saved_mean[j], mu, rtol=0, atol=1e-12
                    ):
                        recompute_fold0_failures.append(
                            f"{f}: mean"
                        )
                    if not np.isclose(
                        saved_std[j], sd, rtol=0, atol=1e-12
                    ):
                        recompute_fold0_failures.append(
                            f"{f}: std"
                        )
                    if int(saved_miss[j]) != miss:
                        recompute_fold0_failures.append(
                            f"{f}: missing_count"
                        )

                # Independently reproduce fold0 FIT target stats only.
                y = np.asarray(
                    merged["pm25_city_mean"].values,
                    dtype=np.float64,
                )[
                    task_target[fit_rows],
                    task_city[fit_rows],
                ]
                if not np.isfinite(y).all():
                    failures.append(
                        "fold 0 FIT target has non-finite values"
                    )
                saved_target = json.loads(
                    (fold_dir / "target_stats.json").read_text(
                        encoding="utf-8"
                    )
                )["statistics"]
                target_checks = {
                    "sample_count": int(len(y)),
                    "mean": float(np.mean(y)),
                    "std": float(np.std(y, ddof=0)),
                    "min": float(np.min(y)),
                    "max": float(np.max(y)),
                }
                for k, v in target_checks.items():
                    sv = saved_target[k]
                    if k == "sample_count":
                        if int(sv) != int(v):
                            failures.append(
                                f"fold0 target stat mismatch {k}"
                            )
                    elif not np.isclose(
                        float(sv), float(v), rtol=0, atol=1e-12
                    ):
                        failures.append(
                            f"fold0 target stat mismatch {k}"
                        )

        if recompute_fold0_failures:
            failures.append(
                "fold0 preprocessing reproduction failures: "
                + ", ".join(recompute_fold0_failures[:20])
            )

        out_root = (
            root / "06_processed" / "audits" / AUDIT_ID
        )
        out_root.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(selector_summary).to_csv(
            out_root / "selector_sample_counts.csv",
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
            "folds": 10,
            "feature_count": 72,
            "formal_model_eligible_global_samples": int(formal.sum()),
            "selector_checks": {
                "scenario_selectors_disjoint": disjoint_all,
                "scenario_selectors_exactly_partition_formal_eligible_rows": union_all,
                "selectors_match_frozen_fold_binding": exact_role_all,
            },
            "fit_boundary": {
                "no_2022_issue_centers_in_fit": no_2022_fit_issue,
                "no_2025_targets_in_fit": no_2025_fit,
                "fold0_all_72_preprocess_stats_recomputed": True,
                "fold0_preprocess_reproduction_failures": recompute_fold0_failures,
                "fold0_target_stats_recomputed": True,
            },
            "preprocessing_policy": {
                "imputation": "FIT-only per-feature median",
                "standardization": "FIT-only post-imputation mean/std where schema requests it",
                "missing_indicator_added": False,
            },
            "sealed_future_policy": {
                "2025_target_values_read": False,
                "2025_target_magnitudes_inspected": False,
                "2025_selectors_checked_structurally": True,
            },
            "outputs": ["selector_sample_counts.csv"],
        }
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
