#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


VERSION = "observed_fold_local_artifacts_v1"
PROTOCOL_ID = "national_pm25_ood_split_v1"
SCHEMA_ID = "national_pm25_observed_feature_schema_v1"
BINDING_ID = "observed_task_split_binding_v1"
ELIGIBILITY_POLICY_ID = "model_sample_eligibility_policy_v1"
EXPECTED_FOLDS = 10
EXPECTED_CITIES = 50

SCENARIOS = (
    "FIT",
    "VALIDATION",
    "SPATIAL_OOD_FINAL",
    "TEMPORAL_OOD_FINAL",
    "JOINT_OOD_FINAL",
    "RESERVED",
)

SCENARIO_KEY = {
    "FIT": "fit",
    "VALIDATION": "validation",
    "SPATIAL_OOD_FINAL": "spatial_ood_final",
    "TEMPORAL_OOD_FINAL": "temporal_ood_final",
    "JOINT_OOD_FINAL": "joint_ood_final",
    "RESERVED": "reserved",
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
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(obj: Any, path: Path):
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
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_feature_masks(schema):
    order = list(schema["feature_order"])
    recs = {r["feature"]: r for r in schema["feature_records"]}

    def by_family(names):
        return np.asarray(
            [recs[f]["family"] in names for f in order],
            dtype=bool,
        )

    default_core = np.asarray(
        [bool(recs[f]["default_core_predictor"]) for f in order],
        dtype=bool,
    )
    pm10 = by_family({"pm10_state"})

    masks = {
        "history_calendar_only": by_family(
            {"pm25_history", "calendar"}
        ),
        "core_without_pm10": default_core,
        "core_plus_pm10": default_core | pm10,
        "all_with_pressure": np.ones(len(order), dtype=bool),
        "meteorology_physics_only": by_family(
            {
                "calendar",
                "transport_dispersion",
                "moisture_stability_context",
                "precipitation_radiation",
            }
        ),
        "pollution_history_only": by_family(
            {
                "pm25_history",
                "calendar",
                "pollution_composition",
                "pm10_state",
            }
        ),
    }

    for fam in schema["families"]:
        masks[f"family__{fam['family']}"] = by_family({fam["family"]})

    for name, expected in EXPECTED_MASK_COUNTS.items():
        got = int(masks[name].sum())
        if got != expected:
            raise RuntimeError(
                f"{name}: expected {expected} features, got {got}"
            )

    manifest = {
        "schema_id": schema["schema_id"],
        "schema_fingerprint_sha256": schema["schema_fingerprint_sha256"],
        "feature_order": order,
        "masks": {
            name: {
                "feature_count": int(mask.sum()),
                "features": [
                    f for f, keep in zip(order, mask) if keep
                ],
            }
            for name, mask in masks.items()
        },
    }
    manifest["mask_manifest_fingerprint_sha256"] = canonical_sha256(manifest)
    return masks, manifest


def transform_policy(schema):
    order = list(schema["feature_order"])
    recs = {r["feature"]: r for r in schema["feature_records"]}
    classes = [str(recs[f]["transform_class"]) for f in order]
    standardize = np.asarray(
        [
            c in {
                "continuous_fold_local_standardization",
                "diagnostic_continuous_fold_local_scaling_if_used",
            }
            for c in classes
        ],
        dtype=bool,
    )
    families = [str(recs[f]["family"]) for f in order]
    statuses = [str(recs[f]["family_status"]) for f in order]
    return standardize, classes, families, statuses


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    bank_path = (
        root / "04_derived" / "issue_time_features"
        / SCHEMA_ID / "issue_time_feature_bank.zarr"
    )
    schema_path = (
        root / "05_manifests" / "features"
        / SCHEMA_ID / "feature_schema.json"
    )
    task_path = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1.parquet"
    )
    merged_path = (
        root / "04_derived" / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )
    binding_root = (
        root / "06_processed" / "split_bindings"
        / PROTOCOL_ID / BINDING_ID
    )
    binding_path = binding_root / "fold_city_year_roles.parquet"
    prior_count_path = binding_root / "fold_scenario_sample_counts.csv"
    eligibility_path = (
        root / "05_manifests" / f"{ELIGIBILITY_POLICY_ID}.json"
    )
    split_protocol_path = (
        root / "05_manifests" / "splits"
        / PROTOCOL_ID / "split_protocol.json"
    )

    required = [
        bank_path,
        schema_path,
        task_path,
        merged_path,
        binding_path,
        prior_count_path,
        eligibility_path,
        split_protocol_path,
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"missing prerequisite: {p}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    eligibility_policy = json.loads(
        eligibility_path.read_text(encoding="utf-8")
    )
    split_protocol = json.loads(
        split_protocol_path.read_text(encoding="utf-8")
    )

    if schema["schema_id"] != SCHEMA_ID:
        raise RuntimeError("schema id mismatch")
    if eligibility_policy["status"] != "FROZEN":
        raise RuntimeError("sample eligibility policy not frozen")
    if split_protocol["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("split protocol mismatch")

    out_root = (
        root / "06_processed" / "fold_artifacts"
        / PROTOCOL_ID / SCHEMA_ID
    )
    if out_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"output exists: {out_root}; use --overwrite intentionally"
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    masks, mask_manifest = build_feature_masks(schema)
    np.savez_compressed(
        out_root / "feature_masks.npz",
        **{k: v.astype(np.uint8) for k, v in masks.items()},
    )
    atomic_json(mask_manifest, out_root / "feature_masks.json")

    task = pd.read_parquet(
        task_path,
        columns=[
            "sample_id",
            "city_index",
            "city_en",
            "region",
            "issue_time_index",
            "target_time_index",
            "issue_time_local",
            "target_time_local",
            "target_local_year",
            "target_valid_benchmark",
        ],
    )

    if len(task) != 1_315_200:
        raise RuntimeError(f"unexpected task rows={len(task)}")
    if not task["sample_id"].is_unique:
        raise RuntimeError("task sample_id is not unique")

    task_city_idx = task["city_index"].to_numpy(dtype=np.int16)
    task_issue_idx = task["issue_time_index"].to_numpy(dtype=np.int32)
    task_target_idx = task["target_time_index"].to_numpy(dtype=np.int32)
    task_year = task["target_local_year"].to_numpy(dtype=np.int16)
    target_valid = task["target_valid_benchmark"].to_numpy(dtype=bool)

    issue_local = pd.DatetimeIndex(task["issue_time_local"])
    target_local = pd.DatetimeIndex(task["target_time_local"])
    same_local_year = (
        issue_local.year.to_numpy(dtype=np.int16)
        == target_local.year.to_numpy(dtype=np.int16)
    )
    formal_eligible = target_valid & same_local_year

    bank = xr.open_zarr(bank_path, consolidated=True, chunks=None)
    merged = xr.open_zarr(merged_path, consolidated=True, chunks=None)

    try:
        if bank.sizes.get("time") != 27040 or bank.sizes.get("city") != 50:
            raise RuntimeError("unexpected feature-bank dimensions")
        if merged.sizes.get("time") != 27040 or merged.sizes.get("city") != 50:
            raise RuntimeError("unexpected merged dimensions")
        if not np.array_equal(bank["time"].values, merged["time"].values):
            raise RuntimeError("bank/merged time coordinate mismatch")
        if not np.array_equal(bank["city"].values, merged["city"].values):
            raise RuntimeError("bank/merged city coordinate mismatch")

        city_en = [str(x) for x in bank["city_en"].values.tolist()]
        for ci, name in enumerate(city_en):
            q = task.loc[task["city_index"].eq(ci), "city_en"].unique()
            if len(q) != 1 or str(q[0]) != name:
                raise RuntimeError(
                    f"task city_index mapping mismatch at index {ci}"
                )

        binding = pd.read_parquet(binding_path)
        if len(binding) != 1500:
            raise RuntimeError("binding must have 1500 rows")

        # Scenario code matrix: fold x city_index x {2023,2024,2025}.
        years = [2023, 2024, 2025]
        year_pos = {y: i for i, y in enumerate(years)}
        scenario_to_code = {s: i for i, s in enumerate(SCENARIOS)}
        code_to_scenario = {i: s for s, i in scenario_to_code.items()}
        scenario_matrix = np.full(
            (EXPECTED_FOLDS, EXPECTED_CITIES, len(years)),
            -1,
            dtype=np.int8,
        )

        city_lookup = {c: i for i, c in enumerate(city_en)}
        for r in binding.itertuples(index=False):
            if r.city_en not in city_lookup:
                raise RuntimeError(f"binding unknown city {r.city_en}")
            if int(r.target_local_year) not in year_pos:
                raise RuntimeError("binding year outside 2023-2025")
            if r.scenario not in scenario_to_code:
                raise RuntimeError(f"unknown scenario {r.scenario}")
            scenario_matrix[
                int(r.fold),
                city_lookup[r.city_en],
                year_pos[int(r.target_local_year)],
            ] = scenario_to_code[r.scenario]

        if np.any(scenario_matrix < 0):
            raise RuntimeError("incomplete fold/city/year scenario matrix")

        ycol = np.asarray([year_pos[int(y)] for y in task_year], dtype=np.int8)

        selector_rows: dict[int, dict[str, np.ndarray]] = {}
        count_rows = []

        for fold in range(EXPECTED_FOLDS):
            row_codes = scenario_matrix[
                fold,
                task_city_idx,
                ycol,
            ]

            fold_selectors = {}
            for code, scenario in code_to_scenario.items():
                raw_role = row_codes == code
                selected = np.flatnonzero(
                    raw_role & formal_eligible
                ).astype(np.int32)
                fold_selectors[scenario] = selected

                count_rows.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "target_valid_before_same_year": int(
                            np.sum(raw_role & target_valid)
                        ),
                        "benchmark_valid_cross_year_excluded": int(
                            np.sum(
                                raw_role
                                & target_valid
                                & ~same_local_year
                            )
                        ),
                        "formal_model_eligible_samples": int(len(selected)),
                    }
                )

            # Every formal-eligible task row belongs to exactly one scenario.
            total = sum(len(v) for v in fold_selectors.values())
            if total != int(formal_eligible.sum()):
                raise RuntimeError(
                    f"fold {fold}: selector union count mismatch"
                )
            selector_rows[fold] = fold_selectors

            fold_dir = out_root / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                fold_dir / "scenario_task_rows.npz",
                **{
                    f"{SCENARIO_KEY[s]}_task_rows": rows
                    for s, rows in fold_selectors.items()
                },
            )

        count_df = pd.DataFrame(count_rows)
        count_df.to_csv(
            out_root / "formal_scenario_sample_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Feature transform policy.
        standardize, transform_classes, families, statuses = (
            transform_policy(schema)
        )
        order = list(schema["feature_order"])
        n_features = len(order)
        if n_features != 72:
            raise RuntimeError("expected 72 features")

        med = np.zeros((EXPECTED_FOLDS, n_features), dtype=np.float64)
        mean_imp = np.zeros_like(med)
        std_imp_raw = np.zeros_like(med)
        obs_min = np.zeros_like(med)
        obs_max = np.zeros_like(med)
        finite_count = np.zeros((EXPECTED_FOLDS, n_features), dtype=np.int64)
        missing_count = np.zeros_like(finite_count)

        fit_issue = {}
        fit_city = {}
        for fold in range(EXPECTED_FOLDS):
            rows = selector_rows[fold]["FIT"]
            if len(rows) == 0:
                raise RuntimeError(f"fold {fold}: empty FIT selector")
            fit_issue[fold] = task_issue_idx[rows]
            fit_city[fold] = task_city_idx[rows]

            # Hard boundary: issue centers used for fitting are local 2023/2024.
            fit_issue_year = issue_local.year.to_numpy(dtype=np.int16)[rows]
            if not np.isin(fit_issue_year, [2023, 2024]).all():
                raise RuntimeError(
                    f"fold {fold}: FIT contains issue centers outside 2023/2024"
                )

        for j, feature in enumerate(
            tqdm(order, desc="fold-local feature stats", unit="feature")
        ):
            a = np.asarray(bank[feature].values, dtype=np.float64)
            if a.shape != (27040, 50):
                raise RuntimeError(
                    f"{feature}: unexpected shape {a.shape}"
                )

            for fold in range(EXPECTED_FOLDS):
                x = a[
                    fit_issue[fold],
                    fit_city[fold],
                ]
                finite = np.isfinite(x)
                nfinite = int(finite.sum())
                nmiss = int(len(x) - nfinite)
                if nfinite == 0:
                    raise RuntimeError(
                        f"fold {fold} feature {feature}: all FIT values missing"
                    )

                xf = x[finite]
                m = float(np.median(xf))
                xi = np.where(finite, x, m)

                med[fold, j] = m
                mean_imp[fold, j] = float(np.mean(xi))
                std_imp_raw[fold, j] = float(np.std(xi, ddof=0))
                obs_min[fold, j] = float(np.min(xf))
                obs_max[fold, j] = float(np.max(xf))
                finite_count[fold, j] = nfinite
                missing_count[fold, j] = nmiss

        # Target statistics: FIT only, observed 2023/2024 only.
        target2d = np.asarray(
            merged["pm25_city_mean"].values,
            dtype=np.float64,
        )
        target_stats = {}

        for fold in range(EXPECTED_FOLDS):
            rows = selector_rows[fold]["FIT"]
            y = target2d[
                task_target_idx[rows],
                task_city_idx[rows],
            ]
            if not np.isfinite(y).all():
                raise RuntimeError(
                    f"fold {fold}: formal FIT contains non-finite target"
                )

            target_stats[fold] = {
                "sample_count": int(len(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y, ddof=0)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
            }

        # Drop the only target-value array before writing future-scenario metadata.
        del target2d

        fold_index = []
        for fold in range(EXPECTED_FOLDS):
            fold_dir = out_root / f"fold_{fold}"

            zero_var = standardize & (std_imp_raw[fold] < 1e-12)
            safe_std = std_imp_raw[fold].copy()
            safe_std[zero_var] = 1.0

            center = np.where(
                standardize,
                mean_imp[fold],
                0.0,
            ).astype(np.float64)
            scale = np.where(
                standardize,
                safe_std,
                1.0,
            ).astype(np.float64)

            np.savez_compressed(
                fold_dir / "preprocess_stats.npz",
                impute_median_train=med[fold],
                feature_mean_after_impute_train=mean_imp[fold],
                feature_std_after_impute_train_raw=std_imp_raw[fold],
                feature_observed_min_train=obs_min[fold],
                feature_observed_max_train=obs_max[fold],
                feature_finite_count_train=finite_count[fold],
                feature_missing_count_train=missing_count[fold],
                apply_standardization=standardize.astype(np.uint8),
                transform_center=center,
                transform_scale=scale,
                zero_variance_guard=zero_var.astype(np.uint8),
            )

            fit_n = int(len(selector_rows[fold]["FIT"]))
            table = pd.DataFrame(
                {
                    "feature_index": np.arange(n_features, dtype=int),
                    "feature": order,
                    "family": families,
                    "family_status": statuses,
                    "transform_class": transform_classes,
                    "fit_sample_count": fit_n,
                    "fit_finite_count": finite_count[fold],
                    "fit_missing_count": missing_count[fold],
                    "fit_missing_fraction": (
                        missing_count[fold] / float(fit_n)
                    ),
                    "impute_median_train": med[fold],
                    "mean_after_impute_train": mean_imp[fold],
                    "std_after_impute_train_raw": std_imp_raw[fold],
                    "apply_standardization": standardize,
                    "effective_center": center,
                    "effective_scale": scale,
                    "zero_variance_guard_applied": zero_var,
                }
            )
            table.to_csv(
                fold_dir / "preprocess_stats.csv",
                index=False,
                encoding="utf-8-sig",
            )

            atomic_json(
                {
                    "fold": fold,
                    "target": "pm25_city_mean at target_time_index",
                    "fit_scope": (
                        "formal FIT samples only: train cities, target years "
                        "2023/2024, target_valid_benchmark=True, same local "
                        "issue/target year"
                    ),
                    "statistics": target_stats[fold],
                    "target_transform_applied_by_builder": False,
                },
                fold_dir / "target_stats.json",
            )

            preprocess_hash = sha256_file(
                fold_dir / "preprocess_stats.npz"
            )
            scenario_hash = sha256_file(
                fold_dir / "scenario_task_rows.npz"
            )

            artifact = {
                "version": VERSION,
                "created_at_utc": now_utc(),
                "fold": fold,
                "protocol_id": PROTOCOL_ID,
                "schema_id": SCHEMA_ID,
                "schema_fingerprint_sha256": schema[
                    "schema_fingerprint_sha256"
                ],
                "eligibility_policy_id": ELIGIBILITY_POLICY_ID,
                "feature_bank": str(bank_path),
                "task_index": str(task_path),
                "merged_target_store": str(merged_path),
                "feature_bank_data_copied": False,
                "target_data_copied": False,
                "fit_sample_count": int(
                    len(selector_rows[fold]["FIT"])
                ),
                "validation_sample_count": int(
                    len(selector_rows[fold]["VALIDATION"])
                ),
                "preprocessing": {
                    "imputation": (
                        "per-feature median fitted on formal FIT only"
                    ),
                    "standardization": (
                        "schema-designated continuous features only; "
                        "mean/std after FIT-median imputation"
                    ),
                    "missing_indicator_added": False,
                    "preprocess_stats_sha256": preprocess_hash,
                    "zero_variance_guard_count": int(zero_var.sum()),
                },
                "target_statistics": {
                    "fit_on": "formal FIT only",
                    "applied_by_builder": False,
                },
                "scenario_task_rows_sha256": scenario_hash,
                "future_2025": {
                    "selectors_materialized": True,
                    "target_values_read": False,
                    "used_for_preprocessing_fit": False,
                    "used_for_model_selection": False,
                },
            }
            artifact["artifact_fingerprint_sha256"] = canonical_sha256(
                artifact
            )
            atomic_json(
                artifact,
                fold_dir / "fold_artifact.json",
            )

            fold_index.append(
                {
                    "fold": fold,
                    "fit_sample_count": int(
                        len(selector_rows[fold]["FIT"])
                    ),
                    "validation_sample_count": int(
                        len(selector_rows[fold]["VALIDATION"])
                    ),
                    "preprocess_stats_sha256": preprocess_hash,
                    "scenario_task_rows_sha256": scenario_hash,
                    "artifact_fingerprint_sha256": artifact[
                        "artifact_fingerprint_sha256"
                    ],
                    "zero_variance_guard_count": int(zero_var.sum()),
                    "target_mean_fit": target_stats[fold]["mean"],
                    "target_std_fit": target_stats[fold]["std"],
                }
            )

        pd.DataFrame(fold_index).to_csv(
            out_root / "fold_artifact_index.csv",
            index=False,
            encoding="utf-8-sig",
        )

        build_report = {
            "version": VERSION,
            "status": "PASS",
            "created_at_utc": now_utc(),
            "protocol_id": PROTOCOL_ID,
            "schema_id": SCHEMA_ID,
            "schema_fingerprint_sha256": schema[
                "schema_fingerprint_sha256"
            ],
            "eligibility_policy_id": ELIGIBILITY_POLICY_ID,
            "fold_count": EXPECTED_FOLDS,
            "feature_count": n_features,
            "global_formal_model_eligible_samples": int(
                formal_eligible.sum()
            ),
            "feature_preprocessing_fit_scope": (
                "per fold: formal FIT samples only"
            ),
            "target_statistics_fit_scope": (
                "per fold: formal FIT samples only"
            ),
            "future_2025_target_values_read": False,
            "future_2025_used_for_preprocessing_fit": False,
            "future_2025_used_for_model_selection": False,
            "provenance": {
                "schema": str(schema_path),
                "schema_sha256": sha256_file(schema_path),
                "task_index": str(task_path),
                "task_index_sha256": sha256_file(task_path),
                "binding": str(binding_path),
                "binding_sha256": sha256_file(binding_path),
                "eligibility_policy": str(eligibility_path),
                "eligibility_policy_sha256": sha256_file(eligibility_path),
                "split_protocol": str(split_protocol_path),
                "split_protocol_sha256": sha256_file(split_protocol_path),
            },
        }
        atomic_json(build_report, out_root / "build_report.json")

    finally:
        bank.close()
        merged.close()


if __name__ == "__main__":
    main()
