#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BINDING_ID = "observed_task_split_binding_v1"
SOURCE_SPLIT_ID = "national_pm25_ood_split_v1"
TASK_INDEX_ID = "pm25_24h_task_index_v1"

ROLE_MAP = {
    ("train", 2023): "FIT",
    ("train", 2024): "FIT",
    ("validation", 2023): "VALIDATION",
    ("validation", 2024): "VALIDATION",
    ("test", 2023): "SPATIAL_OOD_FINAL",
    ("test", 2024): "SPATIAL_OOD_FINAL",
    ("train", 2025): "TEMPORAL_OOD_FINAL",
    ("validation", 2025): "RESERVED",
    ("test", 2025): "JOINT_OOD_FINAL",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    split_root = (
        root / "05_manifests" / "splits" / SOURCE_SPLIT_ID
    )
    protocol_path = split_root / "split_protocol.json"
    fold_roles_path = split_root / "fold_city_roles.csv"
    assignment_path = root / "05_manifests" / "cross_city_fold_assignment.json"

    task_path = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1.parquet"
    )
    task_manifest_path = (
        root / "06_processed" / "task_index"
        / "pm25_24h_task_index_v1_manifest.json"
    )
    benchmark_policy_path = (
        root / "05_manifests" / "benchmark_validity_policy_v1.json"
    )

    required = [
        protocol_path,
        fold_roles_path,
        assignment_path,
        task_path,
        task_manifest_path,
        benchmark_policy_path,
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"missing prerequisite: {p}")

    with protocol_path.open("r", encoding="utf-8") as f:
        protocol = json.load(f)
    with task_manifest_path.open("r", encoding="utf-8") as f:
        task_manifest = json.load(f)
    with benchmark_policy_path.open("r", encoding="utf-8") as f:
        benchmark_policy = json.load(f)

    if protocol.get("protocol_id") != SOURCE_SPLIT_ID:
        raise RuntimeError("unexpected split protocol id")
    if task_manifest.get("index_id") != TASK_INDEX_ID:
        raise RuntimeError("unexpected task index id")
    if benchmark_policy.get("status") != "FROZEN":
        raise RuntimeError("benchmark validity policy is not frozen")

    roles = pd.read_csv(fold_roles_path, encoding="utf-8-sig")
    needed_cols = {"fold", "city", "region", "city_role"}
    if not needed_cols.issubset(roles.columns):
        raise RuntimeError(
            f"fold_city_roles.csv missing columns "
            f"{sorted(needed_cols - set(roles.columns))}"
        )

    roles["fold"] = roles["fold"].astype(int)
    roles["city"] = roles["city"].astype(str)
    roles["region"] = roles["region"].astype(str)
    roles["city_role"] = roles["city_role"].astype(str)

    expected_folds = list(range(10))
    if sorted(roles["fold"].unique().tolist()) != expected_folds:
        raise RuntimeError("fold IDs are not exactly 0..9")

    task = pd.read_parquet(
        task_path,
        columns=[
            "sample_id",
            "city_index",
            "region",
            "city_en",
            "target_local_year",
            "target_valid_benchmark",
        ],
    )
    task["city_en"] = task["city_en"].astype(str)
    task["region"] = task["region"].astype(str)
    task["target_local_year"] = task["target_local_year"].astype(int)

    task_cities = (
        task[["city_en", "region"]]
        .drop_duplicates()
        .sort_values("city_en")
        .reset_index(drop=True)
    )
    role_cities = (
        roles[["city", "region"]]
        .drop_duplicates()
        .rename(columns={"city": "city_en"})
        .sort_values("city_en")
        .reset_index(drop=True)
    )

    if not task_cities.equals(role_cities):
        merged_check = task_cities.merge(
            role_cities,
            on="city_en",
            how="outer",
            suffixes=("_task", "_split"),
            indicator=True,
        )
        raise RuntimeError(
            "task/split city identity mismatch:\n"
            + merged_check.to_string(index=False)
        )

    # Validate each frozen fold JSON against fold_city_roles.csv.
    fold_json_fingerprints = {}
    for fold in expected_folds:
        fp = split_root / f"fold_{fold}.json"
        if not fp.exists():
            raise RuntimeError(f"missing {fp}")
        with fp.open("r", encoding="utf-8") as f:
            fj = json.load(f)

        if int(fj["fold"]) != fold:
            raise RuntimeError(f"fold_{fold}.json contains wrong fold id")

        rg = roles.loc[roles["fold"].eq(fold)]
        expected = {
            "train": sorted(rg.loc[rg["city_role"].eq("train"), "city"].tolist()),
            "validation": sorted(
                rg.loc[rg["city_role"].eq("validation"), "city"].tolist()
            ),
            "test": sorted(rg.loc[rg["city_role"].eq("test"), "city"].tolist()),
        }
        observed = {
            "train": sorted(map(str, fj["train_cities"])),
            "validation": sorted(map(str, fj["validation_cities"])),
            "test": sorted(map(str, fj["test_cities"])),
        }
        if expected != observed:
            raise RuntimeError(f"fold {fold} JSON disagrees with fold_city_roles.csv")

        fold_json_fingerprints[str(fold)] = sha256_file(fp)

    # Create compact authoritative city-year role binding:
    # 10 folds * 50 cities * 3 years = 1500 rows.
    binding_rows = []
    for row in roles.itertuples(index=False):
        for year in (2023, 2024, 2025):
            scenario = ROLE_MAP.get((row.city_role, year))
            if scenario is None:
                raise RuntimeError(
                    f"no scenario mapping for role={row.city_role}, year={year}"
                )
            binding_rows.append(
                {
                    "fold": int(row.fold),
                    "city_en": row.city,
                    "region": row.region,
                    "city_role": row.city_role,
                    "target_local_year": year,
                    "scenario": scenario,
                    "fit_model_weights": scenario == "FIT",
                    "fit_data_dependent_transforms": scenario == "FIT",
                    "allow_model_selection": scenario == "VALIDATION",
                    "final_evaluation": scenario in {
                        "SPATIAL_OOD_FINAL",
                        "TEMPORAL_OOD_FINAL",
                        "JOINT_OOD_FINAL",
                    },
                    "sealed_reserved": scenario == "RESERVED",
                }
            )

    binding = pd.DataFrame(binding_rows).sort_values(
        ["fold", "target_local_year", "region", "city_en"]
    ).reset_index(drop=True)

    # Count benchmark-valid samples by fold/scenario WITHOUT reading target values.
    # This join is metadata + availability-mask only.
    sample_counts = (
        task.groupby(
            ["city_en", "region", "target_local_year"],
            as_index=False,
        )
        .agg(
            structural_samples=("sample_id", "size"),
            benchmark_valid_samples=("target_valid_benchmark", "sum"),
        )
    )

    count_table = binding.merge(
        sample_counts,
        on=["city_en", "region", "target_local_year"],
        how="left",
        validate="many_to_one",
    )
    if count_table[
        ["structural_samples", "benchmark_valid_samples"]
    ].isna().any().any():
        raise RuntimeError("missing task-index counts after role binding")

    fold_scenario_counts = (
        count_table.groupby(
            ["fold", "scenario"],
            as_index=False,
        )
        .agg(
            cities=("city_en", "nunique"),
            structural_samples=("structural_samples", "sum"),
            benchmark_valid_samples=("benchmark_valid_samples", "sum"),
        )
    )
    fold_scenario_counts["benchmark_retention_fraction"] = (
        fold_scenario_counts["benchmark_valid_samples"]
        / fold_scenario_counts["structural_samples"]
    )

    # High-level invariants.
    for fold in expected_folds:
        fr = roles.loc[roles["fold"].eq(fold)]
        counts = fr["city_role"].value_counts().to_dict()
        if counts != {"train": 40, "validation": 5, "test": 5}:
            raise RuntimeError(
                f"fold {fold} role counts are {counts}, expected 40/5/5"
            )
        for region, rg in fr.groupby("region"):
            rc = rg["city_role"].value_counts().to_dict()
            if rc != {"train": 8, "validation": 1, "test": 1}:
                raise RuntimeError(
                    f"fold {fold}, region {region}: {rc}, expected 8/1/1"
                )

    coverage = roles.groupby(["city", "city_role"]).size().unstack(fill_value=0)
    for role in ("validation", "test"):
        if not np.all(coverage[role].to_numpy() == 1):
            raise RuntimeError(f"not every city is {role} exactly once")
    if not np.all(coverage["train"].to_numpy() == 8):
        raise RuntimeError("not every city is train exactly eight times")

    out_root = (
        root / "06_processed" / "split_bindings"
        / SOURCE_SPLIT_ID / BINDING_ID
    )
    out_root.mkdir(parents=True, exist_ok=True)

    binding_path = out_root / "fold_city_year_roles.parquet"
    binding.to_parquet(binding_path, index=False)

    counts_path = out_root / "fold_scenario_sample_counts.csv"
    fold_scenario_counts.to_csv(
        counts_path,
        index=False,
        encoding="utf-8-sig",
    )

    binding_csv = out_root / "fold_city_year_roles.csv"
    binding.to_csv(binding_csv, index=False, encoding="utf-8-sig")

    binding_fingerprint = canonical_json_sha256(
        binding[
            [
                "fold",
                "city_en",
                "region",
                "city_role",
                "target_local_year",
                "scenario",
            ]
        ].to_dict(orient="records")
    )

    manifest = {
        "binding_id": BINDING_ID,
        "created_at_utc": utc_now(),
        "source_split_protocol_id": SOURCE_SPLIT_ID,
        "task_index_id": TASK_INDEX_ID,
        "rows": int(len(binding)),
        "folds": 10,
        "cities": 50,
        "years": [2023, 2024, 2025],
        "scenario_map": {
            "train_x_2023_2024": "FIT",
            "validation_x_2023_2024": "VALIDATION",
            "test_x_2023_2024": "SPATIAL_OOD_FINAL",
            "train_x_2025": "TEMPORAL_OOD_FINAL",
            "validation_x_2025": "RESERVED",
            "test_x_2025": "JOINT_OOD_FINAL",
        },
        "binding_fingerprint_sha256": binding_fingerprint,
        "provenance": {
            "split_protocol": str(protocol_path),
            "split_protocol_sha256": sha256_file(protocol_path),
            "fold_city_roles": str(fold_roles_path),
            "fold_city_roles_sha256": sha256_file(fold_roles_path),
            "cross_city_fold_assignment": str(assignment_path),
            "cross_city_fold_assignment_sha256": sha256_file(assignment_path),
            "fold_json_sha256": fold_json_fingerprints,
            "task_index": str(task_path),
            "task_index_sha256": sha256_file(task_path),
            "task_index_manifest": str(task_manifest_path),
            "task_index_manifest_sha256": sha256_file(task_manifest_path),
            "benchmark_validity_policy": str(benchmark_policy_path),
            "benchmark_validity_policy_sha256": sha256_file(
                benchmark_policy_path
            ),
        },
        "important": {
            "city_assignment_reused_exactly": True,
            "new_random_split_generated": False,
            "target_values_read": False,
            "2025_target_magnitudes_read": False,
            "benchmark_validity_mask_used_only_for_sample_counts": True,
            "authoritative_binding_granularity": "fold x city x target_local_year",
            "expanded_13m_row_sample_assignment_not_materialized": True,
        },
    }

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
