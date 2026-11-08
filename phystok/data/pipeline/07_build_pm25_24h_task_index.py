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


INDEX_ID = "pm25_24h_task_index_v1"
HORIZON_HOURS = 24
TARGET_LOCAL_START = pd.Timestamp("2023-01-01 00:00:00")
TARGET_LOCAL_END = pd.Timestamp("2025-12-31 23:00:00")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    return p.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    merged_path = (
        root / "04_derived" / "observed_era5_merged"
        / "observed_era5_50cities_v1.zarr"
    )
    policy_path = root / "05_manifests" / "benchmark_validity_policy_v1.json"

    if not merged_path.exists():
        raise SystemExit(f"missing merged store: {merged_path}")
    if not policy_path.exists():
        raise SystemExit("run freeze_benchmark_validity_policy.py first")

    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)

    if policy.get("status") != "FROZEN":
        raise SystemExit("benchmark validity policy is not frozen")
    if int(policy["task"]["forecast_horizon_hours"]) != HORIZON_HOURS:
        raise SystemExit("policy horizon mismatch")

    ds = xr.open_zarr(merged_path, consolidated=True)

    utc = pd.DatetimeIndex(ds["time"].values)
    local = pd.DatetimeIndex(ds["local_time"].values)

    if not np.all((local - utc) == pd.Timedelta(hours=8)):
        raise RuntimeError("local/UTC coordinate mismatch")

    target_mask = (
        (local >= TARGET_LOCAL_START)
        & (local <= TARGET_LOCAL_END)
    )
    target_pos = np.flatnonzero(target_mask)

    if len(target_pos) != (8760 + 8784 + 8760):
        raise RuntimeError(
            f"unexpected number of target hours={len(target_pos)}"
        )

    # Structural issue time is exactly 24 hours before target time.
    issue_utc = utc[target_pos] - pd.Timedelta(hours=HORIZON_HOURS)
    issue_local = local[target_pos] - pd.Timedelta(hours=HORIZON_HOURS)

    utc_to_pos = {ts: i for i, ts in enumerate(utc)}
    issue_pos = np.array(
        [utc_to_pos.get(ts, -1) for ts in issue_utc],
        dtype=np.int32,
    )
    if np.any(issue_pos < 0):
        bad = issue_utc[issue_pos < 0]
        raise RuntimeError(
            f"{len(bad)} issue times absent from merged store; first={bad[0]}"
        )

    city_en = [str(x) for x in ds["city_en"].values.tolist()]
    city_zh = [str(x) for x in ds["city_zh"].values.tolist()]
    region = [str(x) for x in ds["region"].values.tolist()]
    n_city = len(city_en)

    valid = ds["pm25_valid_hj_q95"].values[
        target_pos, :
    ].astype(bool)

    # Target magnitudes are deliberately NOT loaded into this index.
    rows = []
    for h, tp in enumerate(target_pos):
        t_utc = utc[tp]
        t_local = local[tp]
        i_utc = issue_utc[h]
        i_local = issue_local[h]
        ip = int(issue_pos[h])

        for ci in range(n_city):
            rows.append(
                {
                    "sample_id": f"{t_local:%Y%m%d%H}_{city_en[ci]}",
                    "city_index": ci,
                    "region": region[ci],
                    "city_en": city_en[ci],
                    "city_zh": city_zh[ci],
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

    index = pd.DataFrame(rows)

    out_dir = root / "06_processed" / "task_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "pm25_24h_task_index_v1.parquet"
    index.to_parquet(out_path, index=False)

    counts = {}
    for year, g in index.groupby("target_local_year"):
        counts[str(year)] = {
            "all_structural_samples": int(len(g)),
            "benchmark_valid_samples": int(
                g["target_valid_benchmark"].sum()
            ),
            "retention_fraction": float(
                g["target_valid_benchmark"].mean()
            ),
        }

    manifest = {
        "index_id": INDEX_ID,
        "created_at_utc": utc_now(),
        "output": str(out_path),
        "rows": int(len(index)),
        "cities": n_city,
        "forecast_horizon_hours": HORIZON_HOURS,
        "target_local_start": str(TARGET_LOCAL_START),
        "target_local_end": str(TARGET_LOCAL_END),
        "target_validity_variable": "pm25_valid_hj_q95",
        "target_value_materialized_in_index": False,
        "counts_by_target_local_year": counts,
        "important": {
            "issue_time_exactly_target_minus_24h": True,
            "all_issue_times_exist_in_merged_store": True,
            "predictor_history_completeness_not_required_here": True,
            "2025_target_magnitudes_read": False,
            "2025_used_for_model_selection": False,
            "task_index_is_split_agnostic": True,
        },
        "provenance": {
            "merged_store": str(merged_path),
            "benchmark_policy": str(policy_path),
            "benchmark_policy_sha256": sha256_file(policy_path),
        },
    }

    manifest_path = out_dir / "pm25_24h_task_index_v1_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
