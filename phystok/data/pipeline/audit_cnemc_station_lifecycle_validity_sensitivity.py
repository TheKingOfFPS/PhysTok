#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


AUDIT_ID = "cnemc_station_lifecycle_validity_sensitivity_v1"
YEARS = (2023, 2024, 2025)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument(
        "--roster-audit-root",
        type=Path,
        default=None,
        help=(
            "Defaults to ROOT/06_processed/audits/"
            "cnemc_station_registry_50city_roster_v1"
        ),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "Defaults to ROOT/06_processed/audits/"
            "cnemc_station_lifecycle_validity_sensitivity_v1"
        ),
    )
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise last if last is not None else RuntimeError("CSV decode failed")


def hj_required_count(expected_count: int) -> int:
    if expected_count <= 0:
        return 1
    fraction = 0.50 if expected_count < 4 else 0.75
    return int(math.ceil(fraction * expected_count))


def mode_positive(values: np.ndarray) -> int:
    x = values[values > 0]
    if x.size == 0:
        return 0
    counts = np.bincount(x.astype(np.int64))
    # In a tie, choose the larger count as the conservative estimate
    # of the expected active network size.
    winners = np.flatnonzero(counts == counts.max())
    return int(winners.max())


def quantile_higher(values: np.ndarray, q: float) -> int:
    x = values[values > 0]
    if x.size == 0:
        return 0
    try:
        return int(np.quantile(x, q, method="higher"))
    except TypeError:
        return int(np.quantile(x, q, interpolation="higher"))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    sites_root = (
        root
        / "01_raw"
        / "air_quality_cnemc_quotsoft"
        / "sites"
        / "daily"
    )
    roster_root = (
        args.roster_audit_root.expanduser().resolve()
        if args.roster_audit_root is not None
        else root
        / "06_processed"
        / "audits"
        / "cnemc_station_registry_50city_roster_v1"
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else root
        / "06_processed"
        / "audits"
        / AUDIT_ID
    )
    out_root.mkdir(parents=True, exist_ok=True)

    roster = pd.read_csv(
        roster_root / "50city_station_roster_review.csv",
        encoding="utf-8-sig",
    )
    monthly = pd.read_csv(
        roster_root / "station_monthly_pm25_coverage.csv",
        encoding="utf-8-sig",
    )

    noncontrol = roster.loc[roster["control_site"].ne("Y")].copy()
    noncontrol_ids = noncontrol["station_id"].astype(str).tolist()

    city_meta = (
        noncontrol[
            ["region", "city_en", "city_zh", "station_id"]
        ]
        .copy()
    )
    station_to_city = city_meta.set_index("station_id")[
        ["region", "city_en", "city_zh"]
    ]

    city_station_ids = {
        key: g["station_id"].astype(str).tolist()
        for key, g in noncontrol.groupby(
            ["region", "city_en", "city_zh"],
            sort=True,
        )
    }

    monthly_nc = monthly.loc[
        monthly["control_site"].ne("Y")
        & monthly["month"].str[:4].astype(int).isin(YEARS)
    ].copy()

    # Station-month activity candidates based only on missingness coverage.
    active_rows = []
    for row in monthly_nc.itertuples(index=False):
        cov = float(row.pm25_archive_coverage)
        active_rows.append(
            {
                "region": row.region,
                "city_en": row.city_en,
                "city_zh": row.city_zh,
                "station_id": row.station_id,
                "station_name": row.station_name,
                "month": row.month,
                "pm25_archive_coverage": cov,
                "active_cov25": bool(cov >= 0.25),
                "active_cov50": bool(cov >= 0.50),
                "active_cov75": bool(cov >= 0.75),
            }
        )
    active_df = pd.DataFrame(active_rows)

    active_count = (
        active_df.groupby(
            ["region", "city_en", "city_zh", "month"],
            as_index=False,
        )
        .agg(
            active_count_cov25=("active_cov25", "sum"),
            active_count_cov50=("active_cov50", "sum"),
            active_count_cov75=("active_cov75", "sum"),
            registry_noncontrol_count=("station_id", "size"),
        )
    )

    # Read only PM2.5 finite/non-finite masks. Numeric target values are never
    # summarized, persisted, or inspected, including 2025.
    hourly_records = []
    daily_files = []
    for year in YEARS:
        daily_files.extend(
            sorted((sites_root / str(year)).glob("china_sites_*.csv"))
        )

    usecols = ["date", "hour", "type"] + noncontrol_ids

    for path in tqdm(
        daily_files,
        desc="Lifecycle validity",
        unit="day",
    ):
        df = read_csv_flexible(path, usecols=usecols)
        pm = df.loc[df["type"].astype(str).str.strip().eq("PM2.5")].copy()
        if pm.empty:
            continue

        date_str = str(int(pm["date"].iloc[0]))
        month = f"{date_str[:4]}-{date_str[4:6]}"

        for (region, city_en, city_zh), ids in city_station_ids.items():
            block = pm[ids]
            nvalid = block.notna().sum(axis=1).to_numpy(dtype=np.int64)

            for hour, nv in zip(pm["hour"].to_numpy(), nvalid):
                hourly_records.append(
                    {
                        "date": date_str,
                        "month": month,
                        "year": int(date_str[:4]),
                        "hour": int(hour),
                        "region": region,
                        "city_en": city_en,
                        "city_zh": city_zh,
                        "n_valid_noncontrol": int(nv),
                    }
                )

    hourly = pd.DataFrame(hourly_records)

    # Infer expected city network size from hourly missingness only.
    inferred_rows = []
    for key, g in hourly.groupby(
        ["region", "city_en", "city_zh", "month"],
        sort=True,
    ):
        region, city_en, city_zh, month = key
        values = g["n_valid_noncontrol"].to_numpy(dtype=np.int64)
        inferred_rows.append(
            {
                "region": region,
                "city_en": city_en,
                "city_zh": city_zh,
                "month": month,
                "expected_count_mode": mode_positive(values),
                "expected_count_q95": quantile_higher(values, 0.95),
                "expected_count_q99": quantile_higher(values, 0.99),
                "source_rows": len(g),
                "zero_station_rows": int((values == 0).sum()),
            }
        )
    inferred = pd.DataFrame(inferred_rows)

    monthly_counts = active_count.merge(
        inferred,
        on=["region", "city_en", "city_zh", "month"],
        how="outer",
        validate="one_to_one",
    )

    for col in (
        "active_count_cov25",
        "active_count_cov50",
        "active_count_cov75",
        "expected_count_mode",
        "expected_count_q95",
        "expected_count_q99",
    ):
        monthly_counts[f"required_hj__{col}"] = (
            monthly_counts[col]
            .fillna(0)
            .astype(int)
            .map(hj_required_count)
        )

    # Candidate gates to compare. COV25/COV75 are retained as sensitivity
    # bounds; the main three candidates are COV50, MODE, and Q95.
    gate_specs = {
        "ACTIVE_COV25": "active_count_cov25",
        "ACTIVE_COV50": "active_count_cov50",
        "ACTIVE_COV75": "active_count_cov75",
        "NVALID_MODE": "expected_count_mode",
        "NVALID_Q95": "expected_count_q95",
        "NVALID_Q99": "expected_count_q99",
    }

    hourly = hourly.merge(
        monthly_counts[
            [
                "region",
                "city_en",
                "city_zh",
                "month",
            ]
            + list(
                dict.fromkeys(
                    list(gate_specs.values())
                    + [
                        f"required_hj__{x}"
                        for x in gate_specs.values()
                    ]
                )
            )
        ],
        on=["region", "city_en", "city_zh", "month"],
        how="left",
        validate="many_to_one",
    )

    for gate_name, expected_col in gate_specs.items():
        req_col = f"required_hj__{expected_col}"
        hourly[f"pass__{gate_name}"] = (
            hourly["n_valid_noncontrol"] >= hourly[req_col]
        )

    retention_rows = []
    for (
        region,
        city_en,
        city_zh,
        year,
    ), g in hourly.groupby(
        ["region", "city_en", "city_zh", "year"],
        sort=True,
    ):
        for gate_name in gate_specs:
            passed = g[f"pass__{gate_name}"]
            retention_rows.append(
                {
                    "region": region,
                    "city_en": city_en,
                    "city_zh": city_zh,
                    "year": int(year),
                    "gate": gate_name,
                    "source_rows": len(g),
                    "pass_rows": int(passed.sum()),
                    "fail_rows": int((~passed).sum()),
                    "pass_fraction": float(passed.mean()),
                }
            )
    retention = pd.DataFrame(retention_rows)

    # Pairwise mask agreement across all city-hours and by year.
    main_gates = (
        "ACTIVE_COV50",
        "NVALID_MODE",
        "NVALID_Q95",
    )
    agreement_rows = []
    year_groups = [("ALL_2023_2025", hourly)]
    year_groups += [
        (str(year), hourly.loc[hourly["year"].eq(year)])
        for year in YEARS
    ]

    for scope, g in year_groups:
        for i, a in enumerate(main_gates):
            for b in main_gates[i + 1 :]:
                aa = g[f"pass__{a}"].to_numpy(dtype=bool)
                bb = g[f"pass__{b}"].to_numpy(dtype=bool)
                both_pass = aa & bb
                union_pass = aa | bb
                agreement_rows.append(
                    {
                        "scope": scope,
                        "gate_a": a,
                        "gate_b": b,
                        "rows": len(g),
                        "same_mask_rows": int((aa == bb).sum()),
                        "agreement_fraction": float(np.mean(aa == bb)),
                        "jaccard_pass": (
                            float(both_pass.sum() / union_pass.sum())
                            if union_pass.any()
                            else 1.0
                        ),
                        "a_pass_b_fail": int((aa & ~bb).sum()),
                        "a_fail_b_pass": int((~aa & bb).sum()),
                    }
                )
    agreement = pd.DataFrame(agreement_rows)

    # Lifecycle changes for the coverage-based expected station counts.
    lifecycle_rows = []
    for key, g in monthly_counts.groupby(
        ["region", "city_en", "city_zh"],
        sort=True,
    ):
        g = g.sort_values("month")
        prev = None
        for row in g.itertuples(index=False):
            cur = int(row.active_count_cov50)
            if prev is None or cur != prev:
                lifecycle_rows.append(
                    {
                        "region": key[0],
                        "city_en": key[1],
                        "city_zh": key[2],
                        "month": row.month,
                        "active_count_cov50": cur,
                        "previous_active_count_cov50": (
                            "" if prev is None else prev
                        ),
                        "delta": "" if prev is None else cur - prev,
                    }
                )
            prev = cur
    lifecycle = pd.DataFrame(lifecycle_rows)

    # Persist masks as compact CSV.gz. No pollutant numeric values are present.
    hourly_out_cols = [
        "date",
        "month",
        "year",
        "hour",
        "region",
        "city_en",
        "city_zh",
        "n_valid_noncontrol",
    ]
    for gate_name, expected_col in gate_specs.items():
        hourly_out_cols.extend(
            [
                expected_col,
                f"required_hj__{expected_col}",
                f"pass__{gate_name}",
            ]
        )
    hourly[hourly_out_cols].to_csv(
        out_root / "hourly_validity_masks_2023_2025.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    active_df.to_csv(
        out_root / "station_month_activity_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_counts.to_csv(
        out_root / "monthly_expected_station_count_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retention.to_csv(
        out_root / "city_year_gate_retention.csv",
        index=False,
        encoding="utf-8-sig",
    )
    agreement.to_csv(
        out_root / "gate_mask_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lifecycle.to_csv(
        out_root / "active_count_change_points_cov50.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_retention = (
        retention.groupby("gate")
        .agg(
            source_rows=("source_rows", "sum"),
            pass_rows=("pass_rows", "sum"),
            fail_rows=("fail_rows", "sum"),
        )
        .reset_index()
    )
    global_retention["pass_fraction"] = (
        global_retention["pass_rows"] / global_retention["source_rows"]
    )

    summary = {
        "audit_id": AUDIT_ID,
        "created_at_utc": utc_now(),
        "status": "PASS",
        "scope": {
            "years": list(YEARS),
            "selected_cities": 50,
            "pollutant": "PM2.5",
            "numeric_pollutant_values_summarized": False,
            "year_2025_numeric_values_inspected": False,
            "only_missingness_masks_used": True,
        },
        "candidate_expected_station_count_methods": {
            "ACTIVE_COV25": (
                "number of non-control stations with monthly PM2.5 "
                "archive coverage >= 0.25"
            ),
            "ACTIVE_COV50": (
                "number of non-control stations with monthly PM2.5 "
                "archive coverage >= 0.50"
            ),
            "ACTIVE_COV75": (
                "number of non-control stations with monthly PM2.5 "
                "archive coverage >= 0.75"
            ),
            "NVALID_MODE": (
                "monthly mode of positive hourly non-control station counts"
            ),
            "NVALID_Q95": (
                "monthly 95th percentile (higher) of positive hourly "
                "non-control station counts"
            ),
            "NVALID_Q99": (
                "monthly 99th percentile (higher) of positive hourly "
                "non-control station counts"
            ),
        },
        "validity_rule": (
            "HJ 663-2013-inspired count gate: require >=75% of expected "
            "city points; when expected count <4, require >=50%. "
            "This audit compares ways to infer the time-varying expected "
            "active station count from archive missingness and does not yet "
            "freeze a benchmark rule."
        ),
        "global_retention": global_retention.to_dict(orient="records"),
        "main_gate_pairwise_agreement": agreement.loc[
            agreement["scope"].eq("ALL_2023_2025")
        ].to_dict(orient="records"),
        "outputs": [
            "station_month_activity_candidates.csv",
            "monthly_expected_station_count_sensitivity.csv",
            "city_year_gate_retention.csv",
            "gate_mask_agreement.csv",
            "active_count_change_points_cov50.csv",
            "hourly_validity_masks_2023_2025.csv.gz",
        ],
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    archive = out_root.parent / f"{AUDIT_ID}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_root, arcname=out_root.name)


if __name__ == "__main__":
    main()
