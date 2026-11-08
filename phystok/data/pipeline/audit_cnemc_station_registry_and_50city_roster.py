#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


AUDIT_ID = "cnemc_station_registry_50city_roster_v1"
POLLUTANTS = ("PM2.5", "PM10", "SO2", "NO2", "O3", "CO")
PROTOCOL_YEARS = (2023, 2024, 2025)

REGISTRY_REQUIRED_COLUMNS = (
    "监测点编码",
    "监测点名称",
    "城市",
    "经度",
    "纬度",
    "对照点",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "Defaults to "
            "ROOT/01_raw/air_quality_cnemc_quotsoft/"
            "station_registry/source_files/站点列表-2022.02.13起.csv"
        ),
    )
    p.add_argument(
        "--city-manifest",
        type=Path,
        default=None,
        help="Defaults to ROOT/05_manifests/selected_cities_v2.csv",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "Defaults to ROOT/06_processed/audits/"
            "cnemc_station_registry_50city_roster_v1"
        ),
    )
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise last if last is not None else RuntimeError("CSV decode failed")


def norm_city(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    for suffix in ("市",):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_selected_cities(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    city_en_col = find_column(
        df,
        (
            "city",
            "city_en",
            "city_name",
            "city_english",
            "name_en",
        ),
    )
    city_zh_col = find_column(
        df,
        (
            "city_zh",
            "city_cn",
            "city_chinese",
            "城市",
            "name_zh",
            "city_name_zh",
        ),
    )
    region_col = find_column(
        df,
        (
            "region",
            "region_name",
            "region_en",
            "区域",
        ),
    )

    if city_zh_col is None:
        raise RuntimeError(
            "selected_cities_v2.csv must contain a Chinese city-name column "
            "(e.g. city_zh / 城市) for deterministic CNEMC mapping"
        )

    out = pd.DataFrame(
        {
            "city_zh": df[city_zh_col].astype(str).str.strip(),
            "city_norm": df[city_zh_col].map(norm_city),
        }
    )
    if city_en_col is not None:
        out["city_en"] = df[city_en_col].astype(str).str.strip()
    else:
        out["city_en"] = ""

    if region_col is not None:
        out["region"] = df[region_col].astype(str).str.strip()
    else:
        out["region"] = ""

    if len(out) != 50:
        raise RuntimeError(f"expected 50 selected cities, found {len(out)}")
    if out["city_norm"].duplicated().any():
        dup = out.loc[out["city_norm"].duplicated(False), "city_zh"].tolist()
        raise RuntimeError(f"duplicate normalized selected-city names: {dup}")
    return out


def load_registry(path: Path) -> pd.DataFrame:
    reg = read_csv_flexible(path)
    missing = [c for c in REGISTRY_REQUIRED_COLUMNS if c not in reg.columns]
    if missing:
        raise RuntimeError(f"registry missing columns: {missing}")

    reg = reg.loc[:, list(REGISTRY_REQUIRED_COLUMNS)].copy()
    reg["监测点编码"] = reg["监测点编码"].astype(str).str.strip()
    reg["监测点名称"] = reg["监测点名称"].astype(str).str.strip()
    reg["城市"] = reg["城市"].astype(str).str.strip()
    reg["city_norm"] = reg["城市"].map(norm_city)
    reg["对照点"] = reg["对照点"].astype(str).str.strip().str.upper()

    if len(reg) != 2026:
        raise RuntimeError(f"expected 2026 registry rows, found {len(reg)}")
    if reg["监测点编码"].nunique() != 2026:
        raise RuntimeError("registry station IDs are not unique")
    return reg


def expected_hours_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    this_month = date(year, month, 1)
    return (next_month - this_month).days * 24


def safe_num(x):
    if pd.isna(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    raw_root = root / "01_raw" / "air_quality_cnemc_quotsoft"
    sites_root = raw_root / "sites" / "daily"

    registry_path = (
        args.registry.expanduser().resolve()
        if args.registry is not None
        else raw_root
        / "station_registry"
        / "source_files"
        / "站点列表-2022.02.13起.csv"
    )
    city_manifest_path = (
        args.city_manifest.expanduser().resolve()
        if args.city_manifest is not None
        else root / "05_manifests" / "selected_cities_v2.csv"
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else root / "06_processed" / "audits" / AUDIT_ID
    )
    out_root.mkdir(parents=True, exist_ok=True)

    reg = load_registry(registry_path)
    selected = load_selected_cities(city_manifest_path)

    mapping = selected.merge(
        reg[["城市", "city_norm"]].drop_duplicates(),
        on="city_norm",
        how="left",
    )

    missing_city_mapping = mapping.loc[mapping["城市"].isna()].copy()
    ambiguous_counts = (
        reg[["城市", "city_norm"]]
        .drop_duplicates()
        .groupby("city_norm")["城市"]
        .nunique()
    )
    ambiguous_selected = selected.loc[
        selected["city_norm"].isin(
            ambiguous_counts[ambiguous_counts > 1].index
        )
    ].copy()

    city_map = selected.set_index("city_norm")[
        ["city_zh", "city_en", "region"]
    ]

    candidate = reg.loc[reg["city_norm"].isin(selected["city_norm"])].copy()
    candidate = candidate.join(city_map, on="city_norm", rsuffix="_selected")
    candidate = candidate.rename(
        columns={
            "监测点编码": "station_id",
            "监测点名称": "station_name",
            "城市": "registry_city",
            "经度": "longitude",
            "纬度": "latitude",
            "对照点": "control_site",
        }
    )

    if candidate.empty:
        raise RuntimeError("no station candidates matched the selected 50 cities")

    station_ids = candidate["station_id"].tolist()
    station_id_set = set(station_ids)

    # Presence/coverage only. Values are never summarized or inspected.
    # For 2025 this reads only finite/non-finite masks, preserving sealed
    # target values from any value-based model/feature analysis.
    periods = ["2022-12", "2023", "2024", "2025", "2023-2025"]
    source_rows = {
        p: {poll: 0 for poll in POLLUTANTS}
        for p in periods
    }
    valid_counts = {
        sid: {
            p: {poll: 0 for poll in POLLUTANTS}
            for p in periods
        }
        for sid in station_ids
    }

    first_valid_date = {
        sid: {poll: None for poll in POLLUTANTS}
        for sid in station_ids
    }
    last_valid_date = {
        sid: {poll: None for poll in POLLUTANTS}
        for sid in station_ids
    }

    monthly_valid = defaultdict(lambda: defaultdict(int))
    monthly_source_rows = defaultdict(int)

    daily_files = sorted(sites_root.glob("*/*.csv"))
    if len(daily_files) != 1127:
        raise RuntimeError(f"expected 1127 site daily CSVs, found {len(daily_files)}")

    usecols = ["date", "hour", "type"] + station_ids

    for path in tqdm(daily_files, desc="Station roster audit", unit="day"):
        stem = path.stem
        ymd = stem.rsplit("_", 1)[-1]
        d = datetime.strptime(ymd, "%Y%m%d").date()
        month_key = f"{d.year:04d}-{d.month:02d}"

        if d.year == 2022 and d.month == 12:
            year_period = "2022-12"
        elif d.year in PROTOCOL_YEARS:
            year_period = str(d.year)
        else:
            continue

        df = read_csv_flexible(path, usecols=usecols)
        df["type"] = df["type"].astype(str).str.strip()

        for poll in POLLUTANTS:
            block = df.loc[df["type"] == poll, station_ids]
            n_source = len(block)

            source_rows[year_period][poll] += n_source
            if d.year in PROTOCOL_YEARS:
                source_rows["2023-2025"][poll] += n_source

            if poll == "PM2.5":
                monthly_source_rows[month_key] += n_source

            if n_source == 0:
                continue

            finite = block.notna()
            counts = finite.sum(axis=0)

            for sid, n in counts.items():
                n_int = int(n)
                valid_counts[sid][year_period][poll] += n_int
                if d.year in PROTOCOL_YEARS:
                    valid_counts[sid]["2023-2025"][poll] += n_int

                if poll == "PM2.5":
                    monthly_valid[(sid, month_key)]["valid_hours"] += n_int

                if n_int > 0:
                    if first_valid_date[sid][poll] is None:
                        first_valid_date[sid][poll] = d
                    last_valid_date[sid][poll] = d

    roster_rows = []
    for _, row in candidate.sort_values(
        ["region", "city_en", "station_id"]
    ).iterrows():
        sid = row["station_id"]
        out = {
            "region": row["region"],
            "city_en": row["city_en"],
            "city_zh": row["city_zh"],
            "registry_city": row["registry_city"],
            "station_id": sid,
            "station_name": row["station_name"],
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "control_site": row["control_site"],
            "preliminary_rule": (
                "EXCLUDE_CONTROL"
                if row["control_site"] == "Y"
                else "REVIEW_NONCONTROL"
            ),
            "manual_decision": "",
            "manual_note": "",
            "pm25_first_valid_date": (
                first_valid_date[sid]["PM2.5"].isoformat()
                if first_valid_date[sid]["PM2.5"] is not None
                else ""
            ),
            "pm25_last_valid_date": (
                last_valid_date[sid]["PM2.5"].isoformat()
                if last_valid_date[sid]["PM2.5"] is not None
                else ""
            ),
        }

        for period in periods:
            denom = source_rows[period]["PM2.5"]
            valid = valid_counts[sid][period]["PM2.5"]
            out[f"pm25_valid_hours_{period}"] = valid
            out[f"pm25_source_hours_{period}"] = denom
            out[f"pm25_archive_coverage_{period}"] = (
                valid / denom if denom > 0 else np.nan
            )

        for poll in POLLUTANTS:
            valid = valid_counts[sid]["2023-2025"][poll]
            denom = source_rows["2023-2025"][poll]
            key = poll.lower().replace(".", "").replace("2.5", "25")
            out[f"{key}_valid_hours_2023_2025"] = valid
            out[f"{key}_archive_coverage_2023_2025"] = (
                valid / denom if denom > 0 else np.nan
            )

        roster_rows.append(out)

    roster = pd.DataFrame(roster_rows)

    monthly_rows = []
    for sid in station_ids:
        meta = candidate.loc[candidate["station_id"] == sid].iloc[0]
        for year in (2022, 2023, 2024, 2025):
            months = [12] if year == 2022 else range(1, 13)
            for month in months:
                month_key = f"{year:04d}-{month:02d}"
                valid = int(
                    monthly_valid[(sid, month_key)].get("valid_hours", 0)
                )
                source = int(monthly_source_rows.get(month_key, 0))
                theoretical = expected_hours_in_month(year, month)
                monthly_rows.append(
                    {
                        "region": meta["region"],
                        "city_en": meta["city_en"],
                        "city_zh": meta["city_zh"],
                        "station_id": sid,
                        "station_name": meta["station_name"],
                        "control_site": meta["control_site"],
                        "month": month_key,
                        "pm25_valid_hours": valid,
                        "pm25_source_rows": source,
                        "pm25_theoretical_hours": theoretical,
                        "pm25_archive_coverage": (
                            valid / source if source > 0 else np.nan
                        ),
                        "pm25_absolute_coverage": valid / theoretical,
                    }
                )
    monthly = pd.DataFrame(monthly_rows)

    city_summary_rows = []
    for (region, city_en, city_zh), g in roster.groupby(
        ["region", "city_en", "city_zh"], dropna=False
    ):
        noncontrol = g["control_site"].ne("Y")
        city_summary_rows.append(
            {
                "region": region,
                "city_en": city_en,
                "city_zh": city_zh,
                "candidate_stations": len(g),
                "control_stations": int(g["control_site"].eq("Y").sum()),
                "noncontrol_stations": int(noncontrol.sum()),
                "noncontrol_with_any_pm25_2023_2025": int(
                    (
                        noncontrol
                        & g["pm25_valid_hours_2023-2025"].gt(0)
                    ).sum()
                ),
                "noncontrol_pm25_archive_coverage_min_2023_2025": safe_num(
                    g.loc[
                        noncontrol,
                        "pm25_archive_coverage_2023-2025",
                    ].min()
                ),
                "noncontrol_pm25_archive_coverage_median_2023_2025": safe_num(
                    g.loc[
                        noncontrol,
                        "pm25_archive_coverage_2023-2025",
                    ].median()
                ),
                "noncontrol_pm25_archive_coverage_max_2023_2025": safe_num(
                    g.loc[
                        noncontrol,
                        "pm25_archive_coverage_2023-2025",
                    ].max()
                ),
            }
        )
    city_summary = pd.DataFrame(city_summary_rows).sort_values(
        ["region", "city_en"]
    )

    mapping_out = selected.copy()
    mapping_out["registry_match_count"] = mapping_out["city_norm"].map(
        reg.groupby("city_norm").size()
    ).fillna(0).astype(int)
    mapping_out["registry_city_label"] = mapping_out["city_norm"].map(
        reg.groupby("city_norm")["城市"].first()
    )

    roster.to_csv(
        out_root / "50city_station_roster_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly.to_csv(
        out_root / "station_monthly_pm25_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    city_summary.to_csv(
        out_root / "50city_station_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mapping_out.to_csv(
        out_root / "selected_city_registry_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_rows_rows = []
    for period in periods:
        for poll in POLLUTANTS:
            source_rows_rows.append(
                {
                    "period": period,
                    "pollutant": poll,
                    "source_rows": source_rows[period][poll],
                }
            )
    pd.DataFrame(source_rows_rows).to_csv(
        out_root / "source_hour_availability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    status = "PASS"
    failures = []
    if not missing_city_mapping.empty:
        status = "FAIL"
        failures.append(
            f"{len(missing_city_mapping)} selected cities lack registry mapping"
        )
    if not ambiguous_selected.empty:
        status = "FAIL"
        failures.append(
            f"{len(ambiguous_selected)} selected cities have ambiguous normalized registry names"
        )
    if len(city_summary) != 50:
        status = "FAIL"
        failures.append(
            f"city summary contains {len(city_summary)} cities instead of 50"
        )
    if (city_summary["noncontrol_stations"] == 0).any():
        status = "FAIL"
        failures.append("at least one selected city has zero non-control stations")
    if (
        city_summary["noncontrol_with_any_pm25_2023_2025"] == 0
    ).any():
        status = "FAIL"
        failures.append(
            "at least one selected city has zero non-control stations with PM2.5 data in 2023-2025"
        )

    summary = {
        "audit_id": AUDIT_ID,
        "created_at_utc": utc_now(),
        "status": status,
        "failures": failures,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "rows": len(reg),
            "unique_station_ids": reg["station_id"].nunique()
            if "station_id" in reg.columns
            else reg["监测点编码"].nunique(),
        },
        "selected_city_manifest": {
            "path": str(city_manifest_path),
            "sha256": sha256_file(city_manifest_path),
            "cities": len(selected),
        },
        "candidate_roster": {
            "stations_all_selected_cities": len(roster),
            "control_stations": int(roster["control_site"].eq("Y").sum()),
            "noncontrol_stations": int(roster["control_site"].ne("Y").sum()),
            "cities_with_candidates": int(
                roster[["city_en", "city_zh"]].drop_duplicates().shape[0]
            ),
        },
        "coverage_semantics": {
            "archive_coverage": (
                "station non-missing hourly rows divided by pollutant rows actually "
                "present in the QuotSoft daily archive for the same period"
            ),
            "absolute_monthly_coverage": (
                "station non-missing PM2.5 hours divided by theoretical calendar hours"
            ),
            "2025_policy": (
                "2025 numeric pollutant values are not summarized or inspected; "
                "only finite/non-finite masks are read for provenance/coverage QC"
            ),
            "no_imputation": True,
            "no_outlier_filtering": True,
            "no_city_aggregation": True,
            "no_station_deletion": True,
        },
        "outputs": [
            "50city_station_roster_review.csv",
            "station_monthly_pm25_coverage.csv",
            "50city_station_coverage_summary.csv",
            "selected_city_registry_mapping.csv",
            "source_hour_availability.csv",
        ],
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    archive = out_root.parent / f"{AUDIT_ID}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_root, arcname=out_root.name)

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
