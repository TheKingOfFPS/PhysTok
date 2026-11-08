"""Pooled forecasting statistics and paired city-cluster uncertainty.

All inputs are error sufficient statistics, never observations. These functions
cannot validate prediction truth or reconstruct the training data.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
from typing import Iterable, Mapping

import numpy as np

SEEDS = (20260826, 20260827, 20260828)
SCENARIOS = ("Spatial", "Future", "Joint", "External-Spatial", "External-Joint")
MULTIPLICITY = dict(zip(SCENARIOS, (1, 8, 1, 10, 10)))


def pooled(records: Iterable[Mapping]) -> dict:
    """Sum legal-exposure SSE/SAE/n, then compute physical-unit metrics.

    Averaging per-fold or per-city RMSE is not an alternative implementation.
    """
    records = list(records)
    if not records:
        raise ValueError("empty metric support")
    for r in records:
        n, sse, sae = float(r["n"]), float(r["sse"]), float(r["sae"])
        if not all(math.isfinite(x) for x in (n, sse, sae)):
            raise ValueError("nonfinite sufficient statistic")
        if n <= 0 or n != int(n) or sse < 0 or sae < 0:
            raise ValueError("positive integer n and nonnegative errors required")
        if sae * sae > n * sse * (1 + 1e-10) + 1e-7:
            raise ValueError("SAE and SSE violate Cauchy--Schwarz")
    n = sum(int(r["n"]) for r in records)
    sse = math.fsum(float(r["sse"]) for r in records)
    sae = math.fsum(float(r["sae"]) for r in records)
    return dict(n=n, sse=sse, sae=sae, rmse=math.sqrt(sse / n), mae=sae / n)


def mean_sample_sd(values: Iterable[float]) -> dict:
    """Across complete training seeds; never across folds or cities."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("at least two finite training-seed values required")
    return dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), ddof=1)


def validate_city_records(records: Iterable[Mapping], seeds=SEEDS) -> list[dict]:
    """Require one complete city x seed sufficient-statistic grid per model/scene."""
    records = list(records)
    seen, panels = set(), defaultdict(dict)
    for r in records:
        key = (r["model"], r["scenario"], int(r["seed"]), r["city"])
        if key in seen:
            raise ValueError("duplicate city/seed/model/scenario statistic")
        seen.add(key)
        if key[2] not in seeds:
            raise ValueError("unexpected training seed")
        pooled([r])
        panels[key[:2]].setdefault(key[2], set()).add(key[3])
    for panel in panels.values():
        if set(panel) != set(seeds) or len({tuple(sorted(v)) for v in panel.values()}) != 1:
            raise ValueError("incomplete city x seed panel")
    return records


def pool_by_seed(records: Iterable[Mapping]) -> list[dict]:
    """Accept city or fold rows, assuming their exposure supports were audited."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["scenario"], int(r["seed"]))].append(r)
    return [dict(model=m, scenario=e, seed=s, **pooled(rs))
            for (m, e, s), rs in sorted(groups.items())]


def paired_seed_summary(records, baseline, comparator, scenario, seeds=SEEDS,
                        percent_denominator="comparator") -> list[dict]:
    """Delta is comparator minus baseline, averaged after per-seed pooling."""
    selected = [r for r in records if r["scenario"] == scenario and r["model"] in (baseline, comparator)]
    index = {(r["model"], int(r["seed"])): r for r in selected}
    if len(index) != len(selected) or set(index) != {(m, s) for m in (baseline, comparator) for s in seeds}:
        raise ValueError("paired complete seed metrics required")
    for s in seeds:
        if int(index[baseline, s]["n"]) != int(index[comparator, s]["n"]):
            raise ValueError("paired legal-exposure counts differ")
    result = []
    for metric in ("rmse", "mae"):
        a = [float(index[baseline, s][metric]) for s in seeds]
        b = [float(index[comparator, s][metric]) for s in seeds]
        d = [y-x for x, y in zip(a, b)]
        summary = mean_sample_sd(d) if len(seeds) > 1 else dict(mean=d[0], sample_sd=None)
        den = np.mean(b if percent_denominator == "comparator" else a)
        result.append(dict(baseline=baseline, comparator=comparator, scenario=scenario,
                           metric=metric, delta_mean=summary["mean"],
                           paired_seed_sample_sd=summary["sample_sd"],
                           baseline_mean=float(np.mean(a)), comparator_mean=float(np.mean(b)),
                           relative_delta_pct=100*summary["mean"]/den,
                           delta_by_seed=d, percent_denominator=percent_denominator))
    return result


@lru_cache(maxsize=4)
def _cached_weights(cities, regions, reps, seed):
    rng = np.random.default_rng(seed)
    weights = np.zeros((reps, len(cities)), dtype=np.int32)
    for region in sorted(set(regions)):
        positions = np.asarray([i for i, r in enumerate(regions) if r == region])
        draws = rng.integers(0, len(positions), size=(reps, len(positions)))
        for j in range(len(positions)):
            weights[:, positions[j]] = np.count_nonzero(draws == j, axis=1)
    weights.setflags(write=False)
    return weights


def city_cluster_weights(cities, regions, reps=100000, seed=20261025):
    """One city weight shared by models, all hours/exposures, and every seed.

    Internal regions are the five frozen strata. For an external panel, callers
    must assign all ten cities to a single actual external stratum.
    """
    cities = tuple(cities)
    if len(cities) != len(set(cities)) or set(cities) != set(regions):
        raise ValueError("unique city IDs and one stratum per city required")
    if isinstance(reps, bool) or reps <= 0 or int(reps) != reps:
        raise ValueError("positive integer number of bootstrap replicates required")
    return _cached_weights(cities, tuple(regions[c] for c in cities), int(reps), int(seed))


def paired_city_bootstrap(city_records, baseline, comparator, scenario, *, reps=100000,
                          seed=20261025, family_size=12, seeds=SEEDS) -> dict:
    """Re-pool each weighted seed, then mean the paired seed-wise differences.

    A matching n grid is necessary but cannot establish original truth/key
    pairing; that belongs to the prediction-level audit in Full Reproduction.
    """
    selected = [r for r in city_records if r["scenario"] == scenario and r["model"] in (baseline, comparator)]
    validate_city_records(selected, seeds=seeds)
    index = {(r["model"], int(r["seed"]), r["city"]): r for r in selected}
    names = sorted({r["city"] for r in selected if r["model"] == baseline})
    if set(index) != {(m, s, c) for m in (baseline, comparator) for s in seeds for c in names}:
        raise ValueError("paired cities/seeds differ; no silent intersection allowed")
    region = {}
    for c in names:
        declared = {index[m, s, c]["region"] for m in (baseline, comparator) for s in seeds}
        if len(declared) != 1:
            raise ValueError("paired city region differs")
        region[c] = next(iter(declared))
    if scenario.startswith("External-") and len(set(region.values())) != 1:
        raise ValueError("external panel must not use fabricated internal strata")
    arrays = {m: {f: np.asarray([[float(index[m, s, c][f]) for s in seeds] for c in names], np.float64)
                  for f in ("n", "sse", "sae")} for m in (baseline, comparator)}
    if not np.array_equal(arrays[baseline]["n"], arrays[comparator]["n"]):
        raise ValueError("paired city/seed supports differ")
    w = city_cluster_weights(names, region, reps=reps, seed=seed)
    den = w @ arrays[baseline]["n"]
    if not (den > 0).all():
        raise ValueError("empty resampled support")
    out = {}
    for metric, field in (("rmse", "sse"), ("mae", "sae")):
        a = (w @ arrays[baseline][field])/den
        b = (w @ arrays[comparator][field])/den
        if metric == "rmse":
            a, b = np.sqrt(a), np.sqrt(b)
        values = (b-a).mean(axis=1)
        levels = [.025, .975]
        if family_size is not None:
            if family_size < 1:
                raise ValueError("positive comparison-family size required")
            levels += [.05/(2*family_size), 1-.05/(2*family_size)]
        q = np.quantile(values, levels)
        interval = dict(ci95_low=float(q[0]), ci95_high=float(q[1]))
        if family_size is not None:
            interval.update(family_ci_low=float(q[2]), family_ci_high=float(q[3]),
                            family_interval_crosses_zero=bool(q[2] <= 0 <= q[3]))
        out[metric] = interval
    return dict(baseline=baseline, comparator=comparator, scenario=scenario,
                cities=len(names), bootstrap_reps=reps, bootstrap_seed=seed,
                family_size=family_size, metrics=out)


def city_diagnostics(city_records, baseline, comparator, scenario, seeds=SEEDS):
    """Descriptive city/region wins, SSE contributions, and leave-one-city-out."""
    selected = [r for r in city_records if r["scenario"] == scenario and r["model"] in (baseline, comparator)]
    validate_city_records(selected, seeds)
    index = {(r["model"], int(r["seed"]), r["city"]): r for r in selected}
    names = sorted({r["city"] for r in selected})
    if set(index) != {(m, s, c) for m in (baseline, comparator) for s in seeds for c in names}:
        raise ValueError("incomplete paired city grid")
    metrics = {k: pooled([r]) for k, r in index.items()}
    totals = {(m, s): pooled([index[m, s, c] for c in names]) for m in (baseline, comparator) for s in seeds}
    for s in seeds:
        for c in names:
            if index[baseline, s, c]["n"] != index[comparator, s, c]["n"]:
                raise ValueError("paired city support mismatch")
    cities, by_seed, regions = [], [], []
    for c in names:
        row = dict(comparator=comparator, scenario=scenario, city=c,
                   region=index[baseline, seeds[0], c]["region"],
                   n_per_seed=int(index[baseline, seeds[0], c]["n"]))
        wins = 0
        for s in seeds:
            p, b = metrics[baseline, s, c], metrics[comparator, s, c]
            winner = baseline if b["rmse"] > p["rmse"] else comparator if b["rmse"] < p["rmse"] else "tie"
            row["winner_"+str(s)] = winner
            wins += winner == baseline
        row["phystok_wins"] = wins
        for metric in ("rmse", "mae", "sse", "sae"):
            for prefix, vals in [("phystok_", [metrics[baseline, s, c][metric] for s in seeds]),
                                 ("comparator_", [metrics[comparator, s, c][metric] for s in seeds]),
                                 ("delta_", [metrics[comparator, s, c][metric]-metrics[baseline, s, c][metric] for s in seeds])]:
                stat = mean_sample_sd(vals)
                row[prefix+metric+"_mean"] = stat["mean"]
                row[prefix+metric+"_sample_sd"] = stat["sample_sd"]
        leave = {"rmse": [], "mae": []}
        for s in seeds:
            pp = {m: pooled([dict(n=totals[m, s]["n"]-metrics[m, s, c]["n"],
                                  sse=totals[m, s]["sse"]-metrics[m, s, c]["sse"],
                                  sae=totals[m, s]["sae"]-metrics[m, s, c]["sae"])]) for m in (baseline, comparator)}
            for metric in leave:
                leave[metric].append(pp[comparator][metric]-pp[baseline][metric])
        for metric in leave:
            row["leave_this_city_out_delta_"+metric+"_mean"] = float(np.mean(leave[metric]))
        cities.append(row)
    net = math.fsum(r["delta_sse_mean"] for r in cities)
    gross = math.fsum(max(0., r["delta_sse_mean"]) for r in cities)
    for r in cities:
        r["net_sse_contribution_pct"] = 100*r["delta_sse_mean"]/net if net else None
        r["gross_positive_sse_gain_share_pct"] = 100*max(0., r["delta_sse_mean"])/gross if gross else 0.
    for s in seeds:
        ws = [r["winner_"+str(s)] for r in cities]
        by_seed.append(dict(comparator=comparator, scenario=scenario, seed=s, cities=len(cities),
                            phystok_city_wins=ws.count(baseline), comparator_city_wins=ws.count(comparator),
                            ties=ws.count("tie"), pooled_delta_rmse=totals[comparator, s]["rmse"]-totals[baseline, s]["rmse"]))
    for region in sorted({r["region"] for r in cities}):
        members = [r["city"] for r in cities if r["region"] == region]
        parts = {(m, s): pooled([index[m, s, c] for c in members]) for m in (baseline, comparator) for s in seeds}
        row = dict(comparator=comparator, scenario=scenario, region=region, cities=len(members), descriptive_only=True)
        for metric in ("rmse", "mae", "sse"):
            for prefix, vals in [("phystok_", [parts[baseline, s][metric] for s in seeds]),
                                 ("comparator_", [parts[comparator, s][metric] for s in seeds]),
                                 ("delta_", [parts[comparator, s][metric]-parts[baseline, s][metric] for s in seeds])]:
                stat = mean_sample_sd(vals)
                row[prefix+metric+"_mean"] = stat["mean"]
                row[prefix+metric+"_sample_sd"] = stat["sample_sd"]
        regions.append(row)
    concentration = dict(comparator=comparator, scenario=scenario, cities=len(cities),
                         net_delta_sse_mean=net, gross_positive_delta_sse_mean=gross,
                         gross_negative_delta_sse_mean=math.fsum(min(0.,r["delta_sse_mean"]) for r in cities),
                         leave_one_city_out_rmse_delta_min=min(r["leave_this_city_out_delta_rmse_mean"] for r in cities),
                         leave_one_city_out_rmse_delta_max=max(r["leave_this_city_out_delta_rmse_mean"] for r in cities),
                         leave_one_city_out_rmse_rank_flip_count=sum(r["leave_this_city_out_delta_rmse_mean"] < 0 for r in cities),
                         descriptive_only=True)
    for wins in range(len(seeds)+1):
        concentration["city_wins"+str(wins)] = sum(r["phystok_wins"] == wins for r in cities)
    return dict(cities=cities, by_seed=by_seed, regions=regions, concentration=concentration)
