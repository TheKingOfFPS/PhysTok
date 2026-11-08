"""Independent arithmetic checks of released frozen aggregate results."""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .core import (SEEDS, SCENARIOS, MULTIPLICITY, pooled, mean_sample_sd,
                   pool_by_seed, paired_seed_summary, paired_city_bootstrap,
                   city_diagnostics, validate_city_records)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_release(results_dir, *, bootstrap=False, bootstrap_ablations=False):
    """Verify aggregate arithmetic; optionally regenerate the frozen bootstrap.

    This does not reread prediction records, authenticate the experimental
    history, verify original data rights, or establish that raw labels matched.
    The CLI returns nonzero on any mismatch and never filters mismatched rows.
    """
    root = Path(results_dir)
    checks, rebuilt_bootstraps = [], []
    def check(name, good, **detail):
        checks.append(dict(check=name, passed=bool(good), **detail))
    def close(name, a, b, atol=1e-8):
        check(name, math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=atol),
              computed=float(a), frozen=float(b))
    manifest = json.loads((root/"PROVENANCE.json").read_text())
    for file in manifest["files"]:
        p = root/file["file"]
        check("sha256:"+file["file"], p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == file["sha256"])
    main = read_csv(root/"main_benchmark.csv")
    seed = read_csv(root/"main_seed_metrics.csv")
    city = validate_city_records(read_csv(root/"main_city_sufficient.csv"))
    exposure = read_csv(root/"main_fold_exposures.csv")
    paired = read_csv(root/"paired_statistics.csv")
    controls = read_csv(root/"ablation_seed_metrics.csv")
    control_city = validate_city_records(read_csv(root/"ablation_city_sufficient.csv"))
    control_paired = read_csv(root/"ablation_results.csv")
    main_index = {(r["model"],r["scenario"]):r for r in main}
    seed_index = {(r["model"],int(r["seed"]),r["scenario"]):r for r in seed}
    check("main_8_models_5_populations", len(main)==len(main_index)==40)
    check("105_trainable_seed_pools", len(seed)==len(seed_index)==105)
    check("12_main_endpoints",len(paired)==12)
    for row in seed+controls:
        p = pooled([row])
        for field in ("rmse","mae"):
            close("seed_metric:"+":".join([row["model"],row["seed"],row["scenario"],field]),p[field],row[field])
    for r in main:
        if r["model"] == "persistence":
            check("persistence_has_no_fabricated_seed_sd:"+r["scenario"],
                  not r["seeds"] and not r["rmse_sample_sd"] and not r["mae_sample_sd"])
            continue
        selected = [seed_index[r["model"],s,r["scenario"]] for s in SEEDS]
        for metric in ("rmse","mae"):
            summary = mean_sample_sd(float(x[metric]) for x in selected)
            for field in ("mean","sample_sd"):
                close("main_mean_sd:"+":".join([r["model"],r["scenario"],metric,field]),summary[field],r[metric+"_"+field])
    # Independent fold and city paths must agree with official seed pools.
    for label, parts in [("city",city),("fold",exposure)]:
        for r in pool_by_seed(parts):
            expected = seed_index[r["model"],r["seed"],r["scenario"]]
            for field in ("n","sse","sae","rmse","mae"):
                close(label+"_repool:"+":".join(map(str,[r["model"],r["seed"],r["scenario"],field])),r[field],expected[field])
    # Recompute all 12 paired means/SD and the common percentage denominator.
    for r in paired:
        got = next(x for x in paired_seed_summary(seed,"phystok",r["comparator"],r["scenario"]) if x["metric"]==r["metric"])
        for field in ("delta_mean","paired_seed_sample_sd","relative_delta_pct"):
            close("main_paired:"+":".join([r["comparator"],r["scenario"],r["metric"],field]),got[field],r[field])
        check("main_family_size:"+r["comparator"]+r["scenario"]+r["metric"],int(r["family_size"])==12)
    # Read legal physical target/checkpoint multiplicity metadata, without raw targets.
    physical = {(r["scenario"],r["city"]):r for r in read_csv(root/"physical_exposures.csv")}
    for r in city:
        meta = physical[r["scenario"],r["city"]]
        check("legal_exposure:"+":".join([r["model"],r["seed"],r["scenario"],r["city"]]),
              int(r["n"])==int(meta["physical_targets"])*MULTIPLICITY[r["scenario"]]
              ==int(meta["exposures_per_seed"]) and int(meta["multiplicity"])==MULTIPLICITY[r["scenario"]])
    # Retain seed/fold/checkpoint hashes, but no original checkpoint bytes or locations.
    registry = read_csv(root/"main_run_registry.csv")
    keys = {(r["model"],int(r["seed"]),int(r["fold"])) for r in registry}
    models = {r["model"] for r in seed}
    check("210_complete_accepted_fit_identities",len(registry)==len(keys)==210 and keys=={(m,s,f) for m in models for s in SEEDS for f in range(10)} and all(r["stop_reason"]=="QUALIFIED_VAL_PLATEAU" for r in registry))
    check("1055_prediction_hash_identities",len(read_csv(root/"prediction_hash_registry.csv"))==1055)
    # City weights differ from target-volume weights: both diagnostics are checked.
    frozen_cities = {(r["comparator"],r["scenario"],r["city"]):r for r in read_csv(root/"city_level_summary.csv")}
    frozen_wins = {(r["comparator"],r["scenario"],int(r["seed"])):r for r in read_csv(root/"city_win_counts.csv")}
    frozen_regions = {(r["comparator"],r["scenario"],r["region"]):r for r in read_csv(root/"regional_summary.csv")}
    frozen_concentration = {(r["comparator"],r["scenario"]):r for r in read_csv(root/"city_concentration.csv")}
    comparisons = [("timexer",s) for s in SCENARIOS]+[("informer","Spatial")]
    city_diagnostic_outputs = []
    for comparator,scenario in comparisons:
        diag = city_diagnostics(city,"phystok",comparator,scenario)
        city_diagnostic_outputs.append(diag)
        for row in diag["cities"]:
            old = frozen_cities[comparator,scenario,row["city"]]
            for key in ["phystok_wins","delta_rmse_mean","delta_rmse_sample_sd","delta_mae_mean","delta_sse_mean","delta_sae_mean","net_sse_contribution_pct","leave_this_city_out_delta_rmse_mean","leave_this_city_out_delta_mae_mean"]:
                close("city_diagnostic:"+":".join([comparator,scenario,row["city"],key]),row[key],old[key])
            for s in SEEDS:
                check("city_winner:"+":".join(map(str,[comparator,scenario,row["city"],s])),row["winner_"+str(s)]==old["winner_"+str(s)])
        for row in diag["by_seed"]:
            old = frozen_wins[comparator,scenario,row["seed"]]
            for key in ("phystok_city_wins","comparator_city_wins","ties","pooled_delta_rmse"):
                close("city_wins:"+":".join(map(str,[comparator,scenario,row["seed"],key])),row[key],old[key])
        for row in diag["regions"]:
            old = frozen_regions[comparator,scenario,row["region"]]
            for key in ("phystok_rmse_mean","comparator_rmse_mean","delta_rmse_mean","delta_rmse_sample_sd","delta_mae_mean","delta_sse_mean"):
                close("region:"+":".join([comparator,scenario,row["region"],key]),row[key],old[key])
        old = frozen_concentration[comparator,scenario]
        for key,value in diag["concentration"].items():
            if key in old and key not in ("comparator","scenario","descriptive_only"):
                close("concentration:"+":".join([comparator,scenario,key]),value,old[key])
    # Verify selected capacity/weather and all three controlled architectures.
    control_index = {(r["model"],int(r["seed"]),r["scenario"]):r for r in controls}
    for row in pool_by_seed(control_city):
        old = control_index[row["model"],row["seed"],row["scenario"]]
        for key in ("n","sse","sae","rmse","mae"):
            close("ablation_city_repool:"+":".join(map(str,[row["model"],row["seed"],row["scenario"],key])),row[key],old[key])
    for row in controls:
        if row["model"] == "full":
            old = seed_index["phystok",int(row["seed"]),row["scenario"]]
            for key in ("n","sse","sae"):
                close("full_reference_reused:"+row["seed"]+row["scenario"]+key,row[key],old[key])
    for r in control_paired:
        got = next(x for x in paired_seed_summary(controls,r["comparator"],r["variant"],r["scenario"],percent_denominator="baseline") if x["metric"]==r["metric"])
        for field in ("delta_mean","paired_seed_sample_sd","relative_delta_pct"):
            close("ablation_paired:"+":".join([r["experiment"],r["comparator"],r["scenario"],r["metric"],field]),got[field],r[field])
    for group in ("capacity","meteorological"):
        totals = read_csv(root/(group+"_results.csv"))
        parts = read_csv(root/(group+"_city_sufficient.csv"))
        for r in totals:
            selected = [p for p in parts if p["configuration"]==r["configuration"] and p["scenario"]==r["scenario"]]
            p = pooled(selected)
            for key in ("n","sse","sae","rmse","mae"):
                close(group+"_repool:"+r["configuration"]+r["scenario"]+key,p[key],r[key])
            check(group+"_single_seed_no_fake_SD:"+r["configuration"]+r["scenario"],r["seed"]=="20260826" and not r["seed_sample_sd"])
    weather = {r["configuration"]:r for r in read_csv(root/"meteorological_results.csv")}
    arm = {"A":"A","B":"abc_b","C":"abc_c"}
    for r in read_csv(root/"meteorological_paired.csv"):
        first,second = r["comparison"].split("_minus_")
        close("ABC_contrast:"+r["comparison"]+r["metric"],float(weather[arm[first]][r["metric"]])-float(weather[arm[second]][r["metric"]]),r["observed_delta"])
    # Paper claims are checked at their displayed precision, not promoted.
    for scenario in SCENARIOS:
        p = float(main_index["phystok",scenario]["rmse_mean"])
        check("observed_RMSE_rank1:"+scenario,all(p<=float(r["rmse_mean"]) for r in main if r["scenario"]==scenario))
    spatial = next(r for r in paired if r["comparator"]=="informer" and r["scenario"]=="Spatial" and r["metric"]=="rmse")
    check("paper_Spatial_6.94_percent",f'{float(spatial["relative_delta_pct"]):.2f}'=="6.94")
    for r in paired:
        if r["comparator"]=="timexer" and r["scenario"]!="Spatial":
            check("shifted_main_family_includes_zero:"+r["scenario"]+r["metric"],float(r["family_ci_low"])<=0<=float(r["family_ci_high"]))
    cohort=["xlinear","timexer","itransformer","patchtst"]
    ranks={s:{r["model"]:i for i,r in enumerate(sorted([r for r in main if r["scenario"]==s and r["model"] in cohort],key=lambda r:float(r["rmse_mean"])))} for s in ("Spatial","Future")}
    flips=sum((ranks["Spatial"][a]-ranks["Spatial"][b])*(ranks["Future"][a]-ranks["Future"][b])<0 for i,a in enumerate(cohort) for b in cohort[i+1:])
    check("paper_four_of_six_same_budget_orders_reverse",flips==4)
    six = [r["model"] for r in main if r["scenario"]=="Spatial" and r["model"] not in ("phystok","persistence")]
    six_ranks = {scene: {r["model"]:i+1 for i,r in enumerate(sorted([r for r in main if r["scenario"]==scene and r["model"] in six],key=lambda r:float(r["rmse_mean"])))} for scene in SCENARIOS}
    rho = 1-6*sum((six_ranks["Spatial"][m]-six_ranks["Future"][m])**2 for m in six)/(6*(6*6-1))
    close("paper_six_baseline_Spatial_Future_Spearman",rho,-0.6)
    for r in read_csv(root/"baseline_ranks.csv"):
        check("baseline_rank:"+r["model"]+r["scenario"],six_ranks[r["scenario"]][r["model"]]==int(r["baseline_rank"]))
    if bootstrap:
        for comparator,scenario in comparisons:
            generated=paired_city_bootstrap(city,"phystok",comparator,scenario)
            rebuilt_bootstraps.append(generated)
            for r in [r for r in paired if r["comparator"]==comparator and r["scenario"]==scenario]:
                for key,value in generated["metrics"][r["metric"]].items():
                    if key!="family_interval_crosses_zero":
                        close("bootstrap_main:"+":".join([comparator,scenario,r["metric"],key]),value,r[key])
    if bootstrap_ablations:
        groups=defaultdict(list)
        for r in control_paired:
            groups[r["experiment"],r["variant"],r["comparator"],r["scenario"]].append(r)
        for (experiment,variant,baseline,scenario),rs in groups.items():
            ref=rs[0]
            generated=paired_city_bootstrap(control_city,baseline,variant,scenario,
                                            reps=int(ref["bootstrap_reps"]),seed=int(ref["bootstrap_seed"]),
                                            family_size=int(ref["family_size"]) if ref["family_size"] else None)
            rebuilt_bootstraps.append(generated)
            for r in rs:
                for key,value in generated["metrics"][r["metric"]].items():
                    if key!="family_interval_crosses_zero":
                        close("bootstrap_ablation:"+":".join([experiment,baseline,scenario,r["metric"],key]),value,r[key])
        weather_city=[]
        for r in read_csv(root/"meteorological_city_sufficient.csv"):
            weather_city.append(dict(r,model=r["configuration"],city=f'{int(r["city_index"]):02d}',seed=20260826))
        for r in read_csv(root/"meteorological_paired.csv"):
            first,second=r["comparison"].split("_minus_")
            generated=paired_city_bootstrap(weather_city,arm[second],arm[first],"Spatial",reps=10000,seed=20260926,family_size=None,seeds=(20260826,))
            for key,value in generated["metrics"][r["metric"]].items():
                close("bootstrap_ABC:"+r["comparison"]+r["metric"]+key,value,r[key])
    failed=[c for c in checks if not c["passed"]]
    return dict(status="PASS" if not failed else "FAIL",checks=len(checks),passed=len(checks)-len(failed),
                mode="bootstrap-all" if bootstrap_ablations else "bootstrap" if bootstrap else "verify",
                main_family_size=12,bootstrap_main_recomputed=bool(bootstrap),bootstrap_ablations_recomputed=bool(bootstrap_ablations),
                checked_main_models=8,checked_training_seed_pools=105,
                source_observations_or_predictions_verified=False,
                persistence_scope="Frozen displayed metrics retained; no raw prediction or Persistence seed-statistic reconstruction from this package.",
                check_details=checks,bootstrap_results=rebuilt_bootstraps,
                interpretation="Aggregate arithmetic/uncertainty verification only. Intervals condition on observed seeds/cities/configuration; crossing zero is not equivalence.")
