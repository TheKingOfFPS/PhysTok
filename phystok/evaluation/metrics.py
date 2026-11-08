from __future__ import annotations
from collections import Counter,defaultdict
import math
import statistics
from phystok.utils.common import require,SEEDS,SCENARIOS,fingerprint


def errors(truth,prediction):
    import numpy as np
    y,p=np.asarray(truth,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    require(y.shape==p.shape and y.ndim==1 and len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'invalid prediction vectors')
    e=p-y
    return dict(n=len(e),sse=float((e*e).sum()),sae=float(abs(e).sum()))


def pooled(parts):
    require(parts,'empty metrics')
    for p in parts:
        require(type(p['n']) is int and p['n']>0 and all(math.isfinite(p[k]) and p[k]>=0 for k in ['sse','sae']),'invalid sufficient statistics')
    n=sum(p['n'] for p in parts);sse=sum(p['sse'] for p in parts);sae=sum(p['sae'] for p in parts)
    return dict(n=n,sse=sse,sae=sae,rmse=math.sqrt(sse/n),mae=sae/n)


def validate_exposure_rows(rows,expected,*,benchmark_id,model,seed,scenario,fold):
    """expected is the independently obtained frozen-selector physical row set."""
    require(len(rows)>0,'missing predictions')
    ids=[]
    for r in rows:
        require(r['benchmark_id']==benchmark_id and r['model']==model and r['seed']==seed and r['scenario']==scenario and r['fold']==fold,
                'mixed benchmark/model/seed/scenario/exposure')
        ids.append(r['task_row'])
    require(len(ids)==len(set(ids)),'duplicate physical task within checkpoint exposure')
    require(set(ids)==set(expected),'prediction support differs from frozen selector')
    return fingerprint(sorted(ids))


def aggregate(records,*,benchmark_id,models):
    require(all(r['benchmark_id']==benchmark_id for r in records),'legacy/mixed benchmark results forbidden')
    index={}
    for r in records:
        key=(r['model'],r['seed'],r['fold'],r['scenario'])
        require(key not in index,'duplicate metric exposure')
        require(r.get('checkpoint_sha256') and r.get('transform_sha256') and r.get('support_sha256'),'missing frozen provenance')
        require(r['scenario'] in SCENARIOS and r['fold'] in range(10) and r['seed'] in SEEDS,'illegal seed/fold/scenario')
        index[key]=r
    expected={(m,s,f,e) for m in models for s in SEEDS for f in range(10) for e in SCENARIOS}
    require(set(index)==expected,'complete ten folds x three seeds x five scenarios required; no partial SD')
    for m in models:
        for s in SEEDS:
            for f in range(10):
                group=[index[m,s,f,e] for e in SCENARIOS]
                require(len({x['checkpoint_sha256'] for x in group})==1 and len({x['transform_sha256'] for x in group})==1,
                        'five scenarios must use same frozen checkpoint and preprocessing')
    # Same scene/fold physical support for all models/seeds.
    for f in range(10):
        for e in SCENARIOS:
            require(len({index[m,s,f,e]['support_sha256'] for m in models for s in SEEDS})==1,'different physical supports across models/seeds')
    seed_metrics=[];summary=[]
    for m in models:
        for e in SCENARIOS:
            group=[]
            for s in SEEDS:
                p=pooled([index[m,s,f,e] for f in range(10)])
                row=dict(model=m,seed=s,scenario=e,**p);seed_metrics.append(row);group.append(row)
            summary.append(dict(model=m,scenario=e,seeds=list(SEEDS),
                           rmse_mean=statistics.mean(x['rmse'] for x in group),rmse_sample_sd=statistics.stdev(x['rmse'] for x in group),
                           mae_mean=statistics.mean(x['mae'] for x in group),mae_sample_sd=statistics.stdev(x['mae'] for x in group),ddof=1))
    return dict(benchmark_id=benchmark_id,seed_metrics=seed_metrics,mean_sample_sd=summary,estimand='pooled checkpoint exposures per seed; not ensemble')


def persistence_metric(records,*,benchmark_id):
    require(len(records)==5 and {r['scenario'] for r in records}==set(SCENARIOS),'five deterministic scenes required')
    require(all(r['benchmark_id']==benchmark_id and r['model']=='persistence' and r['seed'] is None and r['fold'] is None for r in records),'Persistence has no training seed/fold')
    return [dict(scenario=r['scenario'],**pooled([r])) for r in records]


def check_multiplicities(fold_task_ids,scenario):
    require(len(fold_task_ids)==10,'ten exposure supports required')
    c=Counter(i for ids in fold_task_ids for i in ids)
    expected={'Spatial':1,'Future':8,'Joint':1,'External-Spatial':10,'External-Joint':10}[scenario]
    require(c and set(c.values())=={expected},'illegal physical-task exposure multiplicity')
    return dict(physical_tasks=len(c),exposures=sum(c.values()),multiplicity=expected)
