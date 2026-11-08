#!/usr/bin/env python3
"""Plan or execute 210 fits, freeze their cohort, then score all five populations."""
import argparse
import json
from pathlib import Path
from phystok.training.config import load_config
from phystok.training.app import fit_one, freeze_cohort, evaluate_one, persistence, validate_cohort, validate_evaluation_binding
from phystok.utils.common import read_json, verify_artifacts, fingerprint, require, atomic_json, SCENARIOS
from phystok.evaluation.metrics import aggregate, persistence_metric

def plan(config,output):
    models=[m for m in config['models'] if m!='phystok']+['phystok']
    return [{'model':m,'seed':s,'fold':f,'output':str(Path(output)/'fits'/m/str(s)/f'fold{f}')}
            for m in models for s in config['final_seeds'] for f in config['folds']]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/benchmark_v2.json');p.add_argument('--output',default='runs/benchmark_v2')
    p.add_argument('--data-root');p.add_argument('--data-contract','--contract',dest='data_contract');p.add_argument('--device')
    p.add_argument('--execute',action='store_true',help='Actually train; without this flag only the plan is printed')
    p.add_argument('--freeze-only',action='store_true',help='Freeze preexisting complete runs; no training')
    a=p.parse_args();c=load_config(a.config);jobs=plan(c,a.output);out=Path(a.output)
    if not (a.execute or a.freeze_only):
        print(json.dumps({'training_runs':len(jobs),'prediction_exposures':1050,'persistence_populations':5,'jobs':jobs},indent=2));return
    if not a.data_root or not a.data_contract:p.error('execution/freeze requires --data-root and --data-contract')
    contract=read_json(a.data_contract)
    if a.execute and not a.device:p.error('--execute requires explicit --device')
    manifests=[]
    for j in jobs:
        path=Path(j['output'])/'SUCCESS.json'
        if path.exists():
            r=verify_artifacts(path);i=r['identity']
            require((i['model'],i['seed'],i['fold'])==(j['model'],j['seed'],j['fold']) and i['config_sha256']==fingerprint(c),'preexisting run identity drift')
        elif a.freeze_only:
            raise ValueError(f'missing completed fit: {path}')
        else:
            fit_one(config=c,root=a.data_root,contract=contract,model_name=j['model'],fold=j['fold'],seed=j['seed'],output=j['output'],device=a.device)
        manifests.append(path)
    cohort=out/'cohort.json'
    if not cohort.exists():freeze_cohort(manifests=manifests,config=c,output=cohort)
    validate_cohort(cohort,c,manifests)
    if a.freeze_only:print(cohort);return
    records=[]
    for j,path in zip(jobs,manifests):
        for scene in SCENARIOS:
            dest=out/'evaluation'/j['model']/str(j['seed'])/f'fold{j["fold"]}'/scene
            if (dest/'SUCCESS.json').exists():
                metric=validate_evaluation_binding(training_manifest=path,evaluation_manifest=dest/'SUCCESS.json',
                                                   config=c,contract=contract,scenario=scene)
            else:metric=evaluate_one(config=c,root=a.data_root,contract=contract,training_manifest=path,scenario=scene,output=dest,device=a.device)
            records.append(metric)
    result=aggregate(records,benchmark_id=c['benchmark_id'],models=list(c['models']))
    reference=[]
    for scene in SCENARIOS:
        path=out/'persistence'/f'{scene}.json'
        if not path.exists():persistence(root=a.data_root,contract=contract,scenario=scene,output=path,benchmark_id=c['benchmark_id'])
        row=read_json(path)
        require(row['data_manifest_sha256']==fingerprint(contract) and row['scenario']==scene,'cached Persistence data/population drift')
        reference.append(row)
    result['persistence']=persistence_metric(reference,benchmark_id=c['benchmark_id'])
    atomic_json(out/'pooled_results.json',result)
    print(out/'pooled_results.json')

if __name__=='__main__':main()
