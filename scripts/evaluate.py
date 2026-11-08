#!/usr/bin/env python3
"""Evaluate an already frozen best-VAL checkpoint on one legal population."""
import argparse
from pathlib import Path
from phystok.training.app import evaluate_one, persistence, validate_cohort
from phystok.training.config import load_config
from phystok.utils.common import read_json, require, sha256, fingerprint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/benchmark_v2.json')
    p.add_argument('--data-root');p.add_argument('--data-contract','--contract',dest='data_contract')
    p.add_argument('--training-manifest');p.add_argument('--cohort')
    p.add_argument('--scenario');p.add_argument('--output')
    p.add_argument('--device',default='cpu');p.add_argument('--persistence',action='store_true')
    p.add_argument('--synthetic-smoke',action='store_true',help='Evaluate generated CPU tensors only; no checkpoint or scientific result')
    a=p.parse_args();config=load_config(a.config)
    if a.synthetic_smoke:
        import json,torch
        from phystok.data import SyntheticView
        from phystok.models import build_model
        from phystok.training.engine import evaluate_validation,seed_all
        require(a.device=='cpu','synthetic evaluation is CPU-only')
        torch.set_num_threads(2);seed_all(20260826,'cpu')
        g=torch.Generator(device='cpu').manual_seed(42);x=torch.randn(4,72,72,generator=g);y=.1*x[:,-1,0]
        layout=read_json(Path(__file__).resolve().parents[1]/'configs/data_contract.json')
        model=build_model('phystok',config['models']['phystok']['candidates'][0],layout,
                          dict(input_center=0.,input_scale=1.,target_mean=0.,target_std=1.))
        metric=evaluate_validation(model,SyntheticView(x,y,'VAL'),2,config['precision'],'cpu')
        print(json.dumps(dict(status='PASS',synthetic=True,research_data_used=False,**metric)));return
    if not all([a.data_root,a.data_contract,a.cohort,a.scenario,a.output]):
        p.error('formal evaluation requires --data-root, --data-contract, --cohort, --scenario and --output')
    contract=read_json(a.data_contract)
    cohort=validate_cohort(a.cohort,config)
    if a.persistence:
        print(persistence(root=a.data_root,contract=contract,scenario=a.scenario,output=a.output,benchmark_id=config['benchmark_id']));return
    if not a.training_manifest:p.error('--training-manifest is required for trainable models')
    matches=[r for r in cohort['records'] if Path(r['training_manifest']).resolve()==Path(a.training_manifest).resolve()]
    require(len(matches)==1 and sha256(a.training_manifest)==matches[0]['training_manifest_sha256'],'checkpoint is not a frozen cohort member')
    print(evaluate_one(config=config,root=a.data_root,contract=contract,training_manifest=a.training_manifest,
                       scenario=a.scenario,output=a.output,device=a.device))

if __name__=='__main__':main()
