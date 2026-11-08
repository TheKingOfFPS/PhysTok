#!/usr/bin/env python3
"""Train one model/seed/fold using the unchanged frozen scientific recipe."""
import argparse
from phystok.training.app import fit_one
from phystok.training.config import load_config
from phystok.utils.common import read_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/benchmark_v2.json')
    p.add_argument('--data-root',required=True);p.add_argument('--data-contract','--contract',dest='data_contract',required=True)
    p.add_argument('--model',required=True);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--fold',required=True,type=int);p.add_argument('--output',required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--resume',help='Complete resume checkpoint path; never just a best checkpoint')
    p.add_argument('--resume-sha256')
    a=p.parse_args()
    if bool(a.resume)!=bool(a.resume_sha256):p.error('--resume and --resume-sha256 are required together')
    result=fit_one(config=load_config(a.config),root=a.data_root,contract=read_json(a.data_contract),
        model_name=a.model,seed=a.seed,fold=a.fold,output=a.output,device=a.device,
        resume={'path':a.resume,'sha256':a.resume_sha256} if a.resume else None)
    print(result['stop_reason'])

if __name__=='__main__':main()
