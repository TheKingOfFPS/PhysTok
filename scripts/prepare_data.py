#!/usr/bin/env python3
"""Reconstruct user-local inputs from legally obtained CNEMC and ERA5 sources."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phystok.data.rebuild import initialize,rebuild,bind_local_contract

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,help='A separate user-owned data workspace; never the code repository.')
    sub=p.add_subparsers(dest='step',required=True)
    sub.add_parser('init',help='Initialize data-free frozen city/fold/protocol metadata.')
    for name in ['cnemc','internal','external','audit']:
        sub.add_parser(name)
    era=sub.add_parser('era5',help='Plan or download official CDS ERA5 data with your own account.')
    era.add_argument('--plan-only',action='store_true')
    bind=sub.add_parser('bind',help='Create a hash-bound data contract for your reconstructed files.')
    bind.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.step=='init':print(initialize(args.data_root))
    elif args.step=='bind':print(bind_local_contract(args.data_root,args.output))
    else:rebuild(args.data_root,args.step,plan_only=getattr(args,'plan_only',False))

if __name__=='__main__':main()
