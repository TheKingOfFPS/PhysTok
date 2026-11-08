#!/usr/bin/env python3
"""Run only the frozen manuscript controls, using user-rebuilt data/checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','prepare','fit','evaluate'])
    parser.add_argument('--variant',help='Frozen variant or reusable reference name')
    parser.add_argument('--ablation-config',type=Path,default=ROOT/'configs/ablations.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/benchmark_v2.json')
    parser.add_argument('--contract',type=Path,default=ROOT/'configs/data_contract.json')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--prepared-root',type=Path,help='ABC-B fold-local climatology and refitted preprocessing directory')
    parser.add_argument('--fold',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--training-manifest',type=Path)
    parser.add_argument('--scenario',choices=['Spatial','Future','Joint','External-Spatial','External-Joint'])
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',type=Path,help='Complete resume checkpoint, not a best-only checkpoint')
    parser.add_argument('--resume-sha256')
    args=parser.parse_args()
    if bool(args.resume)!=bool(args.resume_sha256):
        parser.error('--resume and --resume-sha256 are required together')
    ablation=json.loads(args.ablation_config.read_text())
    if args.action=='plan':
        names=[args.variant] if args.variant else list(ablation['variants'])
        rows=[]
        for name in names:
            if name not in ablation['variants']:
                parser.error('plan names must identify newly fitted frozen variants, not reused references')
            spec=ablation['variants'][name]
            rows.extend(dict(variant=name,model=spec['model'],seed=s,fold=f,arm=spec['arm'],scenarios=spec['scenarios'])
                        for s in spec['seeds'] for f in spec['folds'])
        result=dict(formal_new_fits=len(rows),tasks=rows,
                    references=ablation['references'],no_hpo=True,training_started=False)
        text=json.dumps(result,indent=2)+'\n'
        if args.output:
            with args.output.open('x') as f:f.write(text)
        else:print(text,end='')
        return
    if not args.variant or not args.data_root:
        parser.error('--variant and --data-root are required')
    config=json.loads(args.config.read_text());contract=json.loads(args.contract.read_text())
    references=ablation['references']
    if args.variant in references:
        if args.action!='evaluate':
            parser.error('Reference models must reuse selected main checkpoints; fitting/preparing a reference is forbidden')
        spec=dict(references[args.variant],arm='A',model_variant=None,
                  scenarios=['Spatial'] if args.variant=='abc_a' else config['task']['scenarios'])
    elif args.variant in ablation['variants']:
        spec=ablation['variants'][args.variant]
    else:parser.error('Unknown frozen variant')
    from phystok.experiments.weather import InterventionData,prepare_fold
    from phystok.training.app import fit_one,evaluate_one
    from phystok.utils.common import require,read_json
    import torch
    torch.set_num_threads(2)
    if args.action=='prepare':
        require(args.variant=='abc_b','Only ABC-B constructs intervention preprocessing')
        require(args.fold in spec['folds'] and args.prepared_root is not None,'--fold and --prepared-root required')
        result=prepare_fold(args.data_root,args.fold,args.prepared_root/f'fold_{args.fold}',contract=contract)
    elif args.action=='fit':
        seed=args.seed if args.seed is not None else 20260826
        require(args.fold in spec['folds'] and seed in spec['seeds'],'seed/fold outside frozen experiment')
        require(args.output is not None,'--output required')
        data=InterventionData(args.data_root,args.fold,spec['arm'],prepared_root=args.prepared_root,contract=contract)
        result=fit_one(config=config,root=args.data_root,contract=contract,model_name=spec['model'],
                       fold=args.fold,seed=seed,output=args.output,device=args.device,
                       variant=spec['model_variant'],data=data,model_config=spec['model_config'],
                       candidate=spec['model_config']['candidates'][0],
                       experiment_id=ablation['benchmark_id']+':'+args.variant,
                       resume={'path':str(args.resume),'sha256':args.resume_sha256} if args.resume else None)
    else:
        require(args.training_manifest is not None and args.output is not None,'--training-manifest and --output required')
        require(args.scenario in spec['scenarios'],'scenario outside frozen experiment')
        training=read_json(args.training_manifest);identity=training['identity']
        fold,seed=int(identity['fold']),int(identity['seed'])
        require(args.fold is None or args.fold==fold,'requested fold differs from selected checkpoint')
        require(args.seed is None or args.seed==seed,'requested seed differs from selected checkpoint')
        require(identity['model']==spec['model'],'wrong base model checkpoint')
        if args.variant in references:
            require(seed==spec['seed'] and identity['benchmark_id']==config['benchmark_id'] and
                    'variant' not in identity,'reference must be the selected main seed826 checkpoint')
        else:
            require(fold in spec['folds'] and seed in spec['seeds'],'checkpoint outside frozen seed/fold matrix')
            require(identity['benchmark_id']==ablation['benchmark_id']+':'+args.variant and
                    identity.get('variant')==spec['model_variant'],'checkpoint belongs to another experimental arm')
        data=InterventionData(args.data_root,fold,spec['arm'],prepared_root=args.prepared_root,contract=contract)
        result=evaluate_one(config=config,root=args.data_root,contract=contract,
                            training_manifest=args.training_manifest,scenario=args.scenario,
                            output=args.output,device=args.device,data=data)
    print(json.dumps(result,default=str))


if __name__=='__main__':main()
