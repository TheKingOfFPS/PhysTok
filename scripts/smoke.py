#!/usr/bin/env python3
"""One full-PhysTok CPU update on generated data; no research data or weights saved."""
import argparse,json
from pathlib import Path
import torch
from phystok.models import build_model
from phystok.training.engine import seed_all
from phystok.training.config import load_config
from phystok.utils.common import read_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads);seed_all(20260826,'cpu')
    root=Path(__file__).resolve().parents[1];c=load_config(root/'configs/benchmark_v2.json')
    candidate=c['models']['phystok']['candidates'][0]
    model=build_model('phystok',candidate,read_json(root/'configs/data_contract.json'),
                      dict(input_center=0.,input_scale=1.,target_mean=0.,target_std=1.))
    g=torch.Generator(device='cpu').manual_seed(42)
    batch={'x':torch.randn(2,72,72,generator=g),'marks':torch.zeros(2,72,4),'target_marks':torch.zeros(2,1,4)}
    target=batch['x'][:,-1,0]*.1
    opt=torch.optim.AdamW(model.parameters(),lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    opt.zero_grad(set_to_none=True);prediction=model(batch)
    loss=torch.nn.functional.mse_loss(prediction,target);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    model.eval()
    with torch.inference_mode():final=model(batch)
    assert final.shape==(2,) and torch.isfinite(final).all()
    print(json.dumps({'status':'PASS','synthetic':True,'device':'cpu','parameters':model.parameter_count(),
                      'shape':list(final.shape),'optimizer_updates':1,'loss':float(loss.detach()),'research_data_used':False}))

if __name__=='__main__':main()
