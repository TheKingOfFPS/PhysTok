"""Synthetic counterexamples for resume, isolation, pooling and pairing invariants."""
import copy
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from phystok.models import build_model
from phystok.data import SyntheticView
from phystok.training.engine import run_fit, seed_all, evaluate_validation
from phystok.training.checkpoint import load_checkpoint
from phystok.training.config import load_config
from phystok.evaluation.metrics import pooled, aggregate, check_multiplicities
from phystok.evaluation.audits import paired_predictions, audit_timestamps, checkpoint_exposures

ROOT=Path(__file__).resolve().parents[1]

def synthetic_run(tmp_path,interrupt_after=None,resume=None):
    torch.set_num_threads(2)
    c=load_config(ROOT/'configs/benchmark_v2.json');cfg=copy.deepcopy(c['models']['dlinear'])
    cfg.update(batch_size=4,eval_batch_size=4)
    cfg['budget'].update(min_updates=1,initial_updates=6,max_updates=6,min_passes=.1,initial_passes=1.,
                         max_passes=10.,val_every_updates=2,val_every_passes=.1)
    g=torch.Generator(device='cpu').manual_seed(55)
    x=torch.randn(16,72,72,generator=g);y=x[:,-1,0]*.3
    fit=SyntheticView(x,y);val=SyntheticView(x[:8],y[:8],role='VAL')
    bridge=dict(input_center=0.,input_scale=1.,target_mean=0.,target_std=1.)
    layout=json.loads((ROOT/'configs/data_contract.json').read_text())
    seed_all(20260826,'cpu');model=build_model('dlinear',cfg['candidates'][0],layout,bridge)
    identity=dict(benchmark_id='test_v2_generated_only',phase='final',seed=20260826,model='dlinear',fold=0)
    result=run_fit(model,fit,val,identity=identity,model_config=cfg,candidate=cfg['candidates'][0],precision=c['precision'],
                   output=tmp_path,device='cpu',cpu_test=True,interrupt_after=interrupt_after,resume=resume)
    return model,result,identity,val,c['precision']


def test_complete_cpu_resume_replays_sampler_optimizer_and_rng(tmp_path):
    continuous,finished,_,_,_=synthetic_run(tmp_path/'continuous')
    _,interrupted,_,_,_=synthetic_run(tmp_path/'resume',interrupt_after=3)
    restored,continued,identity,_,_=synthetic_run(tmp_path/'resume',resume=interrupted['resume'])
    assert finished['counters']==continued['counters']
    for k,v in continuous.state_dict().items():assert torch.equal(v,restored.state_dict()[k]),k
    assert [e['val_rmse'] for e in finished['history']]==[e['val_rmse'] for e in continued['history']]
    state=load_checkpoint(tmp_path/'resume'/continued['best_checkpoint']['path'],identity,continued['best_checkpoint']['sha256'])
    assert state['kind']=='best'


def test_validation_rejects_fit_view(tmp_path):
    model,_,_,val,p=synthetic_run(tmp_path/'fit')
    assert evaluate_validation(model,val,4,p,'cpu')['n']==8
    with pytest.raises(ValueError):evaluate_validation(model,SyntheticView(val.x,val.y,'FIT'),4,p,'cpu')


def test_pooled_rmse_is_not_mean_fold_rmse():
    result=pooled([dict(n=1,sse=100.,sae=10.),dict(n=9,sse=0.,sae=0.)])
    assert result['rmse']==pytest.approx(np.sqrt(10.)) and result['mae']==1.
    assert result['rmse']!=5.
    with pytest.raises(ValueError):pooled([dict(n=0,sse=0.,sae=0.)])


def test_temporal_and_exposure_counterexamples():
    assert audit_timestamps(['2024-01-01'],['2024-01-02'],['2024-01-01'])['status']=='PASS'
    with pytest.raises(ValueError):audit_timestamps(['2024-01-01'],['2024-01-02'],['2024-01-01T01:00'])
    with pytest.raises(ValueError):audit_timestamps(['2024-12-31'],['2025-01-01'],['2024-12-31'])
    assert audit_timestamps(['2024-12-31'],['2025-01-01'],['2024-12-31'],internal=False)['status']=='PASS'
    with pytest.raises(ValueError):check_multiplicities([[1]]*10,'Spatial')
    assert check_multiplicities([[1]]*10,'External-Joint')['multiplicity']==10
    base=dict(model='phystok',seed=20260826,fold=0,transform_sha256='t',checkpoint_sha256='a')
    with pytest.raises(ValueError):checkpoint_exposures([dict(base,scenario='Spatial'),dict(base,scenario='Joint',checkpoint_sha256='b')])


def test_prediction_pairing_never_drops_mismatches(tmp_path):
    data=dict(benchmark_id=['synthetic']*2,seed=[1,1],fold=[0,0],scenario=['Spatial']*2,
              task_row=[0,1],sample_id=['fake_a','fake_b'],y_true=[1.,2.])
    a,b=tmp_path/'a.parquet',tmp_path/'b.parquet'
    pq.write_table(pa.table(data),a);pq.write_table(pa.table(data),b)
    assert paired_predictions(a,b)['paired_exposures']==2
    data['y_true']=[1.,3.];pq.write_table(pa.table(data),b)
    with pytest.raises(ValueError):paired_predictions(a,b)


def test_partial_seed_fold_grid_cannot_publish_sd():
    with pytest.raises(ValueError):aggregate([],benchmark_id='synthetic',models=['phystok'])
