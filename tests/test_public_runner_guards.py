"""Synthetic counterexamples for stale cohort and cached-evaluation reuse.

Fixtures are generated metadata and literal placeholder bytes, not trained
weights, observations, or real predictions. No fit or GPU operation is run.
"""
import copy
import json
from pathlib import Path

import pytest

from phystok.training import app
from phystok.training.config import load_config
from phystok.utils.common import MODELS, SEEDS, fingerprint, sha256

ROOT=Path(__file__).resolve().parents[1]


def dump(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,sort_keys=True))
    return path


def fit_fixture(root,config,model='phystok',seed=20260826,fold=0):
    path=Path(root)/model/str(seed)/f'fold{fold}'/'SUCCESS.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    fake=path.parent/'checkpoint.synthetic'
    fake.write_bytes(f'GENERATED_NOT_A_MODEL:{model}:{seed}:{fold}'.encode())
    identity=dict(benchmark_id=config['benchmark_id'],model=model,seed=seed,fold=fold,
                  phase='final',config_sha256=fingerprint(config),
                  source_sha256='synthetic_code_binding',
                  data_manifest_sha256=fingerprint({'synthetic_contract':True}),
                  transform_sha256=f'synthetic_fold_{fold}',
                  training_recipe_sha256=fingerprint(config['models'][model]))
    document=dict(status='SUCCEEDED',identity=identity,stop_reason='QUALIFIED_VAL_PLATEAU',
                  best_checkpoint=dict(path=fake.name,sha256=sha256(fake)),
                  artifacts={fake.name:sha256(fake)})
    dump(path,document)
    return path,document


def evaluation_fixture(root,training,scenario='Joint'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    identity=training['identity']
    metrics=dict(benchmark_id=identity['benchmark_id'],model=identity['model'],seed=identity['seed'],
                 fold=identity['fold'],scenario=scenario,
                 checkpoint_sha256=training['best_checkpoint']['sha256'],
                 transform_sha256=identity['transform_sha256'],support_sha256='synthetic_support',
                 n=2,sse=2.,sae=2.,rmse=1.,mae=1.)
    dump(root/'metrics.json',metrics)
    # The binding validator hashes the artifact; prediction content belongs to
    # a separate pairing audit. This file deliberately is not real Parquet.
    (root/'predictions.synthetic').write_bytes(b'GENERATED_NOT_PREDICTIONS')
    success=dict(status='SUCCEEDED',identity=dict(identity,phase='evaluation',scenario=scenario),
                 artifacts={n:sha256(root/n) for n in ['metrics.json','predictions.synthetic']})
    dump(root/'SUCCESS.json',success)
    return root/'SUCCESS.json',metrics


def publish_changed_metric(evaluation_path, metrics):
    root=evaluation_path.parent
    dump(root/'metrics.json',metrics)
    success=json.loads(evaluation_path.read_text())
    success['artifacts']['metrics.json']=sha256(root/'metrics.json')
    dump(evaluation_path,success)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    config=load_config(ROOT/'configs/benchmark_v2.json')
    monkeypatch.setattr(app,'code_fingerprint',lambda:'synthetic_code_binding')
    path,training=fit_fixture(tmp_path/'fits',config)
    evaluation,metric=evaluation_fixture(tmp_path/'evaluation',training)
    return dict(config=config,contract={'synthetic_contract':True},training_manifest=path,
                evaluation_manifest=evaluation,training=training,metric=metric)


def binding_kwargs(fixture):
    return {k:fixture[k] for k in ['training_manifest','evaluation_manifest','config','contract']}|{'scenario':'Joint'}


def test_current_cached_evaluation_binding_is_accepted(fixture):
    assert app.validate_evaluation_binding(**binding_kwargs(fixture))['n']==2


def test_old_checkpoint_cache_rejected_even_with_valid_local_artifact_hashes(fixture):
    changed=dict(fixture['metric'],checkpoint_sha256='old_selected_checkpoint_hash')
    publish_changed_metric(fixture['evaluation_manifest'],changed)
    with pytest.raises(ValueError,match='selected checkpoint mismatch'):
        app.validate_evaluation_binding(**binding_kwargs(fixture))


def test_cache_from_another_scenario_rejected(fixture):
    with pytest.raises(ValueError,match='another training run'):
        app.validate_evaluation_binding(**(binding_kwargs(fixture)|{'scenario':'Spatial'}))


def test_stale_transform_cache_rejected(fixture):
    changed=dict(fixture['metric'],transform_sha256='old_transform')
    publish_changed_metric(fixture['evaluation_manifest'],changed)
    with pytest.raises(ValueError,match='transform mismatch'):
        app.validate_evaluation_binding(**binding_kwargs(fixture))


def test_cache_with_other_data_contract_rejected(fixture):
    with pytest.raises(ValueError,match='contract drift'):
        app.validate_evaluation_binding(**(binding_kwargs(fixture)|{'contract':{'other_contract':True}}))


def cohort_fixture(root,config):
    manifests=[];entries=[]
    for model in MODELS:
        for seed in SEEDS:
            for fold in range(10):
                path,doc=fit_fixture(Path(root)/'fits',config,model,seed,fold)
                manifests.append(path)
                entries.append(dict(identity=doc['identity'],training_manifest=str(path.resolve()),
                                    training_manifest_sha256=sha256(path),
                                    checkpoint_sha256=doc['best_checkpoint']['sha256']))
    path=dump(Path(root)/'cohort.json',dict(benchmark_id=config['benchmark_id'],
              config_sha256=fingerprint(config),records=entries))
    return path,manifests,entries


def test_complete_cohort_matches_exact_current_run_paths(tmp_path):
    config=load_config(ROOT/'configs/benchmark_v2.json')
    cohort,manifests,_=cohort_fixture(tmp_path,config)
    assert len(app.validate_cohort(cohort,config,manifests)['records'])==210
    # Same model/seed/fold/config at a different attempt path must not be
    # silently substituted into an already frozen cohort.
    replacement,_=fit_fixture(tmp_path/'new_attempt',config,MODELS[0],SEEDS[0],0)
    changed=[replacement,*manifests[1:]]
    with pytest.raises(ValueError,match='different runs'):
        app.validate_cohort(cohort,config,changed)


def test_changed_training_receipt_invalidates_old_cohort(tmp_path):
    config=load_config(ROOT/'configs/benchmark_v2.json')
    cohort,manifests,_=cohort_fixture(tmp_path,config)
    doc=json.loads(manifests[0].read_text());doc['new_generated_marker']='different receipt'
    dump(manifests[0],doc)
    with pytest.raises(ValueError,match='training identity changed'):
        app.validate_cohort(cohort,config,manifests)


def test_cohort_with_a_resource_cap_is_not_accepted(tmp_path):
    config=load_config(ROOT/'configs/benchmark_v2.json')
    cohort,manifests,entries=cohort_fixture(tmp_path,config)
    doc=json.loads(manifests[0].read_text());doc['stop_reason']='RESOURCE_CAP_UNCERTIFIED'
    dump(manifests[0],doc)
    data=json.loads(cohort.read_text());data['records'][0]['training_manifest_sha256']=sha256(manifests[0]);dump(cohort,data)
    with pytest.raises(ValueError,match='qualification changed'):
        app.validate_cohort(cohort,config,manifests)
