"""Portable entry points around the frozen V2 optimization and data contracts."""
from pathlib import Path
import json
import os
import uuid
import numpy as np
import torch

from phystok.data import CityData
from phystok.data.rebuild import verify_local_inputs
from phystok.models import build_model, build_control_model, build_capacity_model
from phystok.utils.common import (require, read_json, sha256, fingerprint, atomic_json,
                                  verify_artifacts, MODELS, SEEDS, SCENARIOS)
from .config import validate_config
from .engine import run_fit, seed_all, set_precision, model_batch, autocast
from .checkpoint import load_checkpoint
from phystok.evaluation.metrics import errors, pooled

CONTROLS = {'delayed_xview', 'strict_late_fusion', 'plain_state_flatten'}
CAPACITY = {'phystok_small', 'itransformer_scaled', 'timexer_scaled'}


def construct(model_name, candidate, data, precision, variant=None):
    if variant in CONTROLS:
        return build_control_model(variant, candidate, data.layout, data.bridge, precision['eval_rng_seed'])
    if variant in CAPACITY:
        return build_capacity_model(variant, candidate, data.layout, data.bridge, precision['eval_rng_seed'])
    require(variant is None, 'unknown experimental model variant')
    return build_model(model_name, candidate, data.layout, data.bridge, precision['eval_rng_seed'])


def code_fingerprint():
    """Bind actual Python code, including locally authorized external cores."""
    root = Path(__file__).resolve().parents[1]
    return fingerprint({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))})


def fit_one(*, config, root, contract, model_name, fold, seed, output, device='cpu',
            variant=None, data=None, model_config=None, candidate=None, experiment_id=None, resume=None):
    validate_config(config)
    require(model_name in MODELS and fold in range(10) and seed in SEEDS, 'model/seed/fold outside frozen protocol')
    verify_local_inputs(root, contract)
    data = data if data is not None else CityData(root, fold, contract=contract)
    require(data.fold == fold, 'data fold mismatch')
    fit, val = data.view('FIT'), data.view('VAL')
    data.warm_features([fit, val])
    cfg = model_config if model_config is not None else config['models'][model_name]
    candidate = candidate if candidate is not None else cfg['candidates'][0]
    if variant is None and experiment_id is None:
        require(cfg == config['models'][model_name] and candidate == cfg['candidates'][0], 'main recipe override forbidden')
    identity = dict(benchmark_id=experiment_id or config['benchmark_id'], config_sha256=fingerprint(config),
                    source_sha256=code_fingerprint(), data_manifest_sha256=fingerprint(contract),
                    model=model_name, fold=int(fold), seed=int(seed), candidate_id=0,
                    candidate_sha256=fingerprint(candidate), phase='final', transform_sha256=data.transform_hash,
                    job_id=f'{variant or model_name}-{seed}-fold{fold}', attempt=1)
    identity['training_recipe_sha256'] = fingerprint(cfg)
    if variant is not None:
        identity['variant'] = variant
    output = Path(output)
    spec = {'model_config':cfg,'candidate':candidate,'variant':variant,'experiment_id':experiment_id}
    if (output/'run_spec.json').exists():
        require(read_json(output/'run_spec.json') == spec and resume is not None, 'existing run requires exact resume')
    else:
        atomic_json(output/'run_spec.json', spec)
    # CPU RNG is initialized BEFORE model construction; CUDA training is seeded separately by run_fit.
    seed_all(seed, 'cpu')
    model = construct(model_name, candidate, data, config['precision'], variant)
    result = run_fit(model, fit, val, identity=identity, model_config=cfg, candidate=candidate,
                     precision=config['precision'], output=Path(output), device=device, config=config, resume=resume)
    return result


def freeze_cohort(*, manifests, config, output):
    """Freeze exactly 210 completed VAL-selected fits before any main TEST evaluation."""
    validate_config(config)
    output = Path(output)
    keys, records = set(), []
    require(len(manifests) == 210, 'main cohort requires all 210 fits')
    for path in manifests:
        path = Path(path).resolve()
        doc = verify_artifacts(path)
        i = doc['identity']; key = (i['model'], i['seed'], i['fold'])
        require(key not in keys and i['phase'] == 'final' and 'variant' not in i, 'duplicate/non-main fit')
        require(i['benchmark_id'] == config['benchmark_id'] and i['config_sha256'] == fingerprint(config), 'cohort config mismatch')
        require(i['training_recipe_sha256'] == fingerprint(config['models'][i['model']]), 'cohort training recipe drift')
        require(doc['stop_reason'] == 'QUALIFIED_VAL_PLATEAU', 'uncertified/resource-capped run cannot enter accepted main cohort')
        cp = path.parent / doc['best_checkpoint']['path']
        state = load_checkpoint(cp, i, doc['best_checkpoint']['sha256'], metadata_only=True)
        require(state['kind'] == 'best' and state['candidate'] == config['models'][i['model']]['candidates'][0], 'wrong selected checkpoint')
        keys.add(key)
        records.append({'identity': i, 'training_manifest': str(path), 'training_manifest_sha256': sha256(path),
                        'checkpoint_sha256': doc['best_checkpoint']['sha256']})
    require(keys == {(m,s,f) for m in MODELS for s in SEEDS for f in range(10)}, 'incomplete model/seed/fold cohort')
    require(len({r['identity']['source_sha256'] for r in records}) == 1, 'code changed between fits')
    require(len({r['identity']['data_manifest_sha256'] for r in records}) == 1, 'data changed between fits')
    atomic_json(output, {'benchmark_id': config['benchmark_id'], 'config_sha256': fingerprint(config), 'records': records})
    return output


def validate_cohort(path, config, manifests=None):
    """Reject stale cohorts even when their own JSON and prior runs are intact."""
    c=read_json(path)
    require(c['benchmark_id']==config['benchmark_id'] and c['config_sha256']==fingerprint(config),'stale cohort config')
    expected={(m,s,f) for m in MODELS for s in SEEDS for f in range(10)}
    require(len(c['records'])==210,'incomplete cohort')
    keys=set();paths=set()
    for entry in c['records']:
        p=Path(entry['training_manifest']).resolve();doc=verify_artifacts(p);i=doc['identity']
        require(sha256(p)==entry['training_manifest_sha256'] and i==entry['identity'],'cohort training identity changed')
        require(i['phase']=='final' and i['config_sha256']==fingerprint(config) and i['benchmark_id']==config['benchmark_id'] and 'variant' not in i,'non-main/stale cohort member')
        require(i['training_recipe_sha256']==fingerprint(config['models'][i['model']]),'cohort recipe changed')
        require(doc['stop_reason']=='QUALIFIED_VAL_PLATEAU' and entry['checkpoint_sha256']==doc['best_checkpoint']['sha256'],'cohort checkpoint/qualification changed')
        key=(i['model'],i['seed'],i['fold']);require(key not in keys,'duplicate cohort member');keys.add(key);paths.add(p)
    require(keys==expected,'cohort grid mismatch')
    require(len({e['identity']['source_sha256'] for e in c['records']})==1,'cohort source drift')
    require(len({e['identity']['data_manifest_sha256'] for e in c['records']})==1,'cohort data drift')
    if manifests is not None:require(paths==set(Path(p).resolve() for p in manifests),'cohort refers to different runs')
    return c


def validate_evaluation_binding(*, training_manifest, evaluation_manifest, config, contract, scenario):
    """A cached result is reusable only for this exact current training identity."""
    training=verify_artifacts(training_manifest);i=training['identity']
    evaluated=verify_artifacts(evaluation_manifest);p=Path(evaluation_manifest).parent
    metric=read_json(p/'metrics.json')
    require(i['config_sha256']==fingerprint(config) and i['data_manifest_sha256']==fingerprint(contract), 'cached fit contract drift')
    require(i['source_sha256']==code_fingerprint(),'cached fit code drift')
    expected=dict(i,phase='evaluation',scenario=scenario)
    require(evaluated['identity']==expected,'cached evaluation belongs to another training run')
    require(all(metric[k]==i[k] for k in ['benchmark_id','model','seed','fold']),'cached prediction exposure mismatch')
    require(metric['scenario']==scenario and metric['checkpoint_sha256']==training['best_checkpoint']['sha256'],'cached selected checkpoint mismatch')
    require(metric['transform_sha256']==i['transform_sha256'],'cached evaluation transform mismatch')
    return metric


def evaluate_one(*, config, root, contract, training_manifest, scenario, output, device='cpu', data=None):
    """Predict using the existing best-VAL checkpoint; no optimization or selection."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    require(scenario in SCENARIOS, 'unknown population (Future is the canonical Future-Year key)')
    path = Path(training_manifest)
    training = verify_artifacts(path); i = training['identity']
    require(i['phase']=='final' and training['stop_reason']=='QUALIFIED_VAL_PLATEAU', 'evaluation requires a qualified final fit')
    require(i['config_sha256'] == fingerprint(config), 'evaluation config drift')
    require(i['source_sha256'] == code_fingerprint(), 'evaluation code differs from fitted code')
    require(i['data_manifest_sha256'] == fingerprint(contract), 'evaluation data contract drift')
    verify_local_inputs(root, contract)
    data = data if data is not None else CityData(root, i['fold'], contract=contract)
    require(data.transform_hash == i['transform_sha256'], 'evaluation transform drift')
    checkpoint = path.parent / training['best_checkpoint']['path']
    state = load_checkpoint(checkpoint, i, training['best_checkpoint']['sha256'])
    require(state['kind'] == 'best', 'only best-VAL checkpoint may be scored')
    view = data.view(scenario, purpose='evaluation', frozen_identity=i)
    data.warm_features([view])
    set_precision(config['precision'], device); seed_all(i['seed'], device)
    model = construct(i['model'], state['candidate'], data, config['precision'], i.get('variant')).to(device)
    model.load_state_dict(state['model_state'], strict=True); model.eval()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    require(not (output/'predictions.parquet').exists(), 'prediction output exists; use a new directory')
    fields = [('benchmark_id',pa.string()),('model',pa.string()),('seed',pa.int64()),('fold',pa.int64()),
              ('scenario',pa.string()),('task_row',pa.int64()),('sample_id',pa.string()),('y_true',pa.float64()),('y_pred',pa.float64())]
    schema = pa.schema(fields); parts = []; seen = set()
    temporary = output/f'.pending-predictions-{uuid.uuid4().hex}.parquet'
    spec = read_json(path.parent/'run_spec.json')
    require(fingerprint(spec['model_config']) == i['training_recipe_sha256'], 'run-specific evaluation recipe drift')
    bs = spec['model_config']['eval_batch_size']
    try:
        with pq.ParquetWriter(temporary, schema, compression='zstd') as writer, torch.inference_mode():
            for st in range(0, len(view), bs):
                b = model_batch(model, view, np.arange(st,min(st+bs,len(view))), device)
                with autocast(config['precision'], device): pred = model(b)
                p = model.bridge.to_raw(pred.float()).cpu().numpy().astype(np.float64)
                y = b['y_raw'].cpu().numpy().astype(np.float64); ids = list(map(int,b['task_rows']))
                require(not (seen & set(ids)) and len(set(ids)) == len(ids), 'duplicate prediction exposure')
                seen.update(ids); parts.append(errors(y,p)); n=len(ids)
                writer.write_table(pa.Table.from_pydict(dict(benchmark_id=[i['benchmark_id']]*n,model=[i['model']]*n,
                    seed=[i['seed']]*n,fold=[i['fold']]*n,scenario=[scenario]*n,task_row=ids,sample_id=b['sample_ids'],y_true=y,y_pred=p),schema=schema))
        require(seen == set(map(int,view.rows)), 'prediction support differs from frozen selector')
        os.link(temporary,output/'predictions.parquet');temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    metric = dict(benchmark_id=i['benchmark_id'],model=i['model'],seed=i['seed'],fold=i['fold'],scenario=scenario,
                  checkpoint_sha256=sha256(checkpoint), transform_sha256=data.transform_hash,
                  support_sha256=fingerprint(sorted(seen)), **pooled(parts))
    if 'variant' in i: metric['variant'] = i['variant']
    atomic_json(output/'metrics.json',metric)
    atomic_json(output/'SUCCESS.json',dict(status='SUCCEEDED',identity=dict(i,phase='evaluation',scenario=scenario),
                artifacts={n:sha256(output/n) for n in ['metrics.json','predictions.parquet']}))
    return metric


def persistence(*, root, contract, scenario, output, benchmark_id='phystok_v2_frozen_recipe_20261017_r1'):
    """One deterministic prediction per physical target, with the frozen past-only fallback."""
    from phystok.data.loader import FOLD_ROOT, ROLE_SELECTOR
    verify_local_inputs(root, contract)
    data = CityData(root,0,contract=contract); external = scenario.startswith('External-')
    task = data.metadata(external)
    if external:
        rows = np.flatnonzero(task.target_valid_benchmark.to_numpy(bool) & (task.external_scenario==scenario).to_numpy())
    else:
        parts=[]
        for f in range(10):
            with np.load(data.root/f'{FOLD_ROOT}/fold_{f}/scenario_task_rows.npz',allow_pickle=False) as s:
                parts.append(s[ROLE_SELECTOR[scenario]])
        rows=np.unique(np.concatenate(parts)).astype(np.int64)
    parts=[]
    for st in range(0,len(rows),4096):
        m=task.iloc[rows[st:st+4096]]; city=m.city_index.to_numpy(np.int64); issue=m.issue_time_index.to_numpy(np.int64)
        times=issue[:,None]+np.arange(-72,1)[None,:]
        h=data.array('pm25_now',external=external).take(np.maximum(times,0),city[:,None]).copy();h[times<0]=np.nan
        finite=np.isfinite(h);require(finite.any(axis=1).all(),'no legal Persistence fallback')
        p=h[np.arange(len(m)),72-np.argmax(finite[:,::-1],axis=1)].astype(np.float64)
        y=data.array('pm25_city_mean',target=True,external=external).take(m.target_time_index.to_numpy(np.int64),city).astype(np.float64)
        parts.append(errors(y,p))
    result=dict(benchmark_id=benchmark_id,model='persistence',scenario=scenario,seed=None,fold=None,
                data_manifest_sha256=fingerprint(contract),support_sha256=fingerprint(list(map(int,rows))),**pooled(parts))
    atomic_json(Path(output),result)
    return result
