"""Local reconstruction orchestration; datasets are never shipped by this package.

Numerical feature/physics/selection functions are retained in pipeline/. Routing
and historical-operational audit setup are the only new preparation components.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from ._common import PACKAGE, require, read_json, sha256, fingerprint
from .loader import (SCHEMA, SPLIT, FOLD_ROOT, INTERNAL_BANK, INTERNAL_TARGET,
                     INTERNAL_TASK, EXTERNAL_BANK, EXTERNAL_TARGET, EXTERNAL_TASK)

PIPELINE = Path(__file__).parent / 'pipeline'
METADATA = PACKAGE / 'data/metadata'
STEPS = {'cnemc', 'era5', 'internal', 'external', 'audit'}
INPUT_DIRECTORIES = (INTERNAL_BANK, INTERNAL_TARGET, EXTERNAL_BANK, EXTERNAL_TARGET)
INPUT_TABLES = (INTERNAL_TASK, EXTERNAL_TASK)

def _safe_local_file(root, relative):
    """Reject path traversal and symlinks before following an input binding."""
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
            'manifest input path must be relative and contained in data-root')
    path = root/relative
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        require(not candidate.is_symlink(), f'symlink data input forbidden: {relative}')
    require(path.resolve().is_relative_to(root), f'input escapes data-root: {relative}')
    require(path.is_file(), f'missing input file: {relative}')
    return path

def _input_file_paths(root):
    """Enumerate only the four fixed continuous stores and two task tables."""
    paths = {}
    for relative in INPUT_DIRECTORIES:
        directory = root/relative
        require(directory.is_dir(), f'missing reconstructed input store: {relative}')
        for candidate in (directory, *directory.parents):
            if candidate == root:
                break
            require(not candidate.is_symlink(), f'symlink input store forbidden: {relative}')
        contents = sorted(directory.rglob('*'))
        require(contents, f'empty input store: {relative}')
        for path in contents:
            require(not path.is_symlink(), f'symlink data input forbidden: {path.relative_to(root)}')
            if path.is_file():
                name = path.relative_to(root).as_posix()
                paths[name] = _safe_local_file(root, name)
    for name in INPUT_TABLES:
        paths[name] = _safe_local_file(root, name)
    return paths

def build_input_file_manifest(root):
    """Hash the actual user-owned feature/target arrays and task tables."""
    root = Path(root).expanduser().resolve()
    return {name: {'bytes':path.stat().st_size, 'sha256':sha256(path)}
            for name,path in _input_file_paths(root).items()}

def _stat_identity(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_dev, s.st_ino

def verify_local_inputs(root, contract):
    """Rehash actual inputs before a formal fit/evaluation.

    Every call re-enumerates the fixed input roots and hashes actual bytes.
    Stat-only caching is deliberately avoided: some filesystems retain the
    same timestamps for distinct same-size writes. New, missing or altered
    files fail. Data workspaces must have no concurrent writers during a run.
    """
    root = Path(root).expanduser().resolve()
    manifest = contract.get('input_files')
    require(isinstance(manifest,dict) and manifest,
            'formal runs require content-bound inputs: run prepare_data.py bind')
    actual = _input_file_paths(root)
    require(set(actual)==set(manifest), 'input file membership changed since data binding')
    bound = contract.get('bound_files',{})
    require(isinstance(bound,dict),'bound_files must be a mapping')
    checks = [(name,path,manifest[name]['sha256'],int(manifest[name]['bytes']))
              for name,path in actual.items()]
    checks += [(name,_safe_local_file(root,name),digest,None) for name,digest in bound.items()]
    for name,path,digest,size in checks:
        stat_before = _stat_identity(path)
        require(size is None or stat_before[0]==size, f'input size changed: {name}')
        require(sha256(path)==digest, f'input content binding failed: {name}')
        require(_stat_identity(path)==stat_before, f'input changed during verification: {name}')
    return {'status':'PASS','input_files':len(actual),'bound_files':len(bound),
            'input_bytes':sum(int(v['bytes']) for v in manifest.values()),
            'content_hashes_checked':len(checks),'hash_cache_hits':0,
            'contract_sha256':fingerprint(contract)}

def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data if isinstance(data, bytes) else (json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode()
    if path.exists():
        require(path.read_bytes() == payload, f'Refusing to replace different existing artifact: {path}')
    else:
        path.write_bytes(payload)

def stage_module(name):
    path = PIPELINE/name
    spec = importlib.util.spec_from_file_location('_public_data_'+path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def run_stage(name, root, *extra):
    subprocess.run([sys.executable, str(PIPELINE/name), '--root', str(root), *map(str, extra)], check=True)

def initialize(root):
    root = Path(root).expanduser().resolve()
    internal = pd.read_csv(METADATA/'internal_cities.csv')
    external = pd.read_csv(METADATA/'external_cities.csv')
    require(len(internal)==50 and len(external)==10, 'frozen city roster counts')
    require(not set(internal.city)&set(external.city), 'external cities overlap development cities')
    manifest = root/'05_manifests'
    mappings = {
        'internal_city_manifest.csv':manifest/'selected_cities_v2.csv',
        'cross_city_fold_assignment.json':manifest/'cross_city_fold_assignment.json',
        'fold_city_roles.csv':manifest/'splits'/SPLIT/'fold_city_roles.csv',
        'split_protocol.json':manifest/'splits'/SPLIT/'split_protocol.json',
    }
    for name,dest in mappings.items():
        write_new(dest,(METADATA/name).read_bytes())
    for name in ['cnemc_station_qc_policy_v1.json','benchmark_validity_policy_v1.json',
                 'model_sample_eligibility_policy_v1.json','external_central_china_protocol_v1.json']:
        write_new(manifest/name,(METADATA/name).read_bytes())
    for k in range(10):
        write_new(manifest/'splits'/SPLIT/f'fold_{k}.json',(METADATA/'folds'/f'fold_{k}.json').read_bytes())
    ext_columns = [c for c in external if c!='city_index']
    write_new(manifest/'external_central_china_10cities_v1.csv',external[ext_columns].to_csv(index=False).encode())
    # This is a fresh metadata-only check, not a copy of a historical PASS receipt.
    write_new(root/'09_diagnostics/external_validation/central_china_external_holdout_v1/protocol_freeze/summary.json',
              {'status':'PASS','source':'public local metadata validation',
               'scope':'city/region protocol only; no acquisition, observations, training or prediction checked',
               'external_city_order':external.city.tolist(), 'development_cities':50,'external_cities':10,
               'checks':{'city_sets_disjoint':True,'internal_regions':internal.region.nunique()==5,
                         'external_one_region':external.region.nunique()==1}})
    return root

def create_station_eligibility(root):
    roster_path = root/'06_processed/audits/cnemc_station_registry_50city_roster_v1/50city_station_roster_review.csv'
    x = pd.read_csv(roster_path,encoding='utf-8-sig')
    cols=['region','city_en','city_zh','registry_city','station_id','station_name','longitude','latitude','control_site']
    out=x[cols].copy()
    out['eligible_city_station']=out.control_site.ne('Y')
    out['exclusion_reason']=np.where(out.eligible_city_station,'','CONTROL_SITE')
    out['policy_id']='cnemc_station_qc_policy_v1'
    write_new(root/'05_manifests/cnemc_station_eligibility_v1.csv',out.to_csv(index=False,encoding='utf-8-sig').encode('utf-8-sig'))

def check_era5_before_extraction(root):
    """Re-execute real frozen interpolation checks, independent of old CAMS gates."""
    import xarray as xr
    m=stage_module('05_extract_city_meteorology.py')
    rows=m.load_manifest(root/'05_manifests/selected_cities_v2.csv')
    stores={}
    reports=[]
    try:
        for region in sorted(m.EXPECTED_REGIONS):
            ds=xr.open_zarr(root/f'04_derived/era5_physical/{region}.zarr',consolidated=True,chunks=None)
            stores[region]=ds
            m.ensure_exact_timeline(ds.time.values,region)
            require(set(m.REQUIRED_SOURCE_VARS)<=set(ds.data_vars),f'{region}: missing physical fields')
            require(ds.pressure_level_hpa.values.tolist()==[925,850],f'{region}: pressure order')
        for row in rows:
            out,mapping=m.extract_city(stores[row['region']],row)
            require(all(np.isfinite(a).all() for a in out.values()),f"{row['city']}: interpolation not finite")
            reports.append({'city':row['city'],'region':row['region'],
                            'renormalized_925_hours':int(mapping['925_renormalized_hours'])})
    finally:
        for ds in stores.values():ds.close()
    write_new(root/'09_diagnostics/pre_city_extraction/full_diagnostic_report.json',
              {'overall_status':'PASS','fail_gates':[],'review_gates':[],
               'audit_version':'public_local_era5_city_support_gate_v1',
               'scope':'all five ERA5 physical stores and all 50 frozen city interpolations',
               'historical_aq_gate':'Not used: obsolete CAMS data are not part of this CNEMC benchmark.',
               'city_checks':reports})

def rebuild(root, step, *, plan_only=False):
    root=Path(root).expanduser().resolve()
    require(step in STEPS,'unknown preprocessing step')
    require((root/'05_manifests/selected_cities_v2.csv').exists(),'run init first')
    if step=='cnemc':
        run_stage('audit_cnemc_station_registry_and_50city_roster.py',root)
        run_stage('audit_cnemc_station_lifecycle_validity_sensitivity.py',root)
        create_station_eligibility(root)
        run_stage('build_cnemc_city_air_quality_canonical.py',root)
    elif step=='era5':
        extra=['--plan-only'] if plan_only else []
        run_stage('acquire_era5_bulk.py',root,*extra)
        run_stage('75b_acquire_central_china_external_era5.py',root,*extra,'--download-backend','cdsapi')
    elif step=='internal':
        run_stage('02_build_era5_canonical.py',root)
        run_stage('03_build_era5_derived.py',root)
        check_era5_before_extraction(root)
        for name in ['05_extract_city_meteorology.py','06_build_observed_era5_merged.py',
                     '07_build_pm25_24h_task_index.py','08_bind_observed_task_splits.py',
                     '09_build_observed_issue_time_feature_bank.py','10_build_observed_fold_local_artifacts.py']:
            run_stage(name,root)
    elif step=='external':
        run_stage('75a_build_central_china_external_cnemc.py',root)
        run_stage('76_build_central_china_external_frozen_inputs.py',root,'--source-dir',PIPELINE)
    elif step=='audit':
        run_stage('audit_observed_issue_time_feature_bank.py',root)
        run_stage('audit_observed_fold_local_artifacts.py',root)

def bind_local_contract(root, output=None, *, template=None):
    """Bind local rebuild metadata; this never pretends to certify original bytes."""
    root=Path(root).expanduser().resolve()
    layout=read_json(template or PACKAGE/'configs/data_contract.json')
    names=[f'05_manifests/features/{SCHEMA}/feature_schema.json',
           f'{FOLD_ROOT}/feature_masks.npz']
    for fold in range(10):
        names += [f'05_manifests/splits/{SPLIT}/fold_{fold}.json']
        names += [f'{FOLD_ROOT}/fold_{fold}/{name}' for name in
                  ['preprocess_stats.npz','target_stats.json','scenario_task_rows.npz']]
    for name in names:require((root/name).is_file(),f'missing rebuilt artifact: {name}')
    schema=read_json(root/names[0])
    require(schema['feature_order']==layout['feature_order'],'feature order differs from frozen recipe')
    layout['bound_files']={name:sha256(root/name) for name in names}
    layout['input_files']=build_input_file_manifest(root)
    layout['binding_note']='Bound to the user-local rebuilt artifacts. Does not certify equality to unavailable original raw archive snapshots.'
    dest=Path(output) if output is not None else root/'data_contract.json'
    write_new(dest,layout)
    return dest
