"""Permission-scoped city data, FIT transforms, on-demand causal windows.

No xarray/Zarr async engine and no original training loader imports. Compressed
chunks may physically contain other cities; only authorized indexed cells are
returned/used. No target array is cached wholesale. See access_log.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import math
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from numcodecs import get_codec
from ._common import require, read_json, sha256, fingerprint, PACKAGE

SCHEMA = 'national_pm25_observed_feature_schema_v1'
SPLIT = 'national_pm25_ood_split_v1'
EXT = 'central_china_external_holdout_v1'
FOLD_ROOT = f'06_processed/fold_artifacts/{SPLIT}/{SCHEMA}'
INTERNAL_BANK = f'04_derived/issue_time_features/{SCHEMA}/issue_time_feature_bank.zarr'
INTERNAL_TARGET = '04_derived/observed_era5_merged/observed_era5_50cities_v1.zarr'
INTERNAL_TASK = '06_processed/task_index/pm25_24h_task_index_v1.parquet'
EXTERNAL_BANK = f'04_derived/external_validation/{EXT}/issue_time_features/{SCHEMA}/issue_time_feature_bank.zarr'
EXTERNAL_TARGET = f'04_derived/external_validation/{EXT}/observed_era5_merged/observed_era5_10cities_v1.zarr'
EXTERNAL_TASK = f'06_processed/external_validation/{EXT}/task_index/pm25_24h_task_index_external_v1.parquet'
ROLE_SELECTOR = {'FIT':'fit_task_rows','VAL':'validation_task_rows','Spatial':'spatial_ood_final_task_rows',
                 'Future':'temporal_ood_final_task_rows','Joint':'joint_ood_final_task_rows'}
_METADATA_CACHE = OrderedDict()


class ZarrV2Array:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read_json(self.path / '.zarray')
        m = self.meta
        require(m['zarr_format'] == 2 and m.get('filters') is None and len(m['shape']) in [1,2], 'unsupported Zarr contract')
        self.shape, self.chunks, self.dtype = tuple(m['shape']), tuple(m['chunks']), np.dtype(m['dtype'])
        require(self.dtype.kind != 'O', 'object Zarr arrays not permitted')
        self.codec = get_codec(m['compressor']) if m['compressor'] else None
        self.cache = OrderedDict()
        self.bytes_read = 0

    def chunk(self, key):
        key = tuple(int(i) for i in key)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        p = self.path / self.meta.get('dimension_separator','.').join(map(str,key))
        if p.exists():
            raw = p.read_bytes()
            self.bytes_read += len(raw)
            raw = self.codec.decode(raw) if self.codec else raw
            a = np.frombuffer(raw,dtype=self.dtype).reshape(self.chunks,order=self.meta['order'])
        else:
            fill = self.meta['fill_value']
            if fill == 'NaN':
                fill = np.nan
            if fill is None:
                fill = '' if self.dtype.kind in 'US' else 0
            a = np.full(self.chunks,fill,dtype=self.dtype)
        self.cache[key] = a
        if len(self.cache) > 2:
            self.cache.popitem(last=False)
        return a

    def take(self, *coords):
        require(len(coords) == len(self.shape), 'Zarr coordinate rank')
        coords = np.broadcast_arrays(*[np.asarray(c,dtype=np.int64) for c in coords])
        shape = coords[0].shape
        flat = [c.ravel() for c in coords]
        for a,n in zip(flat,self.shape):
            require(np.all((a >= 0) & (a < n)), 'Zarr index out of bounds')
        keys = np.stack([a//k for a,k in zip(flat,self.chunks)],axis=1)
        unique, inverse = np.unique(keys,axis=0,return_inverse=True)
        result = np.empty(len(flat[0]),dtype=self.dtype)
        for i,key in enumerate(unique):
            pos = np.flatnonzero(inverse == i)
            arr = self.chunk(key)
            result[pos] = arr[tuple(a[pos] % k for a,k in zip(flat,self.chunks))]
        return result.reshape(shape)

    def all(self):
        return self.take(*np.ix_(*[np.arange(n) for n in self.shape]))


def time_marks(times):
    t = pd.DatetimeIndex(np.asarray(times).reshape(-1))
    out = np.stack([t.hour/23-.5,t.dayofweek/6-.5,(t.day-1)/30-.5,(t.dayofyear-1)/365-.5],axis=-1)
    return out.astype(np.float32)


def validate_split(split):
    require(split['fold'] in range(10), 'invalid outer fold')
    roles = [set(split[k]) for k in ['train_cities','validation_cities','test_cities']]
    require([len(s) for s in roles] == [40,5,5] and len(set.union(*roles)) == 50, '40/5/5 roles must be disjoint')
    for role,n in [('train',8),('validation',1),('test',1)]:
        groups = split['cities_by_region'][role]
        require(len(groups) == 5 and all(len(v)==n for v in groups.values()), 'region balance')
    return split


class CityData:
    def __init__(self, root, fold, *, contract=None):
        self.root = Path(root).resolve()
        require(fold in range(10), 'outer fold')
        self.fold = fold
        self.layout = contract or read_json(PACKAGE / 'configs/data_contract.json')
        schema_path=self.root/f'05_manifests/features/{SCHEMA}/feature_schema.json'
        require(sha256(schema_path)==self.layout['bound_files'].get(str(schema_path.relative_to(self.root))),'schema binding mismatch: run prepare_data.py bind for this locally reconstructed dataset')
        require(read_json(schema_path)['feature_order']==self.layout['feature_order'],'feature channel order mismatch')
        split_path = self.root / f'05_manifests/splits/{SPLIT}/fold_{fold}.json'
        self.split = validate_split(read_json(split_path))
        require(self.split['fold'] == fold, 'fold path/identity mismatch')
        paths = [split_path, self.root/f'{FOLD_ROOT}/fold_{fold}/preprocess_stats.npz',
                 self.root/f'{FOLD_ROOT}/fold_{fold}/target_stats.json',
                 self.root/f'{FOLD_ROOT}/fold_{fold}/scenario_task_rows.npz']
        for p in paths:
            expected = self.layout['bound_files'].get(str(p.relative_to(self.root)))
            require(expected is not None and sha256(p) == expected, f'fold-specific artifact binding failed: {p.name}')
        self.transform_hash = fingerprint({str(p.relative_to(self.root)):sha256(p) for p in paths})
        self.pre = dict(np.load(paths[1],allow_pickle=False))
        self.target_stats = read_json(paths[2])['statistics']
        self.selectors = dict(np.load(paths[3],allow_pickle=False))
        self.task = None
        self.external_task = None
        self.arrays = {}
        self.access_log = []
        self.access_counts = {}
        self.cache = None
        self.bridge = dict(input_center=float(self.pre['transform_center'][0]),input_scale=float(self.pre['transform_scale'][0]),
                           target_mean=float(self.target_stats['mean']),target_std=float(self.target_stats['std']))

    def metadata(self, external=False):
        attr = 'external_task' if external else 'task'
        if getattr(self,attr) is None:
            task_path=self.root/(EXTERNAL_TASK if external else INTERNAL_TASK)
            # Root names can be reused for a legitimate local reconstruction.
            # Bind cached metadata to actual table content, not path or mtime.
            cache_key=(str(self.root),external,sha256(task_path))
            if cache_key in _METADATA_CACHE:
                setattr(self,attr,_METADATA_CACHE[cache_key])
                return getattr(self,attr)
            cols = ['sample_id','city_index','city_en','issue_time_index','target_time_index',
                    'issue_time_local','target_time_local','target_local_year','forecast_horizon_hours','target_valid_benchmark']
            if external:
                cols += ['external_scenario','external_task_row']
            table = pd.read_parquet(task_path,columns=cols)
            require(table.sample_id.is_unique, 'physical task identity duplicates')
            if external:
                require(np.array_equal(table.external_task_row,np.arange(len(table))), 'external row identity')
            setattr(self,attr,table)
            _METADATA_CACHE[cache_key]=table
            if len(_METADATA_CACHE)>2:
                _METADATA_CACHE.popitem(last=False)
        return getattr(self,attr)

    def view(self, role, *, purpose='development', limit=None, frozen_identity=None):
        external = role.startswith('External-')
        if purpose == 'development':
            require(role in ['FIT','VAL'] and not external, 'TEST/2025/external not available to training or HPO')
        else:
            require(purpose == 'evaluation' and role in ['Spatial','Future','Joint','External-Spatial','External-Joint'], 'unknown data purpose')
            require(frozen_identity is not None and frozen_identity['fold'] == self.fold and
                    frozen_identity['transform_sha256'] == self.transform_hash and frozen_identity['phase']=='final', 'frozen fold identity required')
        task = self.metadata(external)
        if external:
            rows = np.flatnonzero(task.target_valid_benchmark.to_numpy(bool) & (task.external_scenario == role).to_numpy())
        else:
            rows = np.asarray(self.selectors[ROLE_SELECTOR[role]],dtype=np.int64)
        require(len(rows) > 0 and len(rows) == len(np.unique(rows)), 'empty/duplicate selector')
        meta = task.iloc[rows]
        years = [2023,2024] if role in ['FIT','VAL','Spatial','External-Spatial'] else [2025]
        # Scenario membership is defined by target year. In particular the
        # external contract retains Dec-31 issues predicting Jan-1 targets.
        # The issue timestamp is causal context, not an additional year filter.
        require(meta.target_local_year.isin(years).all(), 'wrong development/evaluation target year')
        require((meta.target_time_local-meta.issue_time_local==pd.Timedelta(hours=24)).all(), 'timestamp horizon mismatch')
        require((meta.target_time_index-meta.issue_time_index == 24).all() and (meta.forecast_horizon_hours==24).all(), 'horizon is t+24, not output length')
        require(meta.target_valid_benchmark.all() and (meta.issue_time_index>=71).all(), 'invalid task membership/history')
        if not external:
            role_key = {'FIT':'train_cities','VAL':'validation_cities','Spatial':'test_cities','Future':'train_cities','Joint':'test_cities'}[role]
            require(set(meta.city_en) == set(self.split[role_key]), 'wrong city roles')
        if limit is not None:
            require(type(limit) is int and 0 < limit <= 65536, 'bounded CPU/pilot sample count')
            rows = rows[:limit]
        return ScopedView(self,role,rows,external,purpose)

    def array(self, name, *, target=False, external=False):
        key = (name,target,external)
        if key not in self.arrays:
            base = (EXTERNAL_TARGET if external else INTERNAL_TARGET) if target else (EXTERNAL_BANK if external else INTERNAL_BANK)
            self.arrays[key] = ZarrV2Array(self.root/base/name)
        return self.arrays[key]

    def warm_features(self, views):
        """Optional runtime-only continuous cache, never expands windows/labels."""
        require(views and all(v.data is self for v in views), 'cache views belong to this fold')
        require(len({v.external for v in views}) == 1 and len({v.purpose for v in views}) == 1, 'cache may not mix evaluation/development')
        external = views[0].external
        meta = pd.concat([v.meta for v in views])
        lo, hi = int(meta.issue_time_index.min()-71), int(meta.issue_time_index.max())
        cities = np.sort(meta.city_index.unique())
        values = np.empty((hi-lo+1,len(cities),72),dtype=np.float32)
        for j,name in enumerate(self.layout['feature_order']):
            raw = self.array(name,external=external).take(np.arange(lo,hi+1)[:,None],cities[None,:])
            values[:,:,j] = self.transform(raw,j)
        self.cache = (lo,hi,{int(c):i for i,c in enumerate(cities)},values,external,views[0].purpose)

    def transform(self, a, j):
        a = np.asarray(a,dtype=np.float32)
        a = np.where(np.isfinite(a),a,np.float32(self.pre['impute_median_train'][j]))
        return (a-np.float32(self.pre['transform_center'][j]))/np.float32(self.pre['transform_scale'][j])

    def record_access(self, role, purpose, rows, meta):
        """Bounded diagnostics; canonical training exposure is in BudgetController.

        Retaining all sampled row IDs would grow to hundreds of millions for
        DLinear. Full order is reproducible from the serialized sampler instead.
        """
        rows=np.asarray(rows,dtype=np.int64)
        self.access_counts[role]=self.access_counts.get(role,0)+len(rows)
        self.access_log.append(dict(role=role,purpose=purpose,fold=self.fold,row_count=len(rows),
            task_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),cities=sorted(set(meta.city_en)),
            target_years=sorted(set(map(int,meta.target_local_year)))))
        if len(self.access_log)>128:
            self.access_log.pop(0)


class ScopedView:
    def __init__(self,data,role,rows,external,purpose):
        self.data,self.role,self.rows,self.external,self.purpose = data,role,rows,external,purpose
        self.meta = data.metadata(external).iloc[rows]
        self.support_hash = fingerprint(rows.tolist())

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, device='cpu', *, pm_only=False):
        ix = np.asarray(indices,dtype=np.int64)
        require(ix.ndim == 1 and len(ix)>0 and np.all((ix>=0)&(ix<len(self))), 'batch index outside allowed view')
        m = self.meta.iloc[ix]
        city = m.city_index.to_numpy(np.int64)
        issue = m.issue_time_index.to_numpy(np.int64)
        target = m.target_time_index.to_numpy(np.int64)
        times = issue[:,None]+np.arange(-71,1)[None,:]
        cache = self.data.cache
        if cache is not None:
            lo,hi,mapping,arr,external,purpose = cache
            require(self.external == external and self.purpose == purpose and times.min()>=lo and times.max()<=hi, 'cache permission/range mismatch')
            require(set(map(int,city)) <= set(mapping), 'cache city mismatch')
            channels = slice(0,1) if pm_only else slice(None)
            x = arr[times-lo,np.asarray([mapping[int(c)] for c in city])[:,None],channels]
        else:
            x = np.stack([self.data.transform(self.data.array(name,external=self.external).take(times,city[:,None]),j)
                          for j,name in enumerate(self.data.layout['feature_order'][:1] if pm_only else self.data.layout['feature_order'])],axis=-1)
        y_raw = self.data.array('pm25_city_mean',target=True,external=self.external).take(target,city).astype(np.float32)
        require(np.isfinite(y_raw).all() and np.isfinite(x).all(), 'nonfinite selected data')
        self.data.record_access(self.role,self.purpose,self.rows[ix],m)
        b = self.data.bridge
        y = (y_raw-np.float32(b['target_mean']))/np.float32(b['target_std'])
        out = {'x':x,'y':y,'y_raw':y_raw}
        if not pm_only:
            local = m.issue_time_local.to_numpy(dtype='datetime64[ns]')
            history = local[:,None]+np.arange(-71,1)[None,:]*np.timedelta64(1,'h')
            target_time = local+np.timedelta64(24,'h')
            out.update(marks=time_marks(history).reshape(-1,72,4),
                       target_marks=time_marks(target_time).reshape(-1,1,4))
        out = {k:torch.from_numpy(np.ascontiguousarray(v)).to(device) for k,v in out.items()}
        out['task_rows'] = self.rows[ix].tolist()
        out['sample_ids'] = m.sample_id.tolist()
        return out


class SyntheticView:
    """Only used by explicit bounded CPU tests, never selected by formal CLI."""
    def __init__(self, x, y, role='FIT'):
        require(role in ['FIT','VAL'], 'synthetic development only')
        self.x,self.y,self.role = x,y,role
        self.support_hash = fingerprint({'synthetic':True,'n':len(x),'role':role})

    def __len__(self):
        return len(self.x)

    def batch(self, indices, device='cpu'):
        i = torch.as_tensor(indices,dtype=torch.long)
        x,y = self.x[i].to(device),self.y[i].to(device)
        return dict(x=x,y=y,y_raw=y,marks=torch.zeros(len(i),72,4,device=device),target_marks=torch.zeros(len(i),1,4,device=device))
