"""Original B conditional means and C coherent month-half donor episodes.

Target/recipient support always stays in the immutable V2 ScopedView.
B fits only issue-center FIT predictors, then changes each history hour using
that hour's own local month/hour. C changes the 39 ERA5 channels by gathering
one continuous donor window; it never rotates individual history timestamps.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from phystok.utils.common import PACKAGE, require, read_json, sha256, fingerprint, atomic_json
from phystok.data.loader import CityData, ScopedView, SCHEMA, INTERNAL_TASK

FAMILIES = {'transport_dispersion', 'moisture_stability_context', 'precipitation_radiation'}


def donor_centers(local_times):
    t = pd.DatetimeIndex(local_times)
    require(t.tz is None and (t.minute == 0).all() and (t.second == 0).all(), 'local hourly axis required')
    day = ((t.day.to_numpy() - 1 + t.days_in_month.to_numpy() // 2)
           % t.days_in_month.to_numpy()) + 1
    return t + pd.to_timedelta(day - t.day.to_numpy(), unit='D')


def schema_groups(data):
    schema = read_json(data.root / f'05_manifests/features/{SCHEMA}/feature_schema.json')
    recs = {r['feature']: r for r in schema['feature_records']}
    era = [j for j,f in enumerate(data.layout['feature_order']) if recs[f]['family'] in FAMILIES]
    require(len(era) == 39 and len(set(era)) == 39, 'exact 39 ERA5 input features required')
    legacy = read_json(PACKAGE / 'experiments/meteorological_interventions/weather_feature_inventory.json')
    require([data.layout['feature_order'][j] for j in era] == legacy['features'], 'legacy ERA5 channel order changed')
    return schema, era


def city_regions(data):
    names = {}
    for role in ['train', 'validation', 'test']:
        for region, cities in data.split['cities_by_region'][role].items():
            for city in cities:
                require(city not in names, 'duplicate city region')
                names[city] = region
    meta = data.metadata().drop_duplicates('city_index')
    require(len(meta) == 50, '50-city region map required')
    return {int(r.city_index): names[r.city_en] for r in meta.itertuples()}


def bank_local_axis(data):
    meta = data.metadata()
    first = meta.iloc[0]
    origin = pd.Timestamp(first.issue_time_local) - pd.Timedelta(hours=int(first.issue_time_index))
    expected = origin + pd.to_timedelta(meta.issue_time_index.to_numpy(), unit='h')
    require(np.array_equal(expected.to_numpy(), meta.issue_time_local.to_numpy(dtype='datetime64[ns]')), 'time-index alignment changed')
    count = data.array(data.layout['feature_order'][0]).shape[0]
    return pd.date_range(origin, periods=count, freq='h')


def prepare_fold(root, fold, output, *, contract=None):
    """Rebuild frozen B FIT-only rules locally; private legacy arrays are not required."""
    output = Path(output)
    require(not output.exists(), 'prepared inputs are immutable; use a new version')
    output.mkdir(parents=True)
    data = CityData(root, fold, contract=contract)
    schema, era = schema_groups(data)
    fit = data.view('FIT')
    meta = fit.meta
    require(meta.target_local_year.isin([2023,2024]).all(), 'B FIT years')
    require(pd.DatetimeIndex(meta.issue_time_local).year.isin([2023,2024]).all(), 'B FIT issue years')
    region_map = city_regions(data)
    regions = sorted(set(region_map.values()))
    fit_city = meta.city_index.to_numpy(np.int64)
    fit_issue = meta.issue_time_index.to_numpy(np.int64)
    local = pd.DatetimeIndex(meta.issue_time_local)
    groups = pd.DataFrame({'region':[region_map[int(c)] for c in fit_city],
                           'month':local.month.to_numpy(), 'hour':local.hour.to_numpy()})
    names = [data.layout['feature_order'][j] for j in era]
    for name in names:
        raw = data.array(name).take(fit_issue, fit_city)
        require(np.isfinite(raw).all(), 'Original B1 requires finite FIT weather: '+name)
        groups[name] = raw
    clim = groups.groupby(['region','month','hour'], observed=True)[names].mean().reset_index()
    require(len(clim) == 5*12*24, 'incomplete B climatology groups')
    lookup = np.empty((5,12,24,39), dtype=np.float32)
    positions = {r:i for i,r in enumerate(regions)}
    for row in clim.itertuples(index=False, name=None):
        region,month,hour,*vals = row
        lookup[positions[region], int(month)-1, int(hour)] = vals
    with (output/'climatology.npy').open('xb') as f: np.save(f, lookup, allow_pickle=False)
    pre = {k:v.copy() for k,v in data.pre.items()}
    recs = {r['feature']:r for r in schema['feature_records']}
    rp = np.asarray([positions[region_map[int(c)]] for c in fit_city])
    for k,j in enumerate(era):
        x = lookup[rp,local.month.to_numpy()-1,local.hour.to_numpy(),k].astype(np.float64)
        median,mean,std = float(np.median(x)),float(np.mean(x)),float(np.std(x,ddof=0))
        standardize = recs[data.layout['feature_order'][j]]['transform_class'] in {
            'continuous_fold_local_standardization','diagnostic_continuous_fold_local_scaling_if_used'}
        pre['impute_median_train'][j] = median
        pre['transform_center'][j] = mean if standardize else 0.
        pre['transform_scale'][j] = (std if std >= 1e-12 else 1.) if standardize else 1.
        values = {'feature_mean_after_impute_train':mean,'feature_std_after_impute_train_raw':std,
                  'feature_observed_min_train':float(x.min()),'feature_observed_max_train':float(x.max()),
                  'feature_finite_count_train':len(x),'feature_missing_count_train':0,
                  'zero_variance_guard':int(standardize and std < 1e-12)}
        for name,v in values.items():
            if name in pre: pre[name][j] = v
    with (output/'preprocess_stats.npz').open('xb') as f:np.savez_compressed(f,**pre)
    spec = dict(status='PASS',fold=fold,arm='B',regions=regions,city_regions=region_map,era_indices=era,
        fit_support_sha256=fit.support_hash,fit_samples=len(fit),fit_cities=sorted(meta.city_en.unique()),
        fit_target_years=sorted(map(int,meta.target_local_year.unique())),base_transform_sha256=data.transform_hash,
        target_stats_unchanged=True,legacy_private_array_comparison_performed=False,
        rule='FIT-only region x local month x local hour mean; each history hour uses own key; float32 bank then B FIT preprocessing',
        artifacts={n:sha256(output/n) for n in ['climatology.npy','preprocess_stats.npz']})
    atomic_json(output/'MANIFEST.json',spec)
    return spec


class InterventionData(CityData):
    def __init__(self, root, fold, arm, *, prepared_root=None, contract=None):
        super().__init__(root,fold,contract=contract)
        require(arm in ['A','B','C'], 'unknown weather arm')
        self.arm = arm
        _,self.era = schema_groups(self)
        self.local_axis = bank_local_axis(self) if arm != 'A' else None
        self.base_transform_hash = self.transform_hash
        if arm == 'B':
            require(prepared_root is not None, 'ABC B requires --prepared-root with fold-local preparation')
            folder = Path(prepared_root) / f'fold_{fold}'
            info = read_json(folder/'MANIFEST.json')
            require(info['status']=='PASS' and info['fold']==fold and info['base_transform_sha256']==self.transform_hash,'B preprocessing identity')
            for rel,h in info['artifacts'].items():require(sha256(folder/rel)==h,'B input artifact changed')
            self.lookup = np.load(folder/'climatology.npy',allow_pickle=False)
            self.pre = dict(np.load(folder/'preprocess_stats.npz',allow_pickle=False))
            self.region_pos = {int(c):info['regions'].index(r) for c,r in info['city_regions'].items()}
            self.transform_hash = fingerprint({'base':self.base_transform_hash,'B_manifest':sha256(folder/'MANIFEST.json')})
            require(float(self.pre['transform_center'][0])==self.bridge['input_center'] and
                    float(self.pre['transform_scale'][0])==self.bridge['input_scale'],'B changed target-history scale')

    def view(self, role, **kwargs):
        require(self.arm=='A' or role in ['FIT','VAL','Spatial'],'ABC is Spatial only')
        original = super().view(role,**kwargs)
        if self.arm == 'C':
            return DonorView(self,role,original.rows,original.external,original.purpose)
        if self.arm == 'B':
            return ClimatologyView(self,role,original.rows,original.external,original.purpose)
        return original

    def warm_features(self, views):
        if self.arm == 'A':return super().warm_features(views)
        require(views and all(v.data is self and not v.external for v in views),'ABC cache view identity')
        require(len({v.purpose for v in views})==1,'cache cannot mix development and test')
        meta = pd.concat([v.meta for v in views])
        issues = meta.issue_time_index.to_numpy(np.int64)
        lo,hi = int(issues.min()-71),int(issues.max())
        if self.arm == 'C':
            donors = self.donor_indices(meta.issue_time_local,issues)
            lo,hi = min(lo,int(donors.min()-71)),max(hi,int(donors.max()))
        cities = np.sort(meta.city_index.unique())
        times = np.arange(lo,hi+1)
        values = np.empty((len(times),len(cities),72),dtype=np.float32)
        local = self.local_axis.take(times)
        for j,name in enumerate(self.layout['feature_order']):
            if self.arm == 'B' and j in self.era:
                raw = self.lookup[np.asarray([self.region_pos[int(c)] for c in cities])[None,:],
                                  (local.month.to_numpy()-1)[:,None],local.hour.to_numpy()[:,None],self.era.index(j)]
            else:
                raw = self.array(name).take(times[:,None],cities[None,:])
            values[:,:,j] = self.transform(raw,j)
        self.cache=(lo,hi,{int(c):i for i,c in enumerate(cities)},values,False,views[0].purpose)

    def donor_indices(self, local, issues):
        local = pd.DatetimeIndex(local)
        require(local.year.isin([2023,2024]).all(),'C formal years only')
        donor = donor_centers(local)
        hours = np.asarray((donor-local)/pd.Timedelta(hours=1),dtype=np.int64)
        index = np.asarray(issues,dtype=np.int64)+hours
        require(np.isin(np.abs(hours)//24,[14,15,16]).all(),'C donor day distance')
        require((index>=71).all() and (index<len(self.local_axis)).all(),'C donor window out of bounds')
        require(np.array_equal(self.local_axis.take(index).to_numpy(),donor.to_numpy()),'C donor time-index mismatch')
        require((donor.year==local.year).all() and (donor.month==local.month).all() and
                (donor.hour==local.hour).all(),'C condition changed')
        return index


class ClimatologyView(ScopedView):
    def batch(self, indices, device='cpu', **kwargs):
        require(self.data.cache is not None, 'B requires its explicitly constructed conditional-mean cache')
        return super().batch(indices,device,**kwargs)


class DonorView(ScopedView):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.donor = self.data.donor_indices(self.meta.issue_time_local,self.meta.issue_time_index)

    def batch(self, indices, device='cpu', *, pm_only=False):
        require(not pm_only,'C requires PhysTok full input view')
        # Use the unchanged canonical label/recipient/timestamp path on CPU.
        out = super().batch(indices,'cpu')
        ix = np.asarray(indices,dtype=np.int64)
        city = self.meta.iloc[ix].city_index.to_numpy(np.int64)
        times = self.donor[ix,None]+np.arange(-71,1)[None,:]
        cache = self.data.cache
        if cache is not None:
            lo,hi,mapping,values,external,purpose = cache
            require(not external and purpose==self.purpose and times.min()>=lo and times.max()<=hi,'C cache permission/range')
            donors = values[times-lo,np.asarray([mapping[int(c)] for c in city])[:,None]][:,:,self.data.era]
        else:
            donors = np.stack([self.data.transform(self.data.array(self.data.layout['feature_order'][j]).take(times,city[:,None]),j)
                               for j in self.data.era],axis=-1)
        out['x'][:,:,self.data.era] = torch.from_numpy(np.ascontiguousarray(donors))
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in out.items()}
