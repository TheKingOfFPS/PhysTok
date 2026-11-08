"""Checks of retrospective donor rules and the frozen ablation matrix."""
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from phystok.experiments.weather import donor_centers

ROOT=Path(__file__).resolve().parents[1]


class InterventionTests(unittest.TestCase):
    def test_donor_mapping_is_coherent_same_month_bijection(self):
        times=pd.date_range('2023-01-01','2024-12-31 23:00',freq='h')
        donor=donor_centers(times)
        self.assertTrue(donor.is_unique)
        self.assertEqual(set(donor),set(times))
        self.assertTrue(np.array_equal(times.year,donor.year))
        self.assertTrue(np.array_equal(times.month,donor.month))
        self.assertTrue(np.array_equal(times.hour,donor.hour))
        distances=np.abs(np.asarray((donor-times)/pd.Timedelta(hours=24),dtype=int))
        self.assertEqual(set(distances),{14,15,16})
        # Some donors occur after the issue: this is explicitly retrospective.
        self.assertTrue((donor>times).any())
        self.assertTrue((donor<times).any())

    def test_naive_hourly_local_axis_required(self):
        with self.assertRaises(ValueError):donor_centers(pd.date_range('2023-01-01',periods=5,freq='h',tz='UTC'))
        with self.assertRaises(ValueError):donor_centers(pd.DatetimeIndex(['2023-01-01 00:30']))

    def test_b_preparation_reads_only_fit_predictor_cells(self):
        from phystok.experiments.weather import prepare_fold, SCHEMA
        layout=json.loads((ROOT/'configs/data_contract.json').read_text())
        weather=json.loads((ROOT/'experiments/meteorological_interventions/weather_feature_inventory.json').read_text())['features']
        names=layout['feature_order'];era=[i for i,n in enumerate(names) if n in weather]
        # Generated hourly keys span all five regions/months/hours. No research
        # observations or historical preprocessing outputs are used.
        cities=[f'generated_city_{i}' for i in range(50)]
        grouped={role:{f'generated_region_{r}':[] for r in range(5)} for role in ['train','validation','test']}
        for r in range(5):
            grouped['train'][f'generated_region_{r}']=cities[r*8:(r+1)*8]
            grouped['validation'][f'generated_region_{r}']=[cities[40+r]]
            grouped['test'][f'generated_region_{r}']=[cities[45+r]]
        times=[pd.Timestamp(2023,m,15,h) for m in range(1,13) for h in range(24)]
        records=[dict(city_index=c,city_en=cities[c],issue_time_local=t,
                      issue_time_index=int((t-pd.Timestamp('2023-01-01'))/pd.Timedelta(hours=1)),target_local_year=2023)
                 for c in range(50) for t in times]
        meta=pd.DataFrame(records);reads=[]
        class Array:
            def __init__(self,j):self.j=j
            def take(self,t,c):
                reads.extend(map(int,np.asarray(c).ravel()))
                if not (np.asarray(c)<40).all():raise AssertionError('VAL/TEST predictor cell used to fit B')
                return np.asarray(t)*.001+np.asarray(c)*.1+self.j
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);schema=root/f'05_manifests/features/{SCHEMA}/feature_schema.json';schema.parent.mkdir(parents=True)
            schema.write_text(json.dumps({'feature_records':[{'feature':n,'family':'transport_dispersion' if n in weather else 'calendar','transform_class':'continuous_fold_local_standardization'} for n in names]}))
            pre={'impute_median_train':np.zeros(72),'transform_center':np.zeros(72),'transform_scale':np.ones(72),
                 'target_mean':np.asarray(30.),'target_std':np.asarray(20.)}
            data=SimpleNamespace(root=root,layout=layout,split={'cities_by_region':grouped},pre=pre,transform_hash='generated_base_transform')
            data.metadata=lambda:meta
            data.view=lambda role:SimpleNamespace(meta=meta[meta.city_index<40],support_hash='generated_fit_support')
            # The formal view also exposes its sample count.
            class View:
                def __init__(self):self.meta=meta[meta.city_index<40];self.support_hash='generated_fit_support'
                def __len__(self):return len(self.meta)
            data.view=lambda role:View()
            data.array=lambda name:Array(names.index(name))
            with patch('phystok.experiments.weather.CityData',return_value=data):
                result=prepare_fold(root,0,root/'prepared',contract=layout)
            self.assertEqual(set(reads),set(range(40)))
            self.assertEqual(result['fit_samples'],40*12*24)
            self.assertTrue(result['target_stats_unchanged'])
            self.assertFalse(result['legacy_private_array_comparison_performed'])
            lookup=np.load(root/'prepared/climatology.npy',allow_pickle=False)
            self.assertEqual(lookup.shape,(5,12,24,39))
            rebuilt=np.load(root/'prepared/preprocess_stats.npz',allow_pickle=False)
            self.assertEqual(float(rebuilt['target_mean']),30.)
            self.assertEqual(float(rebuilt['target_std']),20.)
            for j in set(range(72))-set(era):
                self.assertEqual(float(rebuilt['transform_center'][j]),0.)
                self.assertEqual(float(rebuilt['transform_scale'][j]),1.)

    def test_exact_frozen_matrix_no_extra_delayed_policies(self):
        cfg=json.loads((ROOT/'configs/ablations.json').read_text());v=cfg['variants']
        self.assertEqual(set(v),{'phystok_small','itransformer_scaled','timexer_scaled','abc_b','abc_c','delayed_xview','strict_late_fusion','plain_state_flatten'})
        total=sum(len(s['seeds'])*len(s['folds']) for s in v.values())
        self.assertEqual(total,140)
        for name in ['abc_b','abc_c']:
            self.assertEqual(v[name]['scenarios'],['Spatial']);self.assertEqual(v[name]['seeds'],[20260826]);self.assertIsNone(v[name]['model_variant'])
        self.assertNotIn('abc_a',v)
        self.assertEqual(cfg['references']['abc_a']['seed'],20260826)
        self.assertEqual(v['delayed_xview']['scientific_control']['layer_policy'],['block_diagonal']*3+['full'])


if __name__=='__main__':unittest.main()
