"""Arithmetic and failure gates using runtime-generated synthetic statistics."""
from pathlib import Path
import math
import unittest

import numpy as np

from phystok.statistics import (pooled, mean_sample_sd, city_cluster_weights,
                                paired_city_bootstrap, paired_seed_summary,
                                verify_release)


class StatisticsTests(unittest.TestCase):
    def test_pool_unequal_city_support_before_root(self):
        # One error 3 in city A; nine errors 1 in city B.
        got=pooled([dict(n=1,sse=9,sae=3),dict(n=9,sse=9,sae=9)])
        self.assertEqual(got['n'],10)
        self.assertAlmostEqual(got['rmse'],math.sqrt(1.8))
        self.assertAlmostEqual(got['mae'],1.2)
        self.assertNotAlmostEqual(got['rmse'],(3+1)/2)

    def test_sample_sd_is_ddof1_over_complete_seeds(self):
        got=mean_sample_sd([1,2,3])
        self.assertEqual(got['mean'],2)
        self.assertEqual(got['sample_sd'],1)
        with self.assertRaises(ValueError):mean_sample_sd([1])

    def test_invalid_error_moments_rejected(self):
        for row in [dict(n=0,sse=0,sae=0),dict(n=1,sse=1,sae=2),dict(n=2.5,sse=4,sae=2),dict(n=1,sse=float('nan'),sae=0)]:
            with self.assertRaises(ValueError):pooled([row])

    def test_stratified_weights_keep_city_counts(self):
        cities=['a','b','c','d'];regions=dict(a='north',b='north',c='south',d='south')
        w=city_cluster_weights(cities,regions,reps=101,seed=7)
        np.testing.assert_array_equal(w[:,:2].sum(axis=1),np.full(101,2))
        np.testing.assert_array_equal(w[:,2:].sum(axis=1),np.full(101,2))
        np.testing.assert_array_equal(w,city_cluster_weights(cities,regions,reps=101,seed=7))

    def city_records(self):
        out=[]
        for model in ['baseline','comparator']:
            for seed in [20260826,20260827,20260828]:
                for city,n,error in [('a',1,3),('b',9,1)]:
                    value=error if model=='baseline' else error+1
                    out.append(dict(model=model,seed=seed,scenario='Spatial',city=city,region='example',n=n,sse=n*value**2,sae=n*value))
        return out

    def test_bootstrap_pools_nonlinear_metric_not_city_rmse(self):
        rows=self.city_records()
        got=paired_city_bootstrap(rows,'baseline','comparator','Spatial',reps=201,seed=9,family_size=12)
        self.assertAlmostEqual(got['metrics']['mae']['ci95_low'],1)
        self.assertAlmostEqual(got['metrics']['mae']['ci95_high'],1)
        # RMSE difference varies with which city was drawn, despite each city
        # having exactly +1 comparator RMSE: this distinguishes pooled risk.
        self.assertLess(got['metrics']['rmse']['ci95_low'],1)
        self.assertAlmostEqual(got['metrics']['rmse']['ci95_high'],1)

    def test_support_mismatch_is_rejected_without_inner_join(self):
        rows=self.city_records();rows[-1]['n']=8
        # Adjust moments so the row itself remains valid; support is the error.
        rows[-1]['sse']=8*4;rows[-1]['sae']=8*2
        with self.assertRaisesRegex(ValueError,'supports differ'):
            paired_city_bootstrap(rows,'baseline','comparator','Spatial',reps=10)
        rows=self.city_records()[:-1]
        with self.assertRaises(ValueError):
            paired_city_bootstrap(rows,'baseline','comparator','Spatial',reps=10)

    def test_duplicate_city_not_counted_twice(self):
        rows=self.city_records();rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            paired_city_bootstrap(rows,'baseline','comparator','Spatial',reps=10)

    def test_external_no_fabricated_internal_regions(self):
        rows=self.city_records()
        for row in rows:
            row['scenario']='External-Joint';row['region']=row['city']
        with self.assertRaisesRegex(ValueError,'fabricated'):
            paired_city_bootstrap(rows,'baseline','comparator','External-Joint',reps=10)

    def test_paired_seed_difference_uses_same_seed(self):
        rows=[]
        for seed,a,b in [(20260826,1,4),(20260827,2,4),(20260828,3,4)]:
            for model,error in [('baseline',a),('comparator',b)]:
                rows.append(dict(model=model,seed=seed,scenario='Spatial',n=1,rmse=error,mae=error))
        got=paired_seed_summary(rows,'baseline','comparator','Spatial')[0]
        self.assertEqual(got['delta_by_seed'],[3,2,1])
        self.assertEqual(got['paired_seed_sample_sd'],1)
        self.assertEqual(got['relative_delta_pct'],50)

    def test_frozen_aggregate_release_arithmetic(self):
        result=verify_release(Path(__file__).resolve().parents[1]/'results')
        self.assertEqual(result['status'],'PASS')
        self.assertFalse(result['source_observations_or_predictions_verified'])
        self.assertGreater(result['checks'],5000)


if __name__=='__main__':unittest.main()
