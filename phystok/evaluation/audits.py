"""Audits for users' locally rebuilt supports, prediction files and fold metadata."""
from itertools import zip_longest
import math
import numpy as np
import pyarrow.parquet as pq
from phystok.utils.common import require
from .metrics import pooled


def paired_predictions(left,right):
    """Require exact exposure IDs, physical sample IDs and truths; never delete rows.

    task_row and sample_id are bound to the local task-index manifest by the
    evaluation writer. The writer separately verifies selector membership.
    This function checks correspondence, not the correctness of source labels.
    """
    keys=['benchmark_id','seed','fold','scenario','task_row','sample_id','y_true']
    a=pq.ParquetFile(left);b=pq.ParquetFile(right)
    require(a.metadata.num_rows==b.metadata.num_rows,'different sample support sizes')
    total=0
    for x,y in zip_longest(a.iter_batches(batch_size=65536,columns=keys),b.iter_batches(batch_size=65536,columns=keys)):
        require(x is not None and y is not None,'different prediction supports')
        require(x.equals(y),'predictions do not pair exactly in frozen selector order')
        total+=len(x)
    require(total>0,'empty predictions')
    return {'status':'PASS','paired_exposures':total,'keys':keys,'estimand':'matched checkpoint exposures; repeated records are not independent cities'}


def audit_city_folds(folds,regions):
    require(len(folds)==10,'ten folds required')
    counts={c:dict(FIT=0,VAL=0,TEST=0) for c in regions}
    require(len(regions)==50 and len(set(regions.values()))==5,'50 cities in five regions')
    for i,fold in enumerate(folds):
        require(fold['fold']==i,'fold order/identity')
        roles=[set(fold[k]) for k in ['train_cities','validation_cities','test_cities']]
        require([len(x) for x in roles]==[40,5,5] and len(set.union(*roles))==50,'invalid/disjoint city roles')
        for role,cities,number in zip(['FIT','VAL','TEST'],roles,[8,1,1]):
            require(cities<=set(regions),'unknown city')
            require(all(sum(regions[c]==r for c in cities)==number for r in set(regions.values())),'unbalanced regional fold')
            for c in cities:counts[c][role]+=1
    require(all(v==dict(FIT=8,VAL=1,TEST=1) for v in counts.values()),'rotation multiplicity mismatch')
    return {'status':'PASS','cities':50,'folds':10,'rotation':'8 FIT / 1 VAL / 1 TEST'}


def audit_timestamps(issue,target,latest_predictor,*,internal=True):
    import pandas as pd
    i,t,p=map(pd.DatetimeIndex,[issue,target,latest_predictor])
    require(len(i)==len(t)==len(p)>0,'timestamp support mismatch')
    require(np.all(p<=i),'future predictor timestamp')
    require(np.all(t-i==pd.Timedelta(hours=24)),'not scalar horizon24')
    if internal:require(np.all(i.year==t.year),'internal issue/target cross-year leakage')
    return {'status':'PASS','targets':len(i),'latest_predictor_at_or_before_issue':True}


def checkpoint_exposures(records):
    groups={}
    for r in records:
        key=(r['model'],r['seed'],r['fold'])
        groups.setdefault(key,[]).append(r)
    for key,rs in groups.items():
        require(len({r['checkpoint_sha256'] for r in rs})==1,'checkpoint changed across populations')
        require(len({r['transform_sha256'] for r in rs})==1,'preprocessing changed across populations')
        require(len({r['scenario'] for r in rs})==len(rs),'duplicate scenario exposure')
    return {'status':'PASS','model_seed_folds':len(groups)}
