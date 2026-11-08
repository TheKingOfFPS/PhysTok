from __future__ import annotations
import math
from phystok.utils.common import MODELS, SEEDS, SCENARIOS, PACKAGE, require, read_json, fingerprint, slug

BUDGET_FIELDS = {
    'min_updates','min_passes','initial_updates','initial_passes','max_updates','max_passes',
    'val_every_updates','val_every_passes','patience_checks','lr_patience_checks','lr_factor','min_lr',
    'required_lr_reductions','recovery_checks','relative_min_delta','recent_improvement_checks',
    'extension_factor','max_skipped_updates','max_nonfinite_batches',
}


def validate_budget(b):
    require(set(b) == BUDGET_FIELDS, f'budget fields must be explicit: missing={BUDGET_FIELDS-set(b)}, extra={set(b)-BUDGET_FIELDS}')
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in b.values()), 'finite nonnegative budget numbers required')
    for unit in ['updates','passes']:
        require(0 < b[f'min_{unit}'] <= b[f'initial_{unit}'] <= b[f'max_{unit}'], f'invalid {unit} budget order')
    for k in ['min_updates','initial_updates','max_updates','val_every_updates','patience_checks','lr_patience_checks',
              'required_lr_reductions','recovery_checks','recent_improvement_checks','max_skipped_updates','max_nonfinite_batches']:
        require(type(b[k]) is int, f'{k} must be integer')
    require(b['val_every_updates'] > 0 and b['val_every_passes'] > 0, 'validation frequency positive')
    require(0 < b['lr_factor'] < 1 and b['extension_factor'] > 1 and b['min_lr'] > 0, 'invalid LR/extension')
    return b


def validate_config(c):
    frozen = c.get('protocol_mode') == 'frozen_recipe'
    extra = {'protocol_mode','recipe_policy'} if frozen else set()
    require(set(c) == {'schema_version','benchmark_id','task','hpo_seed','final_seeds','folds','models','precision','approval','pilot','failure_policy'} | extra, 'unknown/missing top-level config fields')
    if frozen:
        r=c['recipe_policy']
        require(r == dict(development_fold=0,nested_selection_independence=False,historical_test_access_disclosed=True,
            candidate_selection_after_freeze=False,phase_order=['baselines_180','phystok_30','frozen_evaluation'],
            qualification_status='PENDING_REAL_FIT_VAL_REVIEW'), 'frozen recipe must disclose non-nested development; qualification resides in immutable evidence')
    require(c['schema_version'] == 2, 'schema version')
    slug(c['benchmark_id'])
    require(c['benchmark_id'].startswith('phystok_v2_'), 'old benchmark ID forbidden')
    require(c['final_seeds'] == list(SEEDS) and c['folds'] == list(range(10)), 'exact three seeds and ten folds required')
    require(c['hpo_seed'] not in SEEDS and type(c['hpo_seed']) is int, 'separate HPO seed required')
    require(c['task'] == dict(lookback=72, horizon=24, output_length=1, train_years=[2023,2024], evaluation_year=2025,
                              roles={'FIT':40,'VAL':5,'TEST':5}, scenarios=list(SCENARIOS), selection='Fold0_developed_frozen_recipe_non_nested' if frozen else 'per_outer_fold_VAL_RMSE',
                              information_policy='native_inputs_disclosed_not_schema_causal_control'), 'research contract mismatch')
    require(set(c['models']) == set(MODELS), 'all seven models required')
    for model, m in c['models'].items():
        require(set(m) == {'adapter_version','batch_size','eval_batch_size','optimizer','gradient_clip','candidates','budget'}, 'explicit model fields required')
        require(m['optimizer'] in ('Adam','AdamW') and m['batch_size'] > 0 and m['eval_batch_size'] > 0, 'optimizer/batch')
        require(m['gradient_clip'] > 0, 'explicit positive clip')
        count=1 if frozen else 4
        require(len(m['candidates']) == count and len({fingerprint(x) for x in m['candidates']}) == count, 'frozen recipe requires one candidate; legacy HPO requires four')
        for cand in m['candidates']:
            require(set(cand) == {'learning_rate','weight_decay','d_model'}, 'candidate fields')
            require(cand['learning_rate'] > 0 and cand['weight_decay'] >= 0, 'candidate optimizer values')
            require(cand['d_model'] > 0 and cand['d_model'] % 8 == 0, 'model width')
            if model == 'phystok':
                require(cand['d_model'] == 512, 'PhysTok architecture must not change')
        validate_budget(m['budget'])
    require(set(c['precision']) == {'dtype','tf32','matmul_precision','deterministic','cudnn_benchmark','eval_rng_seed','cublas_workspace_config','fp16_reduced_precision_reduction','bf16_reduced_precision_reduction'}, 'explicit precision policy')
    require(c['precision']['cublas_workspace_config'] in [':4096:8',':16:8'],'explicit deterministic CUDA workspace')
    require(c['precision']['dtype'] in ['float32','float16','bfloat16'], 'precision dtype')
    require(c['precision']['matmul_precision'] in ['highest','high','medium'], 'matmul precision')
    require(c['approval']['status'] in ['NOT_APPROVED','REVIEWED'], 'approval status')
    require(c['failure_policy'] == {'retry_test_quality':False,'automatic_retry':False,'new_attempt_on_incomplete_resume':True}, 'failure policy')
    require(set(c['pilot']) == {'fold','seed','fit_samples','val_samples','updates','validation_every_updates','max_seconds'}, 'pilot must be bounded and explicit')
    require(c['pilot']['fold'] in range(10) and 0 < c['pilot']['updates'] <= 1000 and 0 < c['pilot']['max_seconds'] <= 3600, 'pilot bounds')
    require(all(type(c['pilot'][k]) is int and c['pilot'][k]>0 for k in ['fit_samples','val_samples','updates','validation_every_updates','seed']), 'pilot integer fields')
    require(max(c['pilot']['fit_samples'],c['pilot']['val_samples'])<=65536 and
            c['pilot']['updates']>=2 and c['pilot']['updates']%c['pilot']['validation_every_updates']==0,
            'bounded pilot samples and complete validation interval required')
    return c


def load_config(path=None):
    return validate_config(read_json(path or PACKAGE / 'configs/benchmark_v2.json'))
