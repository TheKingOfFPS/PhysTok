from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import math
import os
import random
import time
import platform
import uuid
import numpy as np
import torch
from .budget import BudgetController
from .checkpoint import PermutationSampler, rng_state, restore_rng, save_checkpoint, load_checkpoint
from phystok.utils.common import require, atomic_json, read_json, sha256, safe_output, fingerprint, now, canonical_gpu_uuid
from phystok.data import SyntheticView, ScopedView


class TrainingPaused(RuntimeError):
    """A complete checkpoint exists; no success marker is published."""


def seed_all(seed,device):
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    if str(device).startswith('cuda'):
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def set_precision(p,device):
    torch.set_float32_matmul_precision(p['matmul_precision'])
    torch.use_deterministic_algorithms(p['deterministic'])
    torch.backends.cudnn.benchmark=p['cudnn_benchmark']
    torch.backends.cudnn.deterministic=p['deterministic']
    torch.backends.cuda.matmul.allow_tf32=p['tf32']
    torch.backends.cudnn.allow_tf32=p['tf32']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=p['fp16_reduced_precision_reduction']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=p['bf16_reduced_precision_reduction']
    if str(device).startswith('cuda'):
        require(not torch.cuda.is_initialized() or os.environ.get('CUBLAS_WORKSPACE_CONFIG')==p['cublas_workspace_config'],
                'CUDA was initialized without frozen deterministic workspace')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG',p['cublas_workspace_config'])==p['cublas_workspace_config'],'CUDA workspace environment mismatch')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=p['cublas_workspace_config']
        torch.cuda.set_device(device)


def autocast(precision,device):
    if str(device).startswith('cuda') and precision['dtype'] != 'float32':
        return torch.autocast('cuda',dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}[precision['dtype']])
    return nullcontext()


def runtime_contract(device):
    """Exact resume stays within the recorded numerical runtime/device."""
    result=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                cuda_build=torch.version.cuda,device=str(device),cpu_threads=torch.get_num_threads())
    if str(device).startswith('cuda'):
        props=torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name,gpu_uuid=canonical_gpu_uuid(getattr(props,'uuid','')),
                      compute_capability=list(torch.cuda.get_device_capability(device)),cudnn=torch.backends.cudnn.version())
    return result


def model_batch(model,view,indices,device):
    """Skip unused input construction for the unchanged univariate DLinear.

    All other models use the original full-channel path. Selection, transforms,
    labels, sampler order and access accounting remain ScopedView-owned.
    """
    if getattr(model,'name',None)=='dlinear' and isinstance(view,ScopedView):
        return view.batch(indices,device,pm_only=True)
    return view.batch(indices,device)


def evaluate_validation(model,view,batch_size,precision,device):
    require(view.role=='VAL','evaluation criterion must be legal VAL')
    model.eval(); sse=0.; sae=0.; n=0
    start=time.perf_counter()
    with torch.inference_mode():
        for st in range(0,len(view),batch_size):
            b=model_batch(model,view,np.arange(st,min(len(view),st+batch_size)),device)
            with autocast(precision,device):
                pred=model(b)
            raw=model.bridge.to_raw(pred.float())
            e=(raw-b['y_raw']).double()
            require(torch.isfinite(e).all().item(),'nonfinite VAL predictions')
            sse+=e.square().sum().item();sae+=e.abs().sum().item();n+=len(e)
    require(n==len(view),'validation dropped samples')
    return dict(rmse=math.sqrt(sse/n),mae=sae/n,n=n,sse=sse,sae=sae,seconds=time.perf_counter()-start)


def run_fit(model,fit,val,*,identity,model_config,candidate,precision,output,device,
            approval_file=None,config=None,source_digest=None,data_digest=None,
            resume=None,cpu_test=False,interrupt_after=None,pilot=None,qualification=None):
    """Frozen V2 optimization and checkpoint logic, without lab server authorization.

    Bounded CPU tests require SyntheticView. Formal callers provide the frozen
    config and a new output directory; no private approval files are required.
    """
    require(qualification is None, "qualification probes are not a formal reproduction mode")
    entered=time.perf_counter()
    output=safe_output(output)
    require(fit.role=='FIT' and val.role=='VAL','FIT and VAL views required')
    require(not (pilot is not None and qualification is not None),'pilot and scientific qualification are distinct scopes')
    if qualification is not None:
        require(identity['phase']=='qualification' and identity.get('plan_sha256')==qualification['plan_sha256'],
                'qualification identity must bind the exact plan')
        require(type(qualification['max_updates']) is int and qualification['max_updates']>0 and
                math.isfinite(qualification['max_seconds']) and qualification['max_seconds']>0,'invalid qualification caps')
    if cpu_test:
        require(str(device)=='cpu' and isinstance(fit,SyntheticView) and isinstance(val,SyntheticView)
                and len(fit)<=1024 and len(val)<=1024 and model_config['budget']['max_updates']<=32
                and identity['benchmark_id'].startswith('test_v2_'),'bounded synthetic CPU test only')
    output.mkdir(parents=True,exist_ok=True)
    require(identity['phase'] in ['hpo','final','pilot','qualification'],'invalid fit phase')
    require((identity['phase']=='qualification')==(qualification is not None),'qualification phase/scope mismatch')
    require(not (output/'SUCCESS.json').exists(),'completed attempt immutable')
    if not (output/'identity.json').exists():
        atomic_json(output/'identity.json',identity)
    else:
        require(read_json(output/'identity.json')==identity,'output identity mismatch')
        require(resume is not None,'existing attempt requires explicit complete resume or new attempt')
    set_precision(precision,device);seed_all(identity['seed'],device)
    runtime=runtime_contract(device)
    model=model.to(device)
    opt=getattr(torch.optim,model_config['optimizer'])(model.parameters(),lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    policy=model_config['budget']
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=policy['lr_factor'],
                patience=policy['lr_patience_checks'],threshold=policy['relative_min_delta'],threshold_mode='rel',min_lr=policy['min_lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=str(device).startswith('cuda') and precision['dtype']=='float16')
    sampler=PermutationSampler(len(fit),identity['seed'])
    budget=BudgetController(policy,len(fit));best_rmse=math.inf;best_ref=None;last_resume=None
    updated=[0]
    hook=opt.register_step_post_hook(lambda *_:updated.__setitem__(0,updated[0]+1))
    if resume:
        state=load_checkpoint(resume['path'],identity,resume['sha256'])
        require(state['kind']=='resume' and state['precision']==precision and state['candidate']==candidate
                and state['model_config']==model_config and state['device']==str(device),'incomplete/incompatible resume; new attempt required')
        require(state.get('qualification')==qualification,'resume qualification scope changed')
        require(state['fit_support']==fit.support_hash and state['val_support']==val.support_hash,'resume data order/support changed')
        require(state.get('runtime')==runtime,'resume numerical runtime/device changed; new attempt required')
        expected_events={f'validation-{i:05d}.json' for i in range(1,len(state['budget']['history'])+1)}
        require({p.name for p in output.glob('validation-*.json')}==expected_events,
                'resume validation history differs from persisted events; choose the latest complete snapshot or a new attempt')
        model.load_state_dict(state['model_state'],strict=True);opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        sampler.load_state_dict(state['sampler']);budget=BudgetController.from_state(state['budget'])
        best_rmse=state['best_val_rmse'];best_ref=state['best_ref']
        if best_ref:
            require(sha256(output/best_ref['path'])==best_ref['sha256'],'resume best checkpoint missing')
        restore_rng(state['rng'],device)
    started=time.perf_counter();step_seconds=[];val_seconds=[];last_heartbeat=0.
    elapsed_before=state.get('elapsed_seconds_total',0.) if resume else 0.
    control=os.environ.get('PHYSTOK_CONTROL_PATH')
    if control:control=Path(control)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot():
        return save_checkpoint(output,'resume',dict(format_version=2,kind='resume',identity=identity,model_state=model.state_dict(),
              optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device),sampler=sampler.state_dict(),
              budget=budget.state_dict(),counters=asdict(budget.count),best_val_rmse=best_rmse,best_ref=best_ref,precision=precision,
              candidate=candidate,model_config=model_config,device=str(device),runtime=runtime,fit_support=fit.support_hash,val_support=val.support_hash,
              qualification=qualification,
              elapsed_seconds_total=elapsed_before+time.perf_counter()-started))

    def heartbeat(status='RUNNING'):
        atomic_json(output/'HEARTBEAT.json',dict(status=status,identity=identity,at=now(),pid=os.getpid(),runtime=runtime,
            counters=asdict(budget.count),passes=budget.passes,elapsed_seconds_total=elapsed_before+time.perf_counter()-started,
            best_val_rmse=None if not math.isfinite(best_rmse) else best_rmse,last_resume=last_resume),replace=True)

    try:
        while budget.stop_reason is None:
            if qualification is not None:
                if time.perf_counter()-entered>=qualification['max_seconds']:
                    budget.stop_reason='QUALIFICATION_WALL_LIMIT';break
                if budget.count.attempted_updates>=qualification['max_updates']:
                    budget.stop_reason='QUALIFICATION_BOUND';break
            if control and control.exists() and read_json(control)['mode']=='PAUSE':
                last_resume=snapshot();heartbeat('PAUSED')
                atomic_json(output/f'PAUSED-{uuid.uuid4().hex}.json',dict(status='PAUSED',identity=identity,resume=last_resume,at=now()))
                raise TrainingPaused('COOPERATIVE_PAUSE: complete snapshot retained; explicit same-device resume review required')
            if time.perf_counter()-last_heartbeat>=30:
                heartbeat();last_heartbeat=time.perf_counter()
            if pilot and (budget.count.attempted_updates>=pilot['updates'] or time.perf_counter()-started>=pilot['max_seconds']):
                budget.stop_reason='PILOT_BOUND';break
            model.train();st=time.perf_counter()
            remaining=max(1,math.ceil(policy['max_passes']*len(fit))-budget.count.presentations)
            idx=sampler.next(min(model_config['batch_size'],remaining))
            batch=model_batch(model,fit,idx,device);opt.zero_grad(set_to_none=True)
            with autocast(precision,device):
                pred=model(batch);loss=torch.nn.functional.mse_loss(pred,batch['y'])
            if not torch.isfinite(loss).item():
                budget.count.record(len(idx),updated=False,nonfinite=True)
                raise RuntimeError('NONFINITE_LOSS; no quality-based retry')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),model_config['gradient_clip'])
            if not scaler.is_enabled() and not torch.isfinite(gradnorm).item():
                budget.count.record(len(idx),updated=False,nonfinite=True)
                raise RuntimeError('NONFINITE_GRADIENT')
            before=updated[0];scaler.step(opt);scaler.update()
            budget.count.record(len(idx),updated=updated[0]>before)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter()-st)
            require(budget.failure_reason() is None,str(budget.failure_reason()))
            due = budget.needs_validation() if not pilot else budget.count.attempted_updates % pilot['validation_every_updates']==0
            if due:
                vm=evaluate_validation(model,val,model_config['eval_batch_size'],precision,device)
                val_seconds.append(vm['seconds']);lr_before=opt.param_groups[0]['lr'];scheduler.step(vm['rmse'])
                event=budget.validation(vm['rmse'],lr_before=lr_before,lr_after=opt.param_groups[0]['lr'])
                event['validation']=vm;event['sampler_epochs_started']=sampler.epochs
                if vm['rmse']<best_rmse:
                    best_rmse=vm['rmse']
                    best_ref=save_checkpoint(output,'best',dict(format_version=2,kind='best',identity=identity,model_state=model.state_dict(),
                        counters=asdict(budget.count),val_rmse=best_rmse,precision=precision,candidate=candidate,
                        fit_support=fit.support_hash,val_support=val.support_hash))
                atomic_json(output/f'validation-{len(budget.history):05d}.json',event)
                last_resume=snapshot()
            if interrupt_after is not None and budget.count.attempted_updates>=interrupt_after:
                require(cpu_test or pilot is not None,'controlled interruption only for CPU tests / approved pilot')
                last_resume=snapshot()
                return dict(status='INTERRUPTED_TEST' if cpu_test else 'INTERRUPTED_PILOT',resume={'path':str(output/last_resume['path']),'sha256':last_resume['sha256']},
                            counters=asdict(budget.count),step_seconds=step_seconds,validation_seconds=val_seconds,elapsed_seconds=time.perf_counter()-started)
        if best_ref is None:
            vm=evaluate_validation(model,val,model_config['eval_batch_size'],precision,device)
            best_rmse=vm['rmse'];val_seconds.append(vm['seconds'])
            best_ref=save_checkpoint(output,'best',dict(format_version=2,kind='best',identity=identity,model_state=model.state_dict(),
                counters=asdict(budget.count),val_rmse=best_rmse,precision=precision,candidate=candidate,
                fit_support=fit.support_hash,val_support=val.support_hash))
        last_resume=snapshot()
        if qualification is not None and time.perf_counter()-entered>=qualification['max_seconds']:
            budget.stop_reason='QUALIFICATION_WALL_LIMIT'
            last_resume=snapshot()
        result=dict(status='SUCCEEDED',identity=identity,best_checkpoint=best_ref,resume_checkpoint=last_resume,best_val_rmse=best_rmse,
                    stop_reason=budget.stop_reason,counters=asdict(budget.count),passes=budget.passes,
                    best_counters=load_checkpoint(output/best_ref['path'],identity,best_ref['sha256'])['counters'],
                    history=budget.history,extensions=budget.extensions,step_seconds=step_seconds,validation_seconds=val_seconds,
                    elapsed_seconds=time.perf_counter()-started,elapsed_seconds_total=elapsed_before+time.perf_counter()-started,runtime=runtime,precision=precision,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
                    pilot=pilot is not None,qualification=qualification is not None,
                    artifacts={best_ref['path']:best_ref['sha256'],last_resume['path']:last_resume['sha256'],
                                                     'identity.json':sha256(output/'identity.json')})
        atomic_json(output/'SUCCESS.json',result)
        heartbeat('SUCCEEDED')
        return result
    except Exception as e:
        atomic_json(output/f'FAILURE-{budget.count.attempted_updates:09d}-{uuid.uuid4().hex}.json',dict(status='FAILED',identity=identity,
                    error_type=type(e).__name__,message=str(e),counters=asdict(budget.count),at=now()))
        raise
    finally:
        hook.remove()
