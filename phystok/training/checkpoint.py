from __future__ import annotations
import io
import random
import uuid
import numpy as np
import torch
from phystok.utils.common import atomic_bytes, sha256, assert_identity, require


class PermutationSampler:
    """CPU-owned permutation state, tail batch retained; deterministic resume."""
    def __init__(self,n,seed):
        require(n>0, 'empty sampler')
        self.n,self.position,self.epochs = int(n),0,0
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.order = torch.empty(0,dtype=torch.long)

    def next(self,batch):
        require(batch>0, 'batch positive')
        if self.position >= len(self.order):
            self.order = torch.randperm(self.n,generator=self.generator)
            self.position = 0
            self.epochs += 1
        ix = self.order[self.position:self.position+batch].clone()
        self.position += len(ix)
        return ix.numpy()

    def state_dict(self):
        return dict(n=self.n,position=self.position,epochs=self.epochs,order=self.order.clone(),generator=self.generator.get_state())

    def load_state_dict(self,s):
        require(s['n']==self.n and 0<=s['position']<=len(s['order']), 'sampler identity/range')
        require(len(s['order']) in [0,self.n], 'sampler permutation length')
        if len(s['order']):
            require(torch.equal(s['order'].sort().values,torch.arange(self.n)), 'invalid sampler permutation')
        self.position,self.epochs,self.order = s['position'],s['epochs'],s['order'].clone()
        self.generator.set_state(s['generator'])


def rng_state(device):
    n = np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None)


def restore_rng(s,device):
    random.setstate(s['python'])
    n=s['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu())
    if str(device).startswith('cuda'):
        require(s['cuda'] is not None, 'missing CUDA RNG; new attempt required')
        torch.cuda.set_rng_state(s['cuda'],device)
    else:
        require(s['cuda'] is None, 'device changed; exact resume forbidden')


def save_checkpoint(folder,kind,payload):
    require(kind in ['best','resume'], 'checkpoint kind')
    path = folder / f'{kind}-{payload["counters"]["attempted_updates"]:09d}-{uuid.uuid4().hex[:12]}.pt'
    buffer=io.BytesIO()
    torch.save(payload,buffer)
    atomic_bytes(path,buffer.getvalue())
    return {'path':path.name,'sha256':sha256(path)}


def load_checkpoint(path,expected_identity,expected_sha=None,*,metadata_only=False):
    if expected_sha:
        require(sha256(path)==expected_sha,'checkpoint checksum mismatch')
    if metadata_only:
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            state=torch.load(path,map_location='cpu',weights_only=True)
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
    assert_identity(state['identity'],expected_identity)
    require(state['format_version']==2,'legacy checkpoint forbidden')
    return state
