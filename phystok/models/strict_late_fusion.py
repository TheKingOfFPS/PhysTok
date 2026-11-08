"""Strict A: four isolated shared-weight layers; all 60 outputs enter the head."""
import torch
from torch import nn
from phystok.utils.common import require
from .adapters import build_model

FULL_PARAMETERS = 27403141
A_PARAMETERS = 27409183
HEAD_HIDDEN = 461


class LayerMaskedEncoder(nn.Module):
    def __init__(self, original, policy='strict'):
        super().__init__()
        require(isinstance(original, nn.TransformerEncoder) and len(original.layers)==4, 'original four-layer encoder required')
        require(policy in ('strict','full'), 'only diagnostic all-open or formal strict policy')
        self.layers=original.layers; self.norm=original.norm; self.num_layers=original.num_layers; self.policy=policy
        view=torch.arange(60)>=54
        self.register_buffer('_blocked_edges',view[:,None]!=view[None,:],persistent=False)
        for layer in self.layers:
            require(layer.self_attn.batch_first and not layer.norm_first,'original batch-first/post-LN required')

    def mask_for(self,index):
        require(index in range(4),'layer outside four original layers')
        return self._blocked_edges if self.policy=='strict' else None

    def forward(self,src,mask=None,src_key_padding_mask=None,is_causal=None):
        require(mask is None and src_key_padding_mask is None and is_causal in (None,False),'no caller/padding/causal mask')
        require(src.ndim==3 and src.shape[1:]==(60,512),'60x512 required')
        out=src
        for i,layer in enumerate(self.layers):out=layer(out,src_mask=self.mask_for(i),is_causal=False)
        if self.norm is not None:out=self.norm(out)
        return out


class LateFusionCore(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.lookback=original.lookback;self.num_trajectory_features=original.num_trajectory_features
        self.d_model=original.d_model;self.switches=original.switches
        self.register_buffer('trajectory_family_ids',original.trajectory_family_ids)
        self.register_buffer('state_family_ids',original.state_family_ids)
        self.trajectory_tokenizer=original.trajectory_tokenizer
        self.identity=original.identity
        self.state_tokenizer=original.state_tokenizer
        self.transformer=LayerMaskedEncoder(original.transformer)
        # Original constructor (including its head) ran first. All common tensors
        # keep identical CPU-seed initialization; only this useful head is new.
        self.projection=nn.Sequential(nn.Linear(60*512,HEAD_HIDDEN),nn.GELU(),
                                      nn.Dropout(.1),nn.Linear(HEAD_HIDDEN,1))

    def forward(self,x_seq,x_state=None,*,return_aux=False):
        require(x_seq.ndim==3 and x_seq.shape[1:]==(72,54),'history shape must be 72x54')
        require(x_state is not None and x_state.shape==(len(x_seq),68),'all 68 states required')
        if return_aux:
            traj,scales=self.trajectory_tokenizer(x_seq,return_scale_weights=True)
        else:
            traj=self.trajectory_tokenizer(x_seq,return_scale_weights=False);scales=None
        traj=self.identity.add_to_trajectory(traj,variable_family_ids=self.trajectory_family_ids)
        state=self.state_tokenizer(x_state)
        state=self.identity.add_to_state(state,state_family_ids=self.state_family_ids)
        tokens=torch.cat([traj,state],dim=1)
        encoded=self.transformer(tokens)
        # First cross-view fusion: the dense head sees concatenated final rows.
        pred=self.projection(encoded.reshape(len(encoded),60*512)).squeeze(-1)
        if return_aux:
            return pred,dict(scale_weights=scales,trajectory_tokens=traj,state_tokens=state,
                             encoded_token_count=60,encoded_history=encoded[:,:54],encoded_state=encoded[:,54:])
        return pred


def control_model(candidate,layout,bridge,eval_seed=20261017,policy='strict'):
    model=build_model('phystok',candidate,layout,bridge,eval_seed)
    model.core=LateFusionCore(model.core)
    model.core.transformer.policy=policy
    require(model.parameter_count()==A_PARAMETERS,'A parameter calculation differs')
    return model
