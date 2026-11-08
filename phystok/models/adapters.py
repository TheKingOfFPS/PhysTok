"""Seven explicit scalar t+24 adapters. All return target-standardized [B].

Official source cores are pinned, namespaced, and hashed in vendor_provenance.
No historical training runner is imported. No GPU choice is made here.
"""
from __future__ import annotations
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from phystok.utils.common import require
from .registry import MODEL_NAMES as MODELS


class ScaleBridge(nn.Module):
    def __init__(self, input_center, input_scale, target_mean, target_std):
        super().__init__()
        values = torch.tensor([input_center,input_scale,target_mean,target_std], dtype=torch.float64)
        require(torch.isfinite(values).all().item() and input_scale > 0 and target_std > 0, 'invalid fold scale bridge')
        for k, v in zip(['input_center','input_scale','target_mean','target_std'], values):
            self.register_buffer(k, v.float())

    def from_input_z(self, p):
        return (p * self.input_scale + self.input_center - self.target_mean) / self.target_std

    def to_raw(self, p):
        return p * self.target_std + self.target_mean


class DLinearCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.seasonal = nn.Linear(72, 1)
        self.trend = nn.Linear(72, 1)
        nn.init.constant_(self.seasonal.weight, 1/72)
        nn.init.constant_(self.trend.weight, 1/72)

    def forward(self, x):
        x = x[..., 0]
        trend = F.avg_pool1d(F.pad(x[:, None], (12,12), mode='replicate'), 25, stride=1).squeeze(1)
        return (self.seasonal(x-trend) + self.trend(trend)).squeeze(-1)


class InformerScalar(nn.Module):
    """Pinned encoder-decoder ProbSparse, 36 observed decoder tokens + t+24 query.

No t+1..t+23 trajectory/target is used. Query values are zero. Query timestamp
is t+24, not t+1. Fixed RNG per eval call preserves official sampling law and
prevents consuming training RNG. This direct irregular-query adapter is named
Informer-derived scalar V2, not an unmodified official forecasting task.
"""
    def __init__(self, width, eval_seed):
        super().__init__()
        from .vendor.informer.models.model import Informer
        self.core = Informer(enc_in=72, dec_in=72, c_out=1, seq_len=72, label_len=36, out_len=1,
                             factor=5, d_model=width, n_heads=8, e_layers=2, d_layers=1, d_ff=width*4,
                             dropout=.1, attn='prob', embed='timeF', freq='h', activation='gelu',
                             output_attention=False, distil=True, mix=True, device=torch.device('cpu'))
        self.eval_seed = int(eval_seed)

    def decoder_input(self, batch):
        x = batch['x']
        require(batch['marks'].shape == (len(x),72,4) and batch['target_marks'].shape == (len(x),1,4), 'Informer timestamp marks missing')
        dec = torch.cat([x[:, -36:], torch.zeros_like(x[:, :1])], dim=1)
        marks = torch.cat([batch['marks'][:, -36:], batch['target_marks']], dim=1)
        return dec, marks

    def forward(self, batch):
        dec, marks = self.decoder_input(batch)
        # Official ProbAttention draws sample indices on CPU. In eval all other
        # stochastic modules are disabled; only CPU RNG needs isolating.
        with (torch.random.fork_rng(devices=[]) if not self.training else nullcontext()):
            if not self.training:
                torch.random.default_generator.manual_seed(self.eval_seed)
            return self.core(batch['x'], batch['marks'], dec, marks)[:, 0, 0]


class ScalarAdapter(nn.Module):
    horizon = 24
    output_length = 1
    lookback = 72

    def __init__(self, name, candidate, layout, bridge, eval_seed=20261017, *, capacity_control=False):
        super().__init__()
        require(name in MODELS, 'unknown model')
        require(layout['feature_order'][0] == 'pm25_now' and len(layout['feature_order']) == 72, 'PM channel/order contract')
        self.name = name
        self.bridge = ScaleBridge(**bridge)
        self.register_buffer('trajectory_indices', torch.tensor(layout['trajectory_indices'], dtype=torch.long))
        self.register_buffer('state_indices', torch.tensor(layout['state_indices'], dtype=torch.long))
        self.register_buffer('ms_permutation', torch.tensor(list(range(1,72)) + [0], dtype=torch.long))
        width = candidate['d_model']
        if name == 'phystok':
            require(width == 512 or (capacity_control and width == 256), 'Full PhysTok is frozen at width512; width256 is an explicit capacity control')
            from .vendor.phystok.strong_vtt_model import StrongVTT, StrongVTTSwitches
            self.core = StrongVTT(lookback=72,num_trajectory_features=54,d_model=width,n_heads=width//32,e_layers=4,
                                  d_ff=4*width,dropout=.1,head_hidden=width,activation='gelu',
                                  trajectory_family_ids=layout['trajectory_family_ids'],
                                  state_family_feature_positions=layout['state_family_positions'],switches=StrongVTTSwitches())
        elif name == 'dlinear':
            self.core = DLinearCore()
        elif name == 'informer':
            self.core = InformerScalar(width, eval_seed)
        elif name == 'patchtst':
            from .vendor.patchtst.layers.PatchTST_backbone import PatchTST_backbone
            self.core = PatchTST_backbone(c_in=72,context_window=72,target_window=1,patch_len=12,stride=6,
                                          n_layers=3,d_model=width,n_heads=8,d_ff=4*width,norm='BatchNorm',
                                          attn_dropout=0.,dropout=.1,act='gelu',res_attention=True,pre_norm=False,
                                          pe='zeros',learn_pe=True,head_dropout=0.,padding_patch='end',
                                          individual=False,revin=True,affine=True,subtract_last=False)
        elif name == 'itransformer':
            from .vendor.itransformer.model.iTransformer import Model
            self.core = Model(SimpleNamespace(seq_len=72,pred_len=1,output_attention=False,use_norm=True,
                              embed='timeF',freq='h',dropout=.1,class_strategy='projection',factor=5,
                              d_model=width,n_heads=8,d_ff=2048,e_layers=2,activation='gelu'))
        elif name == 'timexer':
            _require_external('timexer')
            from .vendor.timexer.models.TimeXer import Model
            self.core = Model(SimpleNamespace(task_name='long_term_forecast',features='MS',seq_len=72,pred_len=1,
                              use_norm=True,patch_len=24,enc_in=72,d_model=width,dropout=.1,embed='timeF',freq='h',
                              factor=1,n_heads=8,d_ff=2048,e_layers=2,activation='gelu'))
        else:
            _require_external('xlinear')
            from .vendor.xlinear.models.XLinear import Model
            self.core = Model(SimpleNamespace(seq_len=72,pred_len=1,d_model=width,enc_in=72,t_ff=width//2,
                              c_ff=72,usenorm=1,embed_dropout=0.,head_dropout=0.,t_dropout=0.,c_dropout=0.,features='MS'))

    def forward(self, batch):
        x = batch['x']
        shapes = {(72,1),(72,72)} if self.name == 'dlinear' else {(72,72)}
        require(x.ndim == 3 and tuple(x.shape[1:]) in shapes, 'input shape differs from model native-view contract')
        require(torch.isfinite(x).all().item(), 'input must be FIT-imputed and finite')
        if self.name == 'phystok':
            pred = self.core(x.index_select(2,self.trajectory_indices), x[:, -1].index_select(1,self.state_indices))
        elif self.name == 'informer':
            # A scalar raw-unit readout has no official per-channel denorm.
            # The learned head directly represents benchmark target z units.
            pred = self.core(batch)
        elif self.name == 'dlinear':
            pred = self.bridge.from_input_z(self.core(x))
        elif self.name == 'patchtst':
            pred = self.bridge.from_input_z(self.core(x.permute(0,2,1))[:,0,0])
        elif self.name == 'itransformer':
            pred = self.bridge.from_input_z(self.core(x,None,None,None)[:,0,0])
        elif self.name == 'timexer':
            pred = self.bridge.from_input_z(self.core(x.index_select(2,self.ms_permutation),None,None,None)[:,0,0])
        else:
            pred = self.bridge.from_input_z(self.core(x.index_select(2,self.ms_permutation))[:,0,0])
        require(pred.shape == (len(x),), 'scalar target output shape')
        return pred

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name, candidate, layout, bridge, eval_seed=20261017):
    return ScalarAdapter(name,candidate,layout,bridge,eval_seed)


def _require_external(name):
    module = Path(__file__).parent / "vendor" / name / "models" / ("TimeXer.py" if name == "timexer" else "XLinear.py")
    if not module.is_file():
        raise ImportError(
            f"{name} source is intentionally not redistributed because the pinned upstream license is unconfirmed. "
            "See baselines/README.md and run baselines/install_external.py with an authorized local checkout."
        )
