"""Preserve the frozen encoder's layers; vary only per-layer attention masks.

Bool True means a forbidden query/key edge for TransformerEncoderLayer/MHA.
No new trainable tensor, initialization, branch, residual or normalization order.
"""
import torch
from torch import nn
from phystok.utils.common import require
from .adapters import build_model


class LayerMaskedEncoder(nn.Module):
    def __init__(self, original, policy):
        super().__init__()
        require(isinstance(original, nn.TransformerEncoder), 'requires frozen TransformerEncoder')
        require(policy in ('full', 'delayed'), 'unknown fixed interaction policy')
        require(len(original.layers) == 4, 'exactly four original layers required')
        self.layers = original.layers
        self.norm = original.norm
        self.num_layers = original.num_layers
        self.policy = policy
        view = torch.arange(60) >= 54
        self.register_buffer('_blocked_edges', view[:, None] != view[None, :], persistent=False)
        for layer in self.layers:
            require(layer.self_attn.batch_first and not layer.norm_first,
                    'frozen batch-first, post-normalization layer required')

    def mask_for(self, index):
        return self._blocked_edges if self.policy == 'delayed' and index < 3 else None

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        require(mask is None and src_key_padding_mask is None and is_causal in (None, False),
                'this task has no padding, temporal-causal encoder mask, or caller mask')
        require(src.ndim == 3 and src.shape[1:] == (60, 512), '60 x 512 token contract')
        out = src
        for index, layer in enumerate(self.layers):
            out = layer(out, src_mask=self.mask_for(index),
                        src_key_padding_mask=None, is_causal=False)
        if self.norm is not None:
            out = self.norm(out)
        return out


def control_model(candidate, layout, bridge, eval_seed=20261017, policy='delayed'):
    # Original constructor runs first, with the original RNG consumption/order.
    model = build_model('phystok', candidate, layout, bridge, eval_seed)
    model.core.transformer = LayerMaskedEncoder(model.core.transformer, policy)
    require(model.parameter_count() == 27403141, 'deployed parameter count must match Full')
    return model
