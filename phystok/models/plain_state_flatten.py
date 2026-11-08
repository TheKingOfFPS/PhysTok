"""Full joint encoder/head with six full-vector, non-domain state slots."""
import torch
from torch import nn
from phystok.utils.common import require
from .adapters import build_model

FULL_PARAMETERS = 27403141
PLAIN_PARAMETERS = 27449733
GENERIC_HIDDEN = 128
HEAD_HIDDEN = 512


class GenericStateTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(68,128),nn.GELU(),nn.Linear(128,512),nn.LayerNorm(512))
            for _ in range(6)])

    def forward(self, state):
        require(state.ndim == 2 and state.shape[1] == 68, 'complete ordered68 state vector required')
        # No family positions, index_select, grouping or fixed feature partition.
        return torch.stack([projector(state) for projector in self.projectors], dim=1)


class PlainStateCore(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.lookback = original.lookback
        self.num_trajectory_features = original.num_trajectory_features
        self.d_model = original.d_model
        self.register_buffer('trajectory_family_ids', original.trajectory_family_ids)
        self.trajectory_tokenizer = original.trajectory_tokenizer
        self.identity = original.identity
        self.transformer = original.transformer
        self.projection = original.projection
        # Original Full is constructed first, retaining ALL common initialization.
        # Original family state projector modules/buffers are not retained.
        self.generic_state_tokenizer = GenericStateTokenizer()
        self.generic_slot_identity = nn.Embedding(6,512)
        nn.init.normal_(self.generic_slot_identity.weight, mean=0., std=.02)
        self.register_buffer('_generic_slot_ids',torch.arange(6),persistent=False)
        require(set(map(int,self.trajectory_family_ids)) == set(range(6)),
                'all retained family embedding rows must remain used by history')

    def state_tokens(self, state):
        slots = self.generic_state_tokenizer(state)
        # Slot identities are independent of history family embeddings. Only the
        # original state VIEW marker is retained; no add_to_state/family ID call.
        view = self.identity.view_embedding(torch.tensor(1,device=state.device,dtype=torch.long))
        return slots + self.generic_slot_identity(self._generic_slot_ids)[None,:,:] + view

    def forward(self, x_seq, x_state=None, *, return_aux=False):
        require(x_seq.ndim == 3 and x_seq.shape[1:] == (72,54), '72x54 historical input')
        require(x_state is not None and x_state.shape == (len(x_seq),68), 'all68 state features required')
        if return_aux:
            trajectory, scales = self.trajectory_tokenizer(x_seq,return_scale_weights=True)
        else:
            trajectory = self.trajectory_tokenizer(x_seq,return_scale_weights=False);scales=None
        trajectory = self.identity.add_to_trajectory(trajectory,variable_family_ids=self.trajectory_family_ids)
        state = self.state_tokens(x_state)
        tokens = torch.cat([trajectory,state],dim=1)
        # Unmodified Full Transformer: all60 tokens, no cross-view mask.
        encoded = self.transformer(tokens)
        encoded_history = encoded[:,:54,:]
        prediction = self.projection(encoded_history.reshape(len(x_seq),54*512)).squeeze(-1)
        if return_aux:
            return prediction, dict(scale_weights=scales,trajectory_tokens=trajectory,state_tokens=state,
                encoded_token_count=60,encoded_history=encoded_history,encoded_state=encoded[:,54:,:])
        return prediction


def control_model(candidate, layout, bridge, eval_seed=20261017):
    model = build_model('phystok',candidate,layout,bridge,eval_seed)
    model.core = PlainStateCore(model.core)
    require(model.parameter_count() == PLAIN_PARAMETERS, 'actual Plain-State parameter count differs')
    return model
