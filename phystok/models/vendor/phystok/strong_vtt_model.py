#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


TRAJECTORY_TOKEN_COUNT = 54
STATE_TOKEN_COUNT = 6

FAMILY_ORDER = (
    "pm25_history",
    "calendar",
    "transport_dispersion",
    "moisture_stability_context",
    "pollution_composition",
    "precipitation_radiation",
)


@dataclass(frozen=True)
class StrongVTTSwitches:
    use_identity_embedding: bool = True
    use_multiscale_trajectory: bool = True
    use_family_state_tokens: bool = True


class MultiScaleTrajectoryTokenizer(nn.Module):
    """
    A2. Baseline-preserving multi-scale trajectory tokenizer.

    Base path:
        full 72h trajectory -> shared Linear(72, d_model)

    Optional residual path:
        12h / 24h / 72h embeddings
        -> sample-variable-specific softmax fusion
        -> bottleneck residual
        -> add back to the 72h base token

    residual_up is zero-initialized, so A2 starts as an exact no-op around
    the already-qualified primitive VTT trajectory representation.
    """

    def __init__(
        self,
        *,
        lookback: int,
        d_model: int,
        enabled: bool,
        bottleneck: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if lookback != 72:
            raise ValueError("Strong VTT v1 is frozen at 72h lookback")

        self.lookback = int(lookback)
        self.d_model = int(d_model)
        self.enabled = bool(enabled)

        # Exact primitive-VTT base path.
        self.base_72 = nn.Linear(72, d_model)

        if self.enabled:
            self.branch_12 = nn.Linear(12, d_model)
            self.branch_24 = nn.Linear(24, d_model)
            self.scale_gate = nn.Linear(d_model * 3, 3)

            self.residual_down = nn.Linear(d_model, bottleneck)
            self.residual_act = nn.GELU()
            self.residual_dropout = nn.Dropout(dropout)
            self.residual_up = nn.Linear(bottleneck, d_model)
            nn.init.zeros_(self.residual_up.weight)
            nn.init.zeros_(self.residual_up.bias)

            self.residual_scale = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32)
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_scale_weights: bool = False,
    ):
        if x.ndim != 3:
            raise RuntimeError(f"expected [B,L,C], got {tuple(x.shape)}")
        if x.shape[1] != 72:
            raise RuntimeError(f"expected 72h, got {x.shape[1]}")

        x_var = x.permute(0, 2, 1).contiguous()  # [B,C,72]
        e72 = self.base_72(x_var)                # [B,C,D]

        if not self.enabled:
            if return_scale_weights:
                return e72, None
            return e72

        e12 = self.branch_12(x_var[..., -12:])
        e24 = self.branch_24(x_var[..., -24:])

        gate_in = torch.cat([e12, e24, e72], dim=-1)
        weights = torch.softmax(self.scale_gate(gate_in), dim=-1)

        stacked = torch.stack([e12, e24, e72], dim=-2)  # [B,C,3,D]
        fused = torch.sum(weights.unsqueeze(-1) * stacked, dim=-2)

        delta = self.residual_down(fused)
        delta = self.residual_act(delta)
        delta = self.residual_dropout(delta)
        delta = self.residual_up(delta)

        out = e72 + self.residual_scale.to(e72.dtype) * delta

        if return_scale_weights:
            return out, weights
        return out


class TokenIdentityEmbedding(nn.Module):
    """
    A1. Explicit variable identity + physical-family identity.

    Trajectory tokens:
        variable identity + family identity + trajectory-view identity

    State tokens:
        family identity + state-view identity
    """

    def __init__(
        self,
        *,
        num_variables: int,
        num_families: int,
        d_model: int,
        enabled: bool,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)

        if self.enabled:
            self.variable_embedding = nn.Embedding(num_variables, d_model)
            self.family_embedding = nn.Embedding(num_families, d_model)
            self.view_embedding = nn.Embedding(2, d_model)

            nn.init.normal_(self.variable_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)

    def add_to_trajectory(
        self,
        tokens: torch.Tensor,
        *,
        variable_family_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return tokens

        c = tokens.shape[1]
        var_ids = torch.arange(c, device=tokens.device, dtype=torch.long)
        var_e = self.variable_embedding(var_ids)
        fam_e = self.family_embedding(variable_family_ids)
        view_e = self.view_embedding(
            torch.tensor(0, device=tokens.device, dtype=torch.long)
        )
        return tokens + var_e[None, :, :] + fam_e[None, :, :] + view_e

    def add_to_state(
        self,
        tokens: torch.Tensor,
        *,
        state_family_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return tokens

        fam_e = self.family_embedding(state_family_ids)
        view_e = self.view_embedding(
            torch.tensor(1, device=tokens.device, dtype=torch.long)
        )
        return tokens + fam_e[None, :, :] + view_e


class FamilyStateTokenizer(nn.Module):
    """
    A3. Six family-wise issue-time engineered-state tokens.

    The 68D core_without_pm10 vector is split into:
      PM2.5 history / calendar / transport / moisture-stability /
      pollution composition / precipitation-radiation.

    Each family gets its own compact MLP and produces one state token.
    """

    def __init__(
        self,
        *,
        family_feature_positions: Iterable[Iterable[int]],
        d_model: int,
        enabled: bool,
        bottleneck: int = 128,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)

        positions = [
            tuple(int(i) for i in group)
            for group in family_feature_positions
        ]
        if len(positions) != STATE_TOKEN_COUNT:
            raise ValueError(
                f"expected {STATE_TOKEN_COUNT} state families, got {len(positions)}"
            )
        if any(len(g) == 0 for g in positions):
            raise ValueError("state family cannot be empty")

        self.positions = positions

        if self.enabled:
            self.projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(len(group), bottleneck),
                        nn.GELU(),
                        nn.Linear(bottleneck, d_model),
                        nn.LayerNorm(d_model),
                    )
                    for group in positions
                ]
            )

    def forward(self, state_x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError(
                "FamilyStateTokenizer.forward called while disabled"
            )
        if state_x.ndim != 2:
            raise RuntimeError(
                f"expected state_x [B,F], got {tuple(state_x.shape)}"
            )

        tokens = []
        for group, projector in zip(self.positions, self.projectors):
            pos = torch.tensor(
                group,
                device=state_x.device,
                dtype=torch.long,
            )
            family_x = state_x.index_select(dim=1, index=pos)
            tokens.append(projector(family_x))

        return torch.stack(tokens, dim=1)  # [B,6,D]


class StrongVTT(nn.Module):
    """
    Strong VTT v1
      A1 identity-aware tokens
      A2 multi-scale trajectory residual
      A3 family-wise dual-view state tokens

    Even with A3 enabled, the forecast head still flattens only the 54
    encoded trajectory tokens. State tokens can help only by interacting
    with trajectory tokens inside Transformer attention.
    """

    def __init__(
        self,
        *,
        lookback: int,
        num_trajectory_features: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        head_hidden: int,
        activation: str,
        trajectory_family_ids: Iterable[int],
        state_family_feature_positions: Iterable[Iterable[int]],
        switches: StrongVTTSwitches,
    ) -> None:
        super().__init__()

        if lookback != 72:
            raise ValueError("Strong VTT v1 requires lookback=72")
        if num_trajectory_features != TRAJECTORY_TOKEN_COUNT:
            raise ValueError(
                f"Strong VTT v1 requires {TRAJECTORY_TOKEN_COUNT} trajectory channels"
            )
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.lookback = int(lookback)
        self.num_trajectory_features = int(num_trajectory_features)
        self.d_model = int(d_model)
        self.switches = switches

        traj_family_ids = tuple(int(x) for x in trajectory_family_ids)
        if len(traj_family_ids) != TRAJECTORY_TOKEN_COUNT:
            raise ValueError("trajectory_family_ids length mismatch")
        if min(traj_family_ids) < 0 or max(traj_family_ids) >= STATE_TOKEN_COUNT:
            raise ValueError("trajectory family id outside 0..5")

        self.register_buffer(
            "trajectory_family_ids",
            torch.tensor(traj_family_ids, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "state_family_ids",
            torch.arange(STATE_TOKEN_COUNT, dtype=torch.long),
            persistent=True,
        )

        self.trajectory_tokenizer = MultiScaleTrajectoryTokenizer(
            lookback=lookback,
            d_model=d_model,
            enabled=switches.use_multiscale_trajectory,
            bottleneck=128,
            dropout=0.0,
        )

        self.identity = TokenIdentityEmbedding(
            num_variables=TRAJECTORY_TOKEN_COUNT,
            num_families=STATE_TOKEN_COUNT,
            d_model=d_model,
            enabled=switches.use_identity_embedding,
        )

        self.state_tokenizer = FamilyStateTokenizer(
            family_feature_positions=state_family_feature_positions,
            d_model=d_model,
            enabled=switches.use_family_state_tokens,
            bottleneck=128,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
        )

        # Same head dimensionality as primitive VTT.
        self.projection = nn.Sequential(
            nn.Linear(
                d_model * TRAJECTORY_TOKEN_COUNT,
                head_hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_state: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ):
        if x_seq.ndim != 3:
            raise RuntimeError(
                f"expected x_seq [B,L,C], got {tuple(x_seq.shape)}"
            )
        if x_seq.shape[1:] != (
            self.lookback,
            self.num_trajectory_features,
        ):
            raise RuntimeError(
                f"unexpected sequence shape: {tuple(x_seq.shape)}"
            )

        if return_aux:
            traj, scale_weights = self.trajectory_tokenizer(
                x_seq,
                return_scale_weights=True,
            )
        else:
            traj = self.trajectory_tokenizer(
                x_seq,
                return_scale_weights=False,
            )
            scale_weights = None
        traj = self.identity.add_to_trajectory(
            traj,
            variable_family_ids=self.trajectory_family_ids,
        )

        if self.switches.use_family_state_tokens:
            if x_state is None:
                raise RuntimeError(
                    "Strong VTT state-token path requires x_state"
                )
            state = self.state_tokenizer(x_state)
            state = self.identity.add_to_state(
                state,
                state_family_ids=self.state_family_ids,
            )
            tokens = torch.cat([traj, state], dim=1)
        else:
            state = None
            tokens = traj

        encoded = self.transformer(tokens)

        encoded_traj = encoded[:, :TRAJECTORY_TOKEN_COUNT, :]
        flat = encoded_traj.reshape(encoded_traj.shape[0], -1)
        pred = self.projection(flat).squeeze(-1)

        if return_aux:
            return pred, {
                "scale_weights": scale_weights,
                "trajectory_tokens": traj,
                "state_tokens": state,
                "encoded_token_count": int(encoded.shape[1]),
            }
        return pred
