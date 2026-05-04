"""Graph and perceiver-style architectures.

This file intentionally houses all model definitions so checkpoints created by
`train.py` and consumed by `infer.py` always reference one canonical module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from emulator.data.station_metadata import StationMetaEncoder


class MLP(nn.Module):
    """Small one-hidden-layer MLP."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool dense node tensors while ignoring padded positions."""
    mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
    h = h * mask_f
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return h.sum(dim=1) / denom


class SpatialOnlyGraphSAGEBatch(nn.Module):
    """Baseline encoder for `history_steps == 0`.

    Pipeline:
      GraphSAGE -> global mean pool -> linear forecast head.

    Optional p_mean pathway:
      If enabled and `batch.p_mean_curr` is provided, a small encoder embeds the
      scalar and concatenates it to the pooled graph representation.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.0,
        use_pmean: bool = False,
        pmean_dim: int = 32,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = float(dropout)

        self.use_pmean = bool(use_pmean)
        if self.use_pmean:
            self.pmean_curr_enc = nn.Sequential(
                nn.Linear(1, int(pmean_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(int(pmean_dim), int(pmean_dim)),
            )
            self.lin_out = nn.Linear(hidden_channels + int(pmean_dim), out_channels)
        else:
            self.pmean_curr_enc = None
            self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h_pool = global_mean_pool(h, batch_index)
        h_pool = F.dropout(h_pool, p=self.dropout, training=self.training)

        if self.pmean_curr_enc is not None and hasattr(batch, "p_mean_curr"):
            p_mean_curr = batch.p_mean_curr
            if p_mean_curr.dim() == 1:
                p_mean_curr = p_mean_curr.unsqueeze(-1)
            p_mean_curr = p_mean_curr.to(dtype=h_pool.dtype)
            p_mean_emb = self.pmean_curr_enc(p_mean_curr)
            h_pool = torch.cat([h_pool, p_mean_emb], dim=-1)

        return self.lin_out(h_pool)


class SpatialMLP0h(nn.Module):
    """0h spatial-only baseline: mean-pool nodes, then apply a tiny MLP head."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        use_pmean: bool = False,
        pmean_dim: int = 32,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.use_pmean = bool(use_pmean)
        self.hidden_channels = int(hidden_channels)
        self.pmean_dim = int(pmean_dim)

        self.feature_proj = nn.Linear(int(in_channels), self.hidden_channels)
        self.feature_act = nn.ReLU(inplace=True)

        if self.use_pmean:
            self.pmean_curr_enc = nn.Sequential(
                nn.Linear(1, self.pmean_dim),
                nn.ReLU(inplace=True),
            )
            head_in = self.hidden_channels + self.pmean_dim
        else:
            self.pmean_curr_enc = None
            head_in = self.hidden_channels

        self.lin_out = nn.Linear(head_in, int(out_channels))

    def forward(self, batch):
        h = global_mean_pool(batch.x, batch.batch)
        h = self.feature_act(self.feature_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.pmean_curr_enc is not None and hasattr(batch, "p_mean_curr"):
            p_mean_curr = batch.p_mean_curr
            if p_mean_curr.dim() == 1:
                p_mean_curr = p_mean_curr.unsqueeze(-1)
            p_mean_emb = self.pmean_curr_enc(p_mean_curr.to(dtype=h.dtype))
            h = torch.cat([h, p_mean_emb], dim=-1)

        return self.lin_out(h)


class TemporalCNN12h(nn.Module):
    """12h temporal-only baseline: pool space first, then run a small CNN over time."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        temporal_hidden=None,
        use_pmean: bool = False,
        pmean_T: int | None = None,
        pmean_dim: int = 32,
    ):
        super().__init__()
        hidden_dim = int(hidden_channels if temporal_hidden is None else temporal_hidden)
        self.dropout = float(dropout)
        self.use_pmean = bool(use_pmean)
        self.pmean_T = int(pmean_T) if (pmean_T is not None) else None
        self.pmean_dim = int(pmean_dim)

        self.time_conv = nn.Conv1d(int(in_channels), hidden_dim, kernel_size=3, padding=1)
        self.time_act = nn.ReLU(inplace=True)

        if self.use_pmean:
            if self.pmean_T is None or self.pmean_T <= 0:
                raise ValueError("TemporalCNN12h: use_pmean=True requires pmean_T=W (window length).")
            self.pmean_hist_enc = nn.Sequential(
                nn.Linear(self.pmean_T, self.pmean_dim),
                nn.ReLU(inplace=True),
            )
            head_in = hidden_dim + self.pmean_dim
        else:
            self.pmean_hist_enc = None
            head_in = hidden_dim

        self.lin_out = nn.Linear(head_in, int(out_channels))

    def forward(self, batch):
        x_hist = batch.x_hist
        pooled_seq = [global_mean_pool(x_hist[:, t, :], batch.batch) for t in range(x_hist.size(1))]
        h = torch.stack(pooled_seq, dim=1).transpose(1, 2)
        h = self.time_act(self.time_conv(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.mean(dim=-1)

        if self.pmean_hist_enc is not None and hasattr(batch, "p_mean_hist"):
            p_mean_hist = batch.p_mean_hist
            if p_mean_hist.dim() == 3 and p_mean_hist.size(-1) == 1:
                p_mean_hist = p_mean_hist.squeeze(-1)
            p_mean_emb = self.pmean_hist_enc(p_mean_hist.to(dtype=h.dtype))
            h = torch.cat([h, p_mean_emb], dim=-1)

        return self.lin_out(h)


class TemporalLSTM12h(nn.Module):
    """12h temporal-only baseline: pool space first, then run a small LSTM over time."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        temporal_hidden=None,
        use_pmean: bool = False,
        pmean_T: int | None = None,
        pmean_dim: int = 32,
    ):
        super().__init__()
        hidden_dim = int(hidden_channels)
        temporal_dim = int(hidden_channels if temporal_hidden is None else temporal_hidden)
        self.dropout = float(dropout)
        self.use_pmean = bool(use_pmean)
        self.pmean_T = int(pmean_T) if (pmean_T is not None) else None
        self.pmean_dim = int(pmean_dim)

        self.input_proj = nn.Linear(int(in_channels), hidden_dim)
        self.input_act = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True,
        )

        if self.use_pmean:
            if self.pmean_T is None or self.pmean_T <= 0:
                raise ValueError("TemporalLSTM12h: use_pmean=True requires pmean_T=W (window length).")
            self.pmean_hist_enc = nn.Sequential(
                nn.Linear(self.pmean_T, self.pmean_dim),
                nn.ReLU(inplace=True),
            )
            head_in = temporal_dim + self.pmean_dim
        else:
            self.pmean_hist_enc = None
            head_in = temporal_dim

        self.lin_out = nn.Linear(head_in, int(out_channels))

    def forward(self, batch):
        x_hist = batch.x_hist
        pooled_seq = [global_mean_pool(x_hist[:, t, :], batch.batch) for t in range(x_hist.size(1))]
        h = torch.stack(pooled_seq, dim=1)
        h = self.input_act(self.input_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out, _ = self.lstm(h)
        h_final = F.dropout(out[:, -1, :], p=self.dropout, training=self.training)

        if self.pmean_hist_enc is not None and hasattr(batch, "p_mean_hist"):
            p_mean_hist = batch.p_mean_hist
            if p_mean_hist.dim() == 3 and p_mean_hist.size(-1) == 1:
                p_mean_hist = p_mean_hist.squeeze(-1)
            p_mean_emb = self.pmean_hist_enc(p_mean_hist.to(dtype=h_final.dtype))
            h_final = torch.cat([h_final, p_mean_emb], dim=-1)

        return self.lin_out(h_final)


class SpatioTemporalGraphSAGEBatch(nn.Module):
    """Baseline encoder for `history_steps > 0`.

    For each history step:
      GraphSAGE -> global mean pool

    Then:
      stacked pooled sequence -> LSTM -> linear forecast head.

    Optional p_mean pathway:
      If enabled and `batch.p_mean_hist` exists, a compact history encoder is
      concatenated to the final LSTM state before prediction.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.0,
        temporal_hidden=None,
        use_pmean: bool = False,
        pmean_T: int | None = None,
        pmean_dim: int = 32,
    ):
        super().__init__()
        if temporal_hidden is None:
            temporal_hidden = hidden_channels

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = float(dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.use_pmean = bool(use_pmean)
        self.pmean_T = int(pmean_T) if (pmean_T is not None) else None
        if self.use_pmean:
            if self.pmean_T is None or self.pmean_T <= 0:
                raise ValueError("SpatioTemporalGraphSAGEBatch: use_pmean=True requires pmean_T=W (window length).")

            self.pmean_hist_enc = nn.Sequential(
                nn.Linear(self.pmean_T, int(pmean_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(int(pmean_dim), int(pmean_dim)),
            )
            self.lin_out = nn.Linear(int(temporal_hidden) + int(pmean_dim), out_channels)
        else:
            self.pmean_hist_enc = None
            self.lin_out = nn.Linear(int(temporal_hidden), out_channels)

    def forward(self, batch):
        x_hist = batch.x_hist
        edge_index = batch.edge_index
        batch_index = batch.batch
        window = x_hist.size(1)

        pooled_seq = []
        for t in range(window):
            x_t = x_hist[:, t, :]
            h = x_t
            for conv in self.convs:
                h = conv(h, edge_index)
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h_pool = global_mean_pool(h, batch_index)
            pooled_seq.append(h_pool)

        h_seq = torch.stack(pooled_seq, dim=1)
        out, _ = self.lstm(h_seq)
        h_final = out[:, -1, :]
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)

        if self.pmean_hist_enc is not None and hasattr(batch, "p_mean_hist"):
            p_mean_hist = batch.p_mean_hist
            if p_mean_hist.dim() == 3 and p_mean_hist.size(-1) == 1:
                p_mean_hist = p_mean_hist.squeeze(-1)
            p_mean_hist = p_mean_hist.to(dtype=h_final.dtype)
            p_mean_emb = self.pmean_hist_enc(p_mean_hist)
            h_final = torch.cat([h_final, p_mean_emb], dim=-1)

        return self.lin_out(h_final)


class TransformerBlock(nn.Module):
    """Lightweight Transformer block for short token sequences."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=float(dropout), batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(h))
        h = self.ff(x)
        x = self.ln2(x + self.drop(h))
        return x


class PACT(nn.Module):
    """Perceiver-like spatio-temporal model with optional p_mean pathways.

    Core flow:
      1. Node encoder encodes node tokens per timestep.
      2. Station query cross-attends node tokens.
      3. Temporal transformer processes station memory tokens.
      4. Horizon queries read memory via cross-attention.
      5. Peak-aware gated head predicts output horizon.

    Optional p_mean pathways:
      - `use_pmean_tokens`: append time-aligned p_mean tokens (context tokens).
      - `use_pmean_global`: encode full p_mean history and concatenate to head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_node_read_heads: int = 8,
        n_time_read_heads: int = 8,
        n_transformer_layers: int = 2,
        transformer_ff_mult: float = 4.0,
        transformer_dropout: float = 0.05,
        head_hidden: int | None = None,
        head_dropout: float = 0.0,
        gate_mode: str = "window",
        gate_bias_init: float = -2.0,
        tail_tanh_clip: float = 2.5,
        alpha_init_logit: float = -2.0,
        max_time_steps: int = 32,
        station_feat_dim: int = 0,
        use_station_meta: bool = True,
        use_pmean_tokens: bool = False,
        use_pmean_global: bool = False,
        pmean_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.station_token = nn.Parameter(torch.zeros(1, hidden_channels))

        self.use_station_meta = bool(use_station_meta and station_feat_dim > 0)
        if self.use_station_meta:
            self.station_meta_encoder = StationMetaEncoder(station_feat_dim, hidden_channels)
            self.station_meta_ln = nn.LayerNorm(hidden_channels)

        self.node_read_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=int(n_node_read_heads),
            dropout=0.0,
            batch_first=True,
        )

        self.time_embed = nn.Embedding(int(max_time_steps), hidden_channels)

        ff_dim = int(hidden_channels * float(transformer_ff_mult))
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(hidden_channels, n_heads=int(n_time_read_heads), ff_dim=ff_dim, dropout=float(transformer_dropout))
                for _ in range(int(n_transformer_layers))
            ]
        )

        self.horizon_embed = nn.Embedding(self.out_channels, hidden_channels)

        self.forecast_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=int(n_time_read_heads),
            dropout=0.0,
            batch_first=True,
        )

        if head_hidden is None:
            head_hidden = hidden_channels * 2

        self.use_pmean_global = bool(use_pmean_global)
        self.pmean_dim = int(pmean_dim) if self.use_pmean_global else 0
        self._pmean_max_steps = int(max_time_steps)
        if self.use_pmean_global:
            self.pmean_global_enc = nn.Sequential(
                nn.Linear(self._pmean_max_steps, int(pmean_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(int(pmean_dim), int(pmean_dim)),
                nn.LayerNorm(int(pmean_dim)),
            )
        else:
            self.pmean_global_enc = None

        head_in_dim = int(hidden_channels) + (self.pmean_dim if self.use_pmean_global else 0)

        self.gate_mode = str(gate_mode)
        self.tail_tanh_clip = float(tail_tanh_clip)

        self.mlp_base = MLP(head_in_dim, head_hidden, 1, dropout=head_dropout)
        self.mlp_tail = MLP(head_in_dim, head_hidden, 1, dropout=head_dropout)

        if self.gate_mode == "window":
            self.mlp_gate = MLP(head_in_dim, head_hidden, 1, dropout=head_dropout)
        elif self.gate_mode == "horizon":
            self.mlp_gate = MLP(head_in_dim, head_hidden, 1, dropout=head_dropout)
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        with torch.no_grad():
            self.mlp_gate.fc2.bias.fill_(float(gate_bias_init))

        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init_logit), dtype=torch.float32))

        self.use_pmean_tokens = bool(use_pmean_tokens)
        if self.use_pmean_tokens:
            self.pmean_token_proj = nn.Linear(1, hidden_channels)
            self.pmean_token_ln = nn.LayerNorm(hidden_channels)
        else:
            self.pmean_token_proj = None
            self.pmean_token_ln = None

    def _encode_nodes_one_time(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        batch_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x_t
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, batch, station_feat: torch.Tensor | None = None, return_aux: bool = False):
        x_hist = batch.x_hist
        edge_index = batch.edge_index
        batch_vec = batch.batch
        steps = x_hist.size(1)
        horizons = self.out_channels

        if steps > self.time_embed.num_embeddings:
            raise ValueError(
                f"T={steps} exceeds max_time_steps={self.time_embed.num_embeddings}. Increase --max_time_steps."
            )

        q_station = self.station_token
        if self.use_station_meta and station_feat is not None and station_feat.numel() > 0:
            station_emb = self.station_meta_encoder(station_feat.to(q_station.device))
            station_emb = self.station_meta_ln(station_emb)
            q_station = q_station + station_emb

        batch_size = int(batch.num_graphs)
        query = q_station.unsqueeze(0).expand(batch_size, -1, -1)

        z_list = []
        node_attn_max_list = []
        node_attn_entropy_list = []

        for t in range(steps):
            node_tokens = self._encode_nodes_one_time(x_hist[:, t, :], edge_index, batch_vec)
            node_dense, mask = to_dense_batch(node_tokens, batch_vec)

            if return_aux:
                z_t, attn_w = self.node_read_attn(
                    query,
                    node_dense,
                    node_dense,
                    key_padding_mask=(~mask),
                    need_weights=True,
                    average_attn_weights=True,
                )

                w = attn_w.squeeze(1)
                w = w.masked_fill(~mask, 0.0)
                w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                p = w / w_sum
                node_attn_max_list.append(p.max(dim=1).values)
                node_attn_entropy_list.append(-(p * (p.clamp_min(1e-12).log())).sum(dim=1))
            else:
                z_t, _ = self.node_read_attn(
                    query,
                    node_dense,
                    node_dense,
                    key_padding_mask=(~mask),
                    need_weights=False,
                )

            z_list.append(z_t)

        z_seq = torch.stack(z_list, dim=1).squeeze(2)

        if self.pmean_token_proj is not None and hasattr(batch, "p_mean_hist"):
            p_mean_hist = batch.p_mean_hist
            if p_mean_hist.dim() == 2:
                p_mean_hist = p_mean_hist.unsqueeze(-1)
            elif p_mean_hist.dim() == 3 and p_mean_hist.size(-1) == 1:
                pass
            else:
                raise ValueError(
                    "Expected batch.p_mean_hist to have shape (B,T) or (B,T,1), "
                    f"got {tuple(p_mean_hist.shape)}"
                )
            p_tokens = self.pmean_token_proj(p_mean_hist.to(dtype=z_seq.dtype))
            p_tokens = self.pmean_token_ln(p_tokens)
            z_seq = torch.cat([z_seq, p_tokens], dim=1)

        time_ids = torch.arange(steps, device=z_seq.device)
        time_emb = self.time_embed(time_ids).unsqueeze(0)
        if z_seq.size(1) == steps:
            z_seq = z_seq + time_emb
        else:
            z_seq = z_seq + torch.cat([time_emb, time_emb], dim=1)

        memory = z_seq
        for block in self.transformer:
            memory = block(memory)

        horizon_ids = torch.arange(horizons, device=memory.device)
        horizon_query = self.horizon_embed(horizon_ids).unsqueeze(0).expand(batch_size, horizons, -1)

        if return_aux:
            context, attn_time = self.forecast_attn(horizon_query, memory, memory, need_weights=True, average_attn_weights=True)
        else:
            context, _ = self.forecast_attn(horizon_query, memory, memory, need_weights=False)
            attn_time = None

        p_global = None
        if self.pmean_global_enc is not None:
            if hasattr(batch, "p_mean_hist"):
                p_mean_hist = batch.p_mean_hist
                if p_mean_hist.dim() == 3 and p_mean_hist.size(-1) == 1:
                    p_mean_hist = p_mean_hist.squeeze(-1)
                if p_mean_hist.dim() != 2:
                    raise ValueError(
                        "Expected batch.p_mean_hist to have shape (B,T) or (B,T,1), "
                        f"got {tuple(p_mean_hist.shape)}"
                    )
                if p_mean_hist.size(0) != batch_size:
                    raise ValueError(
                        "batch.p_mean_hist batch dim mismatch: "
                        f"got B={p_mean_hist.size(0)} but num_graphs={batch_size}"
                    )
                if p_mean_hist.size(1) != steps:
                    raise ValueError(
                        "p_mean_hist length mismatch: "
                        f"p_mean_hist has T={p_mean_hist.size(1)} but x_hist has T={steps}."
                    )
                if steps > self._pmean_max_steps:
                    raise ValueError(
                        f"T={steps} exceeds p_mean encoder max_time_steps={self._pmean_max_steps}. Increase --max_time_steps."
                    )

                p_mean_pad = p_mean_hist.new_zeros((batch_size, self._pmean_max_steps))
                p_mean_pad[:, self._pmean_max_steps - steps : self._pmean_max_steps] = p_mean_hist
                p_global = self.pmean_global_enc(p_mean_pad.to(dtype=context.dtype))
            else:
                p_global = torch.zeros((batch_size, self.pmean_dim), device=context.device, dtype=context.dtype)

        c_flat = context.reshape(batch_size * horizons, self.hidden_channels)

        if p_global is not None:
            p_rep = p_global.unsqueeze(1).expand(batch_size, horizons, -1).reshape(batch_size * horizons, self.pmean_dim)
            c_flat = torch.cat([c_flat, p_rep.to(dtype=c_flat.dtype)], dim=-1)

        y_base = self.mlp_base(c_flat).view(batch_size, horizons)
        r_tail = self.mlp_tail(c_flat).view(batch_size, horizons)

        if self.tail_tanh_clip is not None and self.tail_tanh_clip > 0:
            clip_val = float(self.tail_tanh_clip)
            r_tail = clip_val * torch.tanh(r_tail / clip_val)

        alpha = torch.sigmoid(self.alpha_logit)

        if self.gate_mode == "window":
            g_in = context.mean(dim=1)
            if p_global is not None:
                g_in = torch.cat([g_in, p_global.to(dtype=g_in.dtype)], dim=-1)

            gate = torch.sigmoid(self.mlp_gate(g_in)).view(batch_size, 1)
            y = y_base + gate * (alpha * r_tail)
        else:
            gate = torch.sigmoid(self.mlp_gate(c_flat)).view(batch_size, horizons)
            y = y_base + gate * (alpha * r_tail)

        if not return_aux:
            return y

        aux = {}

        if node_attn_max_list:
            node_attn_max = torch.stack(node_attn_max_list, dim=1).mean(dim=1)
            node_attn_ent = torch.stack(node_attn_entropy_list, dim=1).mean(dim=1)
        else:
            node_attn_max = torch.zeros(batch_size, device=memory.device)
            node_attn_ent = torch.zeros(batch_size, device=memory.device)

        if attn_time is not None:
            idx_recent = steps - 1
            time_attn_recent = attn_time[..., idx_recent].mean(dim=1)
        else:
            time_attn_recent = torch.zeros(batch_size, device=memory.device)

        if self.gate_mode == "window":
            gate_mean = gate.view(-1)
        else:
            gate_mean = gate.mean(dim=1)

        aux.update(
            node_attn_max_per_sample=node_attn_max.detach(),
            node_attn_entropy_per_sample=node_attn_ent.detach(),
            time_attn_recent_per_sample=time_attn_recent.detach(),
            gate_mean_per_sample=gate_mean.detach(),
            alpha_scalar=float(alpha.detach().cpu().item()),
        )
        return y, aux


class PACTCNN(PACT):
    """PACT variant that swaps the GraphSAGE node encoder for a CNN encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.convs = None

    def _encode_nodes_one_time(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        batch_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_vec is None:
            raise ValueError("PACTCNN requires batch indices for dense CNN encoding.")
        x_dense, mask = to_dense_batch(x_t, batch_vec)
        h = x_dense.transpose(1, 2)
        h = self.act(self.conv1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act(self.conv2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.transpose(1, 2)
        return h[mask]


__all__ = [
    "MLP",
    "SpatialMLP0h",
    "SpatialOnlyGraphSAGEBatch",
    "TemporalCNN12h",
    "TemporalLSTM12h",
    "SpatioTemporalGraphSAGEBatch",
    "TransformerBlock",
    "PACT",
    "PACTCNN",
]
