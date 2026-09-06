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


def _canonical_encoder_type(encoder_type: str) -> str:
    """Normalize the public encoder name while keeping a strict option set."""
    value = str(encoder_type).strip().lower()
    if value == "graphsage":
        return "GraphSAGE"
    if value == "cnn":
        return "CNN"
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Expected 'GraphSAGE' or 'CNN'.")


def canonical_temporal_block(temporal_block: str) -> str:
    """Normalize temporal block names; ``attn`` aliases the current Transformer."""
    value = str(temporal_block).strip().lower()
    names = {
        "mlp": "MLP",
        "lstm": "LSTM",
        "gru": "GRU",
        "attn": "Transformer",
        "transformer": "Transformer",
    }
    if value in names:
        return names[value]
    raise ValueError(
        f"Unknown temporal_block: {temporal_block!r}. "
        "Expected 'MLP', 'LSTM', 'GRU', or 'Transformer' (alias: 'attn')."
    )


def _make_graphsage_layers(in_channels: int, hidden_channels: int, num_layers: int) -> nn.ModuleList:
    """Build the historical GraphSAGE stack without changing state-dict keys."""
    convs = nn.ModuleList()
    convs.append(SAGEConv(in_channels, hidden_channels))
    for _ in range(num_layers - 2):
        convs.append(SAGEConv(hidden_channels, hidden_channels))
    convs.append(SAGEConv(hidden_channels, hidden_channels))
    return convs


def _uniform_grid_dim(batch, name: str, batch_size: int) -> int:
    """Read one positive grid dimension and require it to match across a batch."""
    if not hasattr(batch, name):
        raise ValueError(
            f"CNN encoder requires batch.{name}. Ensure ForcingGraphView receives graphs "
            "with grid_H/grid_W metadata."
        )

    raw_value = getattr(batch, name)
    if torch.is_tensor(raw_value):
        values = raw_value.reshape(-1)
        if values.numel() == 0:
            raise ValueError(f"CNN encoder received empty batch.{name} metadata.")
        if values.numel() not in (1, batch_size):
            raise ValueError(
                f"Expected batch.{name} to contain 1 or B={batch_size} values, "
                f"got {values.numel()}."
            )
        first = int(values[0].item())
        if bool((values != first).any().item()):
            found = values.detach().cpu().tolist()
            raise ValueError(f"CNN encoder requires uniform {name} within a batch, got {found}.")
    else:
        values = list(raw_value) if isinstance(raw_value, (list, tuple)) else [raw_value]
        if len(values) not in (1, batch_size):
            raise ValueError(
                f"Expected batch.{name} to contain 1 or B={batch_size} values, got {len(values)}."
            )
        parsed = [int(value) for value in values]
        first = parsed[0]
        if any(value != first for value in parsed[1:]):
            raise ValueError(f"CNN encoder requires uniform {name} within a batch, got {parsed}.")

    if first <= 0:
        raise ValueError(f"CNN encoder requires positive {name}, got {first}.")
    return first


def _infer_grid_batch_shape(batch, total_nodes: int) -> tuple[int, int, int]:
    """Infer `(B, H, W)` from PyG batch metadata and validate node layout."""
    batch_size = int(batch.num_graphs)
    if batch_size <= 0:
        raise ValueError(f"CNN encoder requires a non-empty batch, got B={batch_size}.")

    grid_h = _uniform_grid_dim(batch, "grid_H", batch_size)
    grid_w = _uniform_grid_dim(batch, "grid_W", batch_size)
    nodes_per_graph = grid_h * grid_w
    expected_nodes = batch_size * nodes_per_graph
    if int(total_nodes) != expected_nodes:
        raise ValueError(
            "CNN encoder grid/node mismatch: "
            f"B={batch_size}, H={grid_h}, W={grid_w} imply {expected_nodes} flattened nodes, "
            f"but input has {int(total_nodes)}."
        )

    if not hasattr(batch, "batch"):
        raise ValueError("CNN encoder requires the PyG batch assignment vector `batch.batch`.")
    batch_vec = batch.batch
    if int(batch_vec.numel()) != int(total_nodes):
        raise ValueError(
            f"CNN encoder batch vector has {batch_vec.numel()} entries for {int(total_nodes)} nodes."
        )

    counts = torch.bincount(batch_vec, minlength=batch_size)
    counts_ok = counts.numel() == batch_size and bool((counts == nodes_per_graph).all().item())
    if not counts_ok:
        found = counts.detach().cpu().tolist()
        raise ValueError(
            f"CNN encoder requires exactly H*W={nodes_per_graph} nodes per graph, got {found}."
        )

    return batch_size, grid_h, grid_w


class GridCNNEncoder(nn.Module):
    """Same-resolution 2D CNN that returns one hidden token per grid node."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        depth = int(num_layers)
        if depth <= 0:
            raise ValueError(f"CNN encoder requires num_layers >= 1, got {depth}.")

        layers = []
        current_channels = int(in_channels)
        for _ in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=int(hidden_channels),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            current_channels = int(hidden_channels)

        self.layers = nn.ModuleList(layers)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CNN encoder expects flattened node features (B*N,F), got {tuple(x.shape)}.")

        batch_size, grid_h, grid_w = grid_shape
        h = x.reshape(batch_size, grid_h, grid_w, x.size(-1)).permute(0, 3, 1, 2).contiguous()
        for layer in self.layers:
            h = layer(h)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Return the same node-token interface as GraphSAGE: (B*N, hidden_channels).
        return h.permute(0, 2, 3, 1).contiguous().reshape(batch_size * grid_h * grid_w, -1)


class SpatialOnlyGraphSAGEBatch(nn.Module):
    """Spatial baseline for `history_steps == 0` (legacy class name).

    Pipeline:
      selected spatial encoder -> global mean pool -> linear forecast head.

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
        encoder_type: str = "GraphSAGE",
    ):
        super().__init__()
        self.encoder_type = _canonical_encoder_type(encoder_type)
        if self.encoder_type == "GraphSAGE":
            self.convs = _make_graphsage_layers(in_channels, hidden_channels, num_layers)
            self.cnn_encoder = None
        else:
            self.convs = nn.ModuleList()
            self.cnn_encoder = GridCNNEncoder(in_channels, hidden_channels, num_layers, dropout)

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

        if self.cnn_encoder is not None:
            grid_shape = _infer_grid_batch_shape(batch, x.size(0))
            h = self.cnn_encoder(x, grid_shape)
        else:
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


class SpatioTemporalGraphSAGEBatch(nn.Module):
    """Spatio-temporal baseline for `history_steps > 0` (legacy class name).

    For each history step:
      selected spatial encoder -> global mean pool

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
        encoder_type: str = "GraphSAGE",
    ):
        super().__init__()
        if temporal_hidden is None:
            temporal_hidden = hidden_channels

        self.encoder_type = _canonical_encoder_type(encoder_type)
        if self.encoder_type == "GraphSAGE":
            self.convs = _make_graphsage_layers(in_channels, hidden_channels, num_layers)
            self.cnn_encoder = None
        else:
            self.convs = nn.ModuleList()
            self.cnn_encoder = GridCNNEncoder(in_channels, hidden_channels, num_layers, dropout)

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
        grid_shape = None
        if self.cnn_encoder is not None:
            grid_shape = _infer_grid_batch_shape(batch, x_hist.size(0))

        pooled_seq = []
        for t in range(window):
            x_t = x_hist[:, t, :]
            if self.cnn_encoder is not None:
                h = self.cnn_encoder(x_t, grid_shape)
            else:
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


class TemporalMLPBlock(nn.Module):
    """Residual token-wise MLP used as a no-middle-attention ablation."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(p=float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.ff(x)))


class PACT(nn.Module):
    """Perceiver-like spatio-temporal model with optional p_mean pathways.

    Core flow:
      1. Node encoder encodes node tokens per timestep.
      2. Station query cross-attends node tokens.
      3. Selected temporal block processes station memory tokens.
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
        encoder_type: str = "GraphSAGE",
        temporal_block: str = "Transformer",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)

        self.encoder_type = _canonical_encoder_type(encoder_type)
        if self.encoder_type == "GraphSAGE":
            self.convs = _make_graphsage_layers(self.in_channels, hidden_channels, num_layers)
            self.cnn_encoder = None
        else:
            self.convs = nn.ModuleList()
            self.cnn_encoder = GridCNNEncoder(self.in_channels, hidden_channels, num_layers, dropout)
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

        self.temporal_block = canonical_temporal_block(temporal_block)
        temporal_depth = int(n_transformer_layers)
        ff_dim = int(hidden_channels * float(transformer_ff_mult))

        # Keep the historical attribute name and parameter keys for strict loading
        # of existing Transformer checkpoints.
        self.transformer = nn.ModuleList()
        self.temporal_mlp = nn.ModuleList()
        self.temporal_rnn = None
        self.temporal_rnn_dropout = None
        self.temporal_rnn_norm = None

        if self.temporal_block == "Transformer":
            self.transformer.extend(
                TransformerBlock(
                    hidden_channels,
                    n_heads=int(n_time_read_heads),
                    ff_dim=ff_dim,
                    dropout=float(transformer_dropout),
                )
                for _ in range(temporal_depth)
            )
        elif self.temporal_block == "MLP":
            if temporal_depth <= 0:
                raise ValueError("MLP temporal block requires n_transformer_layers >= 1.")
            self.temporal_mlp.extend(
                TemporalMLPBlock(hidden_channels, ff_dim=ff_dim, dropout=float(transformer_dropout))
                for _ in range(temporal_depth)
            )
        else:
            if temporal_depth <= 0:
                raise ValueError(f"{self.temporal_block} temporal block requires n_transformer_layers >= 1.")
            rnn_cls = nn.LSTM if self.temporal_block == "LSTM" else nn.GRU
            self.temporal_rnn = rnn_cls(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                num_layers=temporal_depth,
                batch_first=True,
                dropout=float(transformer_dropout) if temporal_depth > 1 else 0.0,
            )
            self.temporal_rnn_dropout = nn.Dropout(p=float(transformer_dropout))
            self.temporal_rnn_norm = nn.LayerNorm(hidden_channels)

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
        grid_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if self.cnn_encoder is not None:
            if grid_shape is None:
                raise ValueError("CNN encoder requires an inferred (B,H,W) grid shape.")
            return self.cnn_encoder(x_t, grid_shape)

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
        grid_shape = None
        if self.cnn_encoder is not None:
            grid_shape = _infer_grid_batch_shape(batch, x_hist.size(0))

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
            node_tokens = self._encode_nodes_one_time(
                x_hist[:, t, :],
                edge_index,
                batch_vec,
                grid_shape=grid_shape,
            )
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
            # NOTE: this preserves the historical [forcing T][p_mean T] token
            # layout. LSTM/GRU therefore process it as a length-2T sequence.
            # Shipped configs use USE_PMEAN=0; if token-mode p_mean is enabled
            # with a recurrent block, consider time-wise fusion/interleaving.
            z_seq = torch.cat([z_seq, p_tokens], dim=1)

        time_ids = torch.arange(steps, device=z_seq.device)
        time_emb = self.time_embed(time_ids).unsqueeze(0)
        if z_seq.size(1) == steps:
            z_seq = z_seq + time_emb
        else:
            z_seq = z_seq + torch.cat([time_emb, time_emb], dim=1)

        memory = z_seq
        if self.temporal_block == "Transformer":
            for block in self.transformer:
                memory = block(memory)
        elif self.temporal_block == "MLP":
            for block in self.temporal_mlp:
                memory = block(memory)
        else:
            recurrent_memory, _ = self.temporal_rnn(memory)
            memory = self.temporal_rnn_norm(
                memory + self.temporal_rnn_dropout(recurrent_memory)
            )

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


__all__ = [
    "MLP",
    "GridCNNEncoder",
    "TemporalMLPBlock",
    "SpatialOnlyGraphSAGEBatch",
    "SpatioTemporalGraphSAGEBatch",
    "TransformerBlock",
    "PACT",
    "canonical_temporal_block",
]
