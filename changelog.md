# Architecture changelog

## 2026-09-07 — Preparation for the core architecture study

Baseline: commit `a8cd656`. These changes apply to new training runs. Existing graph data, target timestamps, station features, and saved experimental results are unchanged. The two history-related changes below are separate design decisions; the CNN width adjustment is recorded afterward.

### Relative-to-present lag embeddings

New PACT training uses `TIME_ENCODING="relative_lag"` (`--time_encoding relative_lag`). For `T = history_hours / 6 + 1` forcing tokens, the embedding indices are `[T-1, ..., 1, 0]`:

| History | Chronological forcing sequence | Embedding indices |
| --- | --- | --- |
| 0h | `t` | `[0]` |
| 12h | `t-12h, t-6h, t` | `[2, 1, 0]` |
| 24h | `t-24h, ..., t` | `[4, 3, 2, 1, 0]` |
| 48h | `t-48h, ..., t` | `[8, ..., 1, 0]` |

Previously, indices started at the oldest token: `[0, ..., T-1]`. The current forcing therefore used embedding 2 with 12h history and embedding 8 with 48h history. The new convention fixes index 0 to the forecast origin and index 1 to six hours before it, across all history lengths.

Only the embedding lookup indices change. Tokens still enter MLP/LSTM/GRU/Transformer in chronological order. The embedding table size, spatial encoder interface, temporal blocks, and target window remain the same. `MAX_TIME_STEPS` stays fixed at 32 in the common configs; increasing history changes the number of tokens without increasing the model's total parameter count. Optional pressure tokens receive the same lag indices as their corresponding forcing tokens.

This preserves the physical meaning of overlapping lag indices. Independently trained models still learn their own embedding values; equal indices do not imply equal learned representations across runs.

For checkpoint compatibility, the tensor key remains `time_embed.weight`. `sequence_position` retains the old indexing, and inference selects it automatically when a checkpoint has no `time_encoding` argument. New checkpoints save the selected mode. Switching an old checkpoint to relative-lag indexing at inference would change its learned semantics, so an explicit mismatched inference setting is rejected.

Suggested methods wording:

> We index learned temporal embeddings by lag relative to the forecast origin, with lag zero denoting the current forcing. Input tokens remain chronologically ordered. This assigns consistent positional semantics to overlapping lags across history lengths without changing the embedding-table size.

### 0h instantaneous-forcing control

PACT training and inference now accept `history_hours=0`, using one forcing timestep at `t`. The predicted surge window remains `t` through `t+5h`. The station-conditioned readout, chosen temporal block, and prediction head remain present.

With one memory token and no pressure tokens, attention weights are necessarily one regardless of the horizon query. Without an additional query path, the shared prediction head receives the same context for all six horizons and produces the same normalized value. Horizon-specific normalization statistics can make the physical outputs differ, but do not resolve that loss of horizon information. The previous train/infer entry points rejected 0h PACT runs.

The new decoder applies the query residual **only when the number of forcing timesteps is one**:

```python
context, _ = forecast_attn(horizon_query, memory, memory)
if steps == 1:
    context = context + horizon_query
```

This reuses the existing horizon embeddings and adds no parameters. It applies to both prediction-head variants and all four temporal blocks. The condition refers to forcing timesteps even if optional pressure tokens are enabled. Graphs with only current forcing and no stored history can supply a one-step view for this control. The separate spatial baseline keeps its existing mean-pooling and linear-head architecture.

For 6–48h (`T >= 2`), the decoder receives no query residual. Those runs can still change numerically because of the independently selected lag encoding or CNN width; this statement concerns the decoder path specifically.

For paper figures, distinguish the main 6–48h history study from the 0h **instantaneous-forcing control**. The latter has a special decoder treatment and should not be described as an entirely identical architecture differing only in input length.

Suggested methods wording:

> We evaluate instantaneous forcing as a separate zero-history control. For this single-timestep case only, we add the horizon query to the cross-attention output to retain horizon-specific predictions. Models using 6–48h history retain the original decoder without this residual.

### CNN intermediate width and parameter budget

New training uses `CNN_INTERMEDIATE_CHANNEL=29` (`--cnn_intermediate_channel 29`). The setting controls CNN layers before the final spatial layer. The final layer still outputs `HIDDEN_CHANNELS` features at every grid point. With the study's two layers and 128-dimensional output:

```text
GraphSAGE: 5 -> 128 -> 128
CNN:       5 -> 29  -> 128
```

The CNN retains ordinary 3x3 convolutions, stride 1, padding 1, and the existing activation/dropout operations. For deeper CNNs, all intermediate spatial layers use the configured width; a one-layer CNN maps directly to the output dimension.

| Spatial encoder | Trainable parameters, including biases |
| --- | ---: |
| GraphSAGE, two layers, output 128 | 34,304 |
| CNN, old intermediate width 128 | 153,472 |
| CNN, new intermediate width 29 | 34,870 |

For these dimensions, `P_CNN(c) = 1198*c + 128`. Width 29 is the nearest integer match to the GraphSAGE budget, leaving 566 additional parameters (+1.65%). It is selected from parameter counts rather than validation scores. This is a comparison with matched depth/output dimension and approximately matched encoder parameter budgets; internal widths and neighborhood shapes remain different.

Old CNN checkpoints without `cnn_intermediate_channel` use their saved `hidden_channels` for the intermediate layers, reproducing the old structure. Inference config fields for CNN width and time encoding default to empty, allowing checkpoint-driven resolution. Optional explicit values must match the checkpoint. Training snapshots, checkpoint arguments, and inference metadata record the new settings; metadata also identifies when the zero-history query residual is active.

### Validation and scope

Regression coverage checks CNN parameter counts/output shape, lag-index consistency and overlapping memory representations for all temporal families, the conditional decoder residual, and finite forward/backward passes across both encoders, all four temporal blocks, all nine history lengths, and both heads. It also covers legacy checkpoint inference and new 0h/12h CPU training -> checkpoint -> inference workflows.

All 22 regression tests passed on CPU. A separate comparison against the baseline implementation strictly loaded the original state dictionaries and produced bitwise-identical outputs in 388 forward comparisons when using legacy CNN widths and sequence-position encoding. This verifies compatibility on the tested inputs; it does not measure training performance or predictive skill.

This change does not generate or launch the 288-run experiment matrix, alter losses or head defaults, or add a validation-only search mode. The existing training entry point still evaluates test after training; the planned architecture-search workflow must configure its evaluation policy separately.
