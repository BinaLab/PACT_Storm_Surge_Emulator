# Architecture changelog

## 2026-09-07 — Preparation for the core architecture study

Baseline: commit `a8cd656`. These changes apply to new training runs. Existing graph data, target timestamps, station features, and saved experimental results are unchanged. The two history-related changes below are separate design decisions; the CNN width adjustment is recorded afterward. All experiments will be retrained, so the final implementation intentionally removes the old encoding and implicit-width compatibility paths initially included in `45d9529`.

### Relative-to-present lag embeddings

PACT training and inference always use relative-to-present lag embeddings. For `T = history_hours / 6 + 1` forcing tokens, the embedding indices are `[T-1, ..., 1, 0]`:

| History | Chronological forcing sequence | Embedding indices |
| --- | --- | --- |
| 0h | `t` | `[0]` |
| 12h | `t-12h, t-6h, t` | `[2, 1, 0]` |
| 24h | `t-24h, ..., t` | `[4, 3, 2, 1, 0]` |
| 48h | `t-48h, ..., t` | `[8, ..., 1, 0]` |

Previously, indices started at the oldest token: `[0, ..., T-1]`. The current forcing therefore used embedding 2 with 12h history and embedding 8 with 48h history. The new convention fixes index 0 to the forecast origin and index 1 to six hours before it, across all history lengths.

Only the embedding lookup indices change. Tokens still enter MLP/LSTM/GRU/Transformer in chronological order. The embedding table size, spatial encoder interface, temporal blocks, and target window remain the same. `MAX_TIME_STEPS` stays fixed at 32 in the common configs; increasing history changes the number of tokens without increasing the model's total parameter count. Optional pressure tokens receive the same lag indices as their corresponding forcing tokens.

This preserves the physical meaning of overlapping lag indices. Independently trained models still learn their own embedding values; equal indices do not imply equal learned representations across runs.

The parameter is named `lag_embed.weight`. The sequence-position implementation and `TIME_ENCODING` / `--time_encoding` switches have been removed. Old `time_embed.weight` checkpoints fail strict loading; no key conversion is provided. New checkpoints and training/inference metadata record `time_encoding="relative_lag"` as a fixed architecture property, not a configurable mode.

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

The width remains configurable, with default 29. Inference requires `cnn_intermediate_channel` in CNN checkpoint arguments and never infers it from `hidden_channels`. The inference config field defaults to empty to read the saved width; an optional explicit value must match it. Training snapshots and checkpoint arguments record the width. Training/inference metadata record the resolved width, fixed lag encoding, and whether the zero-history query residual is active.

The removal of compatibility support does not change relative-lag predictions for newly trained models, CNN parameter counts, or the 0h-only query residual. It simplifies the supported architecture and makes retraining the path for old checkpoints.

### Validation and scope

Regression coverage checks CNN parameter counts/output shape, lag-index consistency and overlapping memory representations for all temporal families, the conditional decoder residual, and finite forward/backward passes across both encoders, all four temporal blocks, all nine history lengths, and both heads. It also covers current checkpoint inference, rejection of old embedding keys or missing CNN width, and 0h/12h CPU training -> checkpoint -> inference workflows.

All 23 regression tests passed using CPU fixtures. These checks do not measure training performance or predictive skill. The earlier compatibility comparison against the baseline implementation applied to `45d9529`; compatibility with those old configurations is no longer maintained.

This change does not generate or launch the 288-run experiment matrix, alter losses or head defaults, or add a validation-only search mode. The existing training entry point still evaluates test after training; the planned architecture-search workflow must configure its evaluation policy separately.
