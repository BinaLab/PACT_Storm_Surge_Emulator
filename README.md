# PACT Storm-Surge Emulator

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PACT is a config-driven PyTorch Geometric workflow for station-level storm-surge forecasting. It trains and evaluates graph neural emulators on NCEP and CMIP6 forcing graphs, with launchers for single-checkpoint evaluation and multi-dataset sweeps.

The repository currently supports two model families:

| `MODEL` | Use case |
| --- | --- |
| `baseline` | Spatial baseline using the selected GraphSAGE/CNN encoder. With `HISTORY_HOURS=0`, it is spatial only; with history, it adds an LSTM temporal head. |
| `perceiver3` | PACT, a peak-aware cross-attention model using the selected spatial encoder and temporal block for history-aware surge forecasting. |

## Authors and Collaboration

- Author: Zesheng Liu
- Developed in collaboration with Doyup Kwon (Princeton), Ning Lin (Princeton), and Maryam Rahnemoonfar (Lehigh University)
- Corresponding Author: Maryam Rahnemoonfar (maryam@lehigh.edu)

## Repository Layout

```text
Emulator/
|-- README.md
|-- environment_training.yml
|-- environment_dataprep.yml
|-- train.py
|-- train.sh
|-- infer.py
|-- infer.sh
|-- infer_multi.sh
|-- emulator/
|   |-- common/
|   |   |-- distributed.py
|   |   |-- inference_artifacts.sh
|   |   |-- io_utils.py
|   |   `-- runtime.py
|   |-- data/
|   |   |-- graph_store.py
|   |   |-- loaders.py
|   |   |-- normalization.py
|   |   |-- station_metadata.py
|   |   `-- stats.py
|   |-- inference/
|   |   |-- engine.py
|   |   `-- grouping.py
|   |-- models/
|   |   `-- architectures.py
|   `-- training/
|       |-- engine.py
|       `-- losses.py
|-- configs/
|   |-- configs_train/
|   |   |-- train_config_common.sh
|   |   |-- NCEP/
|   |   |-- AWI/
|   |   |-- CNRM/
|   |   |-- EC_EARTH/
|   |   |-- MPI/
|   |   `-- MRI/
|   |-- configs_train_single/  # same experiment layout, single GPU with accumulation
|   `-- configs_infer/
|       |-- infer_config_common.sh
|       |-- infer_config_NCEP.sh
|       |-- infer_config_AWI.sh
|       |-- infer_config_CNRM.sh
|       |-- infer_config_EC_EARTH.sh
|       |-- infer_config_MPI.sh
|       |-- infer_config_MRI.sh
|       |-- infer_multi_config_NCEP.sh
|       |-- infer_multi_config_AWI.sh
|       |-- infer_multi_config_CNRM.sh
|       |-- infer_multi_config_EC_EARTH.sh
|       |-- infer_multi_config_MPI.sh
|       `-- infer_multi_config_MRI.sh
|-- Inference_Checkpoints/
|-- station_json/
|-- preprocessing/
|-- tests/
`-- Data/
    |-- Grid4_New/
    |   |-- NCEP/
    |   |-- CMIP6_AWI/
    |   |-- CMIP6_CNRM/
    |   |-- CMIP6_EC_EARTH/
    |   |-- CMIP6_MPI/
    |   `-- CMIP6_MRI/
    `-- Grid4_New_PastOnly/
```

## Environment

Run commands from your checkout root. The YAML files are environment exports with cluster-specific names. In particular, `environment_training.yml` pins Python 3.11, PyTorch 2.8.0 with CUDA 12.8, and PyG 2.7.0; it does not describe the `torchpyg-cu124` environment named in the launcher configs.

To recreate the exported training environment on Linux, choose an explicit environment name and provide the wheel sources for its pinned CUDA packages:

```bash
cd /path/to/PACT_Storm_Surge_Emulator
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.8.0+cu128.html \
  conda env create -n pact-cu128 -f environment_training.yml
conda activate pact-cu128
```

The wheel sources correspond to the [PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/#v280) and [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html). The export targets its original Linux/CUDA setup; adapt the package versions together if using another platform or CUDA build. `environment_dataprep.yml` records the separate preprocessing environment, including pandas, xarray, and netCDF4.

The launchers can activate conda when `DO_CONDA=1`. Their shared configs currently use these machine-specific values:

```bash
CONDA_ENV="torchpyg-cu124"
CONDA_SH="/software/u22/anaconda/python3.9/etc/profile.d/conda.sh"
```

Set `CONDA_ENV` to the environment you created (for example, `pact-cu128`) and `CONDA_SH` to your conda installation, or activate your environment and set `DO_CONDA=0` in the selected config **after** it sources the common config. The common configs assign these values directly, so prefixing a launch command with `DO_CONDA=0` alone does not override them.

The shell launchers use Bash and Linux utilities. `infer.sh` requires tmux; training uses tmux by default when it is available.

## Data

Training and evaluation expect graph roots containing `*graphs.pt` files. Graph data and checkpoint binaries are excluded from Git. Populate `Data/` (or link it to your existing data directory), and supply a checkpoint at `CKPT_PATH` before inference. The standard configured graph roots are:

```text
./Data/Grid4_New/NCEP/graphs
./Data/Grid4_New/CMIP6_AWI/graphs
./Data/Grid4_New/CMIP6_CNRM/graphs
./Data/Grid4_New/CMIP6_EC_EARTH/graphs
./Data/Grid4_New/CMIP6_MPI/graphs
./Data/Grid4_New/CMIP6_MRI/graphs
```

Each graph file contains a list of PyG `Data` objects and is named `<year0>_<year1>_<station>_<version>_graphs.pt`. Station filtering matches the filename's station field, including case. Inference warns and falls back to all stations if the requested station is absent; direct external evaluation can use `--strict_station_test` to reject that case. PACT also reads optional station metadata from `station_json/<station>.json`.

### Time windows and preprocessing

NCEP forcing is sampled every 6 hours. CMIP6 preprocessing selects every other 3-hour record to produce the same 6-hour cadence. For a forcing cutoff `t`, each graph targets six hourly surge values **`t, t+1h, ..., t+5h`**. The next graph covers `t+6h` through `t+11h`, so the outputs concatenate without overlap. For example, forcing through 00:00 produces surge for 00:00–05:00; forcing through 06:00 produces 06:00–11:00.

The alignment script uses separate input origins: forcing starts at Oct 25 00:00, while hourly station records start at Oct 25 01:00. Graph centers begin at Nov 1 00:00 on both axes, with earlier forcing retained for history; the different recording origins do not shift the labels. CSV `time` values are validated against the hourly convention when present; legacy files without that column use the assumed axis with a warning.

Current `hist48` graphs store forcing through the center time and up to 48 hours of history. A 12-hour model uses three slices (`t-12h`, `t-6h`, `t`); a 0-hour baseline uses only `t`. Select one graph version per year/station in each root. When regenerating data, set the source paths in the forcing scripts and the manual `year_list` in `preprocessing/preprocessing_simulation.py` (currently `[2005]`), then use `preprocessing/time_align_unified.py` to build aligned graphs.

## Model Options

The spatial encoder is selected with `ENCODER_TYPE="GraphSAGE"` or `ENCODER_TYPE="CNN"`. The CNN path reads `grid_H` and `grid_W` from each graph, reshapes flattened node features from `(H*W, F)` to `(F, H, W)`, applies same-resolution 3x3 convolutions, and returns one `hidden_channels` token per grid node. No grid dimensions are hardcoded in the model or launcher.

`NUM_LAYERS` defaults to 2. GraphSAGE requires at least 2 layers and rejects 0 or 1 explicitly; CNN continues to support 1 or more layers.

For PACT, `TEMPORAL_BLOCK` selects the middle sequence processor: `Transformer`, `MLP`, `LSTM`, or `GRU` (`attn` is accepted as an alias for `Transformer`). Every option receives time embeddings and maps `(B,L,hidden_channels)` back to the same shape, where `L=T` normally; the final horizon-query cross-attention is unchanged. `TRANSFORMER_LAYERS` controls the depth of every temporal option, while `TRANSFORMER_FF_MULT` only affects Transformer/MLP. The MLP option is token-wise, so the final horizon cross-attention performs its cross-time aggregation.

`HEAD_TYPE` selects PACT's final prediction head. `single` sends each horizon context directly through the existing base MLP. `dual` is the backward-compatible default and retains the gated tail correction: `y = y_base + gate * sigmoid(alpha_logit) * r_tail`. Both variants consume the same `c_flat` after optional global p_mean concatenation and use the same base-head width and `HEAD_DROPOUT`; `GATE_MODE`, `GATE_BIAS_INIT`, `TAIL_TANH_CLIP`, and `ALPHA_INIT_LOGIT` only affect `dual`.

> **p_mean note:** shipped configs use `USE_PMEAN=0`. With token-mode p_mean enabled, the historical layout is `[forcing tokens for T steps][p_mean tokens for T steps]`; LSTM/GRU therefore process a sequence of length `2T`. If this combination is used later, consider time-wise fusion or interleaving as a separate modeling choice.

When a baseline enables `USE_PMEAN=1` and its batch lacks the required pressure field (`p_mean_curr` for the spatial baseline, `p_mean_hist` for the temporal baseline), it warns and concatenates a zero pressure embedding. It retains the checkpoint's architecture; it does not feed an artificial pressure value of zero through the pressure encoder. Existing pressure fields use the original encoding path. PACT retains its existing missing-pressure behavior: omit pressure tokens and use a zero global embedding when that branch is enabled. These fallbacks support single-process training/evaluation; missing-field training leaves pressure-encoder parameters unused, so the current DDP configuration with `find_unused_parameters=False` is not supported for that case. Mixed samples with different metadata keys within one PyG batch are outside this fallback.

## Training

Training is driven by `train.sh` plus a bash config. For the current single-GPU workflow:

```bash
bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

Useful examples:

```bash
# PACT on NCEP Battery
bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_P3_Best.sh

# 0-hour GraphSAGE baseline
bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_Baseline_0h.sh

# 12-hour GraphSAGE plus LSTM baseline
bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_Baseline_12h.sh
```

The common config in `configs/configs_train_single/` selects `num_gpus=1` with gradient accumulation; the original `configs/configs_train/` tree selects `num_gpus=4`. `train.sh` uses Python for one process and `python -m torch.distributed.run` for multiple processes. Under Slurm, `SLURM_NTASKS_PER_NODE` overrides the configured process count; the launcher also limits it to the count in `CUDA_VISIBLE_DEVICES` when that variable is set.

By default, training starts a detached tmux session when tmux is installed and prints an attach command. To run in the current shell:

```bash
USE_TMUX=0 bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

### Training, validation, and test splits

Splits operate on whole winter groups such as `1979_1980`. With `SHUFFLE_YEARS=0` (the normal setting), groups are sorted chronologically before applying `TRAIN_RATIO` and `VAL_RATIO`; the remainder is test. Counts are rounded with at least one group reserved for each split when there are at least three groups. For the standard data inventories:

| Config / graph root | Train ratio | Validation ratio | Train / validation / test winters |
| --- | --- | --- | --- |
| NCEP, 36 winters | 0.6 | 0.2 | 22 / 7 / 7 |
| Standard CMIP6, 66 winters | 0.4545454545 | 0.0909090909 | 30 / 6 / 30 |
| PastOnly, 36 winters | 0.6 | 0.2 | 22 / 7 / 7 |

The standard CMIP6 split uses historical winters for training/validation and future winters for test. Direct `train.py` defaults to `0.6/0.2`, so pass the CMIP6 ratios explicitly when reproducing those configs. `SHUFFLE_YEARS=1` randomizes whole winter groups with `SEED` before splitting; `FUTURE_ONLY=1` first retains groups with either year greater than `FUTURE_YEAR_THRESHOLD` (default 2030). These are separate experiment designs. PastOnly configs select a different graph root.

Training batches are shuffled within the selected training split even when `SHUFFLE_YEARS=0`; validation and test loaders do not shuffle. Normalization statistics and loss thresholds are fitted on training samples. Validation selects the best checkpoint, which is reloaded for the final test evaluation.

### Single-GPU gradient accumulation

Gradient accumulation is disabled by default (`GRAD_ACCUM_STEPS=1`) and is intentionally restricted to `world_size=1`. The configs under `configs/configs_train_single/` mirror `configs/configs_train/` while requesting a nominal effective batch size of 1024 without changing the microbatch size or learning rate:

```bash
num_gpus=1
BATCH_SIZE=256
GRAD_ACCUM_STEPS=4
```

For example:

```bash
bash train.sh configs/configs_train_single/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

Each accumulation group averages its local-batch losses equally before the optimizer update. A final group containing only `k` microbatches divides by `k`, not by the configured accumulation value and not by raw sample count. This reproduces the existing 4-GPU DDP semantics as closely as possible under the current data-loading design, but is not expected to be numerically or sample-grouping identical to 4-rank DDP, particularly for batch-dependent tail losses. Values greater than 1 are rejected for multi-process DDP; no DDP `no_sync()` path is used.

Training artifacts launched through `train.sh` are kept together under:

```text
All_Results/<timestamp>_<run-name>/
|-- config_used.sh
|-- train_<configuration>.log
|-- best_<configuration>.pth
|-- meta_<configuration>.json
|-- summary_<configuration>.npz
`-- test_preds_<configuration>.npz
```

`run-name` is the qsub/tmux session name when one is supplied, otherwise it is
derived from the training config filename. `config_used.sh` contains the fully
resolved configuration after the common config, selected config, defaults, and
runtime overrides have been applied. A config containing a parameter sweep
writes all of its uniquely named artifacts into the same run folder.

Direct calls to `train.py` save model and metric artifacts in the same layout, but do not create a launcher log or tmux session. If `--output_dir` is omitted,
the script creates `All_Results/<timestamp>_<run-name>/` automatically and
writes a `config_used.sh` record of all parsed arguments. This direct-Python record uses CLI field names and is not a drop-in sweep config for `train.sh`. For example,
`--run_tag manual_test` produces a directory shaped like:

```text
All_Results/20260906_223000_manual_test/
```

## Evaluation

Use `infer.sh` for one config and one target graph root. Always pass a config path.

```bash
bash infer.sh configs/configs_infer/infer_config_NCEP.sh
```

The dataset in each single-run config filename identifies the **checkpoint source**. All six shipped single-run configs currently set `TEST_ROOT_DIR` to CMIP6_EC_EARTH, so this example evaluates NCEP → CMIP6_EC_EARTH. Set `TEST_ROOT_DIR=""` in the selected config for the source's held-out test split, or set another root for all-year evaluation there.

Other single-run examples:

```bash
bash infer.sh configs/configs_infer/infer_config_AWI.sh
bash infer.sh configs/configs_infer/infer_config_CNRM.sh
bash infer.sh configs/configs_infer/infer_config_EC_EARTH.sh
bash infer.sh configs/configs_infer/infer_config_MPI.sh
bash infer.sh configs/configs_infer/infer_config_MRI.sh
```

`infer.sh` starts inference inside a detached tmux session. The command prints the session name, run directory, and runner path. Attach with the printed command:

```bash
tmux attach -t SESSION_NAME  # replace with the printed session name
```

Outputs go to:

```text
All_Inference_Results/<Station>_<ModelLabel>_<Source>_To_<Target>_<timestamp>/
|-- run_infer.sh
|-- infer_config_used.sh
|-- command_used.sh
|-- infer_<STATION>_<target>_<MODEL>_hist<HISTORY_HOURS>h_<timestamp>.log
`-- outputs/
    |-- metrics_per_year_<test_tag>_<station>_<model>.json
    `-- preds_<test_tag>_<station>_<model>_ALLYEARS.npz
```

For example, the shipped AWI single-run config creates
`All_Inference_Results/Battery_P3_Best_CMIP6_AWI_To_CMIP6_EC_EARTH_<timestamp>/`.
`Source` and `Target` are inferred from the graph-root directory names, while
`ModelLabel` comes from `MODEL_LABEL` in the inference config. NCEP is named
`NCEP`; the other current datasets retain their full `CMIP6_AWI`, `CMIP6_CNRM`,
`CMIP6_EC_EARTH`, `CMIP6_MPI`, or `CMIP6_MRI` directory names.

The `.npz` file contains `y_true` and `y_pred` of shape `(n_graphs, 6)` for the standard graphs, plus one `tag` per row in `tags`. Predictions are denormalized before reporting, so RMSE and MAE are in physical target units. Both shell launchers save NPZ output; direct Python evaluation does so only with `--save_npz`. The `ALLYEARS` suffix means all **evaluated** years, including any `YEARS` restriction.

For time-series reconstruction, group tags by winter and station, then use the final integer in each tag as the block index: hourly position is `6 * block_index + horizon_index` within that winter. This also handles older exports whose rows were shuffled. Keep `tags`, `y_true`, and `y_pred` together when reordering rows.

Metrics JSON records `evaluation_scope` and `years_evaluated`. `_overall` aggregates all selected samples/horizons; `_overall_past` covers winter start years 1979–2014 and `_overall_future` covers 2070–2099. These reporting ranges are fixed independently of `FUTURE_YEAR_THRESHOLD`. The average-year timing summary excludes `2014_2015`; its error metrics and predictions are still included when that winter is selected.

`infer_config_used.sh` stores the resolved common/selected config values, launcher defaults, applicable environment values, absolute input paths, and the selected checkpoint path. It can be sourced without the original config files. The tmux runner reads this snapshot and does not re-resolve the checkpoint glob. `command_used.sh` records the actual interpreter, arguments, working directory, and selected CUDA environment values; run `bash <run-directory>/command_used.sh` to repeat that command with the same output paths. Existing outputs at those paths may be replaced. The metrics JSON also records `inference_args` and `checkpoint_args`, including for direct Python invocations.

CNN output filenames append `_CNN` after the model name, non-Transformer PACT outputs append their temporal block name, and single-head PACT outputs append `_single`. GraphSAGE + Transformer + dual keeps the historical filenames unchanged.

## Multi-Target Evaluation

Use `infer_multi.sh` to evaluate one checkpoint over several target graph roots sequentially:

```bash
bash infer_multi.sh configs/configs_infer/infer_multi_config_NCEP.sh
```

Available sweep configs:

```bash
bash infer_multi.sh configs/configs_infer/infer_multi_config_AWI.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_CNRM.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_EC_EARTH.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_MPI.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_MRI.sh
```

Each multi-run config defines a `RUNS` array:

```bash
RUNS=(
  "NCEP_Battery_P3_Best_TO_NCEP|./Data/Grid4_New/NCEP/graphs"
  "NCEP_Battery_P3_Best_TO_CMIP6_AWI|./Data/Grid4_New/CMIP6_AWI/graphs"
)
```

Each item has the form:

```text
<run_name>|<test_root_dir>
```

The first field remains available as a descriptive/config label. Output folder
names are derived from `STATION`, `MODEL_LABEL`, `ROOT_DIR`, and the target graph
path, so stale legacy labels cannot put a run in a misleading directory.

Use an empty test root to evaluate the held-out year split from `ROOT_DIR`. Inference reuses the checkpoint's training/validation ratios, year shuffle, seed, and future-only filter:

```bash
RUNS=(
  "NCEP_Battery_P3_Best_YEAR_SPLIT|"
)
```

Unlike `infer.sh`, `infer_multi.sh` runs in the current shell and does not create a tmux session. Multi-run outputs go to one directory per `RUNS` entry:

```text
All_Inference_Results/<Station>_<ModelLabel>_<Source>_To_<Target>_<timestamp>/
|-- infer_config_used.sh
|-- command_used.sh
|-- infer_<STATION>_<MODEL>_hist<HISTORY_HOURS>h_<timestamp>.log
`-- outputs/
```

Each multi-target snapshot records only its own target in `RUNS`. Pass that snapshot to either launcher to create another run for that target, or use `command_used.sh` to repeat its exact command at the original output location. Snapshots preserve configuration and paths; they do not copy model weights, datasets, source code, or the conda environment.

## Key Config Fields

Training configs source the `train_config_common.sh` in their own tree (`configs_train` or `configs_train_single`). Evaluation configs source `configs/configs_infer/infer_config_common.sh`. Put experiment-specific overrides after the `source` line. An inference config includes:

```bash
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/CMIP6_AWI/graphs"
STATION="Battery"
MODEL="perceiver3"
MODEL_LABEL="P3_Best"
ENCODER_TYPE="GraphSAGE"
TEMPORAL_BLOCK="Transformer"
HEAD_TYPE="dual"
HISTORY_HOURS=12
CKPT_PATH="./Inference_Checkpoints/NCEP_Battery_P3_Best.pth"
BATCH_SIZE=1
YEARS=""
STATION_JSON_DIR="./station_json"
INFERENCE_RESULTS_ROOT="./All_Inference_Results"
```

Field notes:

- `ROOT_DIR`: source graph root used for training. Inference uses it to reconstruct a same-source split when `TEST_ROOT_DIR` is empty and to name the source dataset; normalization statistics are loaded from the checkpoint.
- `TEST_ROOT_DIR`: optional root for all-year evaluation. If empty, inference reconstructs the checkpoint's held-out year split from `ROOT_DIR`. Setting it to the source root includes training/validation years; those results describe full-source performance and must not be reported as held-out test performance. The shipped multi-target configs use all-year evaluation.
- `MODEL`: either `baseline` or `perceiver3`.
- `MODEL_LABEL`: human-readable model/checkpoint label used in run folder names, for example `P3_Best`.
- `INFERENCE_RESULTS_ROOT`: common parent directory for every `infer.sh` and `infer_multi.sh` run.
- `ENCODER_TYPE`: either `GraphSAGE` (the backward-compatible default) or `CNN`.
- `TEMPORAL_BLOCK`: PACT middle block: `Transformer` (backward-compatible default), `MLP`, `LSTM`, or `GRU`.
- `HEAD_TYPE`: PACT prediction head: `dual` (backward-compatible gated tail head) or `single` (base MLP only).
- `HISTORY_HOURS`: inference history window; use `HISTORY_HOURS_LIST=(12)` for a training config. Choose a nonnegative multiple of 6 within the stored history (currently at most 48h). PACT requires a positive history; the baseline also supports 0h. For inference, match the checkpoint's training window.
- `CKPT_PATH`: checkpoint path. Glob patterns are allowed; the newest matching file is used.
- `YEARS`: optional comma-separated winter tags, for example `"2008_2009,2009_2010"`. Leave empty to evaluate every available year in the selected evaluation population. This filter applies to per-year and overall metrics whether or not NPZ output is enabled; it does not expand a held-out split to include other years.
- `USE_AMP`, `AMP_DTYPE`, `USE_TF32`, `TORCH_THREADS`: speed/runtime knobs.
- `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR`, `MP_CONTEXT`: DataLoader knobs.

## Direct Python Evaluation

The shell launchers are preferred, but `infer.py` can also be called directly:

```bash
python -u infer.py \
  --root_dir ./Data/Grid4_New/NCEP/graphs \
  --test_root_dir ./Data/Grid4_New/CMIP6_AWI/graphs \
  --station Battery \
  --station_json_dir ./station_json \
  --model perceiver3 \
  --model_label P3_Best \
  --encoder_type GraphSAGE \
  --temporal_block Transformer \
  --head_type dual \
  --history_hours 12 \
  --batch_size 1 \
  --ckpt ./Inference_Checkpoints/NCEP_Battery_P3_Best.pth \
  --save_npz \
  --amp \
  --amp_dtype bf16 \
  --tf32 \
  --torch_threads 1 \
  --num_workers 0 \
  --prefetch_factor 0 \
  --mp_context fork
```

With `--out_dir` omitted, this direct call creates
`All_Inference_Results/Battery_P3_Best_NCEP_To_CMIP6_AWI_<timestamp>/outputs/`.
`--model_label` may also be omitted for canonical checkpoint filenames such as
`NCEP_Battery_P3_Best.pth`; in that case `P3_Best` is inferred automatically.
Pass `--out_dir <exact-directory>` only when an explicit output location is desired.

Direct `infer.py` calls can omit `--model`, `--encoder_type`, `--temporal_block`, `--head_type`, and `--history_hours` to use checkpoint settings. The shell launchers explicitly pass their configured values, so keep those consistent with the selected checkpoint. Direct calls write metrics and optional predictions; the shell config/command snapshots are produced by the launchers.

## Tests

Run the regression suite from the repository root in an environment with PyTorch, PyG, NumPy, and pandas (pandas is needed for the alignment tests and is absent from the training export):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The suite uses synthetic CPU data and mocked inference launchers to check split recovery, metric filtering, configuration replay, model inputs, missing pressure fields, and timestamp validation.

## Troubleshooting

- `CKPT_PATH does not resolve to a file`: update `CKPT_PATH` or confirm the glob matches at least one `.pth` file.
- `No test samples found`: check `ROOT_DIR`, `TEST_ROOT_DIR`, `STATION`, and `YEARS`.
- Missing station metadata warning: add the station JSON file or confirm the model can run without station features.
- Conda activation warning: update `CONDA_SH`, set `CONDA_ENV`, or set `DO_CONDA=0` in the selected config.
- Inference is only running a few years: check `YEARS` and the logged evaluation scope. With an empty `TEST_ROOT_DIR`, only the checkpoint's held-out split is evaluated even when `YEARS` is empty.
- Single-GPU launch unexpectedly requests DDP: use `configs_train_single`, and check `SLURM_NTASKS_PER_NODE` and `CUDA_VISIBLE_DEVICES`.
- `No matching distribution` for CUDA/PyG packages: use wheel sources matching the versions pinned in the environment export.

## License

This repository is released under the [MIT License](LICENSE).
