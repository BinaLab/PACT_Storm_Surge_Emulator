# PACT: Peak-Aware Cross-Attention Graph Transformer for Storm Surge Forecasting

This repository contains the PACT storm-surge emulator code. The current workflow is config-driven: shell launchers source a bash config file, fill safe defaults, and call `train.py` or `infer.py` with only the arguments supported by the Python entrypoint.

Core tasks:

- Load pre-built PyTorch Geometric forcing graphs.
- Train station-specific models on NCEP or CMIP6 graph roots.
- Run single-target inference from one checkpoint.
- Run sequential multi-target inference sweeps from one checkpoint.
- Save metrics in physical target units and optionally save prediction arrays.

## Project Layout

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
|-- train_shuffle_0429.sh
|-- train_future_only_0430.sh
|-- train_future_only_MPI_MRI_0430.sh
|-- emulator/
|   |-- common/
|   |   |-- distributed.py
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
|   |   |-- MRI/
|   |   `-- Cane5/
|   `-- configs_infer/
|       |-- infer_config_common.sh
|       |-- infer_config_NCEP.sh
|       |-- infer_config_AWI.sh
|       |-- infer_config_CNRM.sh
|       |-- infer_config_EC_EARTH.sh
|       |-- infer_config_MPI.sh
|       |-- infer_config_MRI.sh
|       |-- infer_config_NCEP_Battery_SpatialMLP_0h.sh
|       |-- infer_config_NCEP_Battery_TemporalCNN_12h.sh
|       |-- infer_config_NCEP_Battery_TemporalLSTM_12h.sh
|       |-- infer_multi_config_NCEP.sh
|       |-- infer_multi_config_AWI.sh
|       |-- infer_multi_config_CNRM.sh
|       |-- infer_multi_config_EC_EARTH.sh
|       |-- infer_multi_config_MPI.sh
|       `-- infer_multi_config_MRI.sh
|-- Inference_Checkpoints/
|-- checkpoints_Battery/
|-- station_json/
|-- preprocessing/
`-- Data/
    |-- Grid4_New/
    |   |-- NCEP/
    |   |-- CMIP6_AWI/
    |   |-- CMIP6_CNRM/
    |   |-- CMIP6_EC_EARTH/
    |   |-- CMIP6_MPI/
    |   |-- CMIP6_MRI/
    |   `-- CMIP6_Cane5/
    `-- Grid4_New_PastOnly/
```

## Models

Supported model names are:

- `spatial_mlp_0h`: spatial-only 0-hour baseline.
- `temporal_cnn_12h`: temporal CNN baseline with history.
- `temporal_lstm_12h`: temporal LSTM baseline with history.
- `baseline`: GraphSAGE baseline. With `HISTORY_HOURS=0`, it is spatial-only; with history, it uses the spatiotemporal baseline.
- `perceiver3`: PACT / Perceiver-style peak-aware cross-attention model.
- `perceiver3_cnn`: PACT variant with CNN temporal encoding.

The inference config must match the checkpoint architecture. Set `MODEL` and `HISTORY_HOURS` to the values used for that checkpoint.

## Environment

Run commands from the repository root:

```bash
cd /home/exouser/Documents/PACT-Data/Emulator
```

The launchers can activate conda automatically if `DO_CONDA=1` and `CONDA_SH` exists. The current inference common config expects:

```bash
CONDA_ENV="torchpyg-cu124"
CONDA_SH="/software/u22/anaconda/python3.9/etc/profile.d/conda.sh"
```

If you already activated the right environment, set `DO_CONDA=0` in the selected config after it sources the common config.

## Training

Run training with `train.sh` and a config file:

```bash
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

`train.sh` does not use `srun`. It launches directly with Python for one GPU, or with `python -m torch.distributed.run` when `num_gpus` is greater than 1. It works on a local/interactive GPU node and also inside an allocated Slurm session where `CUDA_VISIBLE_DEVICES` is already set.

Training logs are written under:

```text
launcher_logs_<STATION>/local_<timestamp>/
launcher_logs_<STATION>/idev_<SLURM_JOB_ID>_<timestamp>/
```

## Single Inference

Use `infer.sh` for one config and one target graph root. Always pass a config path.

```bash
bash infer.sh configs/configs_infer/infer_config_NCEP.sh
```

Other single-run examples:

```bash
bash infer.sh configs/configs_infer/infer_config_AWI.sh
bash infer.sh configs/configs_infer/infer_config_CNRM.sh
bash infer.sh configs/configs_infer/infer_config_EC_EARTH.sh
bash infer.sh configs/configs_infer/infer_config_MPI.sh
bash infer.sh configs/configs_infer/infer_config_MRI.sh
bash infer.sh configs/configs_infer/infer_config_NCEP_Battery_SpatialMLP_0h.sh
bash infer.sh configs/configs_infer/infer_config_NCEP_Battery_TemporalCNN_12h.sh
bash infer.sh configs/configs_infer/infer_config_NCEP_Battery_TemporalLSTM_12h.sh
```

`infer.sh` starts inference inside a detached tmux session. The command prints the session name, run directory, and runner path. Attach with the printed command:

```bash
tmux attach -t <session_name>
```

Outputs go to:

```text
logs_infer_<STATION>/<NAME>_<timestamp>/
|-- run_infer.sh
|-- infer_<STATION>_<test_tag>_<MODEL>_hist<HISTORY_HOURS>h_<timestamp>.log
`-- outputs/
    |-- metrics_per_year_<test_tag>_<station>_<model>.json
    `-- preds_<test_tag>_<station>_<model>_ALLYEARS.npz
```

The `.npz` file is saved because the launcher passes `--save_npz`. It contains `y_true`, `y_pred`, and `tags`. Metrics are reported in physical target units after denormalization.

Note: `infer.py` labels the test tag as `NCEP` only when `TEST_ROOT_DIR` is empty. Any nonempty `TEST_ROOT_DIR` is currently labeled `CMIP6` in output filenames, even if the path points to the NCEP graph root.

## Multi Inference Sweep

Use `infer_multi.sh` to evaluate one checkpoint over several target graph roots sequentially:

```bash
bash infer_multi.sh configs/configs_infer/infer_multi_config_NCEP.sh
```

Available multi configs:

```bash
bash infer_multi.sh configs/configs_infer/infer_multi_config_AWI.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_CNRM.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_EC_EARTH.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_MPI.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_MRI.sh
```

Unlike `infer.sh`, `infer_multi.sh` runs in the current shell and does not create a tmux session. It loops over the config's `RUNS` array:

```bash
RUNS=(
  "RunName_TO_NCEP|./Data/Grid4_New/NCEP/graphs"
  "RunName_TO_CMIP6_AWI|./Data/Grid4_New/CMIP6_AWI/graphs"
)
```

Each item has the form:

```text
<run_name>|<test_root_dir>
```

Use an empty test root to fall back to the NCEP year-split test from `ROOT_DIR`:

```bash
RUNS=(
  "RunName_NCEP_year_split|"
)
```

Multi-run outputs go to one directory per `RUNS` entry:

```text
logs_infer_<STATION>/<run_name>_<timestamp>/
|-- infer_config_used.sh
|-- infer_<STATION>_<MODEL>_hist<HISTORY_HOURS>h_<timestamp>.log
`-- outputs/
```

## Inference Config Fields

The inference launchers source `configs/configs_infer/infer_config_common.sh` first, then override values in the selected config. The most important fields are:

```bash
INFER_PY="infer.py"
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
STATION="Battery"
CKPT_PATH="./Inference_Checkpoints/NCEP_Battery_P3_Best.pth"
MODEL="perceiver3"
HISTORY_HOURS=12
BATCH_SIZE=1
STATION_JSON_DIR="./station_json"
YEARS="2094_2095, 2095_2096"
```

Field meanings:

- `ROOT_DIR`: training/NCEP root used for checkpoint-compatible stats and for NCEP year-split inference when `TEST_ROOT_DIR` is empty.
- `TEST_ROOT_DIR`: if nonempty, infer on all matching years from this graph root. This can be NCEP or any CMIP6 graph root.
- `CKPT_PATH`: checkpoint path. Glob patterns are allowed; the newest matching file is used.
- `MODEL`: one of the supported model names listed above.
- `HISTORY_HOURS`: history window expected by the checkpoint; must be a multiple of 6.
- `YEARS`: optional comma-separated year tags. Set `YEARS=""` to evaluate every available year in the selected test set.
- `USE_AMP`, `AMP_DTYPE`, `USE_TF32`, `TORCH_THREADS`: speed/runtime knobs.
- `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR`, `MP_CONTEXT`: DataLoader knobs.

Important: the current `infer_config_common.sh` sets `YEARS` to a small future-year subset. If you want all available years, edit the selected config or common config and set:

```bash
YEARS=""
```

## Direct Python Inference

The shell launchers are preferred, but `infer.py` can also be called directly:

```bash
python -u infer.py \
  --root_dir ./Data/Grid4_New/NCEP/graphs \
  --test_root_dir ./Data/Grid4_New/CMIP6_AWI/graphs \
  --station Battery \
  --station_json_dir ./station_json \
  --model perceiver3 \
  --history_hours 12 \
  --batch_size 1 \
  --ckpt ./Inference_Checkpoints/NCEP_Battery_P3_Best.pth \
  --out_dir infer_outputs \
  --save_npz \
  --amp \
  --amp_dtype bf16 \
  --tf32 \
  --torch_threads 1 \
  --num_workers 0 \
  --prefetch_factor 0 \
  --mp_context fork
```

## Data Notes

Graph roots are expected to contain `*graphs.pt` files. Typical roots are:

```text
./Data/Grid4_New/NCEP/graphs
./Data/Grid4_New/CMIP6_AWI/graphs
./Data/Grid4_New/CMIP6_CNRM/graphs
./Data/Grid4_New/CMIP6_EC_EARTH/graphs
./Data/Grid4_New/CMIP6_MPI/graphs
./Data/Grid4_New/CMIP6_MRI/graphs
./Data/Grid4_New/CMIP6_Cane5/graphs
```

For PACT/Perceiver models, station metadata is loaded from `station_json/<station>.json` when available.

## Quick Troubleshooting

- `CKPT_PATH does not resolve to a file`: update `CKPT_PATH` or confirm the glob matches at least one `.pth` file.
- `No test samples found`: check `ROOT_DIR`, `TEST_ROOT_DIR`, `STATION`, and `YEARS`.
- Missing station metadata warning: add the station JSON file or confirm the model can run without station features.
- Conda activation warning: update `CONDA_SH`, set `CONDA_ENV`, or set `DO_CONDA=0` in the selected config.
- Inference is only running a few years: clear or edit `YEARS` in the selected config/common config.
