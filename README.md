# PACT Storm-Surge Emulator

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PACT is a config-driven PyTorch Geometric workflow for station-level storm-surge forecasting. It trains and evaluates graph neural emulators on NCEP and CMIP6 forcing graphs, with launchers for single-checkpoint evaluation and multi-dataset sweeps.

The repository currently supports two model families:

| `MODEL` | Use case |
| --- | --- |
| `baseline` | Spatial baseline using the selected GraphSAGE/CNN encoder. With `HISTORY_HOURS=0`, it is spatial only; with history, it adds an LSTM temporal head. |
| `perceiver3` | PACT, a peak-aware cross-attention transformer using the selected spatial encoder for history-aware surge forecasting. |

## Authors and Collaboration

- Author: Zesheng Liu
- Developed in collaboration with Doyup Kwon(Princeton), Ning Lin(Princeton), and Maryam Rahnemoonfar(Lehigh University)
- Corresponding Author: Maryam Rahnemoonfar(maryam@lehigh.edu)

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

Create the training environment from the repository root:

```bash
cd /home/exouser/Documents/PACT-Data/Emulator
conda env create -f environment_training.yml
conda activate torchpyg-cu124
```

The launchers can activate conda for you when `DO_CONDA=1`. The shared config files currently expect:

```bash
CONDA_ENV="torchpyg-cu124"
CONDA_SH="/software/u22/anaconda/python3.9/etc/profile.d/conda.sh"
```

If the environment is already active, set `DO_CONDA=0` in the selected config, or override it at launch time when appropriate.

## Data

Training and evaluation expect graph roots containing `*graphs.pt` files. Standard roots in this repository are:

```text
./Data/Grid4_New/NCEP/graphs
./Data/Grid4_New/CMIP6_AWI/graphs
./Data/Grid4_New/CMIP6_CNRM/graphs
./Data/Grid4_New/CMIP6_EC_EARTH/graphs
./Data/Grid4_New/CMIP6_MPI/graphs
./Data/Grid4_New/CMIP6_MRI/graphs
```

Each graph file is filtered by station name when `STATION` or `--station` is set. PACT also reads optional station metadata from `station_json/<station>.json`.

The spatial encoder is selected with `ENCODER_TYPE="GraphSAGE"` or `ENCODER_TYPE="CNN"`. The CNN path reads `grid_H` and `grid_W` from each graph, reshapes flattened node features from `(H*W, F)` to `(F, H, W)`, applies same-resolution 3x3 convolutions, and returns one `hidden_channels` token per grid node. No grid dimensions are hardcoded in the model or launcher.

## Training

Training is driven by `train.sh` plus a bash config:

```bash
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

Useful examples:

```bash
# PACT on NCEP Battery
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh

# 0-hour GraphSAGE baseline
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_Baseline_0h.sh

# 12-hour GraphSAGE plus LSTM baseline
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_Baseline_12h.sh
```

`train.sh` launches directly with Python for one GPU. For multi-GPU runs, set `num_gpus` in the config; the launcher uses `python -m torch.distributed.run`.

Training logs are written under:

```text
launcher_logs_<STATION>/local_<timestamp>/
launcher_logs_<STATION>/idev_<SLURM_JOB_ID>_<timestamp>/
```

Checkpoints and metrics are written to station-specific output folders such as:

```text
checkpoints_Battery/
results_Battery/
```

## Evaluation

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

The `.npz` file contains `y_true`, `y_pred`, and `tags`. Metrics are denormalized before reporting, so RMSE and MAE are in physical target units.
CNN output filenames append `_CNN` after the model name; GraphSAGE keeps the historical filenames unchanged.

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

Use an empty test root to fall back to the NCEP year-split test from `ROOT_DIR`:

```bash
RUNS=(
  "NCEP_Battery_P3_Best_YEAR_SPLIT|"
)
```

Unlike `infer.sh`, `infer_multi.sh` runs in the current shell and does not create a tmux session. Multi-run outputs go to one directory per `RUNS` entry:

```text
logs_infer_<STATION>/<run_name>_<timestamp>/
|-- infer_config_used.sh
|-- infer_<STATION>_<MODEL>_hist<HISTORY_HOURS>h_<timestamp>.log
`-- outputs/
```

## Key Config Fields

Training configs source `configs/configs_train/train_config_common.sh`. Evaluation configs source `configs/configs_infer/infer_config_common.sh`. The most important fields are:

```bash
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/CMIP6_AWI/graphs"
STATION="Battery"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
HISTORY_HOURS=12
CKPT_PATH="./Inference_Checkpoints/NCEP_Battery_P3_Best.pth"
BATCH_SIZE=1
YEARS=""
STATION_JSON_DIR="./station_json"
```

Field notes:

- `ROOT_DIR`: graph root used for training, validation, and checkpoint-compatible statistics.
- `TEST_ROOT_DIR`: optional external evaluation root. If empty, inference uses the NCEP year-split test set from `ROOT_DIR`.
- `MODEL`: either `baseline` or `perceiver3`.
- `ENCODER_TYPE`: either `GraphSAGE` (the backward-compatible default) or `CNN`.
- `HISTORY_HOURS`: history window expected by the model or checkpoint. It must be a multiple of 6.
- `CKPT_PATH`: checkpoint path. Glob patterns are allowed; the newest matching file is used.
- `YEARS`: optional comma-separated year tags. Leave empty to evaluate every available year.
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
  --encoder_type GraphSAGE \
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

## Troubleshooting

- `CKPT_PATH does not resolve to a file`: update `CKPT_PATH` or confirm the glob matches at least one `.pth` file.
- `No test samples found`: check `ROOT_DIR`, `TEST_ROOT_DIR`, `STATION`, and `YEARS`.
- Missing station metadata warning: add the station JSON file or confirm the model can run without station features.
- Conda activation warning: update `CONDA_SH`, set `CONDA_ENV`, or set `DO_CONDA=0` in the selected config.
- Inference is only running a few years: clear or edit `YEARS` in the selected config/common config.



## License

This repository is released under the [MIT License](LICENSE).
