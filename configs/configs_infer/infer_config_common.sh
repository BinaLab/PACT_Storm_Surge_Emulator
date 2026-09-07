#!/usr/bin/env bash

INFER_PY="infer.py"
ENCODER_TYPE="GraphSAGE"
# Empty means the CNN width saved in the checkpoint.
CNN_INTERMEDIATE_CHANNEL=""
TEMPORAL_BLOCK="Transformer"
HEAD_TYPE="dual"
BATCH_SIZE=1
STATION_JSON_DIR="./station_json"
YEARS=""

CUDA_LAUNCH_BLOCKING_FLAG=0
TORCH_GPU_PROBE=1
FAIL_IF_NO_SLURM=0

USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
TORCH_THREADS=1

NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

DO_CONDA=1
CONDA_MODULE="anaconda"
CONDA_SH="/software/u22/anaconda/python3.9/etc/profile.d/conda.sh"
CONDA_ENV="torchpyg-cu124"

# Human-readable checkpoint/model label used in inference run folder names.
# Override this in a model-specific config when evaluating another variant.
MODEL_LABEL="P3_Best"

# Relative paths are resolved from the directory where the launcher is run.
INFERENCE_RESULTS_ROOT="./All_Inference_Results"
