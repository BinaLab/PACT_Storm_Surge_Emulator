#!/usr/bin/env bash
# configs/infer_config.sh
#
# This file is "source"d by infer.sh and infer_sbatch.sh.
# Keep it pure bash: assign variables; avoid running commands.

NAME="NCEP_Battery_P3_Best_AUG_TO_NCEP"

# =========================
# Data
# =========================
ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR="./Data/Grid4_New/NCEP/graphs"   # empty => NCEP year-split test
STATION="Boston"

# =========================
# Inference entrypoint + checkpoint
# =========================
INFER_PY="infer.py"
CKPT_PATH="./Inference_Checkpoints/NCEP_Battery_P3_Best_Aug.pth"   # can be a file OR a glob pattern

# =========================
# Model args (must match infer.py flags)
# =========================
MODEL="perceiver3"          # "" | baseline | perceiver3
HISTORY_HOURS=12            # -1 uses ckpt args; otherwise override
BATCH_SIZE=1
STATION_JSON_DIR="./station_json"

# Optional: only run selected years (comma-separated). Empty => all available years.
YEARS=""

# =========================
# Debug knobs (used by infer.sh / infer_sbatch.sh wrapper)
# =========================
CUDA_LAUNCH_BLOCKING_FLAG=0
TORCH_GPU_PROBE=1
FAIL_IF_NO_SLURM=0

# =========================
# Speed knobs (mapped to infer.py flags)
# =========================
USE_AMP=1
AMP_DTYPE="bf16"            # bf16 | fp16
USE_TF32=1
TORCH_THREADS=1

NUM_WORKERS=2
PIN_MEMORY=1
PERSISTENT_WORKERS=1
PREFETCH_FACTOR=4
MP_CONTEXT="fork"           # fork | spawn

# =========================
# Conda env (used by wrapper)
# =========================
CONDA_SH="/scratch/projects/compilers/intel24.0/oneapi/intelpython/python3.9/etc/profile.d/conda.sh"
CONDA_ENV_1="base"
CONDA_ENV_2="/work/09575/$USER/conda_envs/torchpyg-cu128"
