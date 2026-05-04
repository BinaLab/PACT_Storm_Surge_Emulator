#!/usr/bin/env bash
# configs/train_config.example.sh
#
# This file is "source"d by train.sh.
# Keep it pure bash: assign variables + arrays; avoid running commands.

# =========================
# Train script
# =========================
TRAIN_PY="train.py"   # <-- set this to your CURRENT big train file (e.g., train_clean_step4_pmean_modes.py)

# =========================
# GPU / data
# =========================
num_gpus=4
ROOT_DIR="./Data/Grid4_New_PastOnly/CMIP6_AWI/graphs"
TEST_ROOT_DIR=""          # optional external test root (OOD), leave empty for ID
STATION="Battery"

# =========================
# Core hyperparams
# =========================
BATCH_SIZE=256
LR_LIST=("5e-3")

EPOCHS=300
HIDDEN_CHANNELS=128
NUM_LAYERS=2
DROPOUT=0.05
HEAD_DROPOUT=0.05
HISTORY_HOURS_LIST=(12)

SEED=42
MODEL="perceiver3"          # baseline | perceiver3

TRAIN_RATIO="0.6"
VAL_RATIO="0.2"

# =========================
# p_mean ablation knobs (NEW)
# =========================
USE_PMEAN=0               # 0/1 (maps to --use_pmean)
PMEAN_DIM=32              # maps to --pmean_dim (baseline and perceiver global branch)
PERCEIVER_PMEAN_MODE="tokens"  # tokens | global | both  (only used when MODEL=perceiver3 AND USE_PMEAN=1)

# =========================
# Loss knobs
# =========================
LOSS_MODE_LIST=("mse")    # ("mse" "wmse" "mse_tail" "wmse_tail" "mse_wtail" and add suffix *_slope)

TAIL_FRAC="0.05"
TAIL_LAMBDA_LIST=("0.10")

WMSE_Q_LIST=("95")
WMSE_ALPHA="4.0"
WMSE_S="0.10"
WMSE_USE_ABS="1"

# =========================
# Slope smoothness knobs (NEW: only used when LOSS_MODE ends with *_slope)
# =========================
SLOPE_LAMBDA_LIST=("0.01")   # maps to --slope_lambda
SLOPE_MASK_S_LIST=("0.10")   # maps to --slope_mask_s (soft mask width in meters)
SLOPE_ROBUST="charb"         # charb | huber   (maps to --slope_robust)
SLOPE_CHARB_EPS="1e-3"       # maps to --slope_charb_eps
SLOPE_HUBER_DELTA="0.05"     # maps to --slope_huber_delta (only used if SLOPE_ROBUST=huber)

# =========================
# Scheduler knobs
# =========================
SCHEDULER="cosine"          # cosine | rop
ROP_METRIC="val_rmse_phys"  # val_rmse_phys | val_rmse_peak

# =========================
# Perceiver3 knobs
# =========================
GATE_MODE="window"
GATE_BIAS_INIT="0.0"
TAIL_TANH_CLIP="0"
ALPHA_INIT_LOGIT="0"

NODE_READ_HEADS="8"
TIME_READ_HEADS="8"
TRANSFORMER_LAYERS="2"
TRANSFORMER_FF_MULT="4.0"
TRANSFORMER_DROPOUT="0.0"
MAX_TIME_STEPS="32"

# If your train.py reads station jsons (perceiver3), keep this stable
STATION_JSON_DIR="./station_json"

# =========================
# OOD knobs (train.py supports zscore|robust|mag)
# =========================
X_NORM="robust"            # zscore | robust | mag
X_P_LO="1.0"
X_P_HI="99.0"
X_NODES_PER_GRAPH="0"
X_CLIP="6.0"

X_AUG="1"
X_AUG_PROB="0.5"
X_AUG_SCALE="0.02"
X_AUG_BIAS="0.02"

# Turn OFF OOD tricks (1 disables robust/clip/aug)
DISABLE_OOD="1"

# =========================
# Speed knobs
# =========================
USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
TORCH_THREADS=1

NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

# =========================
# Optional tmux wrapper
# =========================
USE_TMUX="${USE_TMUX:-0}"   # allow override at launch: USE_TMUX=1 bash train.sh configs/xxx.sh

# =========================
# Conda activation (optional)
# =========================
DO_CONDA=1
CONDA_SH="/scratch/projects/compilers/intel24.0/oneapi/intelpython/python3.9/etc/profile.d/conda.sh"
CONDA_ENV_1="base"
CONDA_ENV_2="/work/09575/$USER/conda_envs/torchpyg-cu128"
