#!/usr/bin/env bash
# configs/infer_multi_config.sh
#
# This file is "source"d by infer_sbatch_multi.sh.
# Keep it pure bash: assign variables; avoid running commands.

# ============================================================
# Shared inference settings (same ckpt / model)
# ============================================================

# Data root (common across runs)
ROOT_DIR="./Data/Grid4_New/CMIP6_MPI/graphs"
STATION="Battery"

# Inference entrypoint + checkpoint
INFER_PY="infer.py"
CKPT_PATH="./Inference_Checkpoints/CMIP6_MPI_Battery_P3_Best.pth"   # can be a file OR a glob pattern

# Model args (must match infer.py flags)
MODEL="perceiver3"          # "" | baseline | perceiver3
HISTORY_HOURS=12            # -1 uses ckpt args; otherwise override
BATCH_SIZE=1
STATION_JSON_DIR="./station_json"

# Optional: only run selected years (comma-separated). Empty => all available years.
YEARS="1979_1980, 1980_1981, 1981_1982, 1982_1983, 1983_1984, 1984_1985, 1985_1986, 1986_1987, 1987_1988, 1988_1989, 1989_1990, 1990_1991, 1991_1992, 1992_1993, 1993_1994, 1994_1995, 1995_1996, 1996_1997, 1997_1998, 1998_1999, 1999_2000, 2000_2001, 2001_2002, 2002_2003, 2003_2004, 2004_2005, 2005_2006, 2006_2007, 2007_2008, 2008_2009"

# ============================================================
# Sweep definition
# Format per item: "Name|/absolute/or/relative/test_root_dir"
# - If test_root_dir is empty after the "|", we do NOT pass --test_root_dir,
#   which triggers the default NCEP year-split test in infer.py.
# ============================================================

RUNS=(
  # "NCEP_test|"                                 # default test split (no --test_root_dir)
  "CMIP6_MPI_Battery_P3_Best_TO_NCEP|./Data/Grid4_New/NCEP/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_AWI|./Data/Grid4_New/CMIP6_AWI/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_CNRM|./Data/Grid4_New/CMIP6_CNRM/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_EC_EARTH|./Data/Grid4_New/CMIP6_EC_EARTH/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_MPI|./Data/Grid4_New/CMIP6_MPI/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_MRI|./Data/Grid4_New/CMIP6_MRI/graphs"
  "CMIP6_MPI_Battery_P3_Best_TO_CMIP6_Cane5|./Data/Grid4_New/CMIP6_Cane5/graphs"
)

# ============================================================
# Wrapper debug knobs
# ============================================================
CUDA_LAUNCH_BLOCKING_FLAG=0
TORCH_GPU_PROBE=1

# ============================================================
# Speed knobs (mapped to infer.py flags)
# ============================================================
USE_AMP=1
AMP_DTYPE="bf16"            # bf16 | fp16
USE_TF32=1
TORCH_THREADS=1

NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"           # fork | spawn

# Optional: change the logs folder prefix if you want
LOG_ROOT_PREFIX="logs_infer_"

# ============================================================
# Conda env (used by wrapper)
# ============================================================
CONDA_SH="/scratch/projects/compilers/intel24.0/oneapi/intelpython/python3.9/etc/profile.d/conda.sh"
CONDA_ENV_1="base"
CONDA_ENV_2="/work/09575/$USER/conda_envs/torchpyg-cu128"
