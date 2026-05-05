#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  echo "[TRAP] got SIGINT/SIGTERM — cleaning up children"
  pkill -P $$ 2>/dev/null || true
}
trap cleanup INT TERM

# ============================================================
# train.sh — single-node launcher (interactive idev OR non-Slurm host)
#
# - No srun: uses torchrun / torch.distributed.run directly.
# - All experiment knobs live in a sourced config file.
# - Supports lightweight sweeps over LR / loss / history window, etc.
# ============================================================

# =========================
# Config loading
# =========================
# Usage:
#   bash train.sh configs/train_config_*.sh
#   USE_TMUX=1 bash train.sh configs/train_config_*.sh
#
CONFIG_PATH="${1:-configs/train_config.sh}"
if [[ "${CONFIG_PATH}" == "--_tmux_inner" ]]; then
  CONFIG_PATH="configs/train_config.example.sh"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] config not found: ${CONFIG_PATH}"
  exit 2
fi

# Source config (allow omissions; we fill safe defaults below)
set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

# =========================
# Safe defaults (so older/partial configs don't break)
# =========================
: "${TRAIN_PY:=train.py}"
: "${num_gpus:=1}"

: "${ROOT_DIR:=./Data/NCEP/graphs}"
: "${TEST_ROOT_DIR:=}"
: "${STATION:=}"
: "${MODEL:=baseline}"
: "${STATION_JSON_DIR:=./station_json}"

: "${BATCH_SIZE:=256}"
: "${EPOCHS:=300}"
: "${HIDDEN_CHANNELS:=64}"
: "${NUM_LAYERS:=2}"
: "${DROPOUT:=0.05}"
: "${HEAD_DROPOUT:=0.0}"
: "${SEED:=42}"
: "${TRAIN_RATIO:=0.6}"
: "${VAL_RATIO:=0.2}"
: "${SHUFFLE_YEARS:=0}"
: "${FUTURE_ONLY:=0}"
: "${FUTURE_YEAR_THRESHOLD:=2030}"

case "${SHUFFLE_YEARS,,}" in
  1|true|yes|y|on) SHUFFLE_YEARS=1 ;;
  0|false|no|n|off) SHUFFLE_YEARS=0 ;;
  *) echo "[FATAL] SHUFFLE_YEARS must be 0/1 or true/false; got '${SHUFFLE_YEARS}'"; exit 1 ;;
esac

case "${FUTURE_ONLY,,}" in
  1|true|yes|y|on) FUTURE_ONLY=1 ;;
  0|false|no|n|off) FUTURE_ONLY=0 ;;
  *) echo "[FATAL] FUTURE_ONLY must be 0/1 or true/false; got '${FUTURE_ONLY}'"; exit 1 ;;
esac

# Arrays (declare if missing)
if [[ -z "${LR_LIST+x}" ]]; then LR_LIST=("3e-3"); fi
if [[ -z "${HISTORY_HOURS_LIST+x}" ]]; then HISTORY_HOURS_LIST=(24); fi
if [[ -z "${LOSS_MODE_LIST+x}" ]]; then LOSS_MODE_LIST=("mse"); fi

# Loss knobs
: "${TAIL_FRAC:=0.05}"
if [[ -z "${TAIL_LAMBDA_LIST+x}" ]]; then TAIL_LAMBDA_LIST=("0.10"); fi
if [[ -z "${WMSE_Q_LIST+x}" ]]; then WMSE_Q_LIST=("95"); fi
: "${WMSE_ALPHA:=4.0}"
: "${WMSE_S:=0.10}"
: "${WMSE_USE_ABS:=1}"

# Slope knobs (only used for *_slope modes)
if [[ -z "${SLOPE_LAMBDA_LIST+x}" ]]; then SLOPE_LAMBDA_LIST=("0.01"); fi
if [[ -z "${SLOPE_MASK_S_LIST+x}" ]]; then SLOPE_MASK_S_LIST=("0.10"); fi
: "${SLOPE_ROBUST:=charb}"
: "${SLOPE_CHARB_EPS:=1e-3}"
: "${SLOPE_HUBER_DELTA:=0.05}"

# Scheduler knobs
: "${SCHEDULER:=cosine}"           # cosine | rop
: "${ROP_METRIC:=val_rmse_phys}"   # val_rmse_phys | val_rmse_peak

# OOD knobs
: "${X_NORM:=robust}"              # zscore | robust | mag
: "${X_P_LO:=1.0}"
: "${X_P_HI:=99.0}"
: "${X_NODES_PER_GRAPH:=256}"
: "${X_CLIP:=5.0}"

: "${X_AUG:=1}"
: "${X_AUG_PROB:=1.0}"
: "${X_AUG_SCALE:=0.05}"
: "${X_AUG_BIAS:=0.02}"

: "${DISABLE_OOD:=0}"

# Speed / loader knobs
: "${USE_AMP:=0}"
: "${AMP_DTYPE:=bf16}"             # bf16 | fp16
: "${USE_TF32:=0}"
: "${TORCH_THREADS:=1}"

: "${NUM_WORKERS:=4}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=2}"
: "${MP_CONTEXT:=fork}"

# Optional tmux wrapper
: "${USE_TMUX:=${USE_TMUX:-1}}"

# Optional conda activation
: "${DO_CONDA:=1}"
: "${CONDA_MODULE:=anaconda}"
: "${CONDA_SH:=/software/u22/anaconda/python3.9/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=torchpyg-cu124}"

# p_mean ablation knobs (optional)
: "${USE_PMEAN:=0}"
: "${PMEAN_DIM:=32}"
: "${PERCEIVER_PMEAN_MODE:=tokens}"

# Perceiver3 knobs
: "${GATE_MODE:=window}"
: "${GATE_BIAS_INIT:=-2.0}"
: "${TAIL_TANH_CLIP:=2.5}"
: "${ALPHA_INIT_LOGIT:=-2.0}"

: "${NODE_READ_HEADS:=8}"
: "${TIME_READ_HEADS:=8}"
: "${TRANSFORMER_LAYERS:=2}"
: "${TRANSFORMER_FF_MULT:=4.0}"
: "${TRANSFORMER_DROPOUT:=0.05}"
: "${MAX_TIME_STEPS:=32}"

# =========================
# (Optional) Conda activation
# =========================
if [[ "${DO_CONDA}" -eq 1 ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    set +u
    if command -v module >/dev/null 2>&1; then
      module load "${CONDA_MODULE}" || echo "[WARN] module load ${CONDA_MODULE} failed. Continuing with CONDA_SH=${CONDA_SH}."
    else
      echo "[WARN] module command not found. Continuing with CONDA_SH=${CONDA_SH}."
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    hash -r
    set -u
  else
    echo "[WARN] conda.sh not found at ${CONDA_SH}. Skipping conda activation."
  fi
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

# =========================
# Helper: infer dataset tag
# =========================
infer_dataset_tag () {
  local p="$1"
  p="${p%/}"
  if [[ "$(basename "$p")" == "graphs" ]]; then
    basename "$(dirname "$p")"
  else
    basename "$p"
  fi
}

TRAIN_DATA_TAG="$(infer_dataset_tag "$ROOT_DIR")"
if [[ -n "${TEST_ROOT_DIR}" ]]; then
  TEST_DATA_TAG="$(infer_dataset_tag "$TEST_ROOT_DIR")"
else
  TEST_DATA_TAG="${TRAIN_DATA_TAG}"
fi

# =========================
# Apply DISABLE_OOD override (common “ID-only” baseline)
# =========================
if [[ "${DISABLE_OOD}" == "1" ]]; then
  X_NORM="zscore"
  X_CLIP="0"
  X_AUG="0"
fi

# =========================
# DDP launcher setup (NO srun)
# =========================
_ddp_ngpu="${SLURM_NTASKS_PER_NODE:-${num_gpus}}"
num_gpus="${_ddp_ngpu}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _CVD_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  _NGPU_VISIBLE="${#_CVD_ARR[@]}"
  if [[ "${_NGPU_VISIBLE}" -gt 0 && "${num_gpus}" -gt "${_NGPU_VISIBLE}" ]]; then
    echo "[WARN] num_gpus=${num_gpus} > CUDA_VISIBLE_DEVICES count=${_NGPU_VISIBLE}. Clamping."
    num_gpus="${_NGPU_VISIBLE}"
  fi
fi

MASTER_ADDR="127.0.0.1"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
else
  MASTER_PORT=$((29500 + (RANDOM % 1000)))
fi
export MASTER_ADDR MASTER_PORT

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"

unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# Use the active Python interpreter for DDP launch so we stay inside the
# currently activated conda environment even if PATH or shell hashing points
# somewhere unexpected.
TORCH_LAUNCH=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${num_gpus}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

# =========================
# Launcher logging
# =========================
WORKDIR="$(pwd)"
RUNSTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_ROOT="launcher_logs_${STATION:-ALL}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SWEEP_DIR="${WORKDIR}/${LOG_ROOT}/idev_${SLURM_JOB_ID}_${RUNSTAMP}"
else
  SWEEP_DIR="${WORKDIR}/${LOG_ROOT}/local_${RUNSTAMP}"
fi
mkdir -p "${SWEEP_DIR}"
cp -f "${CONFIG_PATH}" "${SWEEP_DIR}/config_used.sh" 2>/dev/null || true

echo "========================================="
echo "Config file:   ${CONFIG_PATH}"
echo "Host:          $(hostname)"
echo "Workdir:       ${WORKDIR}"
echo "Sweep dir:     ${SWEEP_DIR}"
echo "Train script:  ${TRAIN_PY}"
echo "Model:         ${MODEL}"
echo "ROOT_DIR:      ${ROOT_DIR}"
echo "TEST_ROOT_DIR: ${TEST_ROOT_DIR:-<none>}"
echo "TRAIN_DATA_TAG:${TRAIN_DATA_TAG}"
echo "TEST_DATA_TAG: ${TEST_DATA_TAG}"
echo "LR_LIST:       ${LR_LIST[*]}"
echo "Loss modes:    ${LOSS_MODE_LIST[*]}"
echo "H_LIST:        ${HISTORY_HOURS_LIST[*]}"
echo "Split:         train=${TRAIN_RATIO} val=${VAL_RATIO} shuffle_years=${SHUFFLE_YEARS} future_only=${FUTURE_ONLY} future_year_threshold=${FUTURE_YEAR_THRESHOLD} seed=${SEED}"
echo "MASTER_ADDR:   ${MASTER_ADDR}"
echo "MASTER_PORT:   ${MASTER_PORT}"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID:-<none>}"
echo "CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "num_gpus:      ${num_gpus}"
echo "Scheduler:     ${SCHEDULER} (ROP_METRIC=${ROP_METRIC})"
echo "DISABLE_OOD:   ${DISABLE_OOD} (x_norm=${X_NORM}, x_clip=${X_CLIP}, x_aug=${X_AUG})"
echo "p_mean:        USE_PMEAN=${USE_PMEAN} (PMEAN_DIM=${PMEAN_DIM}, PERCEIVER_PMEAN_MODE=${PERCEIVER_PMEAN_MODE})"
echo "DL:            workers=${NUM_WORKERS} pin=${PIN_MEMORY} pers=${PERSISTENT_WORKERS} prefetch=${PREFETCH_FACTOR} mp=${MP_CONTEXT}"
echo "========================================="

# =========================
# Optional: tmux wrapper
# =========================
TMUX_INNER=0
if [[ "${1:-}" == "${CONFIG_PATH}" && "${2:-}" == "--_tmux_inner" ]]; then
  TMUX_INNER=1
elif [[ "${1:-}" == "--_tmux_inner" ]]; then
  TMUX_INNER=1
fi

if [[ "${USE_TMUX}" == "1" && "${TMUX_INNER}" == "0" ]] && command -v tmux >/dev/null 2>&1; then
  SESSION_NAME="stormsurge_${RUNSTAMP}"
  printf -v TMUX_INNER_CMD 'bash %q %q --_tmux_inner; status=$?; if [[ ${status} -ne 0 ]]; then echo; echo "[tmux] train.sh exited with status ${status}. Type exit or press Ctrl-D to close."; exec bash; fi' "${SCRIPT_PATH}" "${CONFIG_PATH}"
  printf -v TMUX_CMD 'bash -lc %q' "${TMUX_INNER_CMD}"
  echo "[INFO] launching inside tmux session: ${SESSION_NAME}"
  tmux new-session -d -s "${SESSION_NAME}" -c "${WORKDIR}" "${TMUX_CMD}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
  echo "Logs:   ${SWEEP_DIR}"
  exit 0
fi
if [[ "${1:-}" == "${CONFIG_PATH}" && "${2:-}" == "--_tmux_inner" ]]; then
  shift || true
  shift || true
elif [[ "${1:-}" == "--_tmux_inner" ]]; then
  shift || true
fi

# =========================
# Tags for readability
# =========================
EXTRA_TAG=""
case "${MODEL}" in
  baseline) EXTRA_TAG="" ;;
  perceiver3) EXTRA_TAG="_nrh${NODE_READ_HEADS}_trh${TIME_READ_HEADS}_L${TRANSFORMER_LAYERS}_ff${TRANSFORMER_FF_MULT}_td${TRANSFORMER_DROPOUT}_gm${GATE_MODE}" ;;
  *) echo "[FATAL] Unknown MODEL='${MODEL}'. Use baseline|perceiver3"; exit 1 ;;
esac

PMEAN_TAG=""
if [[ "${USE_PMEAN}" == "1" ]]; then
  PMEAN_TAG="_pmean${PMEAN_DIM}"
  if [[ "${MODEL}" == "perceiver3" ]]; then
    PMEAN_TAG+="_${PERCEIVER_PMEAN_MODE}"
  fi
fi

SPLIT_TAG=""
if [[ "${SHUFFLE_YEARS}" == "1" ]]; then
  SPLIT_TAG="_yshuffle"
fi
if [[ "${FUTURE_ONLY}" == "1" ]]; then
  SPLIT_TAG+="_futuregt${FUTURE_YEAR_THRESHOLD}"
fi

# =========================
# Sweep loops
# =========================
for LOSS_MODE in "${LOSS_MODE_LIST[@]}"; do
  for LR_CUR in "${LR_LIST[@]}"; do
    for H in "${HISTORY_HOURS_LIST[@]}"; do
      for WMSE_Q_CUR in "${WMSE_Q_LIST[@]}"; do

        # WMSE knobs only matter for wmse* or *wtail modes
        if [[ "${LOSS_MODE}" != wmse* && "${LOSS_MODE}" != *wtail* ]]; then
          [[ "${WMSE_Q_CUR}" != "${WMSE_Q_LIST[0]}" ]] && continue
        fi

        for TLAMBDA in "${TAIL_LAMBDA_LIST[@]}"; do
          # tail_lambda only matters for *tail*/*wtail modes
          if [[ "${LOSS_MODE}" != *tail* && "${LOSS_MODE}" != *wtail* ]]; then
            [[ "${TLAMBDA}" != "${TAIL_LAMBDA_LIST[0]}" ]] && continue
          fi

          LOSS_TAG="_loss${LOSS_MODE}"
          LOSS_ARGS=(--loss_mode "${LOSS_MODE}")

          # WMSE shaping
          LOSS_ARGS+=(--wmse_q "${WMSE_Q_CUR}" --wmse_alpha "${WMSE_ALPHA}" --wmse_s "${WMSE_S}" --wmse_use_abs "${WMSE_USE_ABS}")
          if [[ "${LOSS_MODE}" == wmse* || "${LOSS_MODE}" == *wtail* ]]; then
            LOSS_TAG+="_q${WMSE_Q_CUR}_a${WMSE_ALPHA}_s${WMSE_S}_abs${WMSE_USE_ABS}"
          fi

          # Tail auxiliary loss
          if [[ "${LOSS_MODE}" == *tail* || "${LOSS_MODE}" == *wtail* ]]; then
            LOSS_ARGS+=(--tail_frac "${TAIL_FRAC}" --tail_lambda "${TLAMBDA}")
            LOSS_TAG+="_tf${TAIL_FRAC}_tl${TLAMBDA}"
          fi

          # Slope smoothness sweep (only used for *_slope modes)
          for SLOPE_LAMBDA in "${SLOPE_LAMBDA_LIST[@]}"; do
            for SLOPE_MASK_S in "${SLOPE_MASK_S_LIST[@]}"; do
              if [[ "${LOSS_MODE}" != *_slope ]]; then
                [[ "${SLOPE_LAMBDA}" != "${SLOPE_LAMBDA_LIST[0]}" ]] && continue
                [[ "${SLOPE_MASK_S}" != "${SLOPE_MASK_S_LIST[0]}" ]] && continue
              fi

              LOSS_TAG2="${LOSS_TAG}"
              LOSS_ARGS2=("${LOSS_ARGS[@]}")
              if [[ "${LOSS_MODE}" == *_slope ]]; then
                LOSS_ARGS2+=(--slope_lambda "${SLOPE_LAMBDA}"
                             --slope_mask_s "${SLOPE_MASK_S}"
                             --slope_robust "${SLOPE_ROBUST}"
                             --slope_charb_eps "${SLOPE_CHARB_EPS}")
                if [[ "${SLOPE_ROBUST}" == "huber" ]]; then
                  LOSS_ARGS2+=(--slope_huber_delta "${SLOPE_HUBER_DELTA}")
                fi
                LOSS_TAG2+="_sl${SLOPE_LAMBDA}_sms${SLOPE_MASK_S}_${SLOPE_ROBUST}"
              fi

              SCHED_ARGS=(--scheduler "${SCHEDULER}")
              if [[ "${SCHEDULER}" == "rop" ]]; then
                SCHED_ARGS+=(--rop_metric "${ROP_METRIC}")
              fi

              RUN_TAG="${STATION:-ALL}_${TRAIN_DATA_TAG}to${TEST_DATA_TAG}_${MODEL}${EXTRA_TAG}${PMEAN_TAG}${SPLIT_TAG}${LOSS_TAG2}_hist${H}h_hid${HIDDEN_CHANNELS}_L${NUM_LAYERS}_bs${BATCH_SIZE}_lr${LR_CUR}_ep${EPOCHS}_sch${SCHEDULER}_xn${X_NORM}"
              LOG_FILE="${SWEEP_DIR}/train_${RUN_TAG}.log"
              : > "${LOG_FILE}"

              BASE_CMD=(
                "${TRAIN_PY}"
                --root_dir "${ROOT_DIR}"
                --model "${MODEL}"
                --batch_size "${BATCH_SIZE}"
                --lr "${LR_CUR}"
                --epochs "${EPOCHS}"
                --hidden_channels "${HIDDEN_CHANNELS}"
                --num_layers "${NUM_LAYERS}"
                --dropout "${DROPOUT}"
                --head_dropout "${HEAD_DROPOUT}"
                --history_hours "${H}"
                --seed "${SEED}"
                --train_ratio "${TRAIN_RATIO}"
                --val_ratio "${VAL_RATIO}"
                --shuffle_years "${SHUFFLE_YEARS}"
                --future_only "${FUTURE_ONLY}"
                --future_year_threshold "${FUTURE_YEAR_THRESHOLD}"
                --run_tag "${RUN_TAG}_${RUNSTAMP}"
                --torch_threads "${TORCH_THREADS}"
                --num_workers "${NUM_WORKERS}"
                --prefetch_factor "${PREFETCH_FACTOR}"
                --mp_context "${MP_CONTEXT}"
                --x_norm "${X_NORM}"
                --x_p_lo "${X_P_LO}"
                --x_p_hi "${X_P_HI}"
                --x_nodes_per_graph "${X_NODES_PER_GRAPH}"
                --x_clip "${X_CLIP}"
                --x_aug "${X_AUG}"
                --x_aug_prob "${X_AUG_PROB}"
                --x_aug_scale "${X_AUG_SCALE}"
                --x_aug_bias "${X_AUG_BIAS}"
                --station_json_dir "${STATION_JSON_DIR}"
              )

              [[ -n "${STATION}" ]]       && BASE_CMD+=(--station "${STATION}")
              [[ -n "${TEST_ROOT_DIR}" ]] && BASE_CMD+=(--test_root_dir "${TEST_ROOT_DIR}")

              # p_mean injection (ablation)
              if [[ "${USE_PMEAN}" == "1" ]]; then
                BASE_CMD+=(--use_pmean --pmean_dim "${PMEAN_DIM}")
                if [[ "${MODEL}" == "perceiver3" ]]; then
                  BASE_CMD+=(--perceiver_pmean_mode "${PERCEIVER_PMEAN_MODE}")
                fi
              fi

              # Speed flags
              if [[ "${USE_AMP}" -eq 1 ]]; then
                BASE_CMD+=(--amp --amp_dtype "${AMP_DTYPE}")
              fi
              if [[ "${USE_TF32}" -eq 1 ]]; then
                BASE_CMD+=(--tf32)
              fi
              if [[ "${PIN_MEMORY}" -eq 1 ]]; then
                BASE_CMD+=(--pin_memory)
              fi
              if [[ "${PERSISTENT_WORKERS}" -eq 1 ]]; then
                BASE_CMD+=(--persistent_workers)
              fi

              # Model-specific knobs
              if [[ "${MODEL}" == "perceiver3" ]]; then
                BASE_CMD+=(--node_read_heads "${NODE_READ_HEADS}"
                           --time_read_heads "${TIME_READ_HEADS}"
                           --transformer_layers "${TRANSFORMER_LAYERS}"
                           --transformer_ff_mult "${TRANSFORMER_FF_MULT}"
                           --transformer_dropout "${TRANSFORMER_DROPOUT}"
                           --max_time_steps "${MAX_TIME_STEPS}"
                           --gate_mode "${GATE_MODE}"
                           --gate_bias_init "${GATE_BIAS_INIT}"
                           --tail_tanh_clip "${TAIL_TANH_CLIP}"
                           --alpha_init_logit "${ALPHA_INIT_LOGIT}")
              fi

              BASE_CMD+=("${SCHED_ARGS[@]}")
              BASE_CMD+=("${LOSS_ARGS2[@]}")

              {
                echo "-----------------------------------------"
                echo "RUN_TAG:  ${RUN_TAG}"
                echo "LOG:      ${LOG_FILE}"
                echo "LAUNCHER: ${TORCH_LAUNCH[*]}"
                echo "-----------------------------------------"
                printf "CMD: "
                printf "%q " "${BASE_CMD[@]}"
                echo
              } | tee -a "${LOG_FILE}"

              if [[ "${num_gpus}" -le 1 ]]; then
                python -u "${BASE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
              else
                "${TORCH_LAUNCH[@]}" "${BASE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
              fi

            done
          done

        done
      done
    done
  done
done

echo "DONE. Launcher logs at: ${SWEEP_DIR}"
