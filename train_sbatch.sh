#!/usr/bin/env bash
#SBATCH -J stormsurge_train
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 24:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=zel220@lehigh.edu

set -euo pipefail

cleanup() {
  echo "[TRAP] got SIGINT/SIGTERM — cleaning up children"
  pkill -P $$ 2>/dev/null || true
}
trap cleanup INT TERM

# ============================================================
# train_sbatch.sh — Slurm batch launcher (single-node DDP)
#
# - Slurm allocates the node; we do NOT use srun.
# - Uses torchrun / torch.distributed.run for single-node DDP.
# - All experiment knobs live in a sourced config file.
# - This script only manages *launcher logs*; train.py writes checkpoints/results.
# ============================================================

# =========================
# Config loading
# =========================
# Usage:
#   sbatch train_sbatch.sh configs/train_config_*.sh
#
CONFIG_PATH="${1:-configs/train_config.example.sh}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] config not found: ${CONFIG_PATH}"
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

# =========================
# Safe defaults (so older/partial configs don't break)
# =========================
: "${TRAIN_PY:=train.py}"
: "${num_gpus:=4}"

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
: "${SCHEDULER:=cosine}"
: "${ROP_METRIC:=val_rmse_phys}"

# OOD knobs
: "${X_NORM:=robust}"
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
: "${AMP_DTYPE:=bf16}"
: "${USE_TF32:=0}"
: "${TORCH_THREADS:=1}"

: "${NUM_WORKERS:=4}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=2}"
: "${MP_CONTEXT:=fork}"

# Optional conda activation (prefer config settings)
: "${DO_CONDA:=0}"
: "${CONDA_SH:=}"
: "${CONDA_ENV_1:=base}"
: "${CONDA_ENV_2:=}"

# p_mean ablation knobs (optional)
: "${USE_PMEAN:=0}"
: "${PMEAN_DIM:=32}"
: "${PERCEIVER_PMEAN_MODE:=tokens}"

# Perceiver3 knobs (only used when MODEL=perceiver3)
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
# Conda env (optional)
# =========================
if [[ "${DO_CONDA}" -eq 1 ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    export CONDA_PKGS_DIRS=/work/09575/"$USER"/conda_pkgs
    conda activate "${CONDA_ENV_1}" >/dev/null 2>&1 || true
    conda activate "${CONDA_ENV_2}"
    set -u
  else
    echo "[WARN] conda.sh not found at ${CONDA_SH}. Skipping conda activation."
  fi
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
num_gpus="${SLURM_NTASKS_PER_NODE:-${num_gpus}}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _CVD_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  _NGPU_VISIBLE="${#_CVD_ARR[@]}"
  if [[ "${_NGPU_VISIBLE}" -gt 0 && "${num_gpus}" -gt "${_NGPU_VISIBLE}" ]]; then
    echo "[WARN] num_gpus=${num_gpus} > CUDA_VISIBLE_DEVICES count=${_NGPU_VISIBLE}. Clamping."
    num_gpus="${_NGPU_VISIBLE}"
  fi
fi

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
export MASTER_ADDR MASTER_PORT

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"

unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCH=(torchrun --nproc_per_node="${num_gpus}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
else
  TORCH_LAUNCH=(python -m torch.distributed.run --nproc_per_node="${num_gpus}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
fi

# =========================
# Launcher log folder (this script only)
# =========================
WORKDIR="$(pwd)"
JOBSTAMP="$(date +"%Y%m%d_%H%M%S")"
LAUNCH_LOG_ROOT="launcher_logs_${STATION:-ALL}"
SWEEP_DIR="${WORKDIR}/${LAUNCH_LOG_ROOT}/slurm_${SLURM_JOB_ID}_${JOBSTAMP}"
mkdir -p "${SWEEP_DIR}"
cp -f "${CONFIG_PATH}" "${SWEEP_DIR}/config_used.sh"

echo "========================================="
echo "JobID:         ${SLURM_JOB_ID}"
echo "Host:          $(hostname)"
echo "NodeList:      ${SLURM_NODELIST}"
echo "Workdir:       ${WORKDIR}"
echo "Config file:   ${CONFIG_PATH}"
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
echo "MASTER_ADDR:   ${MASTER_ADDR}"
echo "MASTER_PORT:   ${MASTER_PORT}"
echo "num_gpus:      ${num_gpus}"
echo "Scheduler:     ${SCHEDULER} (ROP_METRIC=${ROP_METRIC})"
echo "DISABLE_OOD:   ${DISABLE_OOD} (x_norm=${X_NORM}, x_clip=${X_CLIP}, x_aug=${X_AUG})"
echo "p_mean:        USE_PMEAN=${USE_PMEAN} (PMEAN_DIM=${PMEAN_DIM}, PERCEIVER_PMEAN_MODE=${PERCEIVER_PMEAN_MODE})"
echo "DL:            workers=${NUM_WORKERS} pin=${PIN_MEMORY} pers=${PERSISTENT_WORKERS} prefetch=${PREFETCH_FACTOR} mp=${MP_CONTEXT}"
echo "========================================="

# =========================
# Sweep loops
# =========================
RUN_SEQ=0

for LOSS_MODE in "${LOSS_MODE_LIST[@]}"; do
  for LR_CUR in "${LR_LIST[@]}"; do
    for H in "${HISTORY_HOURS_LIST[@]}"; do
      for WMSE_Q_CUR in "${WMSE_Q_LIST[@]}"; do

        if [[ "${LOSS_MODE}" != wmse* && "${LOSS_MODE}" != *wtail* ]]; then
          [[ "${WMSE_Q_CUR}" != "${WMSE_Q_LIST[0]}" ]] && continue
        fi

        for TLAMBDA in "${TAIL_LAMBDA_LIST[@]}"; do
          if [[ "${LOSS_MODE}" != *tail* && "${LOSS_MODE}" != *wtail* ]]; then
            [[ "${TLAMBDA}" != "${TAIL_LAMBDA_LIST[0]}" ]] && continue
          fi

          LOSS_ARGS=(--loss_mode "${LOSS_MODE}")
          LOSS_ARGS+=(--wmse_q "${WMSE_Q_CUR}" --wmse_alpha "${WMSE_ALPHA}" --wmse_s "${WMSE_S}" --wmse_use_abs "${WMSE_USE_ABS}")
          if [[ "${LOSS_MODE}" == *tail* || "${LOSS_MODE}" == *wtail* ]]; then
            LOSS_ARGS+=(--tail_frac "${TAIL_FRAC}" --tail_lambda "${TLAMBDA}")
          fi

          for SLOPE_LAMBDA in "${SLOPE_LAMBDA_LIST[@]}"; do
            for SLOPE_MASK_S in "${SLOPE_MASK_S_LIST[@]}"; do
              if [[ "${LOSS_MODE}" != *_slope ]]; then
                [[ "${SLOPE_LAMBDA}" != "${SLOPE_LAMBDA_LIST[0]}" ]] && continue
                [[ "${SLOPE_MASK_S}" != "${SLOPE_MASK_S_LIST[0]}" ]] && continue
              fi

              LOSS_ARGS2=("${LOSS_ARGS[@]}")
              if [[ "${LOSS_MODE}" == *_slope ]]; then
                LOSS_ARGS2+=(--slope_lambda "${SLOPE_LAMBDA}"
                             --slope_mask_s "${SLOPE_MASK_S}"
                             --slope_robust "${SLOPE_ROBUST}"
                             --slope_charb_eps "${SLOPE_CHARB_EPS}")
                if [[ "${SLOPE_ROBUST}" == "huber" ]]; then
                  LOSS_ARGS2+=(--slope_huber_delta "${SLOPE_HUBER_DELTA}")
                fi
              fi

              SCHED_ARGS=(--scheduler "${SCHEDULER}")
              if [[ "${SCHEDULER}" == "rop" ]]; then
                SCHED_ARGS+=(--rop_metric "${ROP_METRIC}")
              fi

              RUN_SEQ=$((RUN_SEQ+1))
              PER_RUN_STAMP="$(date +"%Y%m%d_%H%M%S")"

              RUN_TAG="${STATION:-ALL}_${TRAIN_DATA_TAG}to${TEST_DATA_TAG}_${MODEL}_loss${LOSS_MODE}_hist${H}h_lr${LR_CUR}_r$(printf '%04d' "${RUN_SEQ}")_${PER_RUN_STAMP}"
              RUN_DIR="${SWEEP_DIR}/${RUN_TAG}"
              mkdir -p "${RUN_DIR}"
              LOG_FILE="${RUN_DIR}/train.log"

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
                --run_tag "${RUN_TAG}"
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

              if [[ "${USE_PMEAN}" == "1" ]]; then
                BASE_CMD+=(--use_pmean --pmean_dim "${PMEAN_DIM}")
                if [[ "${MODEL}" == "perceiver3" ]]; then
                  BASE_CMD+=(--perceiver_pmean_mode "${PERCEIVER_PMEAN_MODE}")
                fi
              fi

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

              BASE_CMD+=("${SCHED_ARGS[@]}")
              BASE_CMD+=("${LOSS_ARGS2[@]}")

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

              {
                echo "-----------------------------------------"
                echo "RUN_TAG:    ${RUN_TAG}"
                echo "RUN_DIR:    ${RUN_DIR}"
                echo "LOG:        ${LOG_FILE}"
                echo "LAUNCHER:   ${TORCH_LAUNCH[*]}"
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
