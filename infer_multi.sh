#!/usr/bin/env bash
set -euo pipefail
cleanup() {
  echo "[TRAP] got SIGINT/SIGTERM — cleaning up children"
  pkill -P $$ 2>/dev/null || true
}
trap cleanup INT TERM

# ============================================================
# infer_multi.sh
# - Non-slurm inference sweep (runs sequentially)
# - Sources configs/infer_multi_config.sh
# - Sweeps over RUNS=("Name|/path/to/test_root_dir" ...)
# - Uses folder name: <Name>_<date>_<time>
#
# Usage:
#   bash infer_multi.sh configs/infer_multi_config.sh
# ============================================================

CONFIG_PATH="${1:-configs/configs_infer/infer_multi_config_NCEP.sh}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] config not found: ${CONFIG_PATH}"
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

: "${INFER_PY:=infer.py}"
: "${ROOT_DIR:=./Data/NCEP/graphs}"
: "${STATION:=Battery}"
: "${MODEL:=baseline}"
: "${HISTORY_HOURS:=12}"
: "${BATCH_SIZE:=1}"
: "${STATION_JSON_DIR:=./station_json}"
: "${YEARS:=}"

: "${CUDA_LAUNCH_BLOCKING_FLAG:=0}"
: "${TORCH_GPU_PROBE:=0}"
: "${DO_CONDA:=1}"
: "${CONDA_MODULE:=anaconda}"
: "${CONDA_SH:=/software/u22/anaconda/python3.9/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=torchpyg-cu124}"

: "${USE_AMP:=1}"
: "${AMP_DTYPE:=bf16}"
: "${USE_TF32:=1}"
: "${TORCH_THREADS:=1}"

: "${NUM_WORKERS:=0}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=0}"
: "${MP_CONTEXT:=fork}"

: "${CKPT_PATH:=}"
: "${LOG_ROOT_PREFIX:=logs_infer_}"

# Resolve checkpoint (supports glob patterns)
resolve_ckpt () {
  local pat="$1"
  if [[ -z "$pat" ]]; then return 1; fi
  if [[ -f "$pat" ]]; then echo "$pat"; return 0; fi
  shopt -s nullglob
  # shellcheck disable=SC2206
  local arr=( $pat )
  shopt -u nullglob
  if [[ "${#arr[@]}" -eq 0 ]]; then return 1; fi
  ls -1t "${arr[@]}" 2>/dev/null | head -n 1
}
CKPT_RESOLVED="$(resolve_ckpt "${CKPT_PATH}" || true)"
if [[ -z "${CKPT_RESOLVED}" ]]; then
  echo "[FATAL] CKPT_PATH does not resolve to a file: ${CKPT_PATH}"
  exit 2
fi

init_conda () {
  if [[ "${DO_CONDA}" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[WARN] conda.sh not found at ${CONDA_SH}. Skipping conda activation."
    return 0
  fi
  set +u
  module load "${CONDA_MODULE}"
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  hash -r
  set -u
}

init_conda

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONUNBUFFERED=1

if [[ "${CUDA_LAUNCH_BLOCKING_FLAG}" -eq 1 ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

TF32_ARGS=()
if [[ "${USE_TF32}" -eq 1 ]]; then TF32_ARGS=(--tf32); fi

AMP_ARGS=()
if [[ "${USE_AMP}" -eq 1 ]]; then AMP_ARGS=(--amp --amp_dtype "${AMP_DTYPE}"); fi

STATION_ARGS=()
if [[ -n "${STATION}" ]]; then STATION_ARGS=(--station "${STATION}"); fi

YEARS_ARGS=()
if [[ -n "${YEARS}" ]]; then YEARS_ARGS=(--years "${YEARS}"); fi

DL_ARGS=(--num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" --mp_context "${MP_CONTEXT}")
if [[ "${PIN_MEMORY}" -eq 1 ]]; then DL_ARGS+=(--pin_memory); fi
if [[ "${PERSISTENT_WORKERS}" -eq 1 ]]; then DL_ARGS+=(--persistent_workers); fi

THREAD_ARGS=(--torch_threads "${TORCH_THREADS}")

# RUNS sanity
if [[ -z "${RUNS+x}" ]]; then
  echo "[FATAL] RUNS array not defined in config. Example:"
  echo '  RUNS=("NCEP_test|" "AWI_future|./Data/CMIP6_AWI/graphs")'
  exit 2
fi
if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "[FATAL] RUNS array is empty."
  exit 2
fi

WORKDIR="$(pwd)"
LOG_ROOT="${LOG_ROOT_PREFIX}${STATION}"

echo "========================================="
echo "host:              $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Config:            ${CONFIG_PATH}"
echo "Infer script:      ${INFER_PY}"
echo "Checkpoint:        ${CKPT_RESOLVED}"
echo "ROOT_DIR:          ${ROOT_DIR}"
echo "Station:           ${STATION}"
echo "Model:             ${MODEL}"
echo "History hours:     ${HISTORY_HOURS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Years:             ${YEARS:-<all>}"
echo "Runs:              ${#RUNS[@]}"
echo "========================================="

if [[ "${TORCH_GPU_PROBE}" -eq 1 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
print("[local][torch] torch:", torch.__version__)
print("[local][torch] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[local][torch] device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        try:
            print(f"[local][torch] {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"[local][torch] {i}: <error> {e}")
PY
fi

# Sweep
for spec in "${RUNS[@]}"; do
  NAME="${spec%%|*}"
  TEST_ROOT_DIR="${spec#*|}"
  if [[ "${NAME}" == "${TEST_ROOT_DIR}" ]]; then
    TEST_ROOT_DIR=""
  fi

  RUNSTAMP=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${WORKDIR}/${LOG_ROOT}/${NAME}_${RUNSTAMP}"
  OUT_DIR="${RUN_DIR}/outputs"
  mkdir -p "${OUT_DIR}"

  LOG_FILE="${RUN_DIR}/infer_${STATION}_${MODEL}_hist${HISTORY_HOURS}h_${RUNSTAMP}.log"
  cp -f "${CONFIG_PATH}" "${RUN_DIR}/infer_config_used.sh" || true

  echo "-----------------------------------------" | tee -a "${LOG_FILE}"
  echo "[RUN] NAME=${NAME}"                         | tee -a "${LOG_FILE}"
  echo "[RUN] TEST_ROOT_DIR=${TEST_ROOT_DIR:-<empty => NCEP year-split test>}" | tee -a "${LOG_FILE}"
  echo "[RUN] OUT_DIR=${OUT_DIR}"                  | tee -a "${LOG_FILE}"
  echo "-----------------------------------------" | tee -a "${LOG_FILE}"

  TEST_ARGS=()
  if [[ -n "${TEST_ROOT_DIR}" ]]; then
    TEST_ARGS=(--test_root_dir "${TEST_ROOT_DIR}")
  fi

  "${PYTHON_BIN}" -u "${INFER_PY}" \
    --root_dir "${ROOT_DIR}" \
    "${TEST_ARGS[@]}" \
    "${STATION_ARGS[@]}" \
    --station_json_dir "${STATION_JSON_DIR}" \
    --model "${MODEL}" \
    --history_hours "${HISTORY_HOURS}" \
    --batch_size "${BATCH_SIZE}" \
    --ckpt "${CKPT_RESOLVED}" \
    --out_dir "${OUT_DIR}" \
    --save_npz \
    "${YEARS_ARGS[@]}" \
    "${AMP_ARGS[@]}" \
    "${TF32_ARGS[@]}" \
    "${THREAD_ARGS[@]}" \
    "${DL_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo "[DONE RUN] outputs in: ${OUT_DIR}" | tee -a "${LOG_FILE}"
done

echo "[DONE ALL] all runs finished."
