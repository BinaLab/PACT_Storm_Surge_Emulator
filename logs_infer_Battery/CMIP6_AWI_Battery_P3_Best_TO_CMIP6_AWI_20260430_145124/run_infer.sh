#!/usr/bin/env bash
set -euo pipefail

cd "/home/exouser/Documents/PACT-Data/Emulator"

# Re-source config inside tmux (single source of truth)
set +u
# shellcheck disable=SC1090
source "configs/configs_infer/infer_config_AWI.sh"
set -u

: "${INFER_PY:=infer.py}"
: "${ROOT_DIR:=./Data/NCEP/graphs}"
: "${TEST_ROOT_DIR:=}"
: "${STATION:=Battery}"
: "${MODEL:=baseline}"
: "${HISTORY_HOURS:=12}"
: "${BATCH_SIZE:=1}"
: "${STATION_JSON_DIR:=./station_json}"
: "${YEARS:=}"

: "${CUDA_LAUNCH_BLOCKING_FLAG:=0}"
: "${TORCH_GPU_PROBE:=0}"
: "${FAIL_IF_NO_SLURM:=0}"
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

# CUDA debug
if [[ "${CUDA_LAUNCH_BLOCKING_FLAG}" -eq 1 ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

# Bind to Slurm-assigned GPU (best effort; harmless when not in Slurm)
if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

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

LOG_FILE="/home/exouser/Documents/PACT-Data/Emulator/logs_infer_Battery/CMIP6_AWI_Battery_P3_Best_TO_CMIP6_AWI_20260430_145124/infer_${STATION}_CMIP6_${MODEL}_hist${HISTORY_HOURS}h_20260430_145124.log"
: > "${LOG_FILE}"

# Optional: basic GPU / torch probe (uses your existing config flags)
{
  echo "========================================="
  echo "[tmux] host=$(hostname)"
  echo "[tmux] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
  echo "[tmux] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[tmux] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
  echo "[tmux] ckpt=${CKPT_RESOLVED}"
  echo "========================================="

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi

  if [[ "${TORCH_GPU_PROBE}" -eq 1 ]]; then
    "${PYTHON_BIN}" - <<'PY'
import torch
print("[tmux][torch] torch:", torch.__version__)
print("[tmux][torch] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[tmux][torch] device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        try:
            print(f"[tmux][torch] {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"[tmux][torch] {i}: <error> {e}")
PY
  fi
  echo "========================================="
} 2>&1 | tee -a "${LOG_FILE}"

if [[ "${FAIL_IF_NO_SLURM}" -eq 1 ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[FATAL] SLURM_JOB_ID is empty." | tee -a "${LOG_FILE}"
  exit 2
fi

TF32_ARGS=()
if [[ "${USE_TF32}" -eq 1 ]]; then TF32_ARGS=(--tf32); fi

AMP_ARGS=()
if [[ "${USE_AMP}" -eq 1 ]]; then AMP_ARGS=(--amp --amp_dtype "${AMP_DTYPE}"); fi

STATION_ARGS=()
if [[ -n "${STATION}" ]]; then STATION_ARGS=(--station "${STATION}"); fi

TEST_ARGS=()
if [[ -n "${TEST_ROOT_DIR}" ]]; then TEST_ARGS=(--test_root_dir "${TEST_ROOT_DIR}"); fi

YEARS_ARGS=()
if [[ -n "${YEARS}" ]]; then YEARS_ARGS=(--years "${YEARS}"); fi

DL_ARGS=(--num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" --mp_context "${MP_CONTEXT}")
if [[ "${PIN_MEMORY}" -eq 1 ]]; then DL_ARGS+=(--pin_memory); fi
if [[ "${PERSISTENT_WORKERS}" -eq 1 ]]; then DL_ARGS+=(--persistent_workers); fi

THREAD_ARGS=(--torch_threads "${TORCH_THREADS}")

# IMPORTANT: pass ONLY args that exist in infer.py
"${PYTHON_BIN}" -u "${INFER_PY}" \
  --root_dir "${ROOT_DIR}" \
  "${TEST_ARGS[@]}" \
  "${STATION_ARGS[@]}" \
  --station_json_dir "${STATION_JSON_DIR}" \
  --model "${MODEL}" \
  --history_hours "${HISTORY_HOURS}" \
  --batch_size "${BATCH_SIZE}" \
  --ckpt "${CKPT_RESOLVED}" \
  --out_dir "/home/exouser/Documents/PACT-Data/Emulator/logs_infer_Battery/CMIP6_AWI_Battery_P3_Best_TO_CMIP6_AWI_20260430_145124/outputs" \
  --save_npz \
  "${YEARS_ARGS[@]}" \
  "${AMP_ARGS[@]}" \
  "${TF32_ARGS[@]}" \
  "${THREAD_ARGS[@]}" \
  "${DL_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[DONE] outputs in: /home/exouser/Documents/PACT-Data/Emulator/logs_infer_Battery/CMIP6_AWI_Battery_P3_Best_TO_CMIP6_AWI_20260430_145124/outputs" | tee -a "${LOG_FILE}"
