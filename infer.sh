#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  echo "[TRAP] got SIGINT/SIGTERM — cleaning up children"
  pkill -P $$ 2>/dev/null || true
}
trap cleanup INT TERM

# ============================================================
# infer.sh
# - Launch inference inside tmux (interactive use)
# - Sources configs/infer_config.sh (single source of truth)
# - Passes ONLY arguments that exist in infer.py
# ============================================================

# =========================
# Config loading
# =========================
CONFIG_PATH="${1:-configs/infer_config.sh}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] config not found: ${CONFIG_PATH}"
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

# Safe defaults (older configs won’t break)
: "${INFER_PY:=infer.py}"
: "${ROOT_DIR:=./Data/NCEP/graphs}"
: "${TEST_ROOT_DIR:=}"
: "${STATION:=Battery}"
: "${NAME:=stormsurge_infer}"
: "${MODEL:=baseline}"
: "${HISTORY_HOURS:=12}"
: "${BATCH_SIZE:=1}"
: "${STATION_JSON_DIR:=./station_json}"
: "${YEARS:=}"

# Debug defaults (if not set in config)
: "${CUDA_LAUNCH_BLOCKING_FLAG:=0}"
: "${TORCH_GPU_PROBE:=0}"
: "${FAIL_IF_NO_SLURM:=0}"
: "${DO_CONDA:=1}"
: "${CONDA_MODULE:=anaconda}"
: "${CONDA_SH:=/software/u22/anaconda/python3.9/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=torchpyg-cu124}"

# Speed defaults (if not set in config)
: "${USE_AMP:=1}"
: "${AMP_DTYPE:=bf16}"
: "${USE_TF32:=1}"
: "${TORCH_THREADS:=1}"

: "${NUM_WORKERS:=0}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=0}"
: "${MP_CONTEXT:=fork}"

# Required
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

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

# =========================
# Helper: resolve checkpoint path (supports glob patterns)
# =========================
resolve_ckpt () {
  local pat="$1"
  if [[ -z "$pat" ]]; then
    return 1
  fi
  if [[ -f "$pat" ]]; then
    echo "$pat"
    return 0
  fi
  shopt -s nullglob
  # shellcheck disable=SC2206
  local arr=( $pat )
  shopt -u nullglob
  if [[ "${#arr[@]}" -eq 0 ]]; then
    return 1
  fi
  ls -1t "${arr[@]}" 2>/dev/null | head -n 1
}

CKPT_RESOLVED="$(resolve_ckpt "${CKPT_PATH}" || true)"
if [[ -z "${CKPT_RESOLVED}" ]]; then
  echo "[FATAL] CKPT_PATH does not resolve to a file: ${CKPT_PATH}"
  exit 2
fi

# =========================
# Logging / tmux
# =========================
SESSION_BASE="${NAME}"
LOG_ROOT="logs_infer_${STATION}"

WORKDIR="$(pwd)"
RUNSTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${WORKDIR}/${LOG_ROOT}/${SESSION_BASE}_${RUNSTAMP}"
mkdir -p "${RUN_DIR}"
SESSION_NAME="${SESSION_BASE}_${RUNSTAMP}"

OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${OUT_DIR}"

TEST_TAG="NCEP"
if [[ -n "${TEST_ROOT_DIR}" ]]; then
  TEST_TAG="CMIP6"
fi

echo "========================================="
echo "Config file:   ${CONFIG_PATH}"
echo "Workdir:       ${WORKDIR}"
echo "Run dir:       ${RUN_DIR}"
echo "Out dir:       ${OUT_DIR}"
echo "Station:       ${STATION}"
echo "Test tag:      ${TEST_TAG}"
echo "Infer script:  ${INFER_PY}"
echo "Checkpoint:    ${CKPT_RESOLVED}"
echo "ROOT_DIR:      ${ROOT_DIR}"
echo "TEST_ROOT_DIR: ${TEST_ROOT_DIR:-<empty => NCEP year-split test>}"
echo "Model:         ${MODEL}"
echo "History hours: ${HISTORY_HOURS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Years:         ${YEARS:-<all>}"
echo "========================================="

RUNNER="${RUN_DIR}/run_infer.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${WORKDIR}"

# Re-source config inside tmux (single source of truth)
set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

: "\${INFER_PY:=infer.py}"
: "\${ROOT_DIR:=./Data/NCEP/graphs}"
: "\${TEST_ROOT_DIR:=}"
: "\${STATION:=Battery}"
: "\${MODEL:=baseline}"
: "\${HISTORY_HOURS:=12}"
: "\${BATCH_SIZE:=1}"
: "\${STATION_JSON_DIR:=./station_json}"
: "\${YEARS:=}"

: "\${CUDA_LAUNCH_BLOCKING_FLAG:=0}"
: "\${TORCH_GPU_PROBE:=0}"
: "\${FAIL_IF_NO_SLURM:=0}"
: "\${DO_CONDA:=1}"
: "\${CONDA_MODULE:=anaconda}"
: "\${CONDA_SH:=/software/u22/anaconda/python3.9/etc/profile.d/conda.sh}"
: "\${CONDA_ENV:=torchpyg-cu124}"

: "\${USE_AMP:=1}"
: "\${AMP_DTYPE:=bf16}"
: "\${USE_TF32:=1}"
: "\${TORCH_THREADS:=1}"

: "\${NUM_WORKERS:=0}"
: "\${PIN_MEMORY:=0}"
: "\${PERSISTENT_WORKERS:=0}"
: "\${PREFETCH_FACTOR:=0}"
: "\${MP_CONTEXT:=fork}"

: "\${CKPT_PATH:=}"

init_conda () {
  if [[ "\${DO_CONDA}" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -f "\${CONDA_SH}" ]]; then
    echo "[WARN] conda.sh not found at \${CONDA_SH}. Skipping conda activation."
    return 0
  fi
  set +u
  module load "\${CONDA_MODULE}"
  # shellcheck disable=SC1090
  source "\${CONDA_SH}"
  conda activate "\${CONDA_ENV}"
  hash -r
  set -u
}

init_conda

if [[ -n "\${CONDA_PREFIX:-}" && -x "\${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="\${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="\$(command -v python)"
fi

export PYTHONUNBUFFERED=1

# CUDA debug
if [[ "\${CUDA_LAUNCH_BLOCKING_FLAG}" -eq 1 ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

# Bind to Slurm-assigned GPU (best effort; harmless when not in Slurm)
if [[ -n "\${SLURM_STEP_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="\${SLURM_STEP_GPUS}"
elif [[ -n "\${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="\${SLURM_JOB_GPUS}"
fi

# Resolve checkpoint (supports glob patterns)
resolve_ckpt () {
  local pat="\$1"
  if [[ -z "\$pat" ]]; then return 1; fi
  if [[ -f "\$pat" ]]; then echo "\$pat"; return 0; fi
  shopt -s nullglob
  # shellcheck disable=SC2206
  local arr=( \$pat )
  shopt -u nullglob
  if [[ "\${#arr[@]}" -eq 0 ]]; then return 1; fi
  ls -1t "\${arr[@]}" 2>/dev/null | head -n 1
}
CKPT_RESOLVED="\$(resolve_ckpt "\${CKPT_PATH}" || true)"
if [[ -z "\${CKPT_RESOLVED}" ]]; then
  echo "[FATAL] CKPT_PATH does not resolve to a file: \${CKPT_PATH}"
  exit 2
fi

LOG_FILE="${RUN_DIR}/infer_\${STATION}_${TEST_TAG}_\${MODEL}_hist\${HISTORY_HOURS}h_${RUNSTAMP}.log"
: > "\${LOG_FILE}"

# Optional: basic GPU / torch probe (uses your existing config flags)
{
  echo "========================================="
  echo "[tmux] host=\$(hostname)"
  echo "[tmux] SLURM_JOB_ID=\${SLURM_JOB_ID:-<none>}"
  echo "[tmux] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[tmux] CUDA_LAUNCH_BLOCKING=\${CUDA_LAUNCH_BLOCKING:-0}"
  echo "[tmux] ckpt=\${CKPT_RESOLVED}"
  echo "========================================="

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi

  if [[ "\${TORCH_GPU_PROBE}" -eq 1 ]]; then
    "\${PYTHON_BIN}" - <<'PY'
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
} 2>&1 | tee -a "\${LOG_FILE}"

if [[ "\${FAIL_IF_NO_SLURM}" -eq 1 ]] && [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "[FATAL] SLURM_JOB_ID is empty." | tee -a "\${LOG_FILE}"
  exit 2
fi

TF32_ARGS=()
if [[ "\${USE_TF32}" -eq 1 ]]; then TF32_ARGS=(--tf32); fi

AMP_ARGS=()
if [[ "\${USE_AMP}" -eq 1 ]]; then AMP_ARGS=(--amp --amp_dtype "\${AMP_DTYPE}"); fi

STATION_ARGS=()
if [[ -n "\${STATION}" ]]; then STATION_ARGS=(--station "\${STATION}"); fi

TEST_ARGS=()
if [[ -n "\${TEST_ROOT_DIR}" ]]; then TEST_ARGS=(--test_root_dir "\${TEST_ROOT_DIR}"); fi

YEARS_ARGS=()
if [[ -n "\${YEARS}" ]]; then YEARS_ARGS=(--years "\${YEARS}"); fi

DL_ARGS=(--num_workers "\${NUM_WORKERS}" --prefetch_factor "\${PREFETCH_FACTOR}" --mp_context "\${MP_CONTEXT}")
if [[ "\${PIN_MEMORY}" -eq 1 ]]; then DL_ARGS+=(--pin_memory); fi
if [[ "\${PERSISTENT_WORKERS}" -eq 1 ]]; then DL_ARGS+=(--persistent_workers); fi

THREAD_ARGS=(--torch_threads "\${TORCH_THREADS}")

# IMPORTANT: pass ONLY args that exist in infer.py
"\${PYTHON_BIN}" -u "\${INFER_PY}" \\
  --root_dir "\${ROOT_DIR}" \\
  "\${TEST_ARGS[@]}" \\
  "\${STATION_ARGS[@]}" \\
  --station_json_dir "\${STATION_JSON_DIR}" \\
  --model "\${MODEL}" \\
  --history_hours "\${HISTORY_HOURS}" \\
  --batch_size "\${BATCH_SIZE}" \\
  --ckpt "\${CKPT_RESOLVED}" \\
  --out_dir "${OUT_DIR}" \\
  --save_npz \\
  "\${YEARS_ARGS[@]}" \\
  "\${AMP_ARGS[@]}" \\
  "\${TF32_ARGS[@]}" \\
  "\${THREAD_ARGS[@]}" \\
  "\${DL_ARGS[@]}" \\
  2>&1 | tee -a "\${LOG_FILE}"

echo "[DONE] outputs in: ${OUT_DIR}" | tee -a "\${LOG_FILE}"
EOF

chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION_NAME}" -c "${WORKDIR}" "${RUNNER}"

echo "Started inference in tmux: ${SESSION_NAME}"
echo "Attach:  tmux attach -t ${SESSION_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Runner:  ${RUNNER}"
