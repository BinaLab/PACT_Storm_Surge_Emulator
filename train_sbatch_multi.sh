#!/usr/bin/env bash
#SBATCH -J stormsurge_AWI
#SBATCH -p h100
#SBATCH -A TG-CIS250588
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # should match GPUs allocated / visible
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -t 4:00:00
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=zel220@lehigh.edu

set -euo pipefail

# -----------------------------
# User knobs
# -----------------------------
TRAIN_SH="./train.sh"

# Put configs here (space-separated). You can also build this list dynamically.
CONFIGS=(
  "configs/AWI/train_config_AWI_Battery_Baseline_12h_PastOnly.sh"
  "configs/AWI/train_config_AWI_Battery_P3_PastOnly.sh"
  "configs/AWI/train_config_AWI_Battery_P3_Best_PastOnly.sh"
)

# Optional: stop after the first failure (1) or continue to next config (0)
STOP_ON_FAIL=0

# Optional: keep a per-job folder for wrapper logs
WRAP_LOG_DIR="slurm_wrapper_logs/${SLURM_JOB_ID}"
mkdir -p "${WRAP_LOG_DIR}"

# -----------------------------
# Basic sanity prints
# -----------------------------
echo "========================================="
echo "JobId:        ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Workdir:      $(pwd)"
echo "TRAIN_SH:     ${TRAIN_SH}"
echo "NTASKS/NODE:  ${SLURM_NTASKS_PER_NODE:-<unset>}"
echo "CVD:          ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Configs:      ${CONFIGS[*]}"
echo "========================================="

# -----------------------------
# Run sequentially
# -----------------------------
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i+1))
  if [[ ! -f "${cfg}" ]]; then
    echo "[FATAL] config not found: ${cfg}"
    exit 2
  fi

  echo
  echo "============================================================"
  echo "[${i}/${#CONFIGS[@]}] Running config: ${cfg}"
  echo "============================================================"

  # Wrapper log (train.sh already writes its own logs_* inside)
  WRAP_LOG="${WRAP_LOG_DIR}/wrapper_${i}_$(basename "${cfg}").log"

  set +e
  bash "${TRAIN_SH}" "${cfg}" 2>&1 | tee "${WRAP_LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] config failed: ${cfg} (exit=${rc})"
    if [[ "${STOP_ON_FAIL}" -eq 1 ]]; then
      echo "[FATAL] STOP_ON_FAIL=1 → exiting."
      exit "${rc}"
    else
      echo "[WARN] continuing to next config (STOP_ON_FAIL=0)."
    fi
  fi
done

echo
echo "ALL DONE. Wrapper logs: ${WRAP_LOG_DIR}"
