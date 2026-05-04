#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-pact_future_only_0430}"

CONFIGS=(
  "configs/configs_train/AWI/train_config_AWI_Battery_P3_Best_FutureOnly.sh"
  "configs/configs_train/CNRM/train_config_CNRM_Battery_P3_Best_FutureOnly.sh"
  "configs/configs_train/EC_EARTH/train_config_EC_EARTH_Battery_P3_Best_FutureOnly.sh"
  "configs/configs_train/MPI/train_config_MPI_Battery_P3_Best_FutureOnly.sh"
  "configs/configs_train/MRI/train_config_MRI_Battery_P3_Best_FutureOnly.sh"
)

run_inner() {
  cd "${SCRIPT_DIR}"

  echo "[driver] workdir: ${SCRIPT_DIR}"
  echo "[driver] started: $(date)"
  echo

  for cfg in "${CONFIGS[@]}"; do
    echo "===== START ${cfg} $(date) ====="
    bash train.sh "${cfg}"
    echo "===== DONE ${cfg} $(date) ====="
    echo
  done

  echo "[driver] all future-only configs finished: $(date)"
}

if [[ "${1:-}" == "--inner" ]]; then
  run_inner
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[FATAL] tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -c "${SCRIPT_DIR}" \
  "bash -lc 'bash ./train_future_only_0430.sh --inner; status=\$?; echo; echo \"[tmux] train_future_only_0430.sh exited with status \$status\"; echo \"[tmux] leaving this shell open so you can inspect output\"; exec bash -l'"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
