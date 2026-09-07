#!/usr/bin/env bash
# Shared artifact helpers for the single-run and multi-target launchers.

resolve_ckpt() {
  local pattern="$1" candidate newest=""
  [[ -n "${pattern}" ]] || return 1
  if [[ -f "${pattern}" ]]; then
    realpath -- "${pattern}"
    return
  fi
  while IFS= read -r candidate; do
    [[ -f "${candidate}" ]] || continue
    if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
      newest="${candidate}"
    fi
  done < <(compgen -G "${pattern}" || true)
  [[ -n "${newest}" ]] || return 1
  realpath -- "${newest}"
}

write_infer_config_snapshot() {
  local output_path="$1" name value
  local config_vars=(
    INFER_PY ROOT_DIR TEST_ROOT_DIR STATION STATION_JSON_DIR MODEL MODEL_LABEL
    ENCODER_TYPE TEMPORAL_BLOCK HEAD_TYPE HISTORY_HOURS BATCH_SIZE YEARS
    CNN_INTERMEDIATE_CHANNEL TIME_ENCODING
    USE_AMP AMP_DTYPE USE_TF32 TORCH_THREADS NUM_WORKERS PIN_MEMORY
    PERSISTENT_WORKERS PREFETCH_FACTOR MP_CONTEXT DO_CONDA CONDA_MODULE
    CONDA_SH CONDA_ENV CUDA_LAUNCH_BLOCKING_FLAG TORCH_GPU_PROBE FAIL_IF_NO_SLURM
    INFERENCE_RESULTS_ROOT WORKDIR RUN_DIR OUT_DIR RUNSTAMP TEST_TAG NAME
  )
  {
    printf '#!/usr/bin/env bash\n# Fully resolved inference configuration; no source-config dependencies.\n'
    printf '# Generated UTC: %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S')"
    printf 'ORIGINAL_CONFIG_PATH=%q\n' "${CONFIG_PATH}"
    for name in "${config_vars[@]}"; do
      if [[ -v "${name}" ]]; then
        value="${!name}"
        case "${name}" in
          INFER_PY|ROOT_DIR|TEST_ROOT_DIR|STATION_JSON_DIR|CONDA_SH|INFERENCE_RESULTS_ROOT)
            if [[ -n "${value}" && "${value}" != /* ]]; then
              value="${WORKDIR}/${value#./}"
            fi ;;
        esac
        printf '%s=%q\n' "${name}" "${value}"
      fi
    done
    printf 'CKPT_PATTERN=%q\n' "${CKPT_PATH}"
    printf 'CKPT_PATH=%q\nCKPT_RESOLVED=%q\n' "${CKPT_RESOLVED}" "${CKPT_RESOLVED}"
    # A per-target snapshot replays only this target when passed to infer_multi.sh.
    value="${TEST_ROOT_DIR:-}"
    if [[ -n "${value}" && "${value}" != /* ]]; then value="${WORKDIR}/${value#./}"; fi
    printf 'RUNS=(%q)\n' "${NAME:-snapshot}|${value}"
  } > "${output_path}"
}

write_infer_command() {
  local output_path="$1" name
  shift
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf '# Exact interpreter and arguments used for this run.\ncd -- %q\n' "${WORKDIR}"
    printf 'export PYTHONUNBUFFERED=1\n'
    for name in CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING; do
      if [[ -v "${name}" ]]; then printf 'export %s=%q\n' "${name}" "${!name}"; fi
    done
    printf 'exec'
    printf ' %q' "$@"
    printf '\n'
  } > "${output_path}"
}
