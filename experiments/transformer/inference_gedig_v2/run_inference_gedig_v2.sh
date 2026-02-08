#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv311/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv311/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

MODELS="${MODELS:-bert-base-uncased,gpt2}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
DEVICE="${DEVICE:-auto}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/results}"

GRID_SEARCH="${GRID_SEARCH:-1}"
SHUFFLE_CONTROL="${SHUFFLE_CONTROL:-1}"
RANDOM_CONTROL="${RANDOM_CONTROL:-0}"
SAVE_SAMPLES="${SAVE_SAMPLES:-1}"

GRID_LAMBDA="${GRID_LAMBDA:-0.01,0.1,0.5,1,2,5,10}"
GRID_GAMMA="${GRID_GAMMA:-0.01,0.1,0.5,1,2,5,10}"
PROJ_DIM="${PROJ_DIM:-128}"

B_DIST="${B_DIST:-}"
B_DEPTH="${B_DEPTH:-}"
TEXT_FILE="${TEXT_FILE:-}"

mkdir -p "${OUTPUT_BASE}"

# Sandbox/CI-friendly runtime defaults
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_BASE}/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUTPUT_BASE}/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

IFS=',' read -r -a MODEL_ARRAY <<< "${MODELS}"

echo "[info] python=${PYTHON_BIN}"
echo "[info] models=${MODELS}"
echo "[info] output_base=${OUTPUT_BASE}"

for model in "${MODEL_ARRAY[@]}"; do
  model="$(echo "${model}" | xargs)"
  [[ -z "${model}" ]] && continue

  safe_model="${model//\//_}"
  model_output="${OUTPUT_BASE}/${safe_model}"
  mkdir -p "${model_output}"

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference_gedig_v2.py"
    --model "${model}"
    --max-samples "${MAX_SAMPLES}"
    --max-length "${MAX_LENGTH}"
    --device "${DEVICE}"
    --output "${model_output}"
    --proj-dim "${PROJ_DIM}"
    --grid-lambda "${GRID_LAMBDA}"
    --grid-gamma "${GRID_GAMMA}"
  )

  if [[ "${GRID_SEARCH}" == "1" ]]; then
    cmd+=(--grid-search)
  fi
  if [[ "${SHUFFLE_CONTROL}" == "1" ]]; then
    cmd+=(--shuffle-control)
  fi
  if [[ "${RANDOM_CONTROL}" == "1" ]]; then
    cmd+=(--random-control)
  fi
  if [[ "${SAVE_SAMPLES}" == "1" ]]; then
    cmd+=(--save-samples)
  fi
  if [[ -n "${B_DIST}" ]]; then
    cmd+=(--b-dist "${B_DIST}")
  fi
  if [[ -n "${B_DEPTH}" ]]; then
    cmd+=(--b-depth "${B_DEPTH}")
  fi
  if [[ -n "${TEXT_FILE}" ]]; then
    cmd+=(--text-file "${TEXT_FILE}")
  fi

  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "[run] model=${model}"
  "${cmd[@]}"

  echo "[plot] model=${model}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_inference_gedig_v2.py" \
    --results-dir "${model_output}" \
    --latest
done

echo "[done] inference geDIG v2 batch run completed"
