#!/usr/bin/env bash
set -euo pipefail

# Matched BabyLM 100M-track NextLat experiment.  The only difference between
# the two arms is the recursively supervised training horizon (d=1 vs d=2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml}"
GPU_D1="${BABYSTEPS_GPU_D1:-1}"
GPU_D2="${BABYSTEPS_GPU_D2:-3}"
SEED="${BABYSTEPS_SEED:-42}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-${RUNS_DIR}/logs/llama_nextlat_paperfixed}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing training config: $CONFIG" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

launch_arm() {
  local horizon="$1"
  local gpu="$2"
  local name="llama_12l896_babylm100m_nextlat_paperfixed_h${horizon}_dynlr4em4_kl0p1_seed${SEED}"
  local log_path="${LOG_DIR}/${name}.log"
  local command=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${SEED}"
    --set "training.epochs=${EPOCHS}"
    --set "nextlat.horizon=${horizon}"
  )

  echo "Launching d=${horizon} on GPU ${gpu}"
  echo "  dynamics AdamW LR: 4e-4; KL lambda: 0.1"
  echo "  rollout diagnostics: h=1,2,4,8 at 200M and 300M words"
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}" >"$log_path" 2>&1 &
  ARM_PID=$!
}

echo "LLaMA paper-fixed NextLat horizon pair"
echo "  config: ${CONFIG}"
echo "  seed: ${SEED}; epochs: ${EPOCHS}"

launch_arm 1 "$GPU_D1"
PID_D1="${ARM_PID:-}"
launch_arm 2 "$GPU_D2"
PID_D2="${ARM_PID:-}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

status=0
wait "$PID_D1" || status=$?
wait "$PID_D2" || status=$?
exit "$status"
