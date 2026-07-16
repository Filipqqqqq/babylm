#!/usr/bin/env bash
set -euo pipefail

# Sequential matched NextLat d=1/d=2 run on one GPU. The only change from the
# paper-fixed kl0p1 pair is nextlat.lambda_kl=0.2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml}"
GPU="${BABYSTEPS_GPU:-1}"
SEED="${BABYSTEPS_SEED:-42}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
KL_LAMBDA="${BABYSTEPS_NEXTLAT_KL_LAMBDA:-0.2}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-${RUNS_DIR}/logs/llama_nextlat_paperfixed_kl0p2}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing training config: $CONFIG" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_arm() {
  local horizon="$1"
  local name="llama_12l896_babylm100m_nextlat_paperfixed_h${horizon}_dynlr4em4_kl0p2_seed${SEED}"
  local log_path="${LOG_DIR}/${name}.log"

  echo
  echo "Running d=${horizon} on GPU ${GPU}"
  echo "  KL lambda: ${KL_LAMBDA}; dynamics AdamW LR: 4e-4"
  echo "  epochs: ${EPOCHS}; seed: ${SEED}"
  echo "  log: ${log_path}"

  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    conda run --no-capture-output -n "$CONDA_ENV" \
    python -u scripts/train/train_causal_lm.py \
      --config "$CONFIG" \
      --set "run.name=${name}" \
      --set "run.seed=${SEED}" \
      --set "training.epochs=${EPOCHS}" \
      --set "nextlat.horizon=${horizon}" \
      --set "nextlat.lambda_kl=${KL_LAMBDA}" \
      >"$log_path" 2>&1
}

echo "LLaMA paper-fixed NextLat KL=0.2 horizon pair"
echo "  config: ${CONFIG}"
echo "  GPU: ${GPU} (d=1 then d=2 sequentially)"

run_arm 1
run_arm 2

echo
echo "Both KL=0.2 NextLat runs finished."
