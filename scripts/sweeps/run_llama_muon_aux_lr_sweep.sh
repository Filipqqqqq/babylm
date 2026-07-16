#!/usr/bin/env bash
set -euo pipefail

# Sequential auxiliary-AdamW learning-rate sweep for the 12L/896 LLaMA Muon run.
#
# The hidden matrices always use Muon at 0.01.  This varies only the AdamW rate
# used for embeddings/the tied LM head, norm gains, and biases.  Each arm uses
# the same data order and seed, and records validation plus EWoK, BLIMP (+
# supplement), Entity Tracking, and COMPS every estimated 100M words.
#
# Default arms: 5e-5 (control), 1e-4, 2e-4.
#
# Preview commands without starting training:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/run_llama_muon_aux_lr_sweep.sh
#
# Run a shorter diagnostic:
#   BABYSTEPS_EPOCHS=1 bash scripts/sweeps/run_llama_muon_aux_lr_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_bpe16384_babylm2026_strict_muon_liger_causal_diagnostics.yaml}"
GPU="${BABYSTEPS_GPU:-0}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/llama_muon_aux_lr_sweep}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"
RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-}"

AUX_LRS=(${BABYSTEPS_AUX_LRS:-0.00005 0.0001 0.0002})
SEED="${BABYSTEPS_SEED:-42}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
BENCHMARK_EVERY_WORDS="${BABYSTEPS_BENCHMARK_EVERY_WORDS:-100000000}"

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing training config: $CONFIG" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

run_one() {
  local aux_lr="$1"
  local name="llama_12l896_bpe16384_strict_muon_auxlr$(slug "$aux_lr")_seed${SEED}${RUN_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local command=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${SEED}"
    --set "training.epochs=${EPOCHS}"
    --set "training.muon_lr=0.01"
    --set "training.learning_rate=${aux_lr}"
    --set "logging.eval_every_steps=0"
    --set "logging.eval_every_words=${BENCHMARK_EVERY_WORDS}"
    --set "ewok_context_sensitivity_full.every_steps=0"
    --set "ewok_context_sensitivity_full.every_words=${BENCHMARK_EVERY_WORDS}"
    --set "blimp_likelihood.every_steps=0"
    --set "blimp_likelihood.every_words=${BENCHMARK_EVERY_WORDS}"
    --set "entity_tracking_likelihood.every_steps=0"
    --set "entity_tracking_likelihood.every_words=${BENCHMARK_EVERY_WORDS}"
    --set "comps_likelihood.every_steps=0"
    --set "comps_likelihood.every_words=${BENCHMARK_EVERY_WORDS}"
  )

  echo "Launching ${name} on GPU ${GPU}"
  echo "  auxiliary AdamW LR: ${aux_lr}; Muon LR: 0.01"
  echo "  epochs: ${EPOCHS}; benchmark interval: ${BENCHMARK_EVERY_WORDS} estimated words"
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}" >"$log_path" 2>&1
  echo "Finished ${name}"
}

echo "LLaMA Muon auxiliary-AdamW LR sweep"
echo "  config: ${CONFIG}"
echo "  GPU: ${GPU}"
echo "  seed: ${SEED}"
echo "  arms: ${AUX_LRS[*]}"

for aux_lr in "${AUX_LRS[@]}"; do
  run_one "$aux_lr"
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Sweep complete. Logs: ${LOG_DIR}"
fi
