#!/usr/bin/env bash
set -euo pipefail

# Run the BabyLM 2026 Strict LLaMA SuperGLUE fine-tuning evals missing from the
# main architecture sweep.
#
# The default 12l768 runs, lr0p01/bs64/seeds 42-44, have already been
# evaluated, as have the 12l896 lr0p01/bs64 runs. The missing architecture
# family is:
#   - 14l768, Muon LR 0.01, batch 64, seeds 42/43/44
#
# If you also want the non-default 12l768 sweep leftovers that have zero-shot
# evals but no SuperGLUE fine-tuning, set:
#   BABYSTEPS_INCLUDE_NONDEFAULT_12L768=1
#
# Dry run:
#   BABYSTEPS_DRY_RUN=1 bash scripts/eval/eval_superglue_llama_missing_100m.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

GPUS="${BABYSTEPS_GPUS:-1 2 3}"
SEEDS="${BABYSTEPS_SEEDS:-42 43 44}"
RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-_babylm2026strict}"
INCLUDE_NONDEFAULT_12L768="${BABYSTEPS_INCLUDE_NONDEFAULT_12L768:-0}"

run_group() {
  local model_size="$1"
  local batch_size="$2"
  local muon_lr="$3"
  local arch_spec="$4"

  echo
  echo "=== SuperGLUE missing 100M group ==="
  echo "  model: ${model_size}"
  echo "  arch spec: ${arch_spec}"
  echo "  Muon LR: ${muon_lr}"
  echo "  train/effective batch size in run name: ${batch_size}"
  echo "  seeds: ${SEEDS}"
  echo "  GPUs: ${GPUS}"

  BABYSTEPS_MODEL_ARCH="${BABYSTEPS_MODEL_ARCH:-llama}" \
  BABYSTEPS_MODEL_SIZE="$model_size" \
  BABYSTEPS_BPES="${BABYSTEPS_BPES:-16384}" \
  BABYSTEPS_MUON_LRS="$muon_lr" \
  BABYSTEPS_BATCH_SIZE="$batch_size" \
  BABYSTEPS_SEEDS="$SEEDS" \
  BABYSTEPS_RUN_SUFFIX="$RUN_SUFFIX" \
  BABYSTEPS_GPUS="$GPUS" \
    bash scripts/eval/eval_superglue_finetune_sweep.sh
}

echo "BabyLM 2026 Strict LLaMA missing SuperGLUE fine-tuning evals"
echo "  default GPUs: ${GPUS}"
echo "  default seeds: ${SEEDS}"
echo "  run suffix: ${RUN_SUFFIX}"
echo "  tasks: ${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli}"
echo "  include non-default 12l768 leftovers: ${INCLUDE_NONDEFAULT_12L768}"

if [[ "$INCLUDE_NONDEFAULT_12L768" == "1" ]]; then
  run_group "12l768" "32" "0.01" "12,768,12,12,2048,32"
  run_group "12l768" "64" "0.02" "12,768,12,12,2048,16"
  run_group "12l768" "32" "0.05" "12,768,12,12,2048,32"
  run_group "12l768" "64" "0.05" "12,768,12,12,2048,16"
fi

run_group "14l768" "64" "0.01" "14,768,12,12,2048,32"

echo
echo "All missing 100M LLaMA SuperGLUE groups finished."
