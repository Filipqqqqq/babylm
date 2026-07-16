#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Transfer test: keep the SuperGLUE fine-tuning recipe fixed while varying only
# the AdamW auxiliary learning rate used during LLaMA pretraining.
GPU="${BABYSTEPS_SUPERGLUE_GPU:-0}"
FINETUNE_LR="${BABYSTEPS_SUPERGLUE_LR:-3e-5}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-${RUNS_DIR}/evaluation_superglue_finetune_llama_auxlr_transfer}"
LOG_DIR="${BABYSTEPS_SUPERGLUE_LOG_DIR:-${RUNS_DIR}/logs/superglue_finetune_llama_auxlr_transfer}"

AUXLR_1E4="${RUNS_DIR}/training/llama_12l896_bpe16384_strict_muon_auxlr0p0001_seed42/20260711_081550/final"
AUXLR_2E4="${RUNS_DIR}/training/llama_12l896_bpe16384_strict_muon_auxlr0p0002_seed42/20260711_120208/final"

for checkpoint in "$AUXLR_1E4" "$AUXLR_2E4"; do
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing pretrained checkpoint: $checkpoint" >&2
    exit 1
  fi
done

exec env \
  BABYSTEPS_SUPERGLUE_GPU="$GPU" \
  BABYSTEPS_SUPERGLUE_LR="$FINETUNE_LR" \
  BABYSTEPS_SUPERGLUE_OUTPUT_ROOT="$OUTPUT_ROOT" \
  BABYSTEPS_SUPERGLUE_LOG_DIR="$LOG_DIR" \
  bash "$ROOT/scripts/eval/eval_superglue_llama_nextlat_gpu3.sh" \
    "llama_auxlr1e4=${AUXLR_1E4}" \
    "llama_auxlr2e4=${AUXLR_2E4}"
