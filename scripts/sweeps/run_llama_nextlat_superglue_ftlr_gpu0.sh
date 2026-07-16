#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

GPU="${BABYSTEPS_SUPERGLUE_GPU:-0}"
FINETUNE_LRS=(${BABYSTEPS_SUPERGLUE_LRS:-8e-5 2e-4})
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
EVAL_SCRIPT="$ROOT/scripts/eval/eval_superglue_llama_nextlat_gpu3.sh"

CHECKPOINT_SPECS=(
  "nonlatent_baseline=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_babylm2026_strict_muon_liger_baseline_causal_diagnostics_50m_bs64/20260704_082711/final"
  "nextlat_paperfixed_d1=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_babylm100m_nextlat_paperfixed_h1_dynlr4em4_kl0p1_seed42/20260712_151122/final"
  "nextlat_paperfixed_d2=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_babylm100m_nextlat_paperfixed_h2_dynlr4em4_kl0p1_seed42/20260712_151122/final"
)

for spec in "${CHECKPOINT_SPECS[@]}"; do
  checkpoint="${spec#*=}"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

for lr in "${FINETUNE_LRS[@]}"; do
  output_root="$RUNS_DIR/evaluation_superglue_finetune_nextlat_paperfixed_ftlr_sweep/lr${lr}"
  log_dir="$RUNS_DIR/logs/superglue_finetune_nextlat_paperfixed_ftlr_sweep/lr${lr}"

  echo
  echo "SuperGLUE sweep: lr=$lr GPU=$GPU"
  env \
    BABYSTEPS_SUPERGLUE_GPU="$GPU" \
    BABYSTEPS_SUPERGLUE_LR="$lr" \
    BABYSTEPS_SUPERGLUE_OUTPUT_ROOT="$output_root" \
    BABYSTEPS_SUPERGLUE_LOG_DIR="$log_dir" \
    "$EVAL_SCRIPT" "${CHECKPOINT_SPECS[@]}"
done

echo
echo "All requested LLaMA baseline/NextLat SuperGLUE fine-tuning runs finished."
