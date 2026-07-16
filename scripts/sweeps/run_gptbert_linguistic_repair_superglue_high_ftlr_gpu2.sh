#!/usr/bin/env bash
set -euo pipefail

# Extend the fixed 1.5% linguistic-repair SuperGLUE fine-tuning LR sweep.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

GPU="${BABYSTEPS_GPU:-2}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
FINETUNE_LRS=(${BABYSTEPS_SUPERGLUE_LRS:-1e-4 2e-4})
CHECKPOINT="${BABYSTEPS_REPAIR_CHECKPOINT:-${RUNS_DIR}/training/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_20260712_144335/final}"
OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-${RUNS_DIR}/evaluation_superglue_finetune_gptbert_ftlr_sensitivity}"
LOG_ROOT="${BABYSTEPS_SUPERGLUE_LOG_ROOT:-${RUNS_DIR}/logs/superglue_finetune_gptbert_ftlr_sensitivity}"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Missing linguistic-repair checkpoint: $CHECKPOINT" >&2
  exit 1
fi

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

for lr in "${FINETUNE_LRS[@]}"; do
  lr_slug="$(slug "$lr")"
  model_name="gptbert_repair1p5_ftlr${lr_slug}"
  arm_output_root="${OUTPUT_ROOT}/repair1p5/lr${lr}"
  arm_log_dir="${LOG_ROOT}/repair1p5/lr${lr}"

  echo
  echo "SuperGLUE linguistic repair: lr=$lr GPU=$GPU"
  env \
    BABYSTEPS_SUPERGLUE_GPU="$GPU" \
    BABYSTEPS_SUPERGLUE_LR="$lr" \
    BABYSTEPS_FINETUNE_SEED="$FINETUNE_SEED" \
    BABYSTEPS_SUPERGLUE_OUTPUT_ROOT="$arm_output_root" \
    BABYSTEPS_SUPERGLUE_LOG_DIR="$arm_log_dir" \
    bash scripts/eval/eval_superglue_llama_nextlat_gpu3.sh \
      "${model_name}=${CHECKPOINT}"
done

echo
echo "Both high-LR linguistic-repair SuperGLUE runs finished."
