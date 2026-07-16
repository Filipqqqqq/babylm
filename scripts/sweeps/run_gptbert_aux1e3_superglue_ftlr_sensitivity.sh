#!/usr/bin/env bash
set -euo pipefail

# Reproduce the original strict-small GPT-BERT corpus with Muon and an AdamW
# auxiliary LR of 1e-3, then test SuperGLUE fine-tuning-LR sensitivity for that
# checkpoint and the fixed 1.5% linguistic-repair checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:-3}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
TRAIN_CONFIG="${BABYSTEPS_TRAIN_CONFIG:-configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_muon.yaml}"
RUN_NAME="${BABYSTEPS_RUN_NAME:-gpt_bert_strict_small_muon_auxlr1e3_repro_seed42_$(date +%Y%m%d_%H%M%S)}"
PRETRAIN_LOG_DIR="${BABYSTEPS_PRETRAIN_LOG_DIR:-${RUNS_DIR}/logs/gptbert_aux1e3_repro}"
PRETRAIN_LOG="${PRETRAIN_LOG_DIR}/${RUN_NAME}.log"
OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-${RUNS_DIR}/evaluation_superglue_finetune_gptbert_ftlr_sensitivity}"
LOG_ROOT="${BABYSTEPS_SUPERGLUE_LOG_ROOT:-${RUNS_DIR}/logs/superglue_finetune_gptbert_ftlr_sensitivity}"
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
FINETUNE_LRS=(${BABYSTEPS_SUPERGLUE_LRS:-1e-5 3e-5 8e-5})

REPAIR_CHECKPOINT="${BABYSTEPS_REPAIR_CHECKPOINT:-${RUNS_DIR}/training/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_20260712_144335/final}"
REPAIR_3E5_OUTPUT_ROOT="${RUNS_DIR}/evaluation_superglue_finetune_linguistic_repair_only/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_20260712_144335"
REPAIR_3E5_LOG_DIR="${RUNS_DIR}/logs/superglue_finetune_linguistic_repair_only/gpt_bert_mixed_babylm2026_strict_small_linguistic_repair_1p5pct_muon_20260712_144335"

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_superglue_arm() {
  local family="$1"
  local checkpoint="$2"
  local lr="$3"
  local lr_slug
  local model_name
  local arm_output_root
  local arm_log_dir

  lr_slug="$(slug "$lr")"
  model_name="gptbert_${family}_ftlr${lr_slug}"
  arm_output_root="${OUTPUT_ROOT}/${family}/lr${lr}"
  arm_log_dir="${LOG_ROOT}/${family}/lr${lr}"

  # Reuse the already completed, identically configured repair 3e-5 arm.
  if [[ "$family" == "repair1p5" && "$lr" == "3e-5" ]]; then
    model_name="gptbert_linguistic_repair_1p5pct"
    arm_output_root="$REPAIR_3E5_OUTPUT_ROOT"
    arm_log_dir="$REPAIR_3E5_LOG_DIR"
  fi

  printf '\nSuperGLUE: family=%s fine-tuning_lr=%s GPU=%s\n' "$family" "$lr" "$GPU"
  env \
    BABYSTEPS_SUPERGLUE_GPU="$GPU" \
    BABYSTEPS_SUPERGLUE_LR="$lr" \
    BABYSTEPS_FINETUNE_SEED="$FINETUNE_SEED" \
    BABYSTEPS_SUPERGLUE_OUTPUT_ROOT="$arm_output_root" \
    BABYSTEPS_SUPERGLUE_LOG_DIR="$arm_log_dir" \
    bash scripts/eval/eval_superglue_llama_nextlat_gpu3.sh \
      "${model_name}=${checkpoint}"
}

if [[ ! -f "$TRAIN_CONFIG" ]]; then
  printf 'Missing training config: %s\n' "$TRAIN_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$REPAIR_CHECKPOINT" ]]; then
  printf 'Missing linguistic-repair checkpoint: %s\n' "$REPAIR_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$PRETRAIN_LOG_DIR" "$OUTPUT_ROOT" "$LOG_ROOT"

printf 'Fresh GPT-BERT Muon reproduction\n'
printf '  GPU: %s\n' "$GPU"
printf '  run: %s\n' "$RUN_NAME"
printf '  AdamW auxiliary LR: 1e-3\n'
printf '  Muon LR: 1e-2\n'
printf '  SuperGLUE LRs: %s\n' "${FINETUNE_LRS[*]}"
printf '  pretraining log: %s\n' "$PRETRAIN_LOG"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  conda run --no-capture-output -n "$CONDA_ENV" \
  python -u scripts/train/train_hybrid_lm.py \
    --config "$TRAIN_CONFIG" \
    --set "run.name=${RUN_NAME}" \
    --set "run.append_timestamp=false" \
    --set "run.seed=42" \
    --set "training.optimizer=muon_pe" \
    --set "training.learning_rate=0.001" \
    --set "training.muon_lr=0.01" \
    >"$PRETRAIN_LOG" 2>&1

BASELINE_CHECKPOINT="${RUNS_DIR}/training/${RUN_NAME}/final"
if [[ ! -d "$BASELINE_CHECKPOINT" ]]; then
  printf 'Pretraining finished without a final checkpoint: %s\n' "$BASELINE_CHECKPOINT" >&2
  exit 1
fi

for lr in "${FINETUNE_LRS[@]}"; do
  run_superglue_arm baseline_aux1e3 "$BASELINE_CHECKPOINT" "$lr"
done

for lr in "${FINETUNE_LRS[@]}"; do
  run_superglue_arm repair1p5 "$REPAIR_CHECKPOINT" "$lr"
done

printf '\nAll GPT-BERT pretraining and SuperGLUE LR-sensitivity arms finished.\n'
