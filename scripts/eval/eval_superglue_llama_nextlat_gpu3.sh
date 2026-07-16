#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Sequential SuperGLUE fine-tuning eval for the LLaMA baseline/NextLat ablation.
# The official BabyLM finetuner names local checkpoint outputs after the
# checkpoint directory basename, so each pretrained model gets its own results
# root to avoid collisions between multiple ".../final" checkpoints.

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_SUPERGLUE_GPU:-3}"
PIPELINE_DIR="${BABYSTEPS_EVAL_PIPELINE_DIR:-${ROOT}/external/babylm-eval/strict}"
EVAL_DATA_DIR="${BABYSTEPS_FULL_EVAL_DATA_DIR:-${PIPELINE_DIR}/evaluation_data/full_eval}"
OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/evaluation_superglue_finetune_nextlat_llama}"
LOG_DIR="${BABYSTEPS_SUPERGLUE_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/superglue_finetune_nextlat_llama}"

TASKS=(${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli})

FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
LR="${BABYSTEPS_SUPERGLUE_LR:-3e-5}"
BSZ="${BABYSTEPS_SUPERGLUE_BATCH_SIZE:-32}"
BIG_BSZ="${BABYSTEPS_SUPERGLUE_BIG_BATCH_SIZE:-16}"
MAX_EPOCHS="${BABYSTEPS_SUPERGLUE_EPOCHS:-10}"
WSC_EPOCHS="${BABYSTEPS_SUPERGLUE_WSC_EPOCHS:-30}"
SEQUENCE_LENGTH="${BABYSTEPS_SUPERGLUE_SEQUENCE_LENGTH:-512}"
VALID_BATCH_SIZE="${BABYSTEPS_SUPERGLUE_VALID_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION="${BABYSTEPS_SUPERGLUE_GRADIENT_ACCUMULATION:-1}"
SAVE_FINETUNED_MODELS="${BABYSTEPS_SAVE_FINETUNED_MODELS:-0}"
RERUN_EXISTING="${BABYSTEPS_RERUN_EXISTING:-0}"
VERBOSE="${BABYSTEPS_VERBOSE:-1}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

DEFAULT_SPECS=(
  "baseline=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_babylm2026_strict_muon_liger_baseline_causal_diagnostics_50m_bs64/20260704_082711/final"
  "nextlat_h1_lam1=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_babylm2026_strict_muon_liger_nextlat_h1_clipdiag_10epoch_gpu3/20260704_132321/final"
  "nextlat_h1_lam0p5=/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_babylm2026_strict_muon_liger_nextlat_h1_lam0p5_clipdiag_10epoch_gpu2/20260704_133044/final"
)

if [[ "$#" -gt 0 ]]; then
  CHECKPOINT_SPECS=("$@")
else
  CHECKPOINT_SPECS=("${DEFAULT_SPECS[@]}")
fi

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  value="${value//\//_}"
  echo "$value"
}

task_settings() {
  local task="$1"

  NUM_LABELS=2
  TASK_BATCH_SIZE="$BSZ"
  TASK_EPOCHS="$MAX_EPOCHS"
  METRIC_FOR_VALID="accuracy"
  METRICS=(accuracy f1 mcc)

  case "$task" in
    boolq|multirc)
      TASK_BATCH_SIZE="$BIG_BSZ"
      ;;
    wsc)
      TASK_EPOCHS="$WSC_EPOCHS"
      ;;
    rte)
      ;;
    mrpc|qqp)
      METRIC_FOR_VALID="f1"
      ;;
    mnli)
      NUM_LABELS=3
      METRICS=(accuracy)
      ;;
    *)
      echo "Unsupported SuperGLUE fine-tuning task: ${task}" >&2
      return 1
      ;;
  esac
}

prediction_path() {
  local results_dir="$1"
  local checkpoint="$2"
  local task="$3"
  local model_name

  model_name="$(basename "$checkpoint")"
  echo "${results_dir}/${model_name}/main/finetune/${task}/predictions.json"
}

run_task() {
  local name="$1"
  local checkpoint="$2"
  local results_dir="$3"
  local save_dir="$4"
  local task="$5"
  local train_data="${EVAL_DATA_DIR}/glue_filtered/${task}.train.jsonl"
  local valid_data="${EVAL_DATA_DIR}/glue_filtered/${task}.valid.jsonl"
  local pred_path
  local task_log
  local command

  task_settings "$task"
  pred_path="$(prediction_path "$results_dir" "$checkpoint" "$task")"
  task_log="${LOG_DIR}/${name}_${task}_ftseed${FINETUNE_SEED}.log"

  if [[ ! -f "$train_data" ]]; then
    echo "Missing train data for ${task}: ${train_data}" >&2
    return 1
  fi
  if [[ ! -f "$valid_data" ]]; then
    echo "Missing validation data for ${task}: ${valid_data}" >&2
    return 1
  fi
  if [[ "$RERUN_EXISTING" != "1" && -f "$pred_path" ]]; then
    echo "[skip] ${name}/${task}: ${pred_path}"
    return 0
  fi

  command=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "TOKENIZERS_PARALLELISM=false"
    conda run --no-capture-output -n "$CONDA_ENV"
    python -m evaluation_pipeline.finetune.run
    --model_name_or_path "$checkpoint"
    --train_data "$train_data"
    --valid_data "$valid_data"
    --predict_data "$valid_data"
    --task "$task"
    --num_labels "$NUM_LABELS"
    --batch_size "$TASK_BATCH_SIZE"
    --valid_batch_size "$VALID_BATCH_SIZE"
    --gradient_accumulation "$GRADIENT_ACCUMULATION"
    --learning_rate "$LR"
    --num_epochs "$TASK_EPOCHS"
    --sequence_length "$SEQUENCE_LENGTH"
    --results_dir "$results_dir"
    --metrics "${METRICS[@]}"
    --metric_for_valid "$METRIC_FOR_VALID"
    --seed "$FINETUNE_SEED"
    --padding_side left
    --take_final
  )

  if [[ "$SAVE_FINETUNED_MODELS" == "1" ]]; then
    command+=(--save --save_dir "$save_dir")
  fi
  if [[ "$VERBOSE" == "1" ]]; then
    command+=(--verbose)
  fi

  echo "[finetune] ${name}/${task}"
  echo "  gpu: ${GPU}"
  echo "  batch size: ${TASK_BATCH_SIZE}"
  echo "  epochs: ${TASK_EPOCHS}"
  echo "  metric for valid: ${METRIC_FOR_VALID}"
  echo "  log: ${task_log}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  "${command[@]}" >"$task_log" 2>&1
}

summarize_model() {
  local name="$1"
  local results_dir="$2"
  local summary_log="${LOG_DIR}/${name}_summary_ftseed${FINETUNE_SEED}.log"
  local command=(
    conda run --no-capture-output -n "$CONDA_ENV"
    python "$ROOT/scripts/eval/summarize_superglue_finetune.py"
    --results-dir "$results_dir"
    --tasks "${TASKS[@]}"
  )

  echo "[summary] ${name}"
  echo "  log: ${summary_log}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  "${command[@]}" | tee "$summary_log"
}

run_model() {
  local spec="$1"
  local name="${spec%%=*}"
  local checkpoint="${spec#*=}"
  local results_dir="${OUTPUT_ROOT}/${name}/results"
  local save_dir="${OUTPUT_ROOT}/${name}/models"
  local task

  if [[ "$name" == "$checkpoint" ]]; then
    echo "Invalid checkpoint spec '${spec}'. Use name=/path/to/checkpoint." >&2
    return 1
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint for ${name}: ${checkpoint}" >&2
    return 1
  fi

  echo
  echo "=== SuperGLUE fine-tuning: ${name} ==="
  echo "checkpoint: ${checkpoint}"
  echo "results: ${results_dir}"
  echo "save dir: ${save_dir}"

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$results_dir" "$LOG_DIR"
    if [[ "$SAVE_FINETUNED_MODELS" == "1" ]]; then
      mkdir -p "$save_dir"
    fi
  fi

  cd "$PIPELINE_DIR"
  for task in "${TASKS[@]}"; do
    run_task "$name" "$checkpoint" "$results_dir" "$save_dir" "$task"
  done

  cd "$ROOT"
  summarize_model "$name" "$results_dir"
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
fi

echo "LLaMA NextLat SuperGLUE sequential fine-tuning eval"
echo "  gpu: ${GPU}"
echo "  tasks: ${TASKS[*]}"
echo "  seed: ${FINETUNE_SEED}"
echo "  output root: ${OUTPUT_ROOT}"
echo "  log dir: ${LOG_DIR}"
echo "  checkpoints: ${#CHECKPOINT_SPECS[@]}"

for spec in "${CHECKPOINT_SPECS[@]}"; do
  run_model "$spec"
done

echo
echo "All requested LLaMA SuperGLUE fine-tuning evals finished."
