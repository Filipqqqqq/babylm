#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Run BabyLM's official fine-tuning evaluation over a checkpoint sweep.
#
# Defaults target the representative strong baseline:
#   model:             GPT-2 8l512
#   BPE:               16384
#   Muon LR:           0.01
#   train batch size:  32
#   pretrained seed:   45
#
# The official BabyLM script writes every final checkpoint under
# results/final/..., so this wrapper calls the same Python module with a
# unique results_dir/save_dir per pretrained run.

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
TRAINING_ROOT="${BABYSTEPS_TRAINING_ROOT:-/SSD2/babylm-baby-steps-runs/training}"
OUTPUT_ROOT="${BABYSTEPS_SUPERGLUE_OUTPUT_ROOT:-/SSD2/babylm-baby-steps-runs/evaluation_superglue_finetune}"
LOG_DIR="${BABYSTEPS_SUPERGLUE_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/superglue_finetune}"
PIPELINE_DIR="${BABYSTEPS_EVAL_PIPELINE_DIR:-${ROOT}/external/babylm-eval/strict}"
EVAL_DATA_DIR="${BABYSTEPS_FULL_EVAL_DATA_DIR:-${PIPELINE_DIR}/evaluation_data/full_eval}"

MODEL_SIZE="${BABYSTEPS_MODEL_SIZE:-8l512}"
MODEL_ARCH="${BABYSTEPS_MODEL_ARCH:-gpt2}"
RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-}"
BPES=(${BABYSTEPS_BPES:-16384})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.01})
PRETRAIN_SEEDS=(${BABYSTEPS_SEEDS:-45})
GPUS=(${BABYSTEPS_GPUS:-2})
TASKS=(${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli})

TRAIN_BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-32}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"

# Official BabyLM fine-tuning defaults.
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
LR="${BABYSTEPS_SUPERGLUE_LR:-3e-5}"
BSZ="${BABYSTEPS_SUPERGLUE_BATCH_SIZE:-32}"
BIG_BSZ="${BABYSTEPS_SUPERGLUE_BIG_BATCH_SIZE:-16}"
MAX_EPOCHS="${BABYSTEPS_SUPERGLUE_EPOCHS:-10}"
WSC_EPOCHS="${BABYSTEPS_SUPERGLUE_WSC_EPOCHS:-30}"
SEQUENCE_LENGTH="${BABYSTEPS_SUPERGLUE_SEQUENCE_LENGTH:-512}"
VALID_BATCH_SIZE="${BABYSTEPS_SUPERGLUE_VALID_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION="${BABYSTEPS_SUPERGLUE_GRADIENT_ACCUMULATION:-1}"
SAVE_FINETUNED_MODELS="${BABYSTEPS_SAVE_FINETUNED_MODELS:-1}"
VERBOSE="${BABYSTEPS_VERBOSE:-1}"
CHECKPOINT_NAME="${BABYSTEPS_CHECKPOINT_NAME:-final}"
RERUN_EXISTING="${BABYSTEPS_RERUN_EXISTING:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
fi

pids=()
names=()
active_jobs=0
gpu_slot=0
failed=0

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

run_name() {
  local bpe="$1"
  local muon_lr="$2"
  local pretrain_seed="$3"

  echo "${MODEL_ARCH}_${MODEL_SIZE}_bpe${bpe}_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")_bs${TRAIN_BATCH_SIZE}_seed${pretrain_seed}${RUN_SUFFIX}"
}

terminate_pid_tree() {
  local pid="$1"

  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
}

force_kill_pid_tree() {
  local pid="$1"

  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status="${1:-$?}"
  local pid

  trap - INT TERM EXIT

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "Stopping active SuperGLUE fine-tuning eval jobs..."
    for pid in "${pids[@]}"; do
      terminate_pid_tree "$pid"
    done
    sleep 3
    for pid in "${pids[@]}"; do
      force_kill_pid_tree "$pid"
    done
    wait 2>/dev/null || true
  fi

  exit "$status"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup "$?"' EXIT

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs listed. Set BABYSTEPS_GPUS, e.g. BABYSTEPS_GPUS=\"2\"."
  exit 1
fi

latest_named_checkpoint() {
  local run_name="$1"
  local checkpoint_name="$2"
  local run_root="${TRAINING_ROOT}/${run_name}"

  if [[ ! -d "$run_root" ]]; then
    return 1
  fi

  find "$run_root" -mindepth 2 -maxdepth 2 -type d -name "$checkpoint_name" | sort | tail -n 1
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
      echo "Unsupported SuperGLUE fine-tuning task: ${task}"
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

run_command() {
  local gpu="$1"
  shift
  local command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "TOKENIZERS_PARALLELISM=false"
    conda run --no-capture-output -n "$CONDA_ENV"
    "$@"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}"
}

run_task_eval() {
  local gpu="$1"
  local checkpoint="$2"
  local results_dir="$3"
  local save_dir="$4"
  local task="$5"
  local train_data="${EVAL_DATA_DIR}/glue_filtered/${task}.train.jsonl"
  local valid_data="${EVAL_DATA_DIR}/glue_filtered/${task}.valid.jsonl"
  local pred_path
  local command

  task_settings "$task"
  pred_path="$(prediction_path "$results_dir" "$checkpoint" "$task")"

  if [[ ! -f "$train_data" ]]; then
    echo "Missing train data for ${task}: ${train_data}"
    return 1
  fi
  if [[ ! -f "$valid_data" ]]; then
    echo "Missing validation data for ${task}: ${valid_data}"
    return 1
  fi
  if [[ "$RERUN_EXISTING" != "1" && -f "$pred_path" ]]; then
    echo "[skip] ${task}: ${pred_path}"
    return 0
  fi

  command=(
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

  echo "[finetune] ${task}"
  echo "  batch size: ${TASK_BATCH_SIZE}"
  echo "  epochs: ${TASK_EPOCHS}"
  echo "  metric for valid: ${METRIC_FOR_VALID}"
  run_command "$gpu" "${command[@]}"
}

run_checkpoint_eval() {
  local gpu="$1"
  local bpe="$2"
  local muon_lr="$3"
  local pretrain_seed="$4"
  local name
  local checkpoint
  local run_root
  local results_dir
  local save_dir
  local task

  name="$(run_name "$bpe" "$muon_lr" "$pretrain_seed")"
  run_root="${OUTPUT_ROOT}/${name}"
  results_dir="${run_root}/results"
  save_dir="${run_root}/models"

  checkpoint="$(latest_named_checkpoint "$name" "$CHECKPOINT_NAME")"
  if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
    echo "Missing ${CHECKPOINT_NAME} checkpoint for ${name} under ${TRAINING_ROOT}"
    return 1
  fi

  echo "Running SuperGLUE fine-tuning eval for ${name} on GPU ${gpu}"
  echo "  checkpoint: ${checkpoint}"
  echo "  eval data: ${EVAL_DATA_DIR}/glue_filtered"
  echo "  results: ${results_dir}"
  echo "  save dir: ${save_dir}"
  echo "  fine-tune seed: ${FINETUNE_SEED}"

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$results_dir"
    if [[ "$SAVE_FINETUNED_MODELS" == "1" ]]; then
      mkdir -p "$save_dir"
    fi
  fi

  cd "$PIPELINE_DIR"

  for task in "${TASKS[@]}"; do
    run_task_eval "$gpu" "$checkpoint" "$results_dir" "$save_dir" "$task"
  done
}

launch_eval() {
  local gpu="$1"
  local bpe="$2"
  local muon_lr="$3"
  local pretrain_seed="$4"
  local name
  local log_path
  local pid

  name="$(run_name "$bpe" "$muon_lr" "$pretrain_seed")"
  log_path="${LOG_DIR}/${name}_superglue_ftseed${FINETUNE_SEED}.log"

  echo "Launching SuperGLUE fine-tuning eval for ${name} on GPU ${gpu}"
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    run_checkpoint_eval "$gpu" "$bpe" "$muon_lr" "$pretrain_seed"
    return
  fi

  run_checkpoint_eval "$gpu" "$bpe" "$muon_lr" "$pretrain_seed" >"$log_path" 2>&1 &
  pid="$!"
  pids+=("$pid")
  names+=("$name")
}

wait_for_batch() {
  local idx
  local status

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "Finished SuperGLUE fine-tuning eval for ${names[$idx]}"
    else
      status=$?
      echo "FAILED SuperGLUE fine-tuning eval for ${names[$idx]} with exit code ${status}"
      failed=1
    fi
  done

  pids=()
  names=()
  active_jobs=0
  gpu_slot=0
}

for bpe in "${BPES[@]}"; do
  for muon_lr in "${MUON_LRS[@]}"; do
    for pretrain_seed in "${PRETRAIN_SEEDS[@]}"; do
      launch_eval "${GPUS[$gpu_slot]}" "$bpe" "$muon_lr" "$pretrain_seed"

      if [[ "$DRY_RUN" == "1" ]]; then
        gpu_slot=$(( (gpu_slot + 1) % ${#GPUS[@]} ))
        continue
      fi

      active_jobs=$((active_jobs + 1))
      gpu_slot=$((gpu_slot + 1))

      if [[ "$active_jobs" -eq "${#GPUS[@]}" ]]; then
        wait_for_batch
      fi
    done
  done
done

if [[ "$DRY_RUN" != "1" && "$active_jobs" -gt 0 ]]; then
  wait_for_batch
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more SuperGLUE fine-tuning eval runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All SuperGLUE fine-tuning eval runs finished."
