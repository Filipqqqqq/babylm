#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Run the official BabyLM full zero-shot tasks over a Muon checkpoint sweep.
#
# Defaults target the strong baseline group:
#   model:      GPT-2 8l512
#   BPE:        16384
#   Muon LR:    0.01
#   warmup:     0.03
#   Muon wd:    0.1
#   train bs:   32
#   seeds:      42, 43, 44, 45
#
# This intentionally does not call scripts/eval_zero_shot.sh because the
# official script writes every final checkpoint under results/final/... .
# Here each training run gets a unique output root before the official
# evaluator adds final/main/zero_shot/...

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
TRAINING_ROOT="${BABYSTEPS_TRAINING_ROOT:-/SSD2/babylm-baby-steps-runs/training}"
OUTPUT_ROOT="${BABYSTEPS_FULL_EVAL_OUTPUT_ROOT:-/SSD2/babylm-baby-steps-runs/evaluation_full_zero_shot}"
LOG_DIR="${BABYSTEPS_FULL_EVAL_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/full_zero_shot_muon}"
PIPELINE_DIR="${BABYSTEPS_EVAL_PIPELINE_DIR:-${ROOT}/external/babylm-eval/strict}"
EVAL_DATA_DIR="${BABYSTEPS_FULL_EVAL_DATA_DIR:-${PIPELINE_DIR}/evaluation_data/full_eval}"

MODEL_SIZE="${BABYSTEPS_MODEL_SIZE:-8l512}"
BPES=(${BABYSTEPS_BPES:-16384})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.01})
SEEDS=(${BABYSTEPS_SEEDS:-42 43 44 45})
GPUS=(${BABYSTEPS_GPUS:-2})

TRAIN_BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-32}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
EVAL_BATCH_SIZE="${BABYSTEPS_EVAL_BATCH_SIZE:-64}"
BACKEND="${BABYSTEPS_EVAL_BACKEND:-causal}"
RERUN_EXISTING="${BABYSTEPS_RERUN_EXISTING:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

TASK_SPECS=(
  "blimp:blimp_filtered"
  "blimp:supplement_filtered"
  "ewok:ewok_filtered"
  "entity_tracking:entity_tracking"
  "comps:comps"
)

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
    echo "Stopping active full zero-shot eval jobs..."
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

latest_final_checkpoint() {
  local run_name="$1"
  local run_root="${TRAINING_ROOT}/${run_name}"

  if [[ ! -d "$run_root" ]]; then
    return 1
  fi

  find "$run_root" -mindepth 2 -maxdepth 2 -type d -name final | sort | tail -n 1
}

sentence_prediction_path() {
  local output_dir="$1"
  local checkpoint="$2"
  local task="$3"
  local dataset="$4"
  local revision="main"
  local model_name

  model_name="$(basename "$checkpoint")"
  echo "${output_dir}/${model_name}/${revision}/zero_shot/${BACKEND}/${task}/${dataset}/predictions.json"
}

reading_prediction_path() {
  local output_dir="$1"
  local checkpoint="$2"
  local revision="main"
  local model_name

  model_name="$(basename "$checkpoint")"
  echo "${output_dir}/${model_name}/${revision}/zero_shot/${BACKEND}/reading/predictions.json"
}

run_command() {
  local gpu="$1"
  shift
  local command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
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

run_seed_eval() {
  local gpu="$1"
  local bpe="$2"
  local muon_lr="$3"
  local seed="$4"
  local name="gpt2_${MODEL_SIZE}_bpe${bpe}_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")_bs${TRAIN_BATCH_SIZE}_seed${seed}"
  local checkpoint
  local output_dir="${OUTPUT_ROOT}/${name}"
  local task_spec
  local task
  local dataset
  local data_path
  local pred_path

  checkpoint="$(latest_final_checkpoint "$name")"
  if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
    echo "Missing final checkpoint for ${name} under ${TRAINING_ROOT}"
    return 1
  fi

  echo "Evaluating ${name} on GPU ${gpu}"
  echo "  checkpoint: ${checkpoint}"
  echo "  eval data: ${EVAL_DATA_DIR}"
  echo "  output: ${output_dir}"

  cd "$PIPELINE_DIR"

  for task_spec in "${TASK_SPECS[@]}"; do
    task="${task_spec%%:*}"
    dataset="${task_spec#*:}"
    data_path="${EVAL_DATA_DIR}/${dataset}"
    pred_path="$(sentence_prediction_path "$output_dir" "$checkpoint" "$task" "$dataset")"

    if [[ ! -d "$data_path" ]]; then
      echo "Missing data directory for ${task}/${dataset}: ${data_path}"
      return 1
    fi
    if [[ "$RERUN_EXISTING" != "1" && -f "$pred_path" ]]; then
      echo "[skip] ${task}/${dataset}: ${pred_path}"
      continue
    fi

    echo "[eval] ${task}/${dataset}"
    run_command "$gpu" python -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$checkpoint" \
      --backend "$BACKEND" \
      --task "$task" \
      --data_path "$data_path" \
      --output_dir "$output_dir" \
      --save_predictions \
      --batch_size "$EVAL_BATCH_SIZE"
  done

  data_path="${EVAL_DATA_DIR}/reading/reading_data.csv"
  pred_path="$(reading_prediction_path "$output_dir" "$checkpoint")"
  if [[ ! -f "$data_path" ]]; then
    echo "Missing reading data file: ${data_path}"
    return 1
  fi
  if [[ "$RERUN_EXISTING" != "1" && -f "$pred_path" ]]; then
    echo "[skip] reading: ${pred_path}"
  else
    echo "[eval] reading"
    run_command "$gpu" python -m evaluation_pipeline.reading.run \
      --model_path_or_name "$checkpoint" \
      --backend "$BACKEND" \
      --data_path "$data_path" \
      --output_dir "$output_dir"
  fi
}

launch_eval() {
  local gpu="$1"
  local bpe="$2"
  local muon_lr="$3"
  local seed="$4"
  local name="gpt2_${MODEL_SIZE}_bpe${bpe}_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")_bs${TRAIN_BATCH_SIZE}_seed${seed}"
  local log_path="${LOG_DIR}/${name}_full_zero_shot.log"
  local pid

  echo "Launching full zero-shot eval for ${name} on GPU ${gpu}"
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    run_seed_eval "$gpu" "$bpe" "$muon_lr" "$seed"
    return
  fi

  run_seed_eval "$gpu" "$bpe" "$muon_lr" "$seed" >"$log_path" 2>&1 &
  pid="$!"
  pids+=("$pid")
  names+=("$name")
}

wait_for_batch() {
  local idx
  local status

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "Finished full zero-shot eval for ${names[$idx]}"
    else
      status=$?
      echo "FAILED full zero-shot eval for ${names[$idx]} with exit code ${status}"
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
    for seed in "${SEEDS[@]}"; do
      launch_eval "${GPUS[$gpu_slot]}" "$bpe" "$muon_lr" "$seed"

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
  echo "One or more full zero-shot eval runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All full zero-shot eval runs finished."
