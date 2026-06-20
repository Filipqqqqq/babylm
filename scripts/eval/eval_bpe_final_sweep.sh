#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Evaluate the final checkpoint for each BPE/LR/seed run from the training
# sweep. Each model writes to a separate output directory so the official
# BabyLM eval outputs do not collide under the shared checkpoint name "final".

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
TRAINING_ROOT="${BABYSTEPS_TRAINING_ROOT:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/training}"
OUTPUT_ROOT="${BABYSTEPS_EVAL_OUTPUT_ROOT:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/evaluation}"
LOG_DIR="${BABYSTEPS_EVAL_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/eval}"
PIPELINE_DIR="${BABYSTEPS_EVAL_PIPELINE_DIR:-external/babylm-eval/strict}"
EVAL_DATA_DIR="${BABYSTEPS_EVAL_DATA_DIR:-external/babylm-eval/strict/evaluation_data/fast_eval}"
BATCH_SIZE="${BABYSTEPS_EVAL_BATCH_SIZE:-64}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
RERUN_EXISTING="${BABYSTEPS_RERUN_EXISTING:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

BPES=(${BABYSTEPS_BPES:-8192 16384})
LRS=(${BABYSTEPS_LRS:-0.0001 0.00005})
SEEDS=(${BABYSTEPS_SEEDS:-44 45})
GPUS=(${BABYSTEPS_GPUS:-1 2})
TASKS=(${BABYSTEPS_EVAL_TASKS:-blimp_fast supplement_fast ewok_fast entity_tracking_fast reading})

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

pids=()
names=()
progress_pids=()
active_jobs=0
gpu_slot=0
failed=0

lr_slug() {
  local slug="$1"
  slug="${slug//./p}"
  slug="${slug//-/m}"
  echo "$slug"
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

  if [[ "${#progress_pids[@]}" -gt 0 ]]; then
    for pid in "${progress_pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
  fi

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "Stopping active eval jobs..."
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

latest_final_checkpoint() {
  local run_name="$1"
  local run_root="${TRAINING_ROOT}/${run_name}"

  if [[ ! -d "$run_root" ]]; then
    return 1
  fi

  find "$run_root" -mindepth 2 -maxdepth 2 -type d -name final | sort | tail -n 1
}

launch_eval() {
  local gpu="$1"
  local bpe="$2"
  local lr="$3"
  local seed="$4"
  local name="gpt2_8l512_bpe${bpe}_lr$(lr_slug "$lr")_seed${seed}"
  local checkpoint
  local output_dir="${OUTPUT_ROOT}/${name}_fast_eval"
  local log_path="${LOG_DIR}/${name}_fast_eval.log"
  local eval_pid
  local progress_pid
  local command

  checkpoint="$(latest_final_checkpoint "$name")"
  if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
    echo "Missing final checkpoint for ${name} under ${TRAINING_ROOT}"
    failed=1
    return
  fi

  echo "Launching eval for ${name} on GPU ${gpu}"
  echo "  checkpoint: ${checkpoint}"
  echo "  output: ${output_dir}"
  echo "  log: ${log_path}"

  command=(
    setsid conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/eval/eval_babylm_checkpoints.py
    --config configs/eval/fast_eval.yaml
    --checkpoint "$checkpoint"
    --pipeline-dir "$PIPELINE_DIR"
    --eval-data-dir "$EVAL_DATA_DIR"
    --output-dir "$output_dir"
    --batch-size "$BATCH_SIZE"
    --device "$gpu"
    --tasks "${TASKS[@]}"
  )

  if [[ "$RERUN_EXISTING" == "1" ]]; then
    command+=(--rerun-existing)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    command+=(--dry-run)
  fi

  "${command[@]}" >"$log_path" 2>&1 &
  eval_pid="$!"
  pids+=("$eval_pid")
  names+=("$name")

  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    tail --pid="$eval_pid" -n 0 -F "$log_path" 2>/dev/null \
      | grep --line-buffered -E "Pipeline:|Eval data:|Output:|\\[eval\\]|\\[skip\\]|Summary written|CSV written|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
      | sed -u "s/^/[${name}] /" &
    progress_pid="$!"
    progress_pids+=("$progress_pid")
  fi
}

wait_for_batch() {
  local idx
  local status
  local progress_pid

  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      echo "Finished eval for ${names[$idx]}"
    else
      status=$?
      echo "FAILED eval for ${names[$idx]} with exit code ${status}"
      failed=1
    fi
  done

  pids=()
  names=()
  for progress_pid in "${progress_pids[@]}"; do
    wait "$progress_pid" 2>/dev/null || true
  done
  progress_pids=()
  active_jobs=0
  gpu_slot=0
}

for bpe in "${BPES[@]}"; do
  for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_eval "${GPUS[$gpu_slot]}" "$bpe" "$lr" "$seed"

      active_jobs=$((active_jobs + 1))
      gpu_slot=$((gpu_slot + 1))

      if [[ "$active_jobs" -eq "${#GPUS[@]}" ]]; then
        wait_for_batch
      fi
    done
  done
done

if [[ "$active_jobs" -gt 0 ]]; then
  wait_for_batch
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more eval runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All final-checkpoint eval runs finished."
