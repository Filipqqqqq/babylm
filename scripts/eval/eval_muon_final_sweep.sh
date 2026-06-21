#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Evaluate final checkpoints produced by scripts/sweeps/run_muon_lr_sweep.sh.

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
TRAINING_ROOT="${BABYSTEPS_TRAINING_ROOT:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/training}"
OUTPUT_ROOT="${BABYSTEPS_EVAL_OUTPUT_ROOT:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/evaluation}"
LOG_DIR="${BABYSTEPS_EVAL_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/eval_muon}"
PIPELINE_DIR="${BABYSTEPS_EVAL_PIPELINE_DIR:-external/babylm-eval/strict}"
EVAL_DATA_DIR="${BABYSTEPS_EVAL_DATA_DIR:-external/babylm-eval/strict/evaluation_data/fast_eval}"
BATCH_SIZE="${BABYSTEPS_EVAL_BATCH_SIZE:-64}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
RERUN_EXISTING="${BABYSTEPS_RERUN_EXISTING:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

BPES=(${BABYSTEPS_BPES:-16384})
SEEDS=(${BABYSTEPS_SEEDS:-42})
GPUS=(${BABYSTEPS_GPUS:-1 2})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.01})
BATCH_SIZE_TRAIN="${BABYSTEPS_BATCH_SIZE:-32}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
TASKS=(${BABYSTEPS_EVAL_TASKS:-blimp_fast supplement_fast ewok_fast entity_tracking_fast reading})

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
fi

pids=()
names=()
progress_pids=()
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

  for pid in "${progress_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "Stopping active Muon eval jobs..."
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
  echo "No GPUs listed. Set BABYSTEPS_GPUS, e.g. BABYSTEPS_GPUS=\"1 2\"."
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

launch_eval() {
  local gpu="$1"
  local bpe="$2"
  local muon_lr="$3"
  local seed="$4"
  local batch_suffix=""
  if [[ -n "$BATCH_SIZE_TRAIN" ]]; then
    batch_suffix="_bs${BATCH_SIZE_TRAIN}"
  fi
  local name="gpt2_8l512_bpe${bpe}_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")${batch_suffix}_seed${seed}"
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

  echo "Launching eval for ${name} on GPU ${gpu}"
  echo "  checkpoint: ${checkpoint}"
  echo "  output: ${output_dir}"
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
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
  echo "One or more Muon eval runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All Muon final-checkpoint eval runs finished."
