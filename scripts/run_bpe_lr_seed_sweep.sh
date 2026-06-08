#!/usr/bin/env bash
set -euo pipefail

# Launch one single-GPU training run per listed GPU. After a batch of jobs
# fills every GPU, the script waits before launching the next batch.

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"

BPES=(${BABYSTEPS_BPES:-8192 16384})
LRS=(${BABYSTEPS_LRS:-0.0001 0.00005 0.000025})
SEEDS=(${BABYSTEPS_SEEDS:-42 43})
GPUS=(${BABYSTEPS_GPUS:-1 2})

mkdir -p "$LOG_DIR"

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
    # Training jobs are launched with setsid, so the PID is also the process
    # group id. Killing the group catches conda-run and its Python child.
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
    echo "Stopping active training jobs..."
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

launch_run() {
  local gpu="$1"
  local bpe="$2"
  local lr="$3"
  local seed="$4"
  local config="configs/train/gpt2_8l512_bpe${bpe}_10m.yaml"
  local name="gpt2_8l512_bpe${bpe}_lr$(lr_slug "$lr")_seed${seed}"
  local log_path="${LOG_DIR}/${name}.log"
  local train_pid
  local progress_pid

  echo "Launching ${name} on GPU ${gpu}"
  echo "  log: ${log_path}"
  CUDA_VISIBLE_DEVICES="$gpu" setsid conda run --no-capture-output -n "$CONDA_ENV" python -u scripts/train_gpt2_small.py \
    --config "$config" \
    --set "run.name=${name}" \
    --set "run.seed=${seed}" \
    --set "training.learning_rate=${lr}" \
    >"$log_path" 2>&1 &
  train_pid="$!"
  pids+=("$train_pid")
  names+=("$name")

  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    tail --pid="$train_pid" -n 0 -F "$log_path" 2>/dev/null \
      | grep --line-buffered -E "Output dir:|Train chunks:|Val chunks:|Model parameters:|\\[step [0-9]+\\]|Epoch [0-9]+|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
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
      echo "Finished ${names[$idx]}"
    else
      status=$?
      echo "FAILED ${names[$idx]} with exit code ${status}"
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
      launch_run "${GPUS[$gpu_slot]}" "$bpe" "$lr" "$seed"

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
  echo "One or more sweep runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All sweep runs finished."
