#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# Muon architecture pretraining sweep.
#
# Default grid:
#   BPE:        16384
#   depth:      12 layers
#   width:      384, 512 hidden dim
#   Muon LR:    0.005, 0.01
#   seeds:      42, 43, 44, 45
#   batch size: 32
#
# That is 32 runs. The base BPE config supplies paths/tokenizer/vocab size;
# this launcher overrides model shape, optimizer settings, seed, and run name.

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/muon_arch}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

BPES=(${BABYSTEPS_BPES:-16384})
LAYERS=(${BABYSTEPS_LAYERS:-12})
HIDDEN_SIZES=(${BABYSTEPS_HIDDEN_SIZES:-384 512})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.005 0.01})
SEEDS=(${BABYSTEPS_SEEDS:-42 43 44 45})
GPUS=(${BABYSTEPS_GPUS:-1 2})

BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-32}"
AUX_LR="${BABYSTEPS_AUX_LR:-0.00005}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_MOMENTUM="${BABYSTEPS_MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${BABYSTEPS_MUON_NS_STEPS:-5}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
GRADIENT_CLIP_NORM="${BABYSTEPS_GRADIENT_CLIP_NORM:-1.0}"
SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-0}"
SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"

MAX_TRAIN_CHUNKS="${BABYSTEPS_MAX_TRAIN_CHUNKS:-}"
MAX_VAL_CHUNKS="${BABYSTEPS_MAX_VAL_CHUNKS:-}"
LOG_EVERY_STEPS="${BABYSTEPS_LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${BABYSTEPS_EVAL_EVERY_STEPS:-500}"
EVAL_MAX_BATCHES="${BABYSTEPS_EVAL_MAX_BATCHES:-20}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
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

heads_for_hidden_size() {
  local hidden_size="$1"

  if (( hidden_size % 64 != 0 )); then
    echo "Hidden size ${hidden_size} is not divisible by 64, so it cannot keep head_dim=64." >&2
    return 1
  fi

  echo $((hidden_size / 64))
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

  for pid in "${progress_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "Stopping active Muon architecture sweep jobs..."
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

launch_run() {
  local gpu="$1"
  local bpe="$2"
  local layers="$3"
  local hidden_size="$4"
  local muon_lr="$5"
  local seed="$6"
  local n_head
  local config="configs/train/gpt2_8l512_bpe${bpe}_10m.yaml"
  local name
  local log_path
  local train_pid
  local progress_pid
  local command

  n_head="$(heads_for_hidden_size "$hidden_size")"
  name="gpt2_${layers}l${hidden_size}_bpe${bpe}_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")_bs${BATCH_SIZE}_seed${seed}"
  log_path="${LOG_DIR}/${name}.log"

  command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
    setsid conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$config"
    --set "run.name=${name}"
    --set "run.seed=${seed}"
    --set "model.size=${layers}l${hidden_size}"
    --set "model.n_layer=${layers}"
    --set "model.n_embd=${hidden_size}"
    --set "model.n_head=${n_head}"
    --set "training.epochs=${EPOCHS}"
    --set "training.batch_size=${BATCH_SIZE}"
    --set "training.optimizer=muon_pe"
    --set "training.learning_rate=${AUX_LR}"
    --set "training.warmup_ratio=${WARMUP_RATIO}"
    --set "training.gradient_clip_norm=${GRADIENT_CLIP_NORM}"
    --set "training.muon_lr=${muon_lr}"
    --set "training.muon_momentum=${MUON_MOMENTUM}"
    --set "training.muon_ns_steps=${MUON_NS_STEPS}"
    --set "training.muon_weight_decay=${MUON_WEIGHT_DECAY}"
    --set "training.muon_nesterov=true"
    --set "training.muon_split_qkv=true"
    --set "checkpointing.save_every_epochs=${SAVE_EVERY_EPOCHS}"
    --set "checkpointing.save_final=${SAVE_FINAL}"
    --set "logging.log_every_steps=${LOG_EVERY_STEPS}"
    --set "logging.eval_every_steps=${EVAL_EVERY_STEPS}"
    --set "logging.eval_max_batches=${EVAL_MAX_BATCHES}"
  )

  if [[ -n "$MAX_TRAIN_CHUNKS" ]]; then
    command+=(--set "data.max_train_chunks=${MAX_TRAIN_CHUNKS}")
  fi
  if [[ -n "$MAX_VAL_CHUNKS" ]]; then
    command+=(--set "data.max_val_chunks=${MAX_VAL_CHUNKS}")
  fi

  echo "Launching ${name} on GPU ${gpu}"
  echo "  layers: ${layers}"
  echo "  hidden size: ${hidden_size}"
  echo "  heads: ${n_head}"
  echo "  aux lr: ${AUX_LR}"
  echo "  muon lr: ${muon_lr}"
  echo "  epochs: ${EPOCHS}"
  echo "  batch size: ${BATCH_SIZE}"
  echo "  warmup ratio: ${WARMUP_RATIO}"
  echo "  muon weight decay: ${MUON_WEIGHT_DECAY}"
  echo "  save every epochs: ${SAVE_EVERY_EPOCHS}"
  echo "  save final: ${SAVE_FINAL}"
  if [[ -n "$MAX_TRAIN_CHUNKS" ]]; then
    echo "  max train chunks: ${MAX_TRAIN_CHUNKS}"
  fi
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  "${command[@]}" >"$log_path" 2>&1 &
  train_pid="$!"
  pids+=("$train_pid")
  names+=("$name")

  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    tail --pid="$train_pid" -n 0 -F "$log_path" 2>/dev/null \
      | grep --line-buffered -E "Output dir:|Train chunks:|Val chunks:|Model parameters:|Optimizer:|Muon hidden|Auxiliary AdamW|Muon split-QKV|\\[step [0-9]+\\]|Epoch [0-9]+|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
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
  for layers in "${LAYERS[@]}"; do
    for hidden_size in "${HIDDEN_SIZES[@]}"; do
      for muon_lr in "${MUON_LRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          launch_run "${GPUS[$gpu_slot]}" "$bpe" "$layers" "$hidden_size" "$muon_lr" "$seed"

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
  done
done

if [[ "$DRY_RUN" != "1" && "$active_jobs" -gt 0 ]]; then
  wait_for_batch
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more Muon architecture sweep runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All Muon architecture sweep runs finished."
