#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# LLaMA + Muon LR/seed sweep for the 16k BPE BabyLM 10M setup.
#
# Defaults:
#   architecture: LLaMA, 8 layers, hidden size 512, 8 heads, 16k BPE
#   MLP width:    parameter-matched round((8/3) * hidden_size) to /256 = 1280
#   Muon LR:      0.02, 0.01, 0.005
#   seeds:        42, 43, 44
#   batch size:   32
#   precision:    bf16
#   Liger:        enabled
#
# Dry-run the exact commands:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Example launch on GPUs 0 and 1:
#   BABYSTEPS_GPUS="0 1" bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Recommended single full 10M LLaMA + Liger run:
#   BABYSTEPS_GPUS="0" BABYSTEPS_MUON_LRS="0.01" BABYSTEPS_SEEDS="42" \
#   bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# For a shorter smoke before the full run:
#   BABYSTEPS_MAX_TRAIN_CHUNKS=1024 BABYSTEPS_MAX_VAL_CHUNKS=128 \
#   BABYSTEPS_EPOCHS=1 BABYSTEPS_SAVE_FINAL=false \
#   bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Token-shard configs can use block limits instead:
#   BABYSTEPS_MAX_TRAIN_BLOCKS=1024 BABYSTEPS_MAX_VAL_BLOCKS=128 \
#   bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Override precision when needed:
#   BABYSTEPS_MIXED_PRECISION=no bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Disable Liger for ablations:
#   BABYSTEPS_USE_LIGER_KERNEL=false bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Add LLaMA dropout:
#   BABYSTEPS_LLAMA_DROPOUT=0.1 BABYSTEPS_RUN_SUFFIX=_drop0p1 bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Sweep batch sizes:
#   BABYSTEPS_BATCH_SIZES="32 64" bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh
#
# Split effective batch sizes into smaller per-step micro-batches:
#   BABYSTEPS_BATCH_SIZES="32 64" BABYSTEPS_MICRO_BATCH_SIZE=32 bash scripts/sweeps/run_llama_muon_lr_seed_sweep.sh

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/llama_muon}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"
RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-}"

CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_8l512_bpe16384_10m_muon_liger.yaml}"
SEEDS=(${BABYSTEPS_SEEDS:-42 43 44})
GPUS=(${BABYSTEPS_GPUS:-1 2})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.02 0.01 0.005})

LAYERS="${BABYSTEPS_LAYERS:-8}"
HIDDEN_SIZE="${BABYSTEPS_HIDDEN_SIZE:-512}"
N_HEAD="${BABYSTEPS_N_HEAD:-8}"
LLAMA_PARAM_ROUND_MULTIPLE="${BABYSTEPS_LLAMA_PARAM_ROUND_MULTIPLE:-256}"
LLAMA_NUM_KEY_VALUE_HEADS="${BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS:-8}"
BATCH_SIZES=(${BABYSTEPS_BATCH_SIZES:-${BABYSTEPS_BATCH_SIZE:-32}})
MICRO_BATCH_SIZE="${BABYSTEPS_MICRO_BATCH_SIZE:-}"
USE_LIGER_KERNEL="${BABYSTEPS_USE_LIGER_KERNEL:-true}"
LLAMA_DROPOUT="${BABYSTEPS_LLAMA_DROPOUT:-}"
LLAMA_EMBD_PDROP="${BABYSTEPS_LLAMA_EMBD_PDROP:-$LLAMA_DROPOUT}"
LLAMA_RESID_PDROP="${BABYSTEPS_LLAMA_RESID_PDROP:-$LLAMA_DROPOUT}"
LLAMA_ATTENTION_DROPOUT="${BABYSTEPS_LLAMA_ATTENTION_DROPOUT:-$LLAMA_DROPOUT}"

AUX_LR="${BABYSTEPS_AUX_LR:-0.00005}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_MOMENTUM="${BABYSTEPS_MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${BABYSTEPS_MUON_NS_STEPS:-5}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
GRADIENT_CLIP_NORM="${BABYSTEPS_GRADIENT_CLIP_NORM:-1.0}"
MIXED_PRECISION="${BABYSTEPS_MIXED_PRECISION:-bf16}"
SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-0}"
SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"

MAX_TRAIN_CHUNKS="${BABYSTEPS_MAX_TRAIN_CHUNKS:-}"
MAX_VAL_CHUNKS="${BABYSTEPS_MAX_VAL_CHUNKS:-}"
MAX_TRAIN_BLOCKS="${BABYSTEPS_MAX_TRAIN_BLOCKS:-}"
MAX_VAL_BLOCKS="${BABYSTEPS_MAX_VAL_BLOCKS:-}"
LOG_EVERY_STEPS="${BABYSTEPS_LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${BABYSTEPS_EVAL_EVERY_STEPS:-500}"
EVAL_MAX_BATCHES="${BABYSTEPS_EVAL_MAX_BATCHES:-20}"
PROFILE_OPTIMIZER_STEPS="${BABYSTEPS_PROFILE_OPTIMIZER_STEPS:-true}"

if (( HIDDEN_SIZE % N_HEAD != 0 )); then
  echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by N_HEAD=${N_HEAD}." >&2
  exit 1
fi

rounded_param_matched_intermediate() {
  local hidden_size="$1"
  local multiple="$2"
  local raw

  raw="$(python -c "print((8.0 / 3.0) * int(${hidden_size}))")"
  if (( multiple <= 1 )); then
    python -c "print(int(round(${raw})))"
  else
    python -c "m=int(${multiple}); print(int(round(${raw} / m) * m))"
  fi
}

LLAMA_INTERMEDIATE_SIZE="${BABYSTEPS_LLAMA_INTERMEDIATE_SIZE:-$(rounded_param_matched_intermediate "$HIDDEN_SIZE" "$LLAMA_PARAM_ROUND_MULTIPLE")}"

if (( N_HEAD % LLAMA_NUM_KEY_VALUE_HEADS != 0 )); then
  echo "N_HEAD=${N_HEAD} must be divisible by LLAMA_NUM_KEY_VALUE_HEADS=${LLAMA_NUM_KEY_VALUE_HEADS}." >&2
  exit 1
fi

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

resolve_micro_batch_size() {
  local effective_batch_size="$1"

  if [[ -n "$MICRO_BATCH_SIZE" ]]; then
    echo "$MICRO_BATCH_SIZE"
  elif (( effective_batch_size > 32 )); then
    echo "32"
  else
    echo "$effective_batch_size"
  fi
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
    echo "Stopping active LLaMA Muon jobs..."
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
  echo "No GPUs listed. Set BABYSTEPS_GPUS, e.g. BABYSTEPS_GPUS=\"0 1\"."
  exit 1
fi

launch_run() {
  local gpu="$1"
  local muon_lr="$2"
  local seed="$3"
  local batch_size="$4"
  local micro_batch_size
  local gradient_accumulation_steps
  local name="llama_${LAYERS}l${HIDDEN_SIZE}_bpe16384_muon_lr$(slug "$muon_lr")_wu$(slug "$WARMUP_RATIO")_mwd$(slug "$MUON_WEIGHT_DECAY")_bs${batch_size}_seed${seed}${RUN_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local train_pid
  local progress_pid
  local command

  micro_batch_size="$(resolve_micro_batch_size "$batch_size")"
  if (( micro_batch_size <= 0 )); then
    echo "Micro-batch size must be positive, got ${micro_batch_size}." >&2
    exit 1
  fi
  if (( batch_size % micro_batch_size != 0 )); then
    echo "Effective batch size ${batch_size} must be divisible by micro-batch size ${micro_batch_size}." >&2
    exit 1
  fi
  gradient_accumulation_steps=$(( batch_size / micro_batch_size ))

  command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
    setsid conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${seed}"
    --set "model.architecture=llama"
    --set "model.size=${LAYERS}l${HIDDEN_SIZE}"
    --set "model.n_layer=${LAYERS}"
    --set "model.n_embd=${HIDDEN_SIZE}"
    --set "model.n_head=${N_HEAD}"
    --set "model.llama_intermediate_size=${LLAMA_INTERMEDIATE_SIZE}"
    --set "model.llama_num_key_value_heads=${LLAMA_NUM_KEY_VALUE_HEADS}"
    --set "model.llama_tie_word_embeddings=true"
    --set "model.use_liger_kernel=${USE_LIGER_KERNEL}"
    --set "training.epochs=${EPOCHS}"
    --set "training.batch_size=${micro_batch_size}"
    --set "training.optimizer=muon_pe"
    --set "training.learning_rate=${AUX_LR}"
    --set "training.warmup_ratio=${WARMUP_RATIO}"
    --set "training.gradient_accumulation_steps=${gradient_accumulation_steps}"
    --set "training.gradient_clip_norm=${GRADIENT_CLIP_NORM}"
    --set "training.mixed_precision=${MIXED_PRECISION}"
    --set "training.muon_lr=${muon_lr}"
    --set "training.muon_momentum=${MUON_MOMENTUM}"
    --set "training.muon_ns_steps=${MUON_NS_STEPS}"
    --set "training.muon_weight_decay=${MUON_WEIGHT_DECAY}"
    --set "training.muon_nesterov=true"
    --set "training.muon_split_qkv=true"
    --set "training.muon_batch_updates=true"
    --set "checkpointing.save_every_epochs=${SAVE_EVERY_EPOCHS}"
    --set "checkpointing.save_final=${SAVE_FINAL}"
    --set "logging.log_every_steps=${LOG_EVERY_STEPS}"
    --set "logging.eval_every_steps=${EVAL_EVERY_STEPS}"
    --set "logging.eval_max_batches=${EVAL_MAX_BATCHES}"
    --set "logging.profile_optimizer_steps=${PROFILE_OPTIMIZER_STEPS}"
  )

  if [[ -n "$LLAMA_EMBD_PDROP" ]]; then
    command+=(--set "model.llama_embd_pdrop=${LLAMA_EMBD_PDROP}")
  fi
  if [[ -n "$LLAMA_RESID_PDROP" ]]; then
    command+=(--set "model.llama_resid_pdrop=${LLAMA_RESID_PDROP}")
  fi
  if [[ -n "$LLAMA_ATTENTION_DROPOUT" ]]; then
    command+=(--set "model.llama_attention_dropout=${LLAMA_ATTENTION_DROPOUT}")
  fi

  if [[ -n "$MAX_TRAIN_CHUNKS" ]]; then
    command+=(--set "data.max_train_chunks=${MAX_TRAIN_CHUNKS}")
  fi
  if [[ -n "$MAX_VAL_CHUNKS" ]]; then
    command+=(--set "data.max_val_chunks=${MAX_VAL_CHUNKS}")
  fi
  if [[ -n "$MAX_TRAIN_BLOCKS" ]]; then
    command+=(--set "data.max_train_blocks=${MAX_TRAIN_BLOCKS}")
  fi
  if [[ -n "$MAX_VAL_BLOCKS" ]]; then
    command+=(--set "data.max_val_blocks=${MAX_VAL_BLOCKS}")
  fi

  echo "Launching ${name} on GPU ${gpu}"
  echo "  config: ${CONFIG}"
  echo "  layers/hidden/heads: ${LAYERS}/${HIDDEN_SIZE}/${N_HEAD}"
  echo "  intermediate/KV heads: ${LLAMA_INTERMEDIATE_SIZE}/${LLAMA_NUM_KEY_VALUE_HEADS}"
  echo "  effective batch size: ${batch_size}"
  echo "  micro batch size: ${micro_batch_size}"
  echo "  gradient accumulation steps: ${gradient_accumulation_steps}"
  echo "  aux lr: ${AUX_LR}"
  echo "  muon lr: ${muon_lr}"
  echo "  mixed precision: ${MIXED_PRECISION}"
  echo "  use Liger: ${USE_LIGER_KERNEL}"
  if [[ -n "$LLAMA_EMBD_PDROP" || -n "$LLAMA_RESID_PDROP" || -n "$LLAMA_ATTENTION_DROPOUT" ]]; then
    echo "  llama dropout attention/embd/resid: ${LLAMA_ATTENTION_DROPOUT:-0}/${LLAMA_EMBD_PDROP:-0}/${LLAMA_RESID_PDROP:-0}"
  fi
  echo "  epochs: ${EPOCHS}"
  echo "  warmup ratio: ${WARMUP_RATIO}"
  echo "  save every epochs: ${SAVE_EVERY_EPOCHS}"
  echo "  save final: ${SAVE_FINAL}"
  if [[ -n "$MAX_TRAIN_CHUNKS" ]]; then
    echo "  max train chunks: ${MAX_TRAIN_CHUNKS}"
  fi
  if [[ -n "$MAX_TRAIN_BLOCKS" ]]; then
    echo "  max train blocks: ${MAX_TRAIN_BLOCKS}"
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
      | grep --line-buffered -E "Output dir:|Train chunks:|Val chunks:|Model architecture:|Liger kernel applied:|Model parameters:|Optimizer:|Muon hidden|Auxiliary AdamW|Muon split-QKV|\\[step [0-9]+\\]|Epoch [0-9]+|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
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

for batch_size in "${BATCH_SIZES[@]}"; do
  for muon_lr in "${MUON_LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_run "${GPUS[$gpu_slot]}" "$muon_lr" "$seed" "$batch_size"

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
  echo "One or more LLaMA Muon sweep runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All LLaMA Muon sweep runs finished."
