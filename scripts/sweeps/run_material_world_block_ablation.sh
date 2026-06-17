#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# GPT-2 8l512 BabyLM 10M block-replacement ablation for material-world data.
#
# Default sweep:
#   baseline, dyn0p5, dynall0p85 x seeds 42/43 on GPUs 5/7
#
# Dry run:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/run_material_world_block_ablation.sh
#
# Smoke run:
#   BABYSTEPS_SMOKE=1 BABYSTEPS_GPUS="5 7" \
#     bash scripts/sweeps/run_material_world_block_ablation.sh
#
# Optional true 1% combined condition:
#   BABYSTEPS_MW_CONDITIONS="baseline dyn0p5 dynall0p85 dynprops1p0" \
#     bash scripts/sweeps/run_material_world_block_ablation.sh

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"
SMOKE="${BABYSTEPS_SMOKE:-0}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
BUILD_MIXES="${BABYSTEPS_BUILD_MIXES:-1}"
REBUILD_MIXES="${BABYSTEPS_REBUILD_MIXES:-0}"

GPUS=(${BABYSTEPS_GPUS:-5 7})
SEEDS=(${BABYSTEPS_SEEDS:-42 43})
CONDITIONS=(${BABYSTEPS_MW_CONDITIONS:-baseline dyn0p5 dynall0p85})

BASE_CONFIG="${BABYSTEPS_CONFIG:-configs/train/gpt2_8l512_bpe16384_10m_muon.yaml}"
BASE_CHUNKS_DIR="${BABYSTEPS_10M_BPE16384_CHUNKS_DIR:-${BABYSTEPS_CACHE_DIR:-/SSD2/babylm-baby-steps-cache}/token_chunks/babylm10m_byte_bpe_16384_block512_eos}"
TOKENIZER_DIR="${BABYSTEPS_10M_BPE16384_TOKENIZER_DIR:-${BABYSTEPS_CACHE_DIR:-/SSD2/babylm-baby-steps-cache}/tokenizers/babylm10m_byte_bpe_16384}"
MIX_CONFIG="${BABYSTEPS_MIX_CONFIG:-configs/data/material_world_token_mix_10m.yaml}"
MIX_ROOT="${BABYSTEPS_MW_MIX_ROOT:-${BABYSTEPS_CACHE_DIR:-/SSD2/babylm-baby-steps-cache}/token_chunks}"
MIX_SEED="${BABYSTEPS_MW_MIX_SEED:-1001}"

DYN_TEXT="${BABYSTEPS_MW_DYN_TEXT:-data/material_world/fireworks_probe_ewok_softmix_1000_dyn/train_audited.txt}"
PROPS_TEXT="${BABYSTEPS_MW_PROPS_TEXT:-data/material_world/fireworks_probe_ewok_softmix_1000/train_audited.txt}"

# Baseline has 31,732 train chunks. 0.5% is floor-rounded to 158 blocks.
DYN_0P5_TOKENS="${BABYSTEPS_MW_DYN_0P5_TOKENS:-81233}"
# Current dyn corpus packs to 270 full 512-token blocks.
DYN_ALL_TOKENS="${BABYSTEPS_MW_DYN_ALL_TOKENS:-138543}"
# True 1% of baseline content tokens: floor(16,246,784 * 0.01).
DYN_PROPS_1P0_TOKENS="${BABYSTEPS_MW_DYN_PROPS_1P0_TOKENS:-162467}"

LOG_DIR="${BABYSTEPS_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/material_world_block_ablation}"

if [[ "$SMOKE" == "1" ]]; then
  EPOCHS="${BABYSTEPS_EPOCHS:-1}"
  MAX_TRAIN_CHUNKS="${BABYSTEPS_MAX_TRAIN_CHUNKS:-1024}"
  MAX_VAL_CHUNKS="${BABYSTEPS_MAX_VAL_CHUNKS:-128}"
  SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-0}"
  SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-false}"
  EVAL_EVERY_STEPS="${BABYSTEPS_EVAL_EVERY_STEPS:-100}"
  EVAL_MAX_BATCHES="${BABYSTEPS_EVAL_MAX_BATCHES:-4}"
else
  EPOCHS="${BABYSTEPS_EPOCHS:-10}"
  MAX_TRAIN_CHUNKS="${BABYSTEPS_MAX_TRAIN_CHUNKS:-}"
  MAX_VAL_CHUNKS="${BABYSTEPS_MAX_VAL_CHUNKS:-}"
  SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-1}"
  SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"
  EVAL_EVERY_STEPS="${BABYSTEPS_EVAL_EVERY_STEPS:-500}"
  EVAL_MAX_BATCHES="${BABYSTEPS_EVAL_MAX_BATCHES:-20}"
fi

BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-32}"
AUX_LR="${BABYSTEPS_AUX_LR:-0.00005}"
MUON_LR="${BABYSTEPS_MUON_LR:-0.01}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
MUON_MOMENTUM="${BABYSTEPS_MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${BABYSTEPS_MUON_NS_STEPS:-5}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
GRADIENT_CLIP_NORM="${BABYSTEPS_GRADIENT_CLIP_NORM:-1.0}"
LOG_EVERY_STEPS="${BABYSTEPS_LOG_EVERY_STEPS:-50}"

pids=()
names=()
progress_pids=()
active_jobs=0
gpu_slot=0
failed=0

usage_conditions="baseline dyn0p5 dynall0p85 dynprops1p0"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs listed. Set BABYSTEPS_GPUS, e.g. BABYSTEPS_GPUS=\"5 7\"."
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

print_command() {
  printf '  command:'
  printf ' %q' "$@"
  printf '\n'
}

condition_requested_tokens() {
  case "$1" in
    dyn0p5) echo "$DYN_0P5_TOKENS" ;;
    dynall0p85) echo "$DYN_ALL_TOKENS" ;;
    dynprops1p0) echo "$DYN_PROPS_1P0_TOKENS" ;;
    baseline) echo "" ;;
    *)
      echo "Unknown condition '$1'. Known: ${usage_conditions}" >&2
      return 1
      ;;
  esac
}

condition_mix_dir() {
  case "$1" in
    dyn0p5)
      echo "${MIX_ROOT}/mw_dyn_k${DYN_0P5_TOKENS}_mixseed${MIX_SEED}"
      ;;
    dynall0p85)
      echo "${MIX_ROOT}/mw_dyn_k${DYN_ALL_TOKENS}_mixseed${MIX_SEED}"
      ;;
    dynprops1p0)
      echo "${MIX_ROOT}/mw_dyn_props_k${DYN_PROPS_1P0_TOKENS}_mixseed${MIX_SEED}"
      ;;
    baseline)
      echo "$BASE_CHUNKS_DIR"
      ;;
    *)
      echo "Unknown condition '$1'. Known: ${usage_conditions}" >&2
      return 1
      ;;
  esac
}

condition_train_chunks() {
  echo "$(condition_mix_dir "$1")/train.pkl"
}

build_mix_if_needed() {
  local condition="$1"
  local output_dir
  local train_chunks
  local requested_tokens
  local command

  if [[ "$condition" == "baseline" ]]; then
    return
  fi
  if [[ "$BUILD_MIXES" != "1" ]]; then
    return
  fi

  output_dir="$(condition_mix_dir "$condition")"
  train_chunks="${output_dir}/train.pkl"
  requested_tokens="$(condition_requested_tokens "$condition")"

  if [[ -f "$train_chunks" && "$REBUILD_MIXES" != "1" ]]; then
    echo "Mix exists for ${condition}: ${train_chunks}"
    return
  fi

  command=(
    conda run --no-capture-output -n "$CONDA_ENV"
    python scripts/data/mix_token_blocks.py
    --config "$MIX_CONFIG"
    --source-train-chunks "${BASE_CHUNKS_DIR}/train.pkl"
    --tokenizer "$TOKENIZER_DIR"
    --requested-tokens "$requested_tokens"
    --seed "$MIX_SEED"
    --output-dir "$output_dir"
  )

  case "$condition" in
    dyn0p5|dynall0p85)
      command+=(--synthetic-file "$DYN_TEXT")
      ;;
    dynprops1p0)
      command+=(--synthetic-file "$DYN_TEXT")
      command+=(--synthetic-file "$PROPS_TEXT")
      ;;
  esac

  if [[ "$REBUILD_MIXES" == "1" ]]; then
    command+=(--overwrite)
  fi

  echo "Building mix for ${condition}"
  echo "  requested tokens: ${requested_tokens}"
  echo "  output dir: ${output_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
    return
  fi

  "${command[@]}"
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
    echo "Stopping active material-world ablation jobs..."
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
  local condition="$2"
  local seed="$3"
  local train_chunks
  local name
  local log_path
  local command
  local train_pid
  local progress_pid

  train_chunks="$(condition_train_chunks "$condition")"
  name="gpt2_8l512_mw_${condition}_seed${seed}"
  if [[ "$SMOKE" == "1" ]]; then
    name="${name}_smoke"
  fi
  log_path="${LOG_DIR}/${name}.log"

  command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
    setsid conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$BASE_CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${seed}"
    --set "data.train_chunks=${train_chunks}"
    --set "data.val_chunks=${BASE_CHUNKS_DIR}/dev.pkl"
    --set "training.epochs=${EPOCHS}"
    --set "training.batch_size=${BATCH_SIZE}"
    --set "training.optimizer=muon_pe"
    --set "training.learning_rate=${AUX_LR}"
    --set "training.warmup_ratio=${WARMUP_RATIO}"
    --set "training.gradient_clip_norm=${GRADIENT_CLIP_NORM}"
    --set "training.muon_lr=${MUON_LR}"
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
  echo "  condition: ${condition}"
  echo "  train chunks: ${train_chunks}"
  echo "  seed: ${seed}"
  echo "  epochs: ${EPOCHS}"
  if [[ -n "$MAX_TRAIN_CHUNKS" ]]; then
    echo "  max train chunks: ${MAX_TRAIN_CHUNKS}"
  fi
  echo "  log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
    return
  fi

  "${command[@]}" >"$log_path" 2>&1 &
  train_pid="$!"
  pids+=("$train_pid")
  names+=("$name")

  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    tail --pid="$train_pid" -n 0 -F "$log_path" 2>/dev/null \
      | grep --line-buffered -E "Output dir:|Train chunks:|Val chunks:|Model parameters:|Optimizer:|Muon hidden|Auxiliary AdamW|\\[step [0-9]+\\]|Epoch [0-9]+|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
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

echo "Material-world block ablation"
echo "  conditions: ${CONDITIONS[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "  GPUs: ${GPUS[*]}"
echo "  smoke: ${SMOKE}"
echo "  dry run: ${DRY_RUN}"
echo "  mix seed: ${MIX_SEED}"
echo "  log dir: ${LOG_DIR}"

for condition in "${CONDITIONS[@]}"; do
  build_mix_if_needed "$condition"
done

for condition in "${CONDITIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    launch_run "${GPUS[$gpu_slot]}" "$condition" "$seed"

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

if [[ "$DRY_RUN" != "1" && "$active_jobs" -gt 0 ]]; then
  wait_for_batch
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more material-world ablation runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All material-world block ablation runs finished."
