#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# GPT-BERT mixed-objective Muon LR/seed sweep on BabyLM 2026 Strict-Small.
#
# Defaults:
#   model/tokenizer: BabyLM-community/babylm-baseline-10m-gpt-bert-mixed
#   objective:       mixed GPT-BERT, 1 causal step : 1 MNTP step
#   Muon LRs:        0.005, 0.01, 0.02
#   seeds:           42, 43, 44
#   batch/seq/accum: 16 x 512 x 2 = 16,384 tokens/update
#   validation:      every 250 update steps
#   benchmarks:      expensive EWoK causal + MNTP likelihood at midpoint and end
#
# Dry-run the exact commands:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/run_gpt_bert_mixed_muon_lr_seed_sweep.sh
#
# Launch on different GPUs:
#   BABYSTEPS_GPUS="0 1" bash scripts/sweeps/run_gpt_bert_mixed_muon_lr_seed_sweep.sh
#
# Short smoke using the same runner:
#   BABYSTEPS_MAX_TRAIN_BLOCKS=300 BABYSTEPS_EPOCHS=1 BABYSTEPS_SAVE_FINAL=false \
#   bash scripts/sweeps/run_gpt_bert_mixed_muon_lr_seed_sweep.sh

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small.yaml}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/gpt_bert_mixed_muon_sweep}"
SHOW_PROGRESS="${BABYSTEPS_SHOW_PROGRESS:-1}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"
RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-}"

SEEDS=(${BABYSTEPS_SEEDS:-42 43 44})
GPUS=(${BABYSTEPS_GPUS:-1 2 3})
MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.005 0.01 0.02})

BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-16}"
SEQ_LEN="${BABYSTEPS_SEQ_LEN:-512}"
EPOCHS="${BABYSTEPS_EPOCHS:-10}"
GRADIENT_ACCUMULATION_STEPS="${BABYSTEPS_GRADIENT_ACCUMULATION_STEPS:-2}"
MIXED_PRECISION="${BABYSTEPS_MIXED_PRECISION:-bf16}"

AUX_LR="${BABYSTEPS_AUX_LR:-0.00005}"
AUX_BETAS="${BABYSTEPS_AUX_BETAS:-[0.9, 0.999]}"
AUX_WEIGHT_DECAY="${BABYSTEPS_AUX_WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${BABYSTEPS_WARMUP_RATIO:-0.03}"
GRADIENT_CLIP_NORM="${BABYSTEPS_GRADIENT_CLIP_NORM:-1.0}"
MUON_WEIGHT_DECAY="${BABYSTEPS_MUON_WEIGHT_DECAY:-0.1}"
MUON_MOMENTUM="${BABYSTEPS_MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${BABYSTEPS_MUON_NS_STEPS:-5}"

LOG_EVERY_STEPS="${BABYSTEPS_LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${BABYSTEPS_EVAL_EVERY_STEPS:-250}"
EVAL_MAX_BATCHES="${BABYSTEPS_EVAL_MAX_BATCHES:-20}"
SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-0}"
SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"
GPU_CLEANUP_TIMEOUT_SECONDS="${BABYSTEPS_GPU_CLEANUP_TIMEOUT_SECONDS:-30}"
GPU_CLEANUP_SLEEP_SECONDS="${BABYSTEPS_GPU_CLEANUP_SLEEP_SECONDS:-3}"

MAX_TRAIN_BLOCKS="${BABYSTEPS_MAX_TRAIN_BLOCKS:-}"
MAX_VAL_BLOCKS="${BABYSTEPS_MAX_VAL_BLOCKS:-}"
MAX_TRAIN_SHARDS="${BABYSTEPS_MAX_TRAIN_SHARDS:-}"
MAX_VAL_SHARDS="${BABYSTEPS_MAX_VAL_SHARDS:-}"

DEFAULT_CACHE_DIR="${BABYSTEPS_CACHE_DIR:-$HOME/babylm-baby-steps-cache}"
TOKENIZER_DIR="${BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR:-${DEFAULT_CACHE_DIR}/tokenizers/gpt_bert_baseline_mixed}"
SHARDS_DIR="${BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_MIXED_SHARDS_DIR:-${DEFAULT_CACHE_DIR}/token_shards/babylm2026_strict_small_gpt_bert_baseline_mixed}"

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

count_train_blocks() {
  local data_dir="$1"
  local batch_size="$2"
  local seq_len="$3"
  local max_shards="$4"
  local max_blocks="$5"
  local stride_tokens=$(( batch_size * seq_len ))
  local block_tokens=$(( stride_tokens + 1 ))
  local total_blocks=0
  local shard_count=0
  local shard_path
  local nbytes
  local tokens
  local blocks

  shopt -s nullglob
  for shard_path in "${data_dir}"/train_*.bin; do
    shard_count=$(( shard_count + 1 ))
    if [[ -n "$max_shards" && "$shard_count" -gt "$max_shards" ]]; then
      break
    fi

    nbytes="$(stat -c%s "$shard_path")"
    tokens=$(( nbytes / 2 ))
    blocks=0
    if (( tokens >= block_tokens )); then
      blocks=$(( 1 + (tokens - block_tokens) / stride_tokens ))
    fi
    total_blocks=$(( total_blocks + blocks ))
  done
  shopt -u nullglob

  if [[ -n "$max_blocks" && "$total_blocks" -gt "$max_blocks" ]]; then
    total_blocks="$max_blocks"
  fi

  echo "$total_blocks"
}

compute_total_update_steps() {
  local total_blocks
  local updates_per_epoch

  total_blocks="$(count_train_blocks "$SHARDS_DIR" "$BATCH_SIZE" "$SEQ_LEN" "$MAX_TRAIN_SHARDS" "$MAX_TRAIN_BLOCKS")"
  if (( total_blocks <= 0 )); then
    echo "Could not find train blocks in ${SHARDS_DIR}." >&2
    exit 1
  fi

  updates_per_epoch=$(( (total_blocks + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS ))
  echo $(( updates_per_epoch * EPOCHS ))
}

if [[ "$DRY_RUN" != "1" ]]; then
  missing=0

  if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "Missing GPT-BERT mixed tokenizer: ${TOKENIZER_DIR}/tokenizer.json" >&2
    missing=1
  fi

  if ! compgen -G "${SHARDS_DIR}/train_*.bin" >/dev/null; then
    echo "Missing GPT-BERT mixed train shards: ${SHARDS_DIR}/train_*.bin" >&2
    missing=1
  fi

  if ! compgen -G "${SHARDS_DIR}/val_*.bin" >/dev/null; then
    echo "Missing GPT-BERT mixed val shards: ${SHARDS_DIR}/val_*.bin" >&2
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    cat >&2 <<'EOF'

Build the GPT-BERT mixed Strict-Small token shards first:

  conda run --no-capture-output -n babylm python scripts/data/build_token_shards.py \
    --config configs/tokenization/babylm2026_strict_small_gpt_bert_baseline_mixed_shards.yaml
EOF
    exit 1
  fi

  mkdir -p "$LOG_DIR"
fi

TOTAL_UPDATE_STEPS="${BABYSTEPS_TOTAL_UPDATE_STEPS:-$(compute_total_update_steps)}"
BENCHMARK_EVERY_STEPS="${BABYSTEPS_BENCHMARK_EVERY_STEPS:-$(( TOTAL_UPDATE_STEPS / 2 ))}"

if (( BENCHMARK_EVERY_STEPS <= 0 )); then
  echo "Benchmark interval must be positive, got ${BENCHMARK_EVERY_STEPS}." >&2
  exit 1
fi

if [[ -z "${BABYSTEPS_BENCHMARK_EVERY_STEPS:-}" && $(( TOTAL_UPDATE_STEPS % 2 )) -ne 0 ]]; then
  cat >&2 <<EOF
Total update steps (${TOTAL_UPDATE_STEPS}) is odd, so a periodic midpoint interval
cannot hit both midpoint and final exactly. Set BABYSTEPS_BENCHMARK_EVERY_STEPS
explicitly, or adjust BABYSTEPS_MAX_TRAIN_BLOCKS / BABYSTEPS_EPOCHS.
EOF
  exit 1
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs listed. Set BABYSTEPS_GPUS, e.g. BABYSTEPS_GPUS=\"0 1\"." >&2
  exit 1
fi

pids=()
names=()
job_gpus=()
progress_pids=()
active_jobs=0
gpu_slot=0
failed=0

process_group_exists() {
  local pid="$1"

  kill -0 -- "-$pid" 2>/dev/null
}

terminate_pid_tree() {
  local pid="$1"

  if process_group_exists "$pid"; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  elif kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

force_kill_pid_tree() {
  local pid="$1"

  if process_group_exists "$pid"; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  elif kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

wait_for_pid_tree_exit() {
  local pid="$1"
  local timeout_seconds="$2"
  local waited=0

  while process_group_exists "$pid" || kill -0 "$pid" 2>/dev/null; do
    if (( waited >= timeout_seconds )); then
      return 1
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done

  return 0
}

cleanup_completed_run() {
  local pid="$1"
  local name="$2"
  local gpu="$3"

  if process_group_exists "$pid" || kill -0 "$pid" 2>/dev/null; then
    echo "Cleaning up ${name} process group on GPU ${gpu}..."
    terminate_pid_tree "$pid"
    if ! wait_for_pid_tree_exit "$pid" "$GPU_CLEANUP_TIMEOUT_SECONDS"; then
      echo "Force-killing ${name} process group on GPU ${gpu}..."
      force_kill_pid_tree "$pid"
      wait_for_pid_tree_exit "$pid" 10 || true
    fi
  fi

  # Give the CUDA driver a short window to release the process context.
  if (( GPU_CLEANUP_SLEEP_SECONDS > 0 )); then
    sleep "$GPU_CLEANUP_SLEEP_SECONDS"
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
    echo "Stopping active GPT-BERT Muon jobs..."
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
  local muon_lr="$2"
  local seed="$3"
  local name="gpt_bert_baseline_mixed_babylm2026_strict_small_muon_lr$(slug "$muon_lr")_seed${seed}${RUN_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local train_pid
  local progress_pid
  local command

  command=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR=${TOKENIZER_DIR}"
    "BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_MIXED_SHARDS_DIR=${SHARDS_DIR}"
    setsid conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_hybrid_lm.py
    --config "$CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${seed}"
    --set "data.seq_len=${SEQ_LEN}"
    --set "training.epochs=${EPOCHS}"
    --set "training.batch_size=${BATCH_SIZE}"
    --set "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    --set "training.optimizer=muon_pe"
    --set "training.learning_rate=${AUX_LR}"
    --set "training.betas=${AUX_BETAS}"
    --set "training.weight_decay=${AUX_WEIGHT_DECAY}"
    --set "training.warmup_ratio=${WARMUP_RATIO}"
    --set "training.gradient_clip_norm=${GRADIENT_CLIP_NORM}"
    --set "training.mixed_precision=${MIXED_PRECISION}"
    --set "training.muon_lr=${muon_lr}"
    --set "training.muon_weight_decay=${MUON_WEIGHT_DECAY}"
    --set "training.muon_momentum=${MUON_MOMENTUM}"
    --set "training.muon_ns_steps=${MUON_NS_STEPS}"
    --set "training.muon_nesterov=true"
    --set "training.muon_split_qkv=false"
    --set "training.muon_batch_updates=true"
    --set "progress.estimate_words_per_token=false"
    --set "progress.benchmark_plot_x_axis=step"
    --set "logging.log_every_steps=${LOG_EVERY_STEPS}"
    --set "logging.eval_every_steps=${EVAL_EVERY_STEPS}"
    --set "logging.eval_max_batches=${EVAL_MAX_BATCHES}"
    --set "logging.benchmark_eval_at_start=false"
    --set "ewok_context_sensitivity_full.enabled=true"
    --set "ewok_context_sensitivity_full.every_steps=${BENCHMARK_EVERY_STEPS}"
    --set "ewok_context_sensitivity_full.every_words=null"
    --set "ewok_context_sensitivity_full.plot=true"
    --set "ewok_mntp_context_sensitivity.enabled=false"
    --set "ewok_mntp_context_sensitivity.every_steps=0"
    --set "ewok_mntp_context_sensitivity.every_words=null"
    --set "mntp_likelihood.enabled=true"
    --set "mntp_likelihood.every_steps=${BENCHMARK_EVERY_STEPS}"
    --set "mntp_likelihood.every_words=null"
    --set "mntp_likelihood.plot=true"
    --set "loss_dynamics.plot=true"
    --set "loss_dynamics.x_axis=step"
    --set "checkpointing.save_every_epochs=${SAVE_EVERY_EPOCHS}"
    --set "checkpointing.save_final=${SAVE_FINAL}"
  )

  if [[ -n "$MAX_TRAIN_BLOCKS" ]]; then
    command+=(--set "data.max_train_blocks=${MAX_TRAIN_BLOCKS}")
  fi
  if [[ -n "$MAX_VAL_BLOCKS" ]]; then
    command+=(--set "data.max_val_blocks=${MAX_VAL_BLOCKS}")
  fi
  if [[ -n "$MAX_TRAIN_SHARDS" ]]; then
    command+=(--set "data.max_train_shards=${MAX_TRAIN_SHARDS}")
  fi
  if [[ -n "$MAX_VAL_SHARDS" ]]; then
    command+=(--set "data.max_val_shards=${MAX_VAL_SHARDS}")
  fi

  echo "Launching ${name} on GPU ${gpu}"
  echo "  config: ${CONFIG}"
  echo "  tokenizer: ${TOKENIZER_DIR}"
  echo "  shards: ${SHARDS_DIR}"
  echo "  batch/seq/accum: ${BATCH_SIZE}/${SEQ_LEN}/${GRADIENT_ACCUMULATION_STEPS}"
  echo "  epochs: ${EPOCHS}"
  echo "  total update steps: ${TOTAL_UPDATE_STEPS}"
  echo "  validation every steps: ${EVAL_EVERY_STEPS}"
  echo "  benchmark every steps: ${BENCHMARK_EVERY_STEPS}"
  echo "  aux lr: ${AUX_LR}"
  echo "  muon lr: ${muon_lr}"
  echo "  seed: ${seed}"
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
  job_gpus+=("$gpu")

  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    tail --pid="$train_pid" -n 0 -F "$log_path" 2>/dev/null \
      | grep --line-buffered -E "Output dir:|Model architecture:|Model parameters:|Optimizer:|Muon hidden|Auxiliary AdamW|\\[step [0-9]+\\]|val_causal_loss|ewok_context|mntp_score|Hybrid epoch|Traceback|ValueError|RuntimeError|ERROR|FAILED" \
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
    cleanup_completed_run "${pids[$idx]}" "${names[$idx]}" "${job_gpus[$idx]}"
  done

  pids=()
  names=()
  job_gpus=()
  for progress_pid in "${progress_pids[@]}"; do
    wait "$progress_pid" 2>/dev/null || true
  done
  progress_pids=()
  active_jobs=0
  gpu_slot=0
}

echo "GPT-BERT mixed Muon sweep"
echo "  LRs: ${MUON_LRS[*]}"
echo "  seeds: ${SEEDS[*]}"
echo "  GPUs: ${GPUS[*]}"
echo "  total update steps: ${TOTAL_UPDATE_STEPS}"
echo "  benchmark every steps: ${BENCHMARK_EVERY_STEPS}"

for muon_lr in "${MUON_LRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    launch_run "${GPUS[$gpu_slot]}" "$muon_lr" "$seed"

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
  echo "One or more GPT-BERT Muon sweep runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

cat <<EOF
All GPT-BERT mixed Muon sweep runs finished.

Analyze the sweep with:

  conda run --no-capture-output -n ${CONDA_ENV} python scripts/analysis/plot_gpt_bert_muon_sweep.py
EOF
