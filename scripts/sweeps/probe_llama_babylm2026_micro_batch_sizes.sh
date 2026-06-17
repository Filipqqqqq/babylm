#!/usr/bin/env bash
set -euo pipefail

# Quick GPU-fit probe for BabyLM 2026 Strict LLaMA architecture candidates.
#
# This runs tiny one-epoch training probes on one GPU to find the largest
# micro-batch size that fits for each architecture. The effective batch stays at
# 64 by default, so only micro-batch sizes that divide 64 are useful.
#
# Defaults probe the width/depth candidates from:
#   scripts/sweeps/run_llama_babylm2026_width_depth_sweep.sh
#
# Dry-run exact commands:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/probe_llama_babylm2026_micro_batch_sizes.sh
#
# Run on GPU 1:
#   BABYSTEPS_GPU=1 bash scripts/sweeps/probe_llama_babylm2026_micro_batch_sizes.sh
#
# Spec format:
#   layers,hidden_size,n_head[,num_key_value_heads[,intermediate_size]]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

ARCH_SPECS=(${BABYSTEPS_ARCH_SPECS:-12,896,14,14 14,768,12,12})
MICRO_BATCH_SIZES=(${BABYSTEPS_MICRO_BATCH_SIZES:-64 32 16 8})

GPU="${BABYSTEPS_GPU:-1}"
EFFECTIVE_BATCH_SIZE="${BABYSTEPS_EFFECTIVE_BATCH_SIZE:-64}"
MUON_LR="${BABYSTEPS_MUON_LR:-0.01}"
SEED="${BABYSTEPS_SEED:-42}"
RUN_SUFFIX_BASE="${BABYSTEPS_RUN_SUFFIX:-_babylm2026strict}"
LOG_DIR="${BABYSTEPS_LOG_DIR:-/SSD2/babylm-baby-steps-runs/logs/llama_microbatch_probe}"
MAX_TRAIN_BLOCKS="${BABYSTEPS_MAX_TRAIN_BLOCKS:-${BABYSTEPS_PROBE_MAX_TRAIN_BLOCKS:-128}}"
MAX_VAL_BLOCKS="${BABYSTEPS_MAX_VAL_BLOCKS:-${BABYSTEPS_PROBE_MAX_VAL_BLOCKS:-32}}"
STOP_ON_FIT="${BABYSTEPS_PROBE_STOP_ON_FIT:-1}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

is_positive_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

if [[ "${#ARCH_SPECS[@]}" -eq 0 ]]; then
  echo "No architecture specs provided. Set BABYSTEPS_ARCH_SPECS." >&2
  exit 1
fi

if [[ "${#MICRO_BATCH_SIZES[@]}" -eq 0 ]]; then
  echo "No micro-batch sizes provided. Set BABYSTEPS_MICRO_BATCH_SIZES." >&2
  exit 1
fi

if ! is_positive_int "$EFFECTIVE_BATCH_SIZE"; then
  echo "BABYSTEPS_EFFECTIVE_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi

echo "BabyLM 2026 LLaMA micro-batch probe"
echo "  GPU: ${GPU}"
echo "  arch specs: ${ARCH_SPECS[*]}"
echo "  candidate micro-batches: ${MICRO_BATCH_SIZES[*]}"
echo "  effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  muon lr: ${MUON_LR}"
echo "  seed: ${SEED}"
echo "  max train blocks: ${MAX_TRAIN_BLOCKS}"
echo "  max val blocks: ${MAX_VAL_BLOCKS}"
echo "  log dir: ${LOG_DIR}"

results=()

for spec in "${ARCH_SPECS[@]}"; do
  IFS=',' read -r layers hidden_size n_head kv_heads intermediate_size extra <<<"$spec"

  if [[ -n "${extra:-}" ]]; then
    echo "Invalid architecture spec '${spec}': too many comma-separated fields." >&2
    exit 1
  fi

  if ! is_positive_int "$layers" || ! is_positive_int "$hidden_size" || ! is_positive_int "$n_head"; then
    echo "Invalid architecture spec '${spec}': layers, hidden_size, and n_head must be positive integers." >&2
    exit 1
  fi

  kv_heads="${kv_heads:-$n_head}"
  if ! is_positive_int "$kv_heads"; then
    echo "Invalid architecture spec '${spec}': num_key_value_heads must be a positive integer." >&2
    exit 1
  fi

  if [[ -n "${intermediate_size:-}" ]] && ! is_positive_int "$intermediate_size"; then
    echo "Invalid architecture spec '${spec}': intermediate_size must be a positive integer when provided." >&2
    exit 1
  fi

  if (( hidden_size % n_head != 0 )); then
    echo "Invalid architecture spec '${spec}': hidden_size must be divisible by n_head." >&2
    exit 1
  fi

  if (( n_head % kv_heads != 0 )); then
    echo "Invalid architecture spec '${spec}': n_head must be divisible by num_key_value_heads." >&2
    exit 1
  fi

  echo
  echo "=== Probing ${layers}l${hidden_size}, heads=${n_head}, kv_heads=${kv_heads} ==="
  found_fit=0

  for micro_batch_size in "${MICRO_BATCH_SIZES[@]}"; do
    if ! is_positive_int "$micro_batch_size"; then
      echo "Skipping invalid micro-batch size '${micro_batch_size}'."
      results+=("${layers}l${hidden_size} micro=${micro_batch_size}: invalid")
      continue
    fi

    if (( EFFECTIVE_BATCH_SIZE % micro_batch_size != 0 )); then
      echo "Skipping micro-batch ${micro_batch_size}: it does not divide effective batch ${EFFECTIVE_BATCH_SIZE}."
      results+=("${layers}l${hidden_size} micro=${micro_batch_size}: skipped")
      continue
    fi

    run_suffix="${RUN_SUFFIX_BASE}_mb${micro_batch_size}_probe"
    run_name="llama_${layers}l${hidden_size}_bpe16384_muon_lr$(slug "$MUON_LR")_wu0p03_mwd0p1_bs${EFFECTIVE_BATCH_SIZE}_seed${SEED}${run_suffix}"
    log_path="${LOG_DIR}/${run_name}.log"

    echo
    echo "Trying micro-batch ${micro_batch_size} for ${layers}l${hidden_size}"
    echo "  run: ${run_name}"
    echo "  log: ${log_path}"

    if [[ -n "${intermediate_size:-}" ]]; then
      command=(
        env
        "BABYSTEPS_GPUS=${GPU}"
        "BABYSTEPS_LAYERS=${layers}"
        "BABYSTEPS_HIDDEN_SIZE=${hidden_size}"
        "BABYSTEPS_N_HEAD=${n_head}"
        "BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS=${kv_heads}"
        "BABYSTEPS_LLAMA_INTERMEDIATE_SIZE=${intermediate_size}"
        "BABYSTEPS_BATCH_SIZES=${EFFECTIVE_BATCH_SIZE}"
        "BABYSTEPS_MICRO_BATCH_SIZE=${micro_batch_size}"
        "BABYSTEPS_MUON_LRS=${MUON_LR}"
        "BABYSTEPS_SEEDS=${SEED}"
        "BABYSTEPS_EPOCHS=1"
        "BABYSTEPS_SAVE_EVERY_EPOCHS=0"
        "BABYSTEPS_SAVE_FINAL=false"
        "BABYSTEPS_MAX_TRAIN_BLOCKS=${MAX_TRAIN_BLOCKS}"
        "BABYSTEPS_MAX_VAL_BLOCKS=${MAX_VAL_BLOCKS}"
        "BABYSTEPS_EVAL_EVERY_STEPS=999999"
        "BABYSTEPS_EVAL_MAX_BATCHES=1"
        "BABYSTEPS_LOG_EVERY_STEPS=1"
        "BABYSTEPS_LOG_DIR=${LOG_DIR}"
        "BABYSTEPS_RUN_SUFFIX=${run_suffix}"
        bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
      )
    else
      command=(
        env -u BABYSTEPS_LLAMA_INTERMEDIATE_SIZE
        "BABYSTEPS_GPUS=${GPU}"
        "BABYSTEPS_LAYERS=${layers}"
        "BABYSTEPS_HIDDEN_SIZE=${hidden_size}"
        "BABYSTEPS_N_HEAD=${n_head}"
        "BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS=${kv_heads}"
        "BABYSTEPS_BATCH_SIZES=${EFFECTIVE_BATCH_SIZE}"
        "BABYSTEPS_MICRO_BATCH_SIZE=${micro_batch_size}"
        "BABYSTEPS_MUON_LRS=${MUON_LR}"
        "BABYSTEPS_SEEDS=${SEED}"
        "BABYSTEPS_EPOCHS=1"
        "BABYSTEPS_SAVE_EVERY_EPOCHS=0"
        "BABYSTEPS_SAVE_FINAL=false"
        "BABYSTEPS_MAX_TRAIN_BLOCKS=${MAX_TRAIN_BLOCKS}"
        "BABYSTEPS_MAX_VAL_BLOCKS=${MAX_VAL_BLOCKS}"
        "BABYSTEPS_EVAL_EVERY_STEPS=999999"
        "BABYSTEPS_EVAL_MAX_BATCHES=1"
        "BABYSTEPS_LOG_EVERY_STEPS=1"
        "BABYSTEPS_LOG_DIR=${LOG_DIR}"
        "BABYSTEPS_RUN_SUFFIX=${run_suffix}"
        bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
      )
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '  command:'
      printf ' %q' "${command[@]}"
      printf '\n'
      results+=("${layers}l${hidden_size} micro=${micro_batch_size}: dry-run")
      continue
    fi

    if "${command[@]}"; then
      echo "FIT: ${layers}l${hidden_size} micro-batch ${micro_batch_size}"
      results+=("${layers}l${hidden_size} micro=${micro_batch_size}: fit")
      found_fit=1
      if [[ "$STOP_ON_FIT" == "1" ]]; then
        break
      fi
    else
      status=$?
      echo "FAILED: ${layers}l${hidden_size} micro-batch ${micro_batch_size} exited ${status}"
      echo "  inspect: ${log_path}"
      results+=("${layers}l${hidden_size} micro=${micro_batch_size}: failed")
    fi
  done

  if [[ "$found_fit" == "0" && "$DRY_RUN" != "1" ]]; then
    echo "No fitting micro-batch found for ${layers}l${hidden_size}."
  fi
done

echo
echo "Micro-batch probe summary:"
for result in "${results[@]}"; do
  echo "  ${result}"
done
