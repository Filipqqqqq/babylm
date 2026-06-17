#!/usr/bin/env bash
set -euo pipefail

# Evaluate BabyLM 2026 Strict LLaMA architecture runs on SuperGLUE fine-tuning.
#
# Default target:
#   width bump: 12 layers, 896 hidden, Muon LR 0.01, effective batch size 64,
#               seeds 42/43/44, run suffix _babylm2026strict, GPUs 3/4/7
#
# The architecture spec format mirrors run_llama_babylm2026_width_depth_sweep.sh:
#   layers,hidden_size,n_head[,num_key_value_heads[,intermediate_size[,micro_batch_size]]]
#
# Only layers and hidden_size are needed to find checkpoints, but accepting the
# full spec keeps train/eval commands copy-paste compatible.
#
# Dry run:
#   BABYSTEPS_DRY_RUN=1 bash scripts/eval/eval_superglue_llama_arch_sweep.sh
#
# Include the depth bump too:
#   BABYSTEPS_ARCH_SPECS="12,896,14,14,2304,16 14,768,12,12,2048,32" \
#   bash scripts/eval/eval_superglue_llama_arch_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

ARCH_SPECS=(${BABYSTEPS_ARCH_SPECS:-12,896,14,14,2304,16})

export BABYSTEPS_MODEL_ARCH="${BABYSTEPS_MODEL_ARCH:-llama}"
export BABYSTEPS_BPES="${BABYSTEPS_BPES:-16384}"
export BABYSTEPS_MUON_LRS="${BABYSTEPS_MUON_LRS:-0.01}"
export BABYSTEPS_BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-64}"
export BABYSTEPS_SEEDS="${BABYSTEPS_SEEDS:-42 43 44}"
export BABYSTEPS_RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-_babylm2026strict}"
export BABYSTEPS_GPUS="${BABYSTEPS_GPUS:-3 4 7}"

is_positive_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

if [[ "${#ARCH_SPECS[@]}" -eq 0 ]]; then
  echo "No architecture specs provided. Set BABYSTEPS_ARCH_SPECS." >&2
  exit 1
fi

echo "BabyLM 2026 LLaMA SuperGLUE fine-tuning eval sweep"
echo "  arch specs: ${ARCH_SPECS[*]}"
echo "  bpes: ${BABYSTEPS_BPES}"
echo "  muon lrs: ${BABYSTEPS_MUON_LRS}"
echo "  train/effective batch size in run names: ${BABYSTEPS_BATCH_SIZE}"
echo "  seeds: ${BABYSTEPS_SEEDS}"
echo "  run suffix: ${BABYSTEPS_RUN_SUFFIX}"
echo "  gpus: ${BABYSTEPS_GPUS}"
echo "  tasks: ${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli}"

for spec in "${ARCH_SPECS[@]}"; do
  IFS=',' read -r layers hidden_size n_head kv_heads intermediate_size spec_micro_batch_size extra <<<"$spec"

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

  if [[ -n "${spec_micro_batch_size:-}" ]] && ! is_positive_int "$spec_micro_batch_size"; then
    echo "Invalid architecture spec '${spec}': micro_batch_size must be a positive integer when provided." >&2
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

  export BABYSTEPS_MODEL_SIZE="${layers}l${hidden_size}"

  echo
  echo "=== Evaluating ${BABYSTEPS_MODEL_ARCH}_${BABYSTEPS_MODEL_SIZE} on SuperGLUE ==="
  bash scripts/eval/eval_superglue_finetune_sweep.sh
done

echo
echo "All BabyLM 2026 LLaMA SuperGLUE architecture evals finished."
