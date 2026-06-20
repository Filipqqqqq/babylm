#!/usr/bin/env bash
set -euo pipefail

# Targeted BabyLM 2026 Strict LLaMA architecture sweep.
#
# This keeps the current LLaMA + Muon recipe fixed and sweeps model shape so we
# can test width-vs-depth hypotheses without rewriting the base launcher.
#
# Default hypotheses:
#   width bump: 12 layers, 896 hidden, 14 heads, 14 KV heads
#   depth bump: 14 layers, 768 hidden, 12 heads, 12 KV heads
#
# Spec format:
#   layers,hidden_size,n_head[,num_key_value_heads[,intermediate_size[,micro_batch_size]]]
#
# If intermediate_size is omitted, scripts/sweeps/run_llama_muon_lr_seed_sweep.sh will
# compute the parameter-matched LLaMA MLP size for that hidden width.
#
# Dry-run exact commands:
#   BABYSTEPS_DRY_RUN=1 bash scripts/sweeps/run_llama_babylm2026_width_depth_sweep.sh
#
# Add larger candidates:
#   BABYSTEPS_ARCH_SPECS="12,896,14,14,2304,16 12,1024,16,16,2816,16 14,768,12,12,2048,32 16,768,12,12,2048,32" \
#   bash scripts/sweeps/run_llama_babylm2026_width_depth_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

ARCH_SPECS=(${BABYSTEPS_ARCH_SPECS:-12,896,14,14,2304,16 14,768,12,12,2048,32})

export BABYSTEPS_LOG_DIR="${BABYSTEPS_LOG_DIR:-${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}/logs/llama_width_depth}"
export BABYSTEPS_MUON_LRS="${BABYSTEPS_MUON_LRS:-0.01}"
export BABYSTEPS_BATCH_SIZES="${BABYSTEPS_BATCH_SIZES:-64}"
export BABYSTEPS_MICRO_BATCH_SIZE="${BABYSTEPS_MICRO_BATCH_SIZE:-32}"
export BABYSTEPS_SEEDS="${BABYSTEPS_SEEDS:-42 43 44}"
export BABYSTEPS_EPOCHS="${BABYSTEPS_EPOCHS:-10}"
export BABYSTEPS_RUN_SUFFIX="${BABYSTEPS_RUN_SUFFIX:-_babylm2026strict}"
export BABYSTEPS_MIXED_PRECISION="${BABYSTEPS_MIXED_PRECISION:-bf16}"
export BABYSTEPS_USE_LIGER_KERNEL="${BABYSTEPS_USE_LIGER_KERNEL:-true}"
export BABYSTEPS_SAVE_EVERY_EPOCHS="${BABYSTEPS_SAVE_EVERY_EPOCHS:-1}"
export BABYSTEPS_SAVE_FINAL="${BABYSTEPS_SAVE_FINAL:-true}"

is_positive_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

if [[ "${#ARCH_SPECS[@]}" -eq 0 ]]; then
  echo "No architecture specs provided. Set BABYSTEPS_ARCH_SPECS." >&2
  exit 1
fi

echo "BabyLM 2026 LLaMA width/depth sweep"
echo "  arch specs: ${ARCH_SPECS[*]}"
echo "  muon lrs: ${BABYSTEPS_MUON_LRS}"
echo "  effective batch sizes: ${BABYSTEPS_BATCH_SIZES}"
echo "  micro batch size: ${BABYSTEPS_MICRO_BATCH_SIZE}"
echo "  seeds: ${BABYSTEPS_SEEDS}"
echo "  run suffix: ${BABYSTEPS_RUN_SUFFIX}"
echo "  log dir: ${BABYSTEPS_LOG_DIR}"

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

  effective_micro_batch_size="${spec_micro_batch_size:-${BABYSTEPS_MICRO_BATCH_SIZE}}"

  if (( hidden_size % n_head != 0 )); then
    echo "Invalid architecture spec '${spec}': hidden_size must be divisible by n_head." >&2
    exit 1
  fi

  if (( n_head % kv_heads != 0 )); then
    echo "Invalid architecture spec '${spec}': n_head must be divisible by num_key_value_heads." >&2
    exit 1
  fi

  echo
  echo "=== Launching architecture ${layers}l${hidden_size}, heads=${n_head}, kv_heads=${kv_heads} ==="
  echo "    micro batch size: ${effective_micro_batch_size}"
  if [[ -n "${intermediate_size:-}" ]]; then
    echo "    intermediate size: ${intermediate_size}"
    env_command=(
      env
      "BABYSTEPS_LAYERS=${layers}"
      "BABYSTEPS_HIDDEN_SIZE=${hidden_size}"
      "BABYSTEPS_N_HEAD=${n_head}"
      "BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS=${kv_heads}"
      "BABYSTEPS_LLAMA_INTERMEDIATE_SIZE=${intermediate_size}"
      "BABYSTEPS_MICRO_BATCH_SIZE=${effective_micro_batch_size}"
      bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
    )
  else
    echo "    intermediate size: auto parameter-matched"
    env_command=(
      env -u BABYSTEPS_LLAMA_INTERMEDIATE_SIZE
      "BABYSTEPS_LAYERS=${layers}"
      "BABYSTEPS_HIDDEN_SIZE=${hidden_size}"
      "BABYSTEPS_N_HEAD=${n_head}"
      "BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS=${kv_heads}"
      "BABYSTEPS_MICRO_BATCH_SIZE=${effective_micro_batch_size}"
      bash scripts/sweeps/run_llama_babylm2026_muon_lr_seed_sweep.sh
    )
  fi

  "${env_command[@]}"
done

echo
echo "All BabyLM 2026 LLaMA width/depth sweep architectures finished."
