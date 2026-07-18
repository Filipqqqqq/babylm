#!/usr/bin/env bash
set -euo pipefail

# Recover the final-only benchmark evaluation requested for the AdamMuon 1%
# linguistic-repair curriculum campaign. The original worker saved the epoch-10
# checkpoints but only ran validation loss, so this evaluates those immutable
# checkpoints post hoc without retraining.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:-0}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_CAMPAIGN_ROOT:-${RUNS_DIR}/experiments/adamuon_linguistic_repair_1pct_curriculum_superglue}"
OUTPUT_ROOT="${BABYSTEPS_FINAL_BENCHMARK_ROOT:-${CAMPAIGN_ROOT}/final_epoch10_benchmarks}"
POLL_SECONDS="${BABYSTEPS_GPU_POLL_SECONDS:-60}"
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 43)
fi

if [[ "$GPU" != "0" ]]; then
  echo "This recovery queue is assigned to physical GPU 0; got GPU=${GPU}." >&2
  exit 2
fi
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "BABYSTEPS_GPU_POLL_SECONDS must be a positive integer." >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/status"

wait_for_campaign() {
  local seed
  for seed in "${SEEDS[@]}"; do
    while [[ ! -f "${CAMPAIGN_ROOT}/status/seed_${seed}.complete" ]]; do
      echo "[$(date --iso-8601=seconds)] Waiting for seed ${seed} campaign completion."
      sleep "$POLL_SECONDS"
    done
  done
}

gpu_compute_pids() {
  nvidia-smi \
    --id="$GPU" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null
}

wait_for_gpu_idle() {
  local output
  while true; do
    if ! output="$(gpu_compute_pids)"; then
      echo "Could not query GPU ${GPU}; refusing to assume it is idle." >&2
      return 1
    fi
    if [[ -z "${output//[[:space:]]/}" || "$output" == *"No running"* ]]; then
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] GPU ${GPU} busy; waiting ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
  done
}

run_seed() {
  local seed="$1"
  local checkpoint="${RUNS_DIR}/training/adamuon-10m-mixed-linguistic-repair1pct-lr0.02-aux0.006-s${seed}-curriculum-tailavg/model_tail_avg"
  local seed_root="${OUTPUT_ROOT}/seed_${seed}_tailavg"
  local status_base="${OUTPUT_ROOT}/status/seed_${seed}"

  if [[ -f "${status_base}.complete" ]]; then
    echo "[skip] final epoch-10 benchmarks already complete for seed ${seed}."
    return 0
  fi
  if [[ ! -f "${checkpoint}/pytorch_model.bin" ]]; then
    echo "Missing tail-averaged checkpoint for seed ${seed}: ${checkpoint}" >&2
    return 1
  fi

  rm -f "${status_base}.failed"
  touch "${status_base}.running"
  mkdir -p "$seed_root"

  echo "=== Final epoch-10 benchmark evaluation: seed ${seed} tail average ==="
  env \
    "CUDA_VISIBLE_DEVICES=${GPU}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
    python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" \
      --tasks blimp blimp_supplement entity_tracking comps \
      --eval-root external/babylm-eval/strict/evaluation_data/full_eval \
      --batch-size 64 \
      --score-reduction sum \
      --trust-remote-code \
      --local-files-only \
      --no-items \
      --output-dir "${seed_root}/mntp_likelihood"

  env \
    "CUDA_VISIBLE_DEVICES=${GPU}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
    python -u scripts/eval/eval_ewok_context_sensitivity.py \
      --checkpoint "$checkpoint" \
      --ewok-data-dir external/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered \
      --batch-size 64 \
      --score-reduction sum \
      --device cuda \
      --output-dir "${seed_root}/ewok_context"

  env \
    "CUDA_VISIBLE_DEVICES=${GPU}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
    python -u scripts/eval/eval_ewok_context_sensitivity.py \
      --checkpoint "$checkpoint" \
      --ewok-data-dir external/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered \
      --batch-size 64 \
      --score-reduction mean \
      --device cuda \
      --append \
      --output-dir "${seed_root}/ewok_context"

  rm -f "${status_base}.running"
  touch "${status_base}.complete"
  echo "[complete] final epoch-10 benchmarks for seed ${seed}."
}

mark_failure() {
  local status="$?"
  local path
  if [[ "$status" -ne 0 ]]; then
    for path in "${OUTPUT_ROOT}/status/"*.running; do
      [[ -e "$path" ]] || continue
      mv "$path" "${path%.running}.failed"
    done
  fi
  exit "$status"
}
trap mark_failure EXIT

echo "AdamMuon 1% linguistic-repair final epoch-10 benchmark recovery queue"
echo "  GPU: ${GPU}"
echo "  seeds: ${SEEDS[*]}"
echo "  output: ${OUTPUT_ROOT}"
wait_for_campaign
wait_for_gpu_idle
for seed in "${SEEDS[@]}"; do
  run_seed "$seed"
done
echo "Final epoch-10 benchmark recovery queue complete."
