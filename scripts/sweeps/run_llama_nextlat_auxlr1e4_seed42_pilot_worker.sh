#!/usr/bin/env bash
set -euo pipefail

# One role in the seed-42 auxiliary-LR/NextLat SuperGLUE pilot.
#
# gpu3: train the h2 NextLat checkpoint, then fine-tune it at 1e-5.
# gpu4: fine-tune the existing baseline at 1e-5, wait for the new checkpoint,
#       then fine-tune the new NextLat checkpoint at 3e-5.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

ROLE="${1:-}"
CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:-}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_PILOT_ROOT:-${RUNS_DIR}/experiments/llama_nextlat_auxlr1e4_seed42_pilot}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml}"
EVAL_SCRIPT="scripts/eval/eval_superglue_llama_nextlat_gpu3.sh"
GPU_POLL_SECONDS="${BABYSTEPS_GPU_POLL_SECONDS:-60}"
CHECKPOINT_POLL_SECONDS="${BABYSTEPS_CHECKPOINT_POLL_SECONDS:-60}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

BASELINE_CHECKPOINT="${BABYSTEPS_BASELINE_CHECKPOINT:-/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_strict_muon_auxlr0p0001_seed42/20260711_081550/final}"
NEXTLAT_NAME="llama_12l896_babylm100m_nextlat_paperfixed_h2_auxlr1e4_dynlr4em4_kl0p1_seed42_pilot"
NEXTLAT_RUN_DIR="${BABYSTEPS_NEXTLAT_RUN_DIR:-${RUNS_DIR}/training/${NEXTLAT_NAME}}"
NEXTLAT_CHECKPOINT="${NEXTLAT_RUN_DIR}/final"

case "$ROLE" in
  gpu3)
    if [[ "$GPU" != "3" ]]; then
      echo "The gpu3 role must use physical GPU 3; got GPU=${GPU:-unset}." >&2
      exit 2
    fi
    ;;
  gpu4)
    if [[ "$GPU" != "4" ]]; then
      echo "The gpu4 role must use physical GPU 4; got GPU=${GPU:-unset}." >&2
      exit 2
    fi
    ;;
  *)
    echo "Usage: BABYSTEPS_GPU=3 $0 gpu3 | BABYSTEPS_GPU=4 $0 gpu4" >&2
    exit 2
    ;;
esac

for path in "$CONFIG" "$EVAL_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if [[ ! -f "${BASELINE_CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ || ! "$CHECKPOINT_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Poll intervals must be positive integers." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Seed-42 auxiliary-LR/NextLat pilot dry run"
  echo "  role: ${ROLE}"
  echo "  physical GPU: ${GPU}"
  echo "  baseline: ${BASELINE_CHECKPOINT}"
  echo "  NextLat output: ${NEXTLAT_RUN_DIR}"
  if [[ "$ROLE" == "gpu3" ]]; then
    echo "  stages: train h2 (base LR 1e-4, dynamics LR 4e-4), NextLat FT LR 1e-5"
  else
    echo "  stages: baseline FT LR 1e-5, wait for h2, NextLat FT LR 3e-5"
  fi
  exit 0
fi

mkdir -p "${CAMPAIGN_ROOT}/logs/gpu${GPU}" "${CAMPAIGN_ROOT}/status"

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
    echo "[$(date --iso-8601=seconds)] GPU ${GPU} busy (PIDs: ${output//$'\n'/,}); waiting ${GPU_POLL_SECONDS}s."
    sleep "$GPU_POLL_SECONDS"
  done
}

CURRENT_STATUS_BASE=""
mark_worker_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_STATUS_BASE" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.running"
    touch "${CURRENT_STATUS_BASE}.failed"
  fi
  exit "$status"
}
trap mark_worker_exit EXIT

begin_stage() {
  local stage="$1"
  CURRENT_STATUS_BASE="${CAMPAIGN_ROOT}/status/${stage}"
  rm -f "${CURRENT_STATUS_BASE}.failed" "${CURRENT_STATUS_BASE}.complete"
  touch "${CURRENT_STATUS_BASE}.running"
}

complete_stage() {
  rm -f "${CURRENT_STATUS_BASE}.running"
  touch "${CURRENT_STATUS_BASE}.complete"
  CURRENT_STATUS_BASE=""
}

train_nextlat() {
  local stage="train_nextlat_h2_seed42"
  local log_path="${CAMPAIGN_ROOT}/logs/gpu${GPU}/train_nextlat_h2_seed42.log"
  local command

  if [[ -f "${NEXTLAT_CHECKPOINT}/model.safetensors" ]]; then
    echo "[skip] completed NextLat checkpoint: ${NEXTLAT_CHECKPOINT}"
    touch "${CAMPAIGN_ROOT}/status/${stage}.complete"
    return 0
  fi
  if [[ -d "$NEXTLAT_RUN_DIR" && -n "$(find "$NEXTLAT_RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Incomplete non-empty run directory requires inspection: ${NEXTLAT_RUN_DIR}" >&2
    return 1
  fi

  begin_stage "$stage"
  wait_for_gpu_idle
  command=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$CONFIG"
    --set "run.name=${NEXTLAT_NAME}"
    --set "run.seed=42"
    --set "run.output_dir=${NEXTLAT_RUN_DIR}"
    --set "run.append_timestamp=false"
    --set "training.learning_rate=0.0001"
    --set "training.extra_learning_rate=0.0004"
    --set "nextlat.enabled=true"
    --set "nextlat.horizon=2"
    --set "nextlat.lambda_next_h=0.5"
    --set "nextlat.lambda_kl=0.1"
  )

  echo "[$(date --iso-8601=seconds)] Training seed-42 h2 NextLat on GPU ${GPU}."
  echo "  base auxiliary LR: 1e-4"
  echo "  dynamics LR: 4e-4"
  echo "  output: ${NEXTLAT_RUN_DIR}"
  echo "  log: ${log_path}"
  "${command[@]}" 2>&1 | tee "$log_path"

  if [[ ! -f "${NEXTLAT_CHECKPOINT}/model.safetensors" ]]; then
    echo "Training finished without a final checkpoint: ${NEXTLAT_CHECKPOINT}" >&2
    return 1
  fi
  complete_stage
}

wait_for_nextlat_checkpoint() {
  while [[ ! -f "${NEXTLAT_CHECKPOINT}/model.safetensors" ]]; do
    if [[ -f "${CAMPAIGN_ROOT}/status/train_nextlat_h2_seed42.failed" ]]; then
      echo "NextLat training failed; not waiting indefinitely for its checkpoint." >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] Waiting ${CHECKPOINT_POLL_SECONDS}s for ${NEXTLAT_CHECKPOINT}."
    sleep "$CHECKPOINT_POLL_SECONDS"
  done
}

run_superglue() {
  local stage="$1"
  local label="$2"
  local checkpoint="$3"
  local lr="$4"
  local lr_slug="${lr//-/_}"
  local output_root="${CAMPAIGN_ROOT}/evaluation/ftlr_${lr_slug}"
  local log_dir="${CAMPAIGN_ROOT}/logs/gpu${GPU}/superglue_ftlr_${lr_slug}"

  if [[ -f "${CAMPAIGN_ROOT}/status/${stage}.complete" ]]; then
    echo "[skip] completed stage: ${stage}"
    return 0
  fi
  if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
    echo "Missing checkpoint for ${stage}: ${checkpoint}" >&2
    return 1
  fi

  begin_stage "$stage"
  wait_for_gpu_idle
  echo "[$(date --iso-8601=seconds)] SuperGLUE ${label}, fine-tuning LR ${lr}, GPU ${GPU}."
  env \
    BABYSTEPS_SUPERGLUE_GPU="$GPU" \
    BABYSTEPS_FINETUNE_SEED=42 \
    BABYSTEPS_SUPERGLUE_LR="$lr" \
    BABYSTEPS_SUPERGLUE_OUTPUT_ROOT="$output_root" \
    BABYSTEPS_SUPERGLUE_LOG_DIR="$log_dir" \
    bash "$EVAL_SCRIPT" "${label}=${checkpoint}"
  complete_stage
}

if [[ "$ROLE" == "gpu3" ]]; then
  train_nextlat
  run_superglue \
    nextlat_h2_ftlr1e5 \
    nextlat_h2_auxlr1e4_dynlr4em4_seed42 \
    "$NEXTLAT_CHECKPOINT" \
    1e-5
else
  run_superglue \
    baseline_auxlr1e4_ftlr1e5 \
    baseline_auxlr1e4_seed42 \
    "$BASELINE_CHECKPOINT" \
    1e-5
  wait_for_nextlat_checkpoint
  run_superglue \
    nextlat_h2_ftlr3e5 \
    nextlat_h2_auxlr1e4_dynlr4em4_seed42 \
    "$NEXTLAT_CHECKPOINT" \
    3e-5
fi

echo "GPU ${GPU} role ${ROLE} complete."
