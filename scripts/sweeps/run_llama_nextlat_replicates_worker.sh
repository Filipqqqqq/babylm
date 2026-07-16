#!/usr/bin/env bash
set -euo pipefail

# Sequential worker for the matched LLaMA NextLat replication campaign.
# It is intended to run inside the tmux session created by
# launch_llama_nextlat_replicates_gpu0_tmux.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml}"
GPU="${BABYSTEPS_GPU:-0}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_NEXTLAT_CAMPAIGN_ROOT:-${RUNS_DIR}/experiments/llama_nextlat_replicates_v1}"
ROLLOUT_BATCHES="${BABYSTEPS_NEXTLAT_ROLLOUT_BATCHES:-20}"
ROLLOUT_WORDS="${BABYSTEPS_NEXTLAT_ROLLOUT_WORDS:-[200000000,300000000,1150000000]}"
GPU_POLL_SECONDS="${BABYSTEPS_GPU_POLL_SECONDS:-60}"
ALLOW_BUSY_GPU="${BABYSTEPS_ALLOW_BUSY_GPU:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 baseline:43 fixed_h1:43 fixed_h2:43 residual_h1:42 [...]" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing training config: $CONFIG" >&2
  exit 1
fi
if [[ ! "$ROLLOUT_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
  echo "BABYSTEPS_NEXTLAT_ROLLOUT_BATCHES must be a positive integer." >&2
  exit 2
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
  if [[ "$DRY_RUN" == "1" || "$ALLOW_BUSY_GPU" == "1" ]]; then
    if [[ "$ALLOW_BUSY_GPU" == "1" ]]; then
      echo "GPU ${GPU} sharing explicitly enabled; skipping the busy-process wait."
    fi
    return
  fi
  while true; do
    if ! output="$(gpu_compute_pids)"; then
      echo "Could not query GPU ${GPU}; refusing to assume it is idle." >&2
      return 1
    fi
    if [[ -z "${output//[[:space:]]/}" || "$output" == *"No running"* ]]; then
      return
    fi
    echo "[$(date --iso-8601=seconds)] GPU ${GPU} busy (PIDs: ${output//$'\n'/,}); waiting ${GPU_POLL_SECONDS}s."
    sleep "$GPU_POLL_SECONDS"
  done
}

completed_checkpoint() {
  local run_root="$1"
  if [[ -d "$run_root" ]]; then
    find "$run_root" -mindepth 2 -maxdepth 2 -type d -name final -print -quit
  fi
}

CURRENT_STATUS_BASE=""
mark_queue_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_STATUS_BASE" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.running"
    touch "${CURRENT_STATUS_BASE}.failed"
  fi
  exit "$status"
}
trap mark_queue_exit EXIT

for spec in "$@"; do
  condition="${spec%%:*}"
  seed="${spec#*:}"
  if [[ "$condition" == "$seed" || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid job '${spec}'. Use condition:seed." >&2
    exit 2
  fi

  horizon=""
  dynamics_lr=""
  nextlat_enabled="true"
  case "$condition" in
    baseline)
      name="llama_12l896_babylm100m_paperfixed_baseline_seed${seed}"
      nextlat_enabled="false"
      ;;
    fixed_h1)
      name="llama_12l896_babylm100m_nextlat_paperfixed_h1_dynlr4em4_kl0p1_seed${seed}_rollout20"
      horizon="1"
      dynamics_lr="0.0004"
      ;;
    fixed_h2)
      name="llama_12l896_babylm100m_nextlat_paperfixed_h2_dynlr4em4_kl0p1_seed${seed}_rollout20"
      horizon="2"
      dynamics_lr="0.0004"
      ;;
    residual_h1)
      name="llama_12l896_babylm100m_nextlat_paperfixed_h1_dynlr5em5_kl0p1_seed${seed}_rollout20"
      horizon="1"
      dynamics_lr="0.00005"
      ;;
    *)
      echo "Unknown condition '${condition}'." >&2
      exit 2
      ;;
  esac

  run_root="${RUNS_DIR}/training/${name}"
  log_path="${CAMPAIGN_ROOT}/logs/gpu${GPU}/${name}.log"
  CURRENT_STATUS_BASE="${CAMPAIGN_ROOT}/status/${condition}_seed_${seed}"

  if checkpoint="$(completed_checkpoint "$run_root")" && [[ -n "$checkpoint" ]]; then
    echo "[skip] completed ${condition} seed ${seed}: ${checkpoint}"
    rm -f "${CURRENT_STATUS_BASE}.running" "${CURRENT_STATUS_BASE}.failed"
    touch "${CURRENT_STATUS_BASE}.complete"
    CURRENT_STATUS_BASE=""
    continue
  fi

  rm -f "${CURRENT_STATUS_BASE}.failed" "${CURRENT_STATUS_BASE}.complete"
  touch "${CURRENT_STATUS_BASE}.running"
  wait_for_gpu_idle

  command=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    conda run --no-capture-output -n "$CONDA_ENV"
    python -u scripts/train/train_causal_lm.py
    --config "$CONFIG"
    --set "run.name=${name}"
    --set "run.seed=${seed}"
    --set "nextlat.enabled=${nextlat_enabled}"
  )

  if [[ "$nextlat_enabled" == "true" ]]; then
    command+=(
      --set "nextlat.horizon=${horizon}"
      --set "nextlat.lambda_next_h=0.5"
      --set "nextlat.lambda_kl=0.1"
      --set "training.extra_learning_rate=${dynamics_lr}"
      --set "nextlat.rollout_eval_max_batches=${ROLLOUT_BATCHES}"
      --set "nextlat.rollout_eval_words=${ROLLOUT_WORDS}"
    )
  fi

  echo
  echo "=== condition=${condition} seed=${seed} GPU=${GPU} ==="
  echo "run: ${name}"
  if [[ "$nextlat_enabled" == "true" ]]; then
    echo "horizon=${horizon} KL=0.1 dynamics_lr=${dynamics_lr}"
    echo "rollout_batches=${ROLLOUT_BATCHES} rollout_words=${ROLLOUT_WORDS}"
  fi
  echo "log: ${log_path}"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "${command[@]}"
    printf '\n'
    rm -f "${CURRENT_STATUS_BASE}.running"
    CURRENT_STATUS_BASE=""
    continue
  fi

  "${command[@]}" 2>&1 | tee "$log_path"

  checkpoint="$(completed_checkpoint "$run_root")"
  if [[ -z "$checkpoint" ]]; then
    echo "Training finished without a final checkpoint under ${run_root}." >&2
    exit 1
  fi
  rm -f "${CURRENT_STATUS_BASE}.running"
  touch "${CURRENT_STATUS_BASE}.complete"
  echo "[complete] ${condition} seed ${seed}: ${checkpoint}"
  CURRENT_STATUS_BASE=""
done

echo "GPU ${GPU} NextLat replication queue complete."
