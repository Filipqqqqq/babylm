#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT:-${RUNS_DIR}/experiments/gpt_bert_linguistic_repair_dose_response_v1}"
EXPERIMENT_CONFIG="${BABYSTEPS_EXPERIMENT_CONFIG:-configs/experiments/linguistic_repair_dose_response_v1/experiment.yaml}"
WORKER="scripts/sweeps/run_gptbert_linguistic_repair_dose_response_worker.sh"
SESSION_GPU2="${BABYSTEPS_TMUX_SESSION_GPU2:-gptbert_repair_dose_gpu2}"
SESSION_GPU3="${BABYSTEPS_TMUX_SESSION_GPU3:-gptbert_repair_dose_gpu3}"
SKIP_GPU_BUSY_CHECK="${BABYSTEPS_SKIP_GPU_BUSY_CHECK:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

GPU2_JOBS=(repair_0pct:42 repair_1p5pct:42 repair_1pct:43 repair_0pct:44 repair_1p5pct:44)
GPU3_JOBS=(repair_1pct:42 repair_0pct:43 repair_1p5pct:43 repair_1pct:44)

for path in "$EXPERIMENT_CONFIG" "$WORKER"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi

if [[ -f "${EXPERIMENT_ROOT}/manifests/experiment_manifest.json" ]]; then
  echo "Validating existing paired token datasets..."
  conda run --no-capture-output -n "$CONDA_ENV" \
    python scripts/experiments/build_linguistic_repair_dose_response.py \
      --config "$EXPERIMENT_CONFIG" \
      --output-root "$EXPERIMENT_ROOT" \
      --validate-only
else
  echo "Building paired token datasets..."
  conda run --no-capture-output -n "$CONDA_ENV" \
    python scripts/experiments/build_linguistic_repair_dose_response.py \
      --config "$EXPERIMENT_CONFIG" \
      --output-root "$EXPERIMENT_ROOT"
fi

gpu_is_busy() {
  local gpu="$1"
  local output
  output="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${output//[[:space:]]/}" && "$output" != *"No running"* ]]
}

for gpu in 2 3; do
  if [[ "$SKIP_GPU_BUSY_CHECK" != "1" ]] && gpu_is_busy "$gpu"; then
    echo "GPU ${gpu} already has a compute process; refusing to launch." >&2
    exit 1
  fi
done

for session in "$SESSION_GPU2" "$SESSION_GPU3"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_ROOT}/logs/gpu2" "${EXPERIMENT_ROOT}/logs/gpu3"

launch_queue() {
  local session="$1"
  local gpu="$2"
  shift 2
  local jobs=("$@")
  local queue_log="${EXPERIMENT_ROOT}/logs/gpu${gpu}/queue.log"
  local command

  printf -v command 'exec env BABYSTEPS_GPU=%q BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT=%q BABYSTEPS_DRY_RUN=%q bash %q' \
    "$gpu" "$EXPERIMENT_ROOT" "$DRY_RUN" "$WORKER"
  for job in "${jobs[@]}"; do
    printf -v command '%s %q' "$command" "$job"
  done
  printf -v command '%s 2>&1 | tee -a %q' "$command" "$queue_log"

  echo "Launching ${session} on GPU ${gpu}: ${jobs[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'tmux new-session -d -s %q -c %q %q\n' "$session" "$ROOT" "$command"
    return
  fi
  tmux new-session -d -s "$session" -c "$ROOT" "$command"
}

launch_queue "$SESSION_GPU2" 2 "${GPU2_JOBS[@]}"
launch_queue "$SESSION_GPU3" 3 "${GPU3_JOBS[@]}"

echo
echo "Launched controlled linguistic-repair experiment."
echo "  experiment root: ${EXPERIMENT_ROOT}"
echo "  GPU 2 session: ${SESSION_GPU2}"
echo "  GPU 3 session: ${SESSION_GPU3}"
echo "Attach with: tmux attach -t ${SESSION_GPU2}"
