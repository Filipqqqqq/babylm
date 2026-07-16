#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

RESUME_GPU1=0
if [[ "${1:-}" == "--resume-gpu1" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --resume-gpu1 PILOT_ID" >&2
    exit 2
  fi
  RESUME_GPU1=1
  PILOT_ID="$2"
else
  PILOT_ID="${BABYSTEPS_PILOT_ID:-$(date +%Y%m%d_%H%M%S)}"
fi
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_PILOT_ROOT:-${RUNS_DIR}/experiments/entity_tracking_counterfactual_pilot_v2/${PILOT_ID}}"
WORKER="scripts/sweeps/run_entity_tracking_counterfactual_pilot_worker.sh"
SESSION_GPU1="${BABYSTEPS_ET_SESSION_GPU1:-et_cf_pilot_${PILOT_ID}_gpu1}"
SESSION_GPU2="${BABYSTEPS_ET_SESSION_GPU2:-et_cf_pilot_${PILOT_ID}_gpu2}"

if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi
if [[ ! -f "$WORKER" ]]; then
  echo "Missing pilot worker: $WORKER" >&2
  exit 1
fi

gpu_is_busy() {
  local gpu="$1"
  local output
  output="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -n "${output//[[:space:]]/}" && "$output" != *"No running"* ]]
}

GPUS=(1 2)
if [[ "$RESUME_GPU1" == "1" ]]; then
  GPUS=(1)
  SESSION_GPU1="${SESSION_GPU1}_matched"
fi
for gpu in "${GPUS[@]}"; do
  if gpu_is_busy "$gpu"; then
    echo "GPU ${gpu} already has a compute process; refusing to launch." >&2
    exit 1
  fi
done
SESSIONS=("$SESSION_GPU1" "$SESSION_GPU2")
if [[ "$RESUME_GPU1" == "1" ]]; then
  SESSIONS=("$SESSION_GPU1")
fi
for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_ROOT}/logs/gpu1" "${EXPERIMENT_ROOT}/logs/gpu2"

launch_queue() {
  local session="$1"
  local gpu="$2"
  shift 2
  local arms=("$@")
  local queue_log="${EXPERIMENT_ROOT}/logs/gpu${gpu}/queue.log"
  local command
  printf -v command 'exec env BABYSTEPS_GPU=%q BABYSTEPS_PILOT_ID=%q BABYSTEPS_ET_PILOT_ROOT=%q bash %q' \
    "$gpu" "$PILOT_ID" "$EXPERIMENT_ROOT" "$WORKER"
  for arm in "${arms[@]}"; do
    printf -v command '%s %q' "$command" "$arm"
  done
  printf -v command '%s 2>&1 | tee -a %q' "$command" "$queue_log"
  echo "Launching ${session} on GPU ${gpu}: ${arms[*]}"
  tmux new-session -d -s "$session" -c "$ROOT" "$command"
}

# Sequential within each GPU; the two queues run concurrently.
launch_queue "$SESSION_GPU1" 1 arm1 arm3
if [[ "$RESUME_GPU1" != "1" ]]; then
  launch_queue "$SESSION_GPU2" 2 arm2 arm4
fi

echo
echo "Launched entity-tracking counterfactual pilot ${PILOT_ID}."
echo "  experiment root: ${EXPERIMENT_ROOT}"
echo "  GPU 1 session: ${SESSION_GPU1} (arm1 -> arm3)"
if [[ "$RESUME_GPU1" != "1" ]]; then
  echo "  GPU 2 session: ${SESSION_GPU2} (arm2 -> arm4)"
else
  echo "  GPU 2 queue was left untouched."
fi
