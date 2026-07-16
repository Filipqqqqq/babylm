#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

EXPERIMENT_ID="${1:-${BABYSTEPS_EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-/home/jorge/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_SEQUENTIAL_ROOT:-${RUNS_DIR}/experiments/entity_tracking_sequential_v1/${EXPERIMENT_ID}}"
WORKER="scripts/sweeps/run_entity_tracking_sequential_worker.sh"
MIN_FREE_MIB="${BABYSTEPS_MIN_FREE_MIB:-30000}"
SESSION_GPU2="et_seq_${EXPERIMENT_ID}_gpu2"
SESSION_GPU3="et_seq_${EXPERIMENT_ID}_gpu3"

if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi
for required in \
  "$WORKER" \
  "/home/jorge/babylm-baby-steps-cache/token_shards/babylm2026_strict_small_midtrain17pct_container_state_multidomain_v3_gpt_bert_mixed/meta.json" \
  "/home/jorge/babylm-baby-steps-cache/token_shards/container_state_multidomain_v3_gpt_bert_mixed/meta.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing prepared sequential input: $required" >&2
    exit 1
  fi
done

for gpu in 2 3; do
  free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ "$free_mib" -lt "$MIN_FREE_MIB" || -n "${processes//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} is not launch-safe (free=${free_mib} MiB, processes=${processes:-none})." >&2
    exit 1
  fi
done
for session in "$SESSION_GPU2" "$SESSION_GPU3"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_ROOT}/logs/gpu2" "${EXPERIMENT_ROOT}/logs/gpu3"
printf '%s\n' \
  '{' \
  "  \"experiment_id\": \"${EXPERIMENT_ID}\"," \
  '  "design": "baseline -> midtrain -> finetune",' \
  '  "gpu_queues": {"2": [42, 44], "3": [43]},' \
  '  "reference_experiment": "entity_tracking_multidomain_v3/20260716_092451"' \
  '}' > "${EXPERIMENT_ROOT}/launch_manifest.json"

launch_queue() {
  local session="$1"
  local gpu="$2"
  shift 2
  local seeds=("$@")
  local queue_log="${EXPERIMENT_ROOT}/logs/gpu${gpu}/queue.log"
  local command
  printf -v command 'exec env BABYSTEPS_GPU=%q BABYSTEPS_EXPERIMENT_ID=%q BABYSTEPS_ET_SEQUENTIAL_ROOT=%q bash %q' \
    "$gpu" "$EXPERIMENT_ID" "$EXPERIMENT_ROOT" "$WORKER"
  for seed in "${seeds[@]}"; do
    printf -v command '%s %q' "$command" "$seed"
  done
  printf -v command '%s 2>&1 | tee -a %q' "$command" "$queue_log"
  tmux new-session -d -s "$session" -c "$ROOT" "$command"
  echo "Launched ${session}: GPU ${gpu}, seeds ${seeds[*]}"
}

launch_queue "$SESSION_GPU2" 2 42 44
launch_queue "$SESSION_GPU3" 3 43

echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "GPU 2 queue: seed 42 -> seed 44"
echo "GPU 3 queue: seed 43"
