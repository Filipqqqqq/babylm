#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_CAMPAIGN_ROOT:-${RUNS_DIR}/experiments/adamuon_linguistic_repair_1pct_curriculum_superglue}"
WORKER="scripts/sweeps/run_adamuon_linguistic_repair_1pct_curriculum_worker.sh"
GPU="${BABYSTEPS_GPU:-0}"
SESSION="${BABYSTEPS_TMUX_SESSION:-adamuon_repair1pct_s42_s43_gpu${GPU}}"
MIN_FREE_MIB="${BABYSTEPS_MIN_FREE_MIB:-28000}"
ALLOW_BUSY_GPU="${BABYSTEPS_ALLOW_BUSY_GPU:-${BABYSTEPS_ALLOW_BUSY_GPU0:-0}}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ ! -f "$WORKER" ]]; then
  echo "Missing worker: $WORKER" >&2
  exit 1
fi
if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi
if [[ ! "$MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "BABYSTEPS_MIN_FREE_MIB must be a positive integer." >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "BABYSTEPS_GPU must be a non-negative physical GPU index; got GPU=${GPU}." >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

free_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
compute_pids="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free memory on GPU ${GPU}." >&2
  exit 1
fi
if (( free_mib < MIN_FREE_MIB )); then
  echo "GPU ${GPU} has ${free_mib} MiB free; at least ${MIN_FREE_MIB} MiB is required." >&2
  exit 1
fi
if [[ -n "${compute_pids//[[:space:]]/}" && "$compute_pids" != *"No running"* && "$ALLOW_BUSY_GPU" != "1" ]]; then
  echo "GPU ${GPU} has compute PID(s) ${compute_pids//$'\n'/,}; set BABYSTEPS_ALLOW_BUSY_GPU=1 only after confirming safe sharing." >&2
  exit 1
fi

mkdir -p "${CAMPAIGN_ROOT}/logs/gpu${GPU}"
queue_log="${CAMPAIGN_ROOT}/logs/gpu${GPU}/queue.log"
printf -v command \
  'exec env BABYSTEPS_GPU=%q BABYSTEPS_RUNS_DIR=%q BABYSTEPS_CAMPAIGN_ROOT=%q BABYSTEPS_DRY_RUN=0 bash %q 42 43 2>&1 | tee -a %q' \
  "$GPU" "$RUNS_DIR" "$CAMPAIGN_ROOT" "$WORKER" "$queue_log"

echo "Launching seed-42/43 AdamMuon 1% linguistic-repair curriculum + SuperGLUE campaign"
echo "  GPU ${GPU} free memory: ${free_mib} MiB"
echo "  session: ${SESSION}"
echo "  campaign root: ${CAMPAIGN_ROOT}"
echo "  queue: seed 42 pretrain + SuperGLUE, then seed 43 pretrain + SuperGLUE"
if [[ -n "${compute_pids//[[:space:]]/}" && "$compute_pids" != *"No running"* ]]; then
  echo "  sharing GPU ${GPU} with existing PID(s): ${compute_pids//$'\n'/,}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'tmux new-session -d -s %q -c %q %q\n' "$SESSION" "$ROOT" "$command"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Attach with: tmux attach -t ${SESSION}"
