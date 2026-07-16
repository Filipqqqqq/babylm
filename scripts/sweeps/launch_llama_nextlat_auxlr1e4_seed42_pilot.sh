#!/usr/bin/env bash
set -euo pipefail

# Launch the seed-42 auxiliary-LR/NextLat pilot as two dependency-aware tmux
# queues on physical GPUs 3 and 4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_PILOT_ROOT:-${RUNS_DIR}/experiments/llama_nextlat_auxlr1e4_seed42_pilot}"
WORKER="scripts/sweeps/run_llama_nextlat_auxlr1e4_seed42_pilot_worker.sh"
SESSION_GPU3="${BABYSTEPS_TMUX_SESSION_GPU3:-llama_nl_aux1e4_s42_g3}"
SESSION_GPU4="${BABYSTEPS_TMUX_SESSION_GPU4:-llama_nl_aux1e4_s42_g4}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

for path in \
  "$WORKER" \
  configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml \
  scripts/eval/eval_superglue_llama_nextlat_gpu3.sh; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi
for session in "$SESSION_GPU3" "$SESSION_GPU4"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi
done

mkdir -p "${CAMPAIGN_ROOT}/logs/gpu3" "${CAMPAIGN_ROOT}/logs/gpu4"
queue_log_gpu3="${CAMPAIGN_ROOT}/logs/gpu3/queue.log"
queue_log_gpu4="${CAMPAIGN_ROOT}/logs/gpu4/queue.log"

printf -v command_gpu3 \
  'exec env BABYSTEPS_GPU=3 BABYSTEPS_RUNS_DIR=%q BABYSTEPS_PILOT_ROOT=%q bash %q gpu3 2>&1 | tee -a %q' \
  "$RUNS_DIR" "$CAMPAIGN_ROOT" "$WORKER" "$queue_log_gpu3"
printf -v command_gpu4 \
  'exec env BABYSTEPS_GPU=4 BABYSTEPS_RUNS_DIR=%q BABYSTEPS_PILOT_ROOT=%q bash %q gpu4 2>&1 | tee -a %q' \
  "$RUNS_DIR" "$CAMPAIGN_ROOT" "$WORKER" "$queue_log_gpu4"

echo "Seed-42 auxiliary-LR/NextLat SuperGLUE pilot"
echo "  GPU 3: train h2 at base LR 1e-4 / dynamics LR 4e-4, then FT at 1e-5"
echo "  GPU 4: existing baseline FT at 1e-5, then new h2 FT at 3e-5"
echo "  reused result: existing baseline FT at 3e-5"
echo "  campaign root: ${CAMPAIGN_ROOT}"
echo "  sessions: ${SESSION_GPU3}, ${SESSION_GPU4}"
echo "  safety: each worker waits for its physical GPU to have no compute process"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'tmux new-session -d -s %q -c %q %q\n' "$SESSION_GPU3" "$ROOT" "$command_gpu3"
  printf 'tmux new-session -d -s %q -c %q %q\n' "$SESSION_GPU4" "$ROOT" "$command_gpu4"
  exit 0
fi

tmux new-session -d -s "$SESSION_GPU3" -c "$ROOT" "$command_gpu3"
tmux new-session -d -s "$SESSION_GPU4" -c "$ROOT" "$command_gpu4"

echo "Attach with: tmux attach -t ${SESSION_GPU3}"
echo "Attach with: tmux attach -t ${SESSION_GPU4}"
