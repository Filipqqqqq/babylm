#!/usr/bin/env bash
set -euo pipefail

# Launch the matched LLaMA baseline/NextLat replication queue on physical GPU 0.
# Existing seed-42 reference runs are reused; this queue adds seeds 43 and 44,
# followed by the h1 dynamics-LR control requested for the residual correction.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

GPU="${BABYSTEPS_GPU:-0}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CAMPAIGN_ROOT="${BABYSTEPS_NEXTLAT_CAMPAIGN_ROOT:-${RUNS_DIR}/experiments/llama_nextlat_replicates_v1}"
WORKER="scripts/sweeps/run_llama_nextlat_replicates_worker.sh"
SESSION="${BABYSTEPS_TMUX_SESSION:-llama_nextlat_repl_gpu0}"
ALLOW_BUSY_GPU="${BABYSTEPS_ALLOW_BUSY_GPU:-0}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ "$GPU" != "0" ]]; then
  echo "This launcher is intentionally pinned to physical GPU 0; got GPU=${GPU}." >&2
  exit 2
fi
for path in "$WORKER" configs/train/llama_12l896_babylm100m_nextlat_paperfixed.yaml; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi

# Together with the new seed-43/44 jobs below, these completed checkpoints give
# the baseline, fixed h1/KL=.1, and fixed h2/KL=.1 arms three seeds (42/43/44).
REFERENCE_CHECKPOINTS=(
  "/home/jorge/babylm-baby-steps-runs/training/llama_12l896_bpe16384_babylm2026_strict_muon_liger_baseline_causal_diagnostics_50m_bs64/20260704_082711/final"
  "/home/jorge/babylm-baby-steps-runs/training/llama_12l896_babylm100m_nextlat_paperfixed_h1_dynlr4em4_kl0p1_seed42/20260712_151122/final"
  "/home/jorge/babylm-baby-steps-runs/training/llama_12l896_babylm100m_nextlat_paperfixed_h2_dynlr4em4_kl0p1_seed42/20260712_151122/final"
)
for checkpoint in "${REFERENCE_CHECKPOINTS[@]}"; do
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing seed-42 reference checkpoint: $checkpoint" >&2
    exit 1
  fi
done

JOBS=(
  baseline:43
  fixed_h1:43
  fixed_h2:43
  baseline:44
  fixed_h1:44
  fixed_h2:44
  residual_h1:42
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "$CAMPAIGN_ROOT/logs/gpu0"
queue_log="${CAMPAIGN_ROOT}/logs/gpu0/queue.log"
printf -v command 'exec env BABYSTEPS_GPU=0 BABYSTEPS_NEXTLAT_CAMPAIGN_ROOT=%q BABYSTEPS_ALLOW_BUSY_GPU=%q BABYSTEPS_DRY_RUN=%q bash %q' \
  "$CAMPAIGN_ROOT" "$ALLOW_BUSY_GPU" "$DRY_RUN" "$WORKER"
for job in "${JOBS[@]}"; do
  printf -v command '%s %q' "$command" "$job"
done
printf -v command '%s 2>&1 | tee -a %q' "$command" "$queue_log"

echo "LLaMA NextLat replication campaign"
echo "  reused seed: 42 (baseline, fixed h1/KL=.1, fixed h2/KL=.1)"
echo "  queued seeds: 43 and 44 for those three arms"
echo "  residual control: h1/KL=.1, dynamics LR 5e-5, seed 42"
echo "  rollout diagnostics: 20 batches at 200M, 300M, and 1.15B words"
echo "  KL=.2: not queued"
if [[ "$ALLOW_BUSY_GPU" == "1" ]]; then
  echo "  GPU: physical 0 (sharing enabled; busy-process wait bypassed)"
else
  echo "  GPU: physical 0 (worker waits while busy)"
fi
echo "  session: ${SESSION}"
echo "  log: ${queue_log}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'tmux new-session -d -s %q -c %q %q\n' "$SESSION" "$ROOT" "$command"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Attach with: tmux attach -t ${SESSION}"
