#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

STAGE="initial"
if [[ "${1:-}" == "--stage" ]]; then
  STAGE="${2:-}"
  shift 2
fi
case "$STAGE" in
  initial) ARMS=(baseline lm_only) ;;
  completion) ARMS=(completion_rank) ;;
  symmetric) ARMS=(symmetric_rank) ;;
  *) echo "--stage must be initial, completion, or symmetric" >&2; exit 2 ;;
esac

EXPERIMENT_ID="${1:-${BABYSTEPS_EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-/home/jorge/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_V3_ROOT:-${RUNS_DIR}/experiments/entity_tracking_multidomain_v3/${EXPERIMENT_ID}}"
WORKER="scripts/sweeps/run_entity_tracking_multidomain_v3_worker.sh"
GPUS=(1 2 5)
SEEDS=(42 43 44)
MIN_FREE_MIB="${BABYSTEPS_MIN_FREE_MIB:-30000}"

if ! command -v tmux >/dev/null; then
  echo "tmux is not installed." >&2
  exit 1
fi
for required in \
  "$WORKER" \
  "data/container_state/container_state_multidomain_v3_seed42/audit_report.json" \
  "/home/jorge/babylm-baby-steps-cache/token_shards/babylm2026_strict_small_replace1pct_container_state_multidomain_v3_gpt_bert_mixed/meta.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing prepared v3 input: $required" >&2
    exit 1
  fi
done

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  seed="${SEEDS[$index]}"
  free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ "$free_mib" -lt "$MIN_FREE_MIB" || -n "${processes//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} is not launch-safe (free=${free_mib} MiB, processes=${processes:-none})." >&2
    exit 1
  fi
  session="et_md_v3_${EXPERIMENT_ID}_s${seed}_${STAGE}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$EXPERIMENT_ROOT"
cat > "${EXPERIMENT_ROOT}/launch_manifest.json" <<EOF
{
  "experiment_id": "${EXPERIMENT_ID}",
  "stage": "${STAGE}",
  "gpus": [1, 2, 5],
  "seeds": [42, 43, 44],
  "arms": ["$(IFS='","'; echo "${ARMS[*]}")"]
}
EOF

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  seed="${SEEDS[$index]}"
  session="et_md_v3_${EXPERIMENT_ID}_s${seed}_${STAGE}"
  queue_log="${EXPERIMENT_ROOT}/logs/gpu${gpu}/seed${seed}_${STAGE}_queue.log"
  mkdir -p "$(dirname "$queue_log")"
  printf -v command 'exec env BABYSTEPS_GPU=%q BABYSTEPS_SEED=%q BABYSTEPS_EXPERIMENT_ID=%q BABYSTEPS_ET_V3_ROOT=%q bash %q' \
    "$gpu" "$seed" "$EXPERIMENT_ID" "$EXPERIMENT_ROOT" "$WORKER"
  for arm in "${ARMS[@]}"; do
    printf -v command '%s %q' "$command" "$arm"
  done
  printf -v command '%s 2>&1 | tee -a %q' "$command" "$queue_log"
  tmux new-session -d -s "$session" -c "$ROOT" "$command"
  echo "Launched ${session}: GPU ${gpu}, seed ${seed}, arms ${ARMS[*]}"
done

echo "Experiment root: ${EXPERIMENT_ROOT}"
