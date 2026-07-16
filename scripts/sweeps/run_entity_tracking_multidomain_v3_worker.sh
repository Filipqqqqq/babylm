#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:?Set BABYSTEPS_GPU to the physical GPU index.}"
SEED="${BABYSTEPS_SEED:?Set BABYSTEPS_SEED.}"
EXPERIMENT_ID="${BABYSTEPS_EXPERIMENT_ID:?Set BABYSTEPS_EXPERIMENT_ID.}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-/home/jorge/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_V3_ROOT:-${RUNS_DIR}/experiments/entity_tracking_multidomain_v3/${EXPERIMENT_ID}}"
TRAIN_CONFIG="configs/train/gpt_bert_entity_tracking_counterfactual_pilot_common_muon.yaml"
BASELINE_SHARDS="${BABYSTEPS_CACHE_DIR:-/home/jorge/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_local_control_gpt_bert_mixed"
SYNTHETIC_SHARDS="${BABYSTEPS_CACHE_DIR:-/home/jorge/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_replace1pct_container_state_multidomain_v3_gpt_bert_mixed"
DATA_ROOT="data/container_state/container_state_multidomain_v3_seed42"
PAIR_ROOT="${DATA_ROOT}/auxiliary"
CUSTOM_EVAL_ROOT="data/container_state/container_state_v1_pool_seed42/validation"
OFFICIAL_EVAL_ROOT="external/babylm-eval/strict/evaluation_data/full_eval"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 baseline [lm_only|completion_rank|symmetric_rank ...]" >&2
  exit 2
fi
for path in \
  "$TRAIN_CONFIG" \
  "$BASELINE_SHARDS/meta.json" \
  "$SYNTHETIC_SHARDS/meta.json" \
  "${DATA_ROOT}/audit_report.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required v3 input: $path" >&2
    exit 1
  fi
done
python - "$DATA_ROOT/audit_report.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    audit = json.load(f)
gate = audit["surface_classifier"]
if not gate["passed"]:
    raise SystemExit(f"Surface gate failed: {gate}")
if audit["leakage"]["normalized_cross_split_collisions"]:
    raise SystemExit("Cross-split text collisions are nonzero.")
PY

mkdir -p "${EXPERIMENT_ROOT}/logs/gpu${GPU}" "${EXPERIMENT_ROOT}/status"
CURRENT_STATUS=""
mark_failure() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_STATUS" ]]; then
    rm -f "${CURRENT_STATUS}.running"
    touch "${CURRENT_STATUS}.failed"
  fi
  exit "$status"
}
trap mark_failure EXIT

for arm in "$@"; do
  case "$arm" in
    baseline)
      label="matched_local_baseline"
      shards="$BASELINE_SHARDS"
      aux_enabled="false"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/iid_completion_rank.jsonl"
      aux_batch=32
      aux_every_causal="null"
      ;;
    lm_only)
      label="multidomain_v3_lm_only"
      shards="$SYNTHETIC_SHARDS"
      aux_enabled="false"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/iid_completion_rank.jsonl"
      aux_batch=32
      aux_every_causal="null"
      ;;
    completion_rank)
      label="multidomain_v3_completion_rank"
      shards="$SYNTHETIC_SHARDS"
      aux_enabled="true"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/iid_completion_rank.jsonl"
      aux_batch=32
      aux_every_causal=5
      ;;
    symmetric_rank)
      label="multidomain_v3_symmetric_rank"
      shards="$SYNTHETIC_SHARDS"
      aux_enabled="true"
      aux_objective="counterfactual_rank"
      aux_train="${PAIR_ROOT}/train_counterfactual_rank.jsonl"
      aux_eval="${PAIR_ROOT}/iid_counterfactual_rank.jsonl"
      aux_batch=16
      aux_every_causal=5
      ;;
    *)
      echo "Unknown v3 arm: $arm" >&2
      exit 2
      ;;
  esac

  run_key="seed${SEED}_${arm}_${label}"
  run_name="et_multidomain_v3_${EXPERIMENT_ID}_${run_key}"
  run_dir="${EXPERIMENT_ROOT}/training/${run_key}"
  checkpoint="${run_dir}/final"
  train_log="${EXPERIMENT_ROOT}/logs/gpu${GPU}/${run_key}_train.log"
  custom_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/01_custom_v1"
  official_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/02_official"
  multidomain_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/03_multidomain_v3"
  CURRENT_STATUS="${EXPERIMENT_ROOT}/status/${run_key}"
  rm -f "${CURRENT_STATUS}.failed" "${CURRENT_STATUS}.complete"
  touch "${CURRENT_STATUS}.running"

  echo
  echo "=== ${arm}: ${label} | seed ${SEED} | GPU ${GPU} | ${EXPERIMENT_ID} ==="
  if [[ -d "$checkpoint" ]]; then
    echo "[skip] final checkpoint exists: $checkpoint"
  else
    if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete run: $run_dir" >&2
      exit 1
    fi
    mkdir -p "$run_dir"
    env \
      "CUDA_VISIBLE_DEVICES=${GPU}" \
      "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      "HF_HUB_OFFLINE=1" \
      "TRANSFORMERS_OFFLINE=1" \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/train/train_hybrid_lm.py \
      --config "$TRAIN_CONFIG" \
      --set "run.name=${run_name}" \
      --set "run.seed=${SEED}" \
      --set "run.output_dir=${run_dir}" \
      --set "run.append_timestamp=false" \
      --set "data.data_dir=${shards}" \
      --set "spatial_auxiliary.enabled=${aux_enabled}" \
      --set "spatial_auxiliary.objective=${aux_objective}" \
      --set "spatial_auxiliary.path=${aux_train}" \
      --set "spatial_auxiliary.eval_path=${aux_eval}" \
      --set "spatial_auxiliary.batch_size=${aux_batch}" \
      --set "spatial_auxiliary.every_causal_steps=${aux_every_causal}" \
      2>&1 | tee "$train_log"
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "Training finished without a final checkpoint: $checkpoint" >&2
    exit 1
  fi
  touch "${CURRENT_STATUS}.pretrained"

  # Co-primary benchmarks are intentionally first and reported separately.
  if find "$custom_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] custom benchmark already complete"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" --tasks entity_tracking \
      --eval-root "$CUSTOM_EVAL_ROOT" --batch-size 64 \
      --score-reduction both --trust-remote-code --local-files-only \
      --output-dir "$custom_output"
  fi
  touch "${CURRENT_STATUS}.custom_evaluated"

  if find "$official_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] official benchmark already complete"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" --tasks entity_tracking \
      --eval-root "$OFFICIAL_EVAL_ROOT" --batch-size 64 \
      --score-reduction both --trust-remote-code --local-files-only \
      --output-dir "$official_output"
  fi
  touch "${CURRENT_STATUS}.official_evaluated"

  if [[ -f "${multidomain_output}/pair_summary.csv" && -f "${multidomain_output}/five_choice_summary.csv" ]]; then
    echo "[skip] multidomain v3 evaluation already complete"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_entity_tracking_multidomain_hf.py \
      --model "$checkpoint" --data-root "$DATA_ROOT" \
      --batch-size 64 --score-reduction both --trust-remote-code \
      --local-files-only --output-dir "$multidomain_output"
  fi

  rm -f "${CURRENT_STATUS}.running"
  touch "${CURRENT_STATUS}.complete"
  echo "[complete] seed ${SEED} ${arm}: ${label}"
  CURRENT_STATUS=""
done

echo "GPU ${GPU}, seed ${SEED} v3 queue complete."
