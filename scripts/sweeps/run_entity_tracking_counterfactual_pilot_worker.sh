#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:?Set BABYSTEPS_GPU to the physical GPU index.}"
PILOT_ID="${BABYSTEPS_PILOT_ID:?Set BABYSTEPS_PILOT_ID.}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_PILOT_ROOT:-${RUNS_DIR}/experiments/entity_tracking_counterfactual_pilot_v2/${PILOT_ID}}"
TRAIN_CONFIG="configs/train/gpt_bert_entity_tracking_counterfactual_pilot_common_muon.yaml"
BASELINE_SHARDS="${BABYSTEPS_CACHE_DIR:-$HOME/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_local_control_gpt_bert_mixed"
COUNTERFACTUAL_SHARDS="${BABYSTEPS_CACHE_DIR:-$HOME/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_replace1pct_container_state_counterfactual_v2_gpt_bert_mixed"
PAIR_ROOT="data/container_state/container_state_counterfactual_v2_seed42/auxiliary"
CUSTOM_EVAL_ROOT="data/container_state/container_state_v1_pool_seed42/validation"
OFFICIAL_EVAL_ROOT="external/babylm-eval/strict/evaluation_data/full_eval"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 arm1 [arm3 ...]" >&2
  exit 2
fi
for path in "$TRAIN_CONFIG" "$BASELINE_SHARDS/meta.json" "$COUNTERFACTUAL_SHARDS/meta.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required pilot input: $path" >&2
    exit 1
  fi
done

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
    arm1)
      label="fresh_matched_local_baseline"
      shards="$BASELINE_SHARDS"
      aux_enabled="false"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/eval_completion_rank.jsonl"
      aux_batch=32
      ;;
    arm2)
      label="counterfactual_v2_lm_only"
      shards="$COUNTERFACTUAL_SHARDS"
      aux_enabled="false"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/eval_completion_rank.jsonl"
      aux_batch=32
      ;;
    arm3)
      label="counterfactual_v2_completion_rank"
      shards="$COUNTERFACTUAL_SHARDS"
      aux_enabled="true"
      aux_objective="completion_rank"
      aux_train="${PAIR_ROOT}/train_completion_rank.jsonl"
      aux_eval="${PAIR_ROOT}/eval_completion_rank.jsonl"
      aux_batch=32
      ;;
    arm4)
      label="counterfactual_v2_symmetric_rank"
      shards="$COUNTERFACTUAL_SHARDS"
      aux_enabled="true"
      aux_objective="counterfactual_rank"
      aux_train="${PAIR_ROOT}/train_counterfactual_rank.jsonl"
      aux_eval="${PAIR_ROOT}/eval_counterfactual_rank.jsonl"
      aux_batch=16
      ;;
    *)
      echo "Unknown pilot arm: $arm" >&2
      exit 2
      ;;
  esac

  run_name="et_cf_pilot_${PILOT_ID}_${arm}_${label}"
  run_dir="${EXPERIMENT_ROOT}/training/${arm}_${label}"
  checkpoint="${run_dir}/final"
  train_log="${EXPERIMENT_ROOT}/logs/gpu${GPU}/${arm}_${label}_train.log"
  custom_output="${EXPERIMENT_ROOT}/evaluation/${arm}_${label}/01_custom_v1"
  official_output="${EXPERIMENT_ROOT}/evaluation/${arm}_${label}/02_official"
  CURRENT_STATUS="${EXPERIMENT_ROOT}/status/${arm}_${label}"
  rm -f "${CURRENT_STATUS}.failed" "${CURRENT_STATUS}.complete"
  touch "${CURRENT_STATUS}.running"

  echo
  echo "=== ${arm}: ${label} | GPU ${GPU} | pilot ${PILOT_ID} ==="
  if [[ -d "$checkpoint" ]]; then
    echo "[skip] final checkpoint exists: $checkpoint"
  else
    if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete run: $run_dir" >&2
      exit 1
    fi
    mkdir -p "$run_dir"
    echo "[train] ${run_name}; log=${train_log}"
    env \
      "CUDA_VISIBLE_DEVICES=${GPU}" \
      "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      "HF_HUB_OFFLINE=1" \
      "TRANSFORMERS_OFFLINE=1" \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/train/train_hybrid_lm.py \
      --config "$TRAIN_CONFIG" \
      --set "run.name=${run_name}" \
      --set "run.output_dir=${run_dir}" \
      --set "run.append_timestamp=false" \
      --set "data.data_dir=${shards}" \
      --set "spatial_auxiliary.enabled=${aux_enabled}" \
      --set "spatial_auxiliary.objective=${aux_objective}" \
      --set "spatial_auxiliary.path=${aux_train}" \
      --set "spatial_auxiliary.eval_path=${aux_eval}" \
      --set "spatial_auxiliary.batch_size=${aux_batch}" \
      2>&1 | tee "$train_log"
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "Training finished without a final checkpoint: $checkpoint" >&2
    exit 1
  fi
  touch "${CURRENT_STATUS}.pretrained"

  # The custom benchmark is intentionally evaluated first and is co-primary.
  if find "$custom_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] custom benchmark already complete: $custom_output"
  else
    echo "[eval 1/2] custom 4,000-item benchmark"
    env \
      "CUDA_VISIBLE_DEVICES=${GPU}" \
      "HF_HUB_OFFLINE=1" \
      "TRANSFORMERS_OFFLINE=1" \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" \
      --tasks entity_tracking \
      --eval-root "$CUSTOM_EVAL_ROOT" \
      --batch-size 64 \
      --score-reduction both \
      --trust-remote-code \
      --local-files-only \
      --output-dir "$custom_output"
  fi
  touch "${CURRENT_STATUS}.custom_evaluated"

  if find "$official_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] official benchmark already complete: $official_output"
  else
    echo "[eval 2/2] official Entity Tracking benchmark"
    env \
      "CUDA_VISIBLE_DEVICES=${GPU}" \
      "HF_HUB_OFFLINE=1" \
      "TRANSFORMERS_OFFLINE=1" \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" \
      --tasks entity_tracking \
      --eval-root "$OFFICIAL_EVAL_ROOT" \
      --batch-size 64 \
      --score-reduction both \
      --trust-remote-code \
      --local-files-only \
      --output-dir "$official_output"
  fi

  rm -f "${CURRENT_STATUS}.running"
  touch "${CURRENT_STATUS}.complete"
  echo "[complete] ${arm}: ${label}"
  CURRENT_STATUS=""
done

echo "GPU ${GPU} pilot queue complete."
