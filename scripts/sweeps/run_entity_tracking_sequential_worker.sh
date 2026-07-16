#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:?Set BABYSTEPS_GPU to the physical GPU index.}"
EXPERIMENT_ID="${BABYSTEPS_EXPERIMENT_ID:?Set BABYSTEPS_EXPERIMENT_ID.}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-/home/jorge/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_ET_SEQUENTIAL_ROOT:-${RUNS_DIR}/experiments/entity_tracking_sequential_v1/${EXPERIMENT_ID}}"
REFERENCE_ROOT="${BABYSTEPS_ET_V3_REFERENCE_ROOT:-${RUNS_DIR}/experiments/entity_tracking_multidomain_v3/20260716_092451}"
TRAIN_CONFIG="configs/train/gpt_bert_entity_tracking_sequential_common_muon.yaml"
CACHE_DIR="${BABYSTEPS_CACHE_DIR:-/home/jorge/babylm-baby-steps-cache}"
MIDTRAIN_SHARDS="${CACHE_DIR}/token_shards/babylm2026_strict_small_midtrain17pct_container_state_multidomain_v3_gpt_bert_mixed"
FINETUNE_SHARDS="${CACHE_DIR}/token_shards/container_state_multidomain_v3_gpt_bert_mixed"
DATA_ROOT="data/container_state/container_state_multidomain_v3_seed42"
CUSTOM_EVAL_ROOT="data/container_state/container_state_v1_pool_seed42/validation"
OFFICIAL_EVAL_ROOT="external/babylm-eval/strict/evaluation_data/full_eval"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 SEED [SEED ...]" >&2
  exit 2
fi
for path in "$TRAIN_CONFIG" "$MIDTRAIN_SHARDS/meta.json" "$FINETUNE_SHARDS/meta.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required sequential input: $path" >&2
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

evaluate_stage() {
  local run_key="$1"
  local checkpoint="$2"
  local custom_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/01_custom_v1"
  local official_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/02_official"
  local multidomain_output="${EXPERIMENT_ROOT}/evaluation/${run_key}/03_multidomain_v3"

  if find "$custom_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] custom evaluation exists: $custom_output"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" --tasks entity_tracking \
      --eval-root "$CUSTOM_EVAL_ROOT" --batch-size 64 \
      --score-reduction both --trust-remote-code --local-files-only \
      --output-dir "$custom_output"
  fi

  if find "$official_output" -name likelihood_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] official evaluation exists: $official_output"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_mntp_likelihood_hf.py \
      --model "$checkpoint" --tasks entity_tracking \
      --eval-root "$OFFICIAL_EVAL_ROOT" --batch-size 64 \
      --score-reduction both --trust-remote-code --local-files-only \
      --output-dir "$official_output"
  fi

  if [[ -f "${multidomain_output}/pair_summary.csv" && -f "${multidomain_output}/five_choice_summary.csv" ]]; then
    echo "[skip] multidomain evaluation exists: $multidomain_output"
  else
    env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      conda run --no-capture-output -n "$CONDA_ENV" \
      python -u scripts/eval/eval_entity_tracking_multidomain_hf.py \
      --model "$checkpoint" --data-root "$DATA_ROOT" \
      --batch-size 64 --score-reduction both --trust-remote-code \
      --local-files-only --output-dir "$multidomain_output"
  fi
}

train_stage() {
  local seed="$1"
  local stage="$2"
  local init_from="$3"
  local shards="$4"
  local learning_rate="$5"
  local muon_lr="$6"
  local eval_every_steps="$7"
  local run_key="seed${seed}_${stage}"
  local run_dir="${EXPERIMENT_ROOT}/training/${run_key}"
  local checkpoint="${run_dir}/final"
  local train_log="${EXPERIMENT_ROOT}/logs/gpu${GPU}/${run_key}_train.log"

  CURRENT_STATUS="${EXPERIMENT_ROOT}/status/${run_key}"
  rm -f "${CURRENT_STATUS}.failed" "${CURRENT_STATUS}.complete"
  touch "${CURRENT_STATUS}.running"
  echo "=== seed ${seed} | ${stage} | GPU ${GPU} | init ${init_from} ==="

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
      --set "run.name=et_sequential_${EXPERIMENT_ID}_${run_key}" \
      --set "run.seed=${seed}" \
      --set "run.output_dir=${run_dir}" \
      --set "run.append_timestamp=false" \
      --set "model.init_from=${init_from}" \
      --set "data.data_dir=${shards}" \
      --set "training.learning_rate=${learning_rate}" \
      --set "training.muon_lr=${muon_lr}" \
      --set "logging.eval_every_steps=${eval_every_steps}" \
      2>&1 | tee "$train_log"
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "Training finished without a final checkpoint: $checkpoint" >&2
    exit 1
  fi
  touch "${CURRENT_STATUS}.trained"
  evaluate_stage "$run_key" "$checkpoint"
  rm -f "${CURRENT_STATUS}.running"
  touch "${CURRENT_STATUS}.complete"
  CURRENT_STATUS=""
}

for seed in "$@"; do
  baseline="${REFERENCE_ROOT}/training/seed${seed}_baseline_matched_local_baseline/final"
  if [[ ! -f "${baseline}/pytorch_model.bin" ]]; then
    echo "Missing matched baseline checkpoint for seed ${seed}: $baseline" >&2
    exit 1
  fi
  train_stage "$seed" midtrain "$baseline" "$MIDTRAIN_SHARDS" 0.0002 0.002 100
  midtrain_final="${EXPERIMENT_ROOT}/training/seed${seed}_midtrain/final"
  train_stage "$seed" finetune "$midtrain_final" "$FINETUNE_SHARDS" 0.0001 0.001 25
done

echo "GPU ${GPU} sequential queue complete."
