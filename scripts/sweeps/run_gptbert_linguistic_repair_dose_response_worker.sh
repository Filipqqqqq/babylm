#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:?Set BABYSTEPS_GPU to the physical GPU index.}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
EXPERIMENT_ROOT="${BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT:-${RUNS_DIR}/experiments/gpt_bert_linguistic_repair_dose_response_v1}"
TRAIN_CONFIG="${BABYSTEPS_TRAIN_CONFIG:-configs/experiments/linguistic_repair_dose_response_v1/train.yaml}"
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
FINETUNE_LRS=(${BABYSTEPS_SUPERGLUE_LRS:-3e-5 8e-5})
TASKS="${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 repair_0pct:42 [repair_1pct:43 ...]" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "Missing training config: $TRAIN_CONFIG" >&2
  exit 1
fi

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

CURRENT_STATUS_BASE=""
mark_queue_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_STATUS_BASE" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.running"
    touch "${CURRENT_STATUS_BASE}.failed"
  fi
  exit "$status"
}
trap mark_queue_exit EXIT

mkdir -p "${EXPERIMENT_ROOT}/logs/gpu${GPU}" "${EXPERIMENT_ROOT}/status"

for spec in "$@"; do
  condition="${spec%%:*}"
  pretrain_seed="${spec#*:}"
  if [[ "$condition" == "$pretrain_seed" || ! "$pretrain_seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid job '${spec}'. Use condition:seed." >&2
    exit 2
  fi

  shards_dir="${EXPERIMENT_ROOT}/data/token_shards/${condition}"
  pretrain_dir="${EXPERIMENT_ROOT}/pretraining/${condition}/seed_${pretrain_seed}"
  checkpoint="${pretrain_dir}/final"
  pretrain_log="${EXPERIMENT_ROOT}/logs/gpu${GPU}/${condition}_seed${pretrain_seed}_pretrain.log"
  CURRENT_STATUS_BASE="${EXPERIMENT_ROOT}/status/${condition}_seed_${pretrain_seed}"

  if [[ ! -f "${shards_dir}/condition_manifest.json" ]]; then
    echo "Missing validated condition shards: ${shards_dir}" >&2
    exit 1
  fi

  rm -f "${CURRENT_STATUS_BASE}.failed" "${CURRENT_STATUS_BASE}.complete"
  touch "${CURRENT_STATUS_BASE}.running"

  echo
  echo "=== Paired repair job: condition=${condition} pretrain_seed=${pretrain_seed} GPU=${GPU} ==="
  if [[ -d "$checkpoint" ]]; then
    echo "[skip] final checkpoint already exists: ${checkpoint}"
  else
    if [[ -d "$pretrain_dir" && -n "$(find "$pretrain_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete pretraining output: ${pretrain_dir}" >&2
      exit 1
    fi
    mkdir -p "$pretrain_dir"
    pretrain_command=(
      env
      "CUDA_VISIBLE_DEVICES=${GPU}"
      "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      "BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
      "BABYSTEPS_LINGUISTIC_REPAIR_SHARDS_DIR=${shards_dir}"
      conda run --no-capture-output -n "$CONDA_ENV"
      python -u scripts/train/train_hybrid_lm.py
      --config "$TRAIN_CONFIG"
      --set "run.name=gptbert_${condition}_seed${pretrain_seed}"
      --set "run.seed=${pretrain_seed}"
      --set "run.output_dir=${pretrain_dir}"
      --set "run.append_timestamp=false"
      --set "data.data_dir=${shards_dir}"
    )
    echo "[pretrain] log=${pretrain_log}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '+'
      printf ' %q' "${pretrain_command[@]}"
      printf ' 2>&1 | tee %q\n' "$pretrain_log"
    else
      "${pretrain_command[@]}" 2>&1 | tee "$pretrain_log"
    fi
  fi

  if [[ "$DRY_RUN" != "1" && ! -d "$checkpoint" ]]; then
    echo "Pretraining finished without final checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  touch "${CURRENT_STATUS_BASE}.pretrained"

  for lr in "${FINETUNE_LRS[@]}"; do
    lr_slug="$(slug "$lr")"
    model_name="gptbert_${condition}_ptseed${pretrain_seed}_ftseed${FINETUNE_SEED}_lr${lr_slug}"
    arm_root="${EXPERIMENT_ROOT}/finetuning/${condition}/pretrain_seed_${pretrain_seed}/lr_${lr}/finetune_seed_${FINETUNE_SEED}"
    arm_log_dir="${EXPERIMENT_ROOT}/logs/gpu${GPU}/finetuning/${condition}/pretrain_seed_${pretrain_seed}/lr_${lr}/finetune_seed_${FINETUNE_SEED}"
    summary_path="${arm_root}/${model_name}/results/superglue_summary.json"

    if [[ -f "$summary_path" ]]; then
      echo "[skip] completed SuperGLUE arm: ${summary_path}"
      continue
    fi
    echo "[finetune] condition=${condition} pretrain_seed=${pretrain_seed} lr=${lr}"
    run_command env \
      "BABYSTEPS_SUPERGLUE_GPU=${GPU}" \
      "BABYSTEPS_SUPERGLUE_LR=${lr}" \
      "BABYSTEPS_FINETUNE_SEED=${FINETUNE_SEED}" \
      "BABYSTEPS_SUPERGLUE_TASKS=${TASKS}" \
      "BABYSTEPS_SUPERGLUE_OUTPUT_ROOT=${arm_root}" \
      "BABYSTEPS_SUPERGLUE_LOG_DIR=${arm_log_dir}" \
      bash scripts/eval/eval_superglue_llama_nextlat_gpu3.sh \
      "${model_name}=${checkpoint}"
  done

  rm -f "${CURRENT_STATUS_BASE}.running"
  touch "${CURRENT_STATUS_BASE}.complete"
  echo "[complete] condition=${condition} pretrain_seed=${pretrain_seed}"
  CURRENT_STATUS_BASE=""
done

echo "GPU ${GPU} queue complete."
