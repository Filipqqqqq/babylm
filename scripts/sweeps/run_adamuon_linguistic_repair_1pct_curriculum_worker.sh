#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

CONDA_ENV="${BABYSTEPS_CONDA_ENV:-babylm}"
GPU="${BABYSTEPS_GPU:-0}"
RUNS_DIR="${BABYSTEPS_RUNS_DIR:-$HOME/babylm-baby-steps-runs}"
CONFIG="${BABYSTEPS_CONFIG:-configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_adamuon_lr0p02_aux0p006_tailavg_seed42.yaml}"
REPAIR_EXPERIMENT_ROOT="${BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT:-${RUNS_DIR}/experiments/gpt_bert_linguistic_repair_dose_response_v1}"
SHARDS_DIR="${BABYSTEPS_LINGUISTIC_REPAIR_SHARDS_DIR:-${REPAIR_EXPERIMENT_ROOT}/data/token_shards/repair_1pct}"
CAMPAIGN_ROOT="${BABYSTEPS_CAMPAIGN_ROOT:-${RUNS_DIR}/experiments/adamuon_linguistic_repair_1pct_curriculum_superglue}"
EVAL_SCRIPT="scripts/eval/eval_superglue_llama_nextlat_gpu3.sh"
FINETUNE_SEED="${BABYSTEPS_FINETUNE_SEED:-42}"
FINETUNE_LRS=(${BABYSTEPS_SUPERGLUE_LRS:-3e-5 8e-5})
TASKS="${BABYSTEPS_SUPERGLUE_TASKS:-boolq multirc rte wsc mrpc qqp mnli}"
DRY_RUN="${BABYSTEPS_DRY_RUN:-0}"
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 43)
fi

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "BABYSTEPS_GPU must be a non-negative physical GPU index; got GPU=${GPU}." >&2
  exit 2
fi
for path in "$CONFIG" "$EVAL_SCRIPT" "${SHARDS_DIR}/meta.json" "${SHARDS_DIR}/condition_manifest.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid pretraining seed: $seed" >&2
    exit 2
  fi
done

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

CURRENT_STATUS_BASE=""
mark_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_STATUS_BASE" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.running"
    touch "${CURRENT_STATUS_BASE}.failed"
  fi
  exit "$status"
}
trap mark_exit EXIT

mkdir -p "${CAMPAIGN_ROOT}/logs/gpu${GPU}" "${CAMPAIGN_ROOT}/status"

echo "AdamMuon 1% linguistic-repair curriculum + SuperGLUE campaign"
echo "  physical GPU: ${GPU}"
echo "  pretraining seeds: ${SEEDS[*]}"
echo "  data: ${SHARDS_DIR}"
echo "  recipe: AdamMuon 0.02 / auxiliary AdamW 0.006"
echo "  schedule: 1.6% LR warmup; sequence lengths 128 -> 256 -> 512"
echo "  pretraining eval: validation loss once, after epoch 10"
echo "  disabled: startup/periodic benchmarks, plots, and epoch checkpoints"
echo "  objective mix: 1 causal step : 1 masked step"
echo "  evaluated checkpoint: final-20% tail average"
echo "  SuperGLUE: ${TASKS}; LRs ${FINETUNE_LRS[*]}; fine-tuning seed ${FINETUNE_SEED}"

for seed in "${SEEDS[@]}"; do
  run_name="adamuon-10m-mixed-linguistic-repair1pct-lr0.02-aux0.006-s${seed}-curriculum-tailavg"
  run_dir="${RUNS_DIR}/training/${run_name}"
  final_checkpoint="${run_dir}/final"
  eval_checkpoint="${run_dir}/model_tail_avg"
  train_log="${CAMPAIGN_ROOT}/logs/gpu${GPU}/pretrain_seed${seed}.log"
  CURRENT_STATUS_BASE="${CAMPAIGN_ROOT}/status/seed_${seed}"

  if [[ "$DRY_RUN" != "1" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.failed" "${CURRENT_STATUS_BASE}.complete"
    touch "${CURRENT_STATUS_BASE}.running"
  fi

  echo
  echo "=== Pretraining seed ${seed} ==="
  if [[ -f "${eval_checkpoint}/pytorch_model.bin" && -f "${final_checkpoint}/pytorch_model.bin" ]]; then
    echo "[skip] completed checkpoints already exist: ${run_dir}"
  else
    if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete non-empty run directory: ${run_dir}" >&2
      exit 1
    fi
    command=(
      env
      "CUDA_VISIBLE_DEVICES=${GPU}"
      "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      conda run --no-capture-output -n "$CONDA_ENV"
      python -u scripts/train/train_hybrid_lm.py
      --config "$CONFIG"
      --set "run.name=${run_name}"
      --set "run.seed=${seed}"
      --set "run.output_dir=${run_dir}"
      --set "run.append_timestamp=false"
      --set "data.data_dir=${SHARDS_DIR}"
      --set "logging.eval_every_steps=0"
      --set "logging.eval_at_end=true"
      --set "logging.benchmark_eval_mode=final_only"
      --set "ewok_context_sensitivity_full.plot=false"
      --set "mntp_likelihood.plot=false"
      --set "loss_dynamics.plot=false"
      --set "checkpointing.save_every_epochs=0"
    )
    echo "  output: ${run_dir}"
    echo "  log: ${train_log}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '+'
      printf ' %q' "${command[@]}"
      printf ' 2>&1 | tee -a %q\n' "$train_log"
    else
      "${command[@]}" 2>&1 | tee -a "$train_log"
    fi
  fi

  if [[ "$DRY_RUN" != "1" && ! -f "${eval_checkpoint}/pytorch_model.bin" ]]; then
    echo "Training exited without the tail-averaged checkpoint: ${eval_checkpoint}" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    touch "${CURRENT_STATUS_BASE}.pretrained"
  fi

  for lr in "${FINETUNE_LRS[@]}"; do
    lr_slug="$(slug "$lr")"
    model_name="gptbert_adamuon_repair1pct_curriculum_ptseed${seed}_tailavg_ftseed${FINETUNE_SEED}_lr${lr_slug}"
    arm_root="${CAMPAIGN_ROOT}/finetuning/pretrain_seed_${seed}/checkpoint_tailavg/lr_${lr}/finetune_seed_${FINETUNE_SEED}"
    arm_logs="${CAMPAIGN_ROOT}/logs/gpu${GPU}/finetuning/pretrain_seed_${seed}/checkpoint_tailavg/lr_${lr}/finetune_seed_${FINETUNE_SEED}"
    summary="${arm_root}/${model_name}/results/superglue_summary.json"

    if [[ -f "$summary" ]]; then
      echo "[skip] completed SuperGLUE arm: ${summary}"
      continue
    fi
    echo "[SuperGLUE] pretraining seed=${seed} checkpoint=tailavg lr=${lr}"
    eval_command=(
      env
      "BABYSTEPS_SUPERGLUE_GPU=${GPU}"
      "BABYSTEPS_SUPERGLUE_LR=${lr}"
      "BABYSTEPS_FINETUNE_SEED=${FINETUNE_SEED}"
      "BABYSTEPS_SUPERGLUE_TASKS=${TASKS}"
      "BABYSTEPS_SUPERGLUE_OUTPUT_ROOT=${arm_root}"
      "BABYSTEPS_SUPERGLUE_LOG_DIR=${arm_logs}"
      "BABYSTEPS_DRY_RUN=${DRY_RUN}"
      bash "$EVAL_SCRIPT"
      "${model_name}=${eval_checkpoint}"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '+'
      printf ' %q' "${eval_command[@]}"
      printf '\n'
    else
      "${eval_command[@]}"
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    rm -f "${CURRENT_STATUS_BASE}.running"
    touch "${CURRENT_STATUS_BASE}.complete"
  fi
  CURRENT_STATUS_BASE=""
  echo "[complete] pretraining seed ${seed} and all SuperGLUE arms"
done

echo "GPU ${GPU} campaign complete."
