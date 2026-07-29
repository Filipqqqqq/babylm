#!/usr/bin/env bash
set -euo pipefail

# Evaluate a complete Hub-hosted BabyLM Strict (100M-data) submission.
# Intermediate checkpoints are resolved as Hub revisions through chck_1000M,
# while the final model is read from main.

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 HUB_REPO_ID [FINETUNE_LR] [BACKEND]" >&2
  exit 2
fi

MODEL_REPO="$1"
FINETUNE_LR="${2:-${BABYSTEPS_SUPERGLUE_LR:-8e-5}}"
BACKEND="${3:-${BABYSTEPS_EVAL_BACKEND:-causal}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STRICT_DIR="${ROOT}/external/babylm-eval/strict"

case "$BACKEND" in
  causal|mlm|mntp|enc_dec_mask|enc_dec_prefix) ;;
  *)
    echo "Unsupported evaluation backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ ! -d "$STRICT_DIR" ]]; then
  echo "Missing official BabyLM strict evaluator: $STRICT_DIR" >&2
  exit 1
fi

cd "$STRICT_DIR"
echo "Evaluating BabyLM Strict submission: ${MODEL_REPO}"
echo "Fine-tuning LR: ${FINETUNE_LR}"
echo "Zero-shot backend: ${BACKEND}"

# Propagate errexit into the official scripts and the child bash processes they
# launch. Several upstream scripts do not set it themselves, which can
# otherwise hide an earlier failed task behind a later successful one.
export SHELLOPTS

# main: full zero-shot and final SuperGLUE.
bash -e scripts/eval_zero_shot.sh "$MODEL_REPO" "$BACKEND"
bash -e scripts/eval_finetuning.sh --model_path "$MODEL_REPO" --lr "$FINETUNE_LR" --seed 42

# All 28 official checkpoint revisions through chck_1000M: fast zero-shot,
# both GlobalPIQA splits, and AoA.
bash -e scripts/eval_zero_shot_fast_all_revisions.sh "$MODEL_REPO" "$BACKEND" strict
bash -e scripts/eval_zero_shot_global_piqa.sh "$MODEL_REPO" "$BACKEND" strict
# AoA calls the 2025 small-track utility, where `non-strict-small` selects the
# same 28-revision schedule used by the 100M Strict track. That utility does not
# accept the top-level evaluator's `strict` spelling.
bash -e scripts/eval_aoa.sh "$MODEL_REPO" "$BACKEND" non-strict-small

# Write results/<repo-name>/all_full_preds_and_fast_scores_causal.json.
bash -e scripts/collate_preds.sh "${MODEL_REPO##*/}" "$BACKEND" strict

COLLATED_PATH="results/${MODEL_REPO##*/}/all_full_preds_and_fast_scores_${BACKEND}.json"
if [[ ! -s "$COLLATED_PATH" ]]; then
  echo "Missing collated submission results: ${STRICT_DIR}/${COLLATED_PATH}" >&2
  exit 1
fi
echo "Collated submission results: ${STRICT_DIR}/${COLLATED_PATH}"
